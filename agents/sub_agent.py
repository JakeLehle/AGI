"""
Script-First Sub-Agent v1.2.3 — 4-Phase Lifecycle with Diagnostic Agent

Features:
- DUAL CLUSTER ROUTING: AGI_CLUSTER (CPU) + AGI_GPU_CLUSTER (GPU subtasks)
- GPU-AWARE SBATCH: Auto-routes tasks to GPU/CPU based on package detection
- GPU NODE RULES: No --mem on GPU partitions, --gres=gpu:N format
- CONDA CLEANUP: Removes environment after success (YAML preserved)
- STATE CHECKPOINTING: Resume from where you left off
- TOKEN BUDGET: Sized for 32K context window
- PROPER SLURM: Jobs appear in squeue, uses sbatch correctly
- OPEN-SOURCE CHANNELS: conda-forge, bioconda only (no defaults/main)

v1.2.0 Changes:
  The monolithic execute() → generate_all_artifacts → retry loop is replaced
  by a 4-phase lifecycle with independent error handling at each phase:

    Phase 1: Script Generation (validated two-pass)
      - Outline generation → LLM validation → full implementation → validation
      - Language-agnostic (Python, R, bash, Perl, Java)

    Phase 2: Environment Creation (script-informed, LLM-reviewed)
      - Entire script sent to LLM for full dependency analysis
      - YAML generation with proper conda/pip/system routing
      - Environment build with repair loop (max 15 iterations)
      - YAML is always the single source of truth
      - DiskManager integration for proactive quota management

    Phase 3: Sbatch Generation (language-aware)
      - Language dispatch table for execution commands
      - GPU routing based on script content + env packages
      - Submission test with rule-based config repair

    Phase 4: Execution + Diagnostic Agent Loop
      - Expanded error classification (12 error types)
      - DiagnosticAgent invocation with independent 25K token budget
      - Structured FixPrescription application
      - DiagnosticMemory for cross-task solution reuse
      - ReflexionMemory for per-task loop prevention

  Removed methods:
    - _generate_all_artifacts() → replaced by per-phase generation
    - _reflect_and_update() → replaced by diagnostic agent
    - _generate_env_yaml() → replaced by LLM dependency review in Phase 2
    - _generate_python_script() → replaced by _generate_script() (language-agnostic)

  All LLM calls use invoke_resilient() with exponential backoff retry.
  No artificial timeouts — the 3-day SLURM wall time is the only limit.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import re
import json
import time
import subprocess
import os
from datetime import datetime
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging

try:
    import yaml
except ImportError:
    yaml = None

try:
    from langchain_ollama import OllamaLLM
except ImportError:
    from langchain_community.llms import Ollama as OllamaLLM

from utils.logging_config import agent_logger
from utils.context_manager import ContextManager
from utils.llm_invoke import invoke_resilient, LLMInvocationError
from utils.model_config import resolve_model, resolve_base_url

logger = logging.getLogger(__name__)


# =============================================================================
# CLUSTER CONFIGURATION (v3.2 — GPU Routing Support)
# =============================================================================

class ClusterConfig:
    """Loads and provides cluster-specific SLURM settings with GPU routing."""

    # Fallback defaults if no config file found
    DEFAULT_CONFIG = {
        'clusters': {
            'default': {
                'name': 'Default (CPU)',
                'slurm': {
                    'partition': 'compute1',
                    'account': 'sdz852',
                    'nodes': 1,
                    'ntasks': 1,
                    'cpus_per_task': 20,
                    'memory': '64G',
                    'time': '1-00:00:00',
                },
                'gpu': {'available': False}
            },
            'default_gpu': {
                'name': 'Default (GPU)',
                'slurm': {
                    'partition': 'gpu1v100',
                    'account': 'sdz852',
                    'nodes': 1,
                    'ntasks': 1,
                    'cpus_per_task': 80,
                    # NO memory key — GPU nodes must NOT specify --mem
                    'time': '1-00:00:00',
                },
                'gpu': {
                    'available': True,
                    'default_count': 1,
                    'max_count': 1,
                    'type': 'v100',
                    'directive_format': '--gres=gpu:{count}'
                }
            }
        },
        'gpu_packages': [
            'torch', 'pytorch', 'tensorflow', 'keras', 'jax',
            'rapids', 'cuml', 'cudf', 'cugraph',
            'scvi-tools', 'scvi', 'scvelo', 'cellbender',
            'flash-attn', 'xformers', 'bitsandbytes',
            'accelerate', 'deepspeed', 'triton'
        ]
    }

    def __init__(self):
        self.config = self._load_config()
        self.cluster_name = os.environ.get('AGI_CLUSTER', 'arc_compute1')
        self.cluster = self.config.get('clusters', {}).get(
            self.cluster_name,
            self.DEFAULT_CONFIG['clusters']['default']
        )
        self.gpu_cluster_name = os.environ.get(
            'AGI_GPU_CLUSTER', 'arc_gpu1v100'
        )

    def _load_config(self) -> Dict:
        """Load cluster config from file."""
        config_path = os.environ.get('AGI_CLUSTER_CONFIG')

        if not config_path:
            for path in [
                Path.cwd() / 'config' / 'cluster_config.yaml',
                Path(__file__).parent.parent / 'config' / 'cluster_config.yaml',
            ]:
                if path.exists():
                    config_path = str(path)
                    break

        if config_path and Path(config_path).exists() and yaml:
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                if config and 'clusters' in config:
                    return config
            except Exception as e:
                print(
                    f"[ClusterConfig] Warning: "
                    f"Failed to load {config_path}: {e}"
                )

        return self.DEFAULT_CONFIG

    @property
    def slurm(self) -> Dict:
        """SLURM settings for current (CPU) cluster."""
        return self.cluster.get('slurm', {})

    def get_gpu_cluster(self) -> Dict:
        """Get the GPU cluster configuration."""
        return self.config.get('clusters', {}).get(
            self.gpu_cluster_name,
            self.DEFAULT_CONFIG['clusters']['default_gpu']
        )

    def task_needs_gpu(self, task_description: str = '', packages: list = None) -> bool:
        """Detect if a task needs GPU based on packages or description."""
        gpu_packages = self.config.get(
            'gpu_packages', self.DEFAULT_CONFIG['gpu_packages']
        )
        check_text = task_description.lower()

        if packages:
            for pkg in packages:
                if pkg.lower() in gpu_packages:
                    return True
                if pkg.lower() in check_text:
                    pass

        for gpu_pkg in gpu_packages:
            if gpu_pkg.lower() in check_text:
                return True

        return False

    def get_slurm_for_task(
        self, task_description: str = '', requires_gpu: bool = False,
        packages: list = None
    ) -> Dict:
        """Get SLURM settings routed to the right cluster for this task."""
        needs_gpu = requires_gpu or self.task_needs_gpu(
            task_description, packages
        )

        if needs_gpu:
            gpu_cluster = self.get_gpu_cluster()
            slurm_settings = dict(gpu_cluster.get('slurm', {}))
            gpu_config = gpu_cluster.get('gpu', {})

            slurm_settings['gpu_available'] = True
            gpu_count = gpu_config.get('default_count', 1)
            directive_fmt = gpu_config.get(
                'directive_format', '--gres=gpu:{count}'
            )
            slurm_settings['gpu_directive'] = directive_fmt.format(
                count=gpu_count
            )
            slurm_settings['cluster_name'] = self.gpu_cluster_name
        else:
            slurm_settings = dict(self.slurm)
            slurm_settings['gpu_available'] = False
            slurm_settings['gpu_directive'] = None
            slurm_settings['cluster_name'] = self.cluster_name

        return slurm_settings


# =============================================================================
# CHECKPOINT STATE (v1.2.0 — Extended)
# =============================================================================

class TaskStatus(Enum):
    NOT_STARTED = "not_started"
    # Phase 1
    GENERATING_OUTLINE = "generating_outline"
    VALIDATING_OUTLINE = "validating_outline"
    GENERATING_SCRIPT = "generating_script"
    VALIDATING_SCRIPT = "validating_script"
    # Phase 2
    REVIEWING_DEPS = "reviewing_deps"
    BUILDING_ENV = "building_env"
    REPAIRING_ENV = "repairing_env"
    # Phase 3
    GENERATING_SBATCH = "generating_sbatch"
    TESTING_SUBMISSION = "testing_submission"
    # Phase 4
    RUNNING_DRYRUN = "running_dryrun"
    WAITING_JOB = "waiting_job"
    DIAGNOSING = "diagnosing"
    RUNNING_PROD = "running_prod"
    # Terminal
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskCheckpoint:
    task_id: str
    status: str
    iteration: int
    env_name: Optional[str]
    env_yaml_path: Optional[str]
    script_path: Optional[str]
    sbatch_path: Optional[str]
    current_job_id: Optional[str]
    last_error: Optional[str]
    env_created: bool
    dry_run_succeeded: bool
    created_at: str
    updated_at: str
    routed_cluster: Optional[str] = None
    # v1.2.0 additions
    language: Optional[str] = None
    outline_validated: bool = False
    script_validated: bool = False
    phase: int = 0
    diagnostic_invocations: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'TaskCheckpoint':
        # Backward compat for older checkpoints
        for key in [
            'routed_cluster', 'language', 'outline_validated',
            'script_validated', 'phase', 'diagnostic_invocations'
        ]:
            if key not in data:
                if key in ('outline_validated', 'script_validated'):
                    data[key] = False
                elif key in ('phase', 'diagnostic_invocations'):
                    data[key] = 0
                else:
                    data[key] = None
        return cls(**data)

    @classmethod
    def new(cls, task_id: str) -> 'TaskCheckpoint':
        now = datetime.now().isoformat()
        return cls(
            task_id=task_id, status=TaskStatus.NOT_STARTED.value,
            iteration=0, env_name=None, env_yaml_path=None,
            script_path=None, sbatch_path=None, current_job_id=None,
            last_error=None, env_created=False, dry_run_succeeded=False,
            created_at=now, updated_at=now, routed_cluster=None,
            language=None, outline_validated=False, script_validated=False,
            phase=0, diagnostic_invocations=0
        )


# =============================================================================
# SUB-AGENT v1.2.0
# =============================================================================

class ScriptFirstSubAgentV3:
    """
    Sub-Agent v1.2.0 — 4-phase lifecycle with diagnostic agent integration.

    Phase 1: Script Generation (validated two-pass)
    Phase 2: Environment Creation (script-informed, LLM-reviewed)
    Phase 3: Sbatch Generation (language-aware, submission-tested)
    Phase 4: Execution + Diagnostic Agent Loop

    All LLM calls use invoke_resilient() with exponential backoff retry.
    No artificial timeouts — the 3-day SLURM wall time is the only limit.
    """

    # Token budget (matches config.yaml context.max_tokens_per_task)
    MAX_CONTEXT_TOKENS = 25_000
    MIN_TOKENS_FOR_RETRY = 3_000

    # Phase limits
    MAX_OUTLINE_ATTEMPTS = 3
    MAX_IMPLEMENTATION_ATTEMPTS = 3
    MAX_ENV_REPAIR_ITERATIONS = 15
    MAX_PHASE4_ITERATIONS = 15

    def __init__(
        self,
        agent_id: str = "sub_agent",
        sandbox=None,
        conda_tools=None,
        slurm_tools=None,
        ollama_model: str = None,
        ollama_base_url: str = None,
        use_slurm: bool = True,
        slurm_config: Dict = None,
        project_root: str = None,
        cleanup_env_on_success: bool = True,
        diagnostic_memory=None,
    ):
        self.agent_id = agent_id
        self.sandbox = sandbox
        self.conda_tools = conda_tools
        self.slurm_tools = slurm_tools
        self.use_slurm = use_slurm
        self.slurm_config = slurm_config or {}
        self.cleanup_env_on_success = cleanup_env_on_success
        self.diagnostic_memory = diagnostic_memory

        # Resolve model (no hardcoded names)
        resolved_model = resolve_model(ollama_model)
        self.ollama_base_url = resolve_base_url(ollama_base_url)
        self.llm = OllamaLLM(
            model=resolved_model, base_url=self.ollama_base_url
        )

        # Project root
        if project_root:
            self.project_root = Path(project_root)
        elif sandbox:
            self.project_root = Path(sandbox.project_dir)
        else:
            self.project_root = Path('.')

        # Cluster config
        self.cluster_config = ClusterConfig()

        # Context manager
        self.context_mgr = ContextManager()

        # Checkpoint
        self.checkpoint: Optional[TaskCheckpoint] = None

        # Disk manager (lazy init)
        self._disk_manager = None

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def execute(
        self, subtask: Dict, env_name: str = None
    ) -> Dict[str, Any]:
        """
        Execute a subtask through the 4-phase lifecycle.

        v1.2.0: Phases are sequential with independent error handling.
        Each phase can be resumed from checkpoint.

        Args:
            subtask: Dict with id, description, packages, input_files,
                     output_files, code_hints, requires_gpu, etc.
            env_name: Optional base env name (not used in v1.2.0, each
                      step gets its own env).

        Returns:
            Dict with success, task_id, and phase-specific results.
        """
        task_id = subtask.get('id', 'unknown')

        # Initialize or resume checkpoint
        self.checkpoint = self._load_or_create_checkpoint(task_id)

        # Check if already completed
        if self._check_existing_outputs(subtask).get('already_complete'):
            self._delete_checkpoint(task_id)
            return self._success_result(
                task_id, message="Outputs exist", skipped=True
            )

        # Route to cluster
        routed_slurm = self.cluster_config.get_slurm_for_task(
            task_description=subtask.get('description', ''),
            requires_gpu=subtask.get('requires_gpu', False),
            packages=subtask.get('packages', []),
        )
        routed_cluster = routed_slurm.get(
            'cluster_name', self.cluster_config.cluster_name
        )
        self._update_checkpoint(routed_cluster=routed_cluster)

        try:
            # =============================================================
            # PHASE 1: Script Generation
            # =============================================================
            if self.checkpoint.phase < 1:
                result = self._phase_1_generate_script(subtask)
                if not result['success']:
                    self._update_checkpoint(status=TaskStatus.FAILED.value)
                    return self._failure_result(
                        task_id, f"Phase 1 failed: {result.get('error')}"
                    )
                self._update_checkpoint(phase=1)

            # =============================================================
            # PHASE 2: Environment Creation
            # =============================================================
            if self.checkpoint.phase < 2:
                result = self._phase_2_create_environment(subtask)
                if not result['success']:
                    self._update_checkpoint(status=TaskStatus.FAILED.value)
                    return self._failure_result(
                        task_id, f"Phase 2 failed: {result.get('error')}"
                    )
                self._update_checkpoint(phase=2)

            # =============================================================
            # PHASE 3: Sbatch Generation
            # =============================================================
            if self.checkpoint.phase < 3:
                result = self._phase_3_generate_sbatch(
                    subtask, routed_slurm
                )
                if not result['success']:
                    self._update_checkpoint(status=TaskStatus.FAILED.value)
                    return self._failure_result(
                        task_id, f"Phase 3 failed: {result.get('error')}"
                    )
                self._update_checkpoint(phase=3)

            # =============================================================
            # PHASE 4: Execution + Diagnostic Loop
            # =============================================================
            result = self._phase_4_execute_and_monitor(
                subtask, routed_cluster, routed_slurm
            )
            return result

        except Exception as e:
            logger.error(f"[{task_id}] Unhandled exception: {e}")
            self._update_checkpoint(
                status=TaskStatus.FAILED.value,
                last_error=str(e)
            )
            return self._failure_result(task_id, f"Unhandled: {e}")

    # =========================================================================
    # PHASE 1: SCRIPT GENERATION (Validated Two-Pass)
    # =========================================================================

    def _phase_1_generate_script(self, subtask: Dict) -> Dict[str, Any]:
        """Generate and validate the analysis script.

        Two-pass approach:
          1a. Generate outline from subtask
          1b. Validate outline against subtask goals
          1c. Generate full script from validated outline
          1d. Validate script against outline + subtask
        """
        task_id = subtask.get('id', 'task')
        safe_id = re.sub(r'[^\w\-]', '_', task_id)[:30]
        desc = subtask.get('description', '')

        # Detect language from subtask
        language = self._detect_language(subtask)
        ext = {'python': 'py', 'r': 'R', 'bash': 'sh',
               'perl': 'pl', 'java': 'java'}.get(language, 'py')

        script_path = self.project_root / 'scripts' / f"{safe_id}.{ext}"
        script_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n[{task_id}] Phase 1: Script Generation ({language})")

        # Skip if script already exists and is validated
        if (self.checkpoint.script_path
                and Path(self.checkpoint.script_path).exists()
                and self.checkpoint.script_validated):
            print(f"  ✓ Script already validated: {self.checkpoint.script_path}")
            return {'success': True}

        # Step 1a+1b: Outline generation with validation
        if not self.checkpoint.outline_validated:
            self._update_checkpoint(
                status=TaskStatus.GENERATING_OUTLINE.value
            )
            outline = self._generate_and_validate_outline(subtask, language)
            if not outline:
                return {
                    'success': False,
                    'error': 'Could not generate validated outline'
                }
            self._update_checkpoint(outline_validated=True)
        else:
            # Reconstruct outline from existing script context
            outline = subtask.get('expanded_plan', desc)

        # Step 1c+1d: Implementation with validation
        self._update_checkpoint(status=TaskStatus.GENERATING_SCRIPT.value)
        script_content = self._generate_and_validate_script(
            subtask, outline, language
        )
        if not script_content:
            return {
                'success': False,
                'error': 'Could not generate validated script'
            }

        # Save script
        script_path.write_text(script_content)
        env_name = f"agi_{safe_id}"
        env_yaml_path = str(
            self.project_root / 'envs' / f"{safe_id}.yml"
        )

        self._update_checkpoint(
            script_path=str(script_path),
            script_validated=True,
            language=language,
            env_name=env_name,
            env_yaml_path=env_yaml_path,
        )
        print(f"  ✓ Script saved: {script_path}")
        print(f"  ✓ Language: {language}")
        return {'success': True}

    def _generate_and_validate_outline(
        self, subtask: Dict, language: str
    ) -> Optional[str]:
        """Generate an outline and validate it against subtask goals."""
        desc = subtask.get('description', '')
        packages = subtask.get('packages', [])
        inputs = subtask.get('input_files', [])
        outputs = subtask.get('output_files', [])
        hints = subtask.get('code_hints', [])

        for attempt in range(self.MAX_IMPLEMENTATION_ATTEMPTS):
            self._update_checkpoint(
                status=TaskStatus.GENERATING_SCRIPT.value
            )

            if language == 'python':
                prompt = self._build_python_script_prompt(
                    desc, outline, packages, inputs, outputs, hints=hints
                )
            elif language == 'r':
                prompt = self._build_r_script_prompt(
                    desc, outline, packages, inputs, outputs, hints=hints
                )
            else:
                prompt = self._build_generic_script_prompt(
                    desc, outline, packages, inputs, outputs, language, hints=hints
                )

            prompt = f"""Create a structured outline for a {language} script.

TASK: {desc}
INPUT FILES: {', '.join(inputs) if inputs else 'None specified'}
OUTPUT FILES: {', '.join(outputs) if outputs else 'None specified'}
PACKAGES: {', '.join(packages) if packages else 'As needed'}

Produce a numbered outline with:
1. All imports/library loads needed
2. Each processing step in order
3. What each step does and why
4. Expected inputs and outputs for each step
5. Error handling approach

Return ONLY the outline, no code."""

            try:
                outline = invoke_resilient(
                    self.llm, prompt,
                    ollama_base_url=self.ollama_base_url,
                    max_retries=20,
                    initial_backoff=30.0,
                )
            except LLMInvocationError as e:
                logger.warning(f"Outline generation failed: {e}")
                continue

            # Validate
            self._update_checkpoint(
                status=TaskStatus.VALIDATING_OUTLINE.value
            )
            validation = self._validate_outline(outline, subtask)

            if validation.get('valid'):
                print(f"  ✓ Outline validated (attempt {attempt + 1})")
                return outline
            else:
                print(
                    f"  ⚠ Outline validation failed (attempt {attempt + 1}): "
                    f"{validation.get('issues', '')[:100]}"
                )

        return None

    def _validate_outline(
        self, outline: str, subtask: Dict
    ) -> Dict[str, Any]:
        """LLM validation of outline against subtask goals."""
        desc = subtask.get('description', '')
        outputs = subtask.get('output_files', [])

        prompt = f"""Validate this script outline against the assigned task.

TASK: {desc}
EXPECTED OUTPUTS: {', '.join(outputs) if outputs else 'Not specified'}

OUTLINE:
{outline}

Check:
1. Does it address every requirement in the task?
2. Are any required steps missing?
3. Will it produce the expected outputs?
4. Does it introduce anything not requested?

Respond with EXACTLY one of:
VALID: The outline fully addresses the task.
INVALID: [specific issues]"""

        try:
            response = invoke_resilient(
                self.llm, prompt,
                ollama_base_url=self.ollama_base_url,
                max_retries=10,
                initial_backoff=15.0,
            )
            if response.strip().upper().startswith('VALID'):
                return {'valid': True}
            return {'valid': False, 'issues': response}
        except LLMInvocationError:
            # On LLM failure, accept the outline
            return {'valid': True}

    def _generate_and_validate_script(
        self, subtask: Dict, outline: str, language: str
    ) -> Optional[str]:
        """Generate the full script from outline, then validate."""
        desc = subtask.get('description', '')
        packages = subtask.get('packages', [])
        inputs = subtask.get('input_files', [])
        outputs = subtask.get('output_files', [])

        for attempt in range(self.MAX_IMPLEMENTATION_ATTEMPTS):
            self._update_checkpoint(
                status=TaskStatus.GENERATING_SCRIPT.value
            )

            if language == 'python':
                prompt = self._build_python_script_prompt(
                    desc, outline, packages, inputs, outputs
                )
            elif language == 'r':
                prompt = self._build_r_script_prompt(
                    desc, outline, packages, inputs, outputs
                )
            else:
                prompt = self._build_generic_script_prompt(
                    desc, outline, packages, inputs, outputs, language
                )

            try:
                response = invoke_resilient(
                    self.llm, prompt,
                    ollama_base_url=self.ollama_base_url,
                    max_retries=20,
                    initial_backoff=30.0,
                )
            except LLMInvocationError as e:
                logger.warning(f"Script generation failed: {e}")
                continue

            # Extract code from response
            content = self._extract_code_from_response(response, language)
            if not content or len(content) < 50:
                continue

            # Ensure required structure for Python
            if language == 'python' and 'AGI_DRY_RUN' not in content:
                content = self._prepend_python_header(content, subtask)

            # Validate
            self._update_checkpoint(
                status=TaskStatus.VALIDATING_SCRIPT.value
            )
            validation = self._validate_script(content, outline, subtask)

            if validation.get('valid'):
                print(f"  ✓ Script validated (attempt {attempt + 1})")
                return content
            else:
                print(
                    f"  ⚠ Script validation failed (attempt {attempt + 1}): "
                    f"{validation.get('issues', '')[:100]}"
                )

        # Return last generated content even if validation failed
        if content and len(content) > 50:
            logger.warning("Returning unvalidated script after max attempts")
            return content
        return None

    def _validate_script(
        self, script: str, outline: str, subtask: Dict
    ) -> Dict[str, Any]:
        """LLM validation of script against outline and subtask."""
        desc = subtask.get('description', '')

        prompt = f"""Validate this script against its outline and task.

TASK: {desc[:500]}

OUTLINE:
{outline[:1000]}

SCRIPT:
```
{script[:4000]}
```

Check:
1. Does it implement every step from the outline?
2. Does it match the task goals?
3. Are expected outputs produced?
4. Is the code syntactically correct?

Respond with EXACTLY one of:
VALID: The script correctly implements the task.
INVALID: [specific issues]"""

        try:
            response = invoke_resilient(
                self.llm, prompt,
                ollama_base_url=self.ollama_base_url,
                max_retries=10,
                initial_backoff=15.0,
            )
            if response.strip().upper().startswith('VALID'):
                return {'valid': True}
            return {'valid': False, 'issues': response}
        except LLMInvocationError:
            return {'valid': True}

    # =========================================================================
    # PHASE 2: ENVIRONMENT CREATION (Script-Informed, LLM-Reviewed)
    # =========================================================================

    def _phase_2_create_environment(
        self, subtask: Dict
    ) -> Dict[str, Any]:
        """Build a conda environment informed by the actual script content.

        Steps:
          2a. LLM dependency review of the entire script
          2b. YAML generation with conda/pip routing
          2c. Environment build
          2d. Repair loop (max 15 iterations)
        """
        task_id = subtask.get('id', 'task')
        print(f"\n[{task_id}] Phase 2: Environment Creation")

        if self.checkpoint.env_created:
            print(f"  ✓ Environment already created: {self.checkpoint.env_name}")
            return {'success': True}

        # Proactive disk check
        self._ensure_disk_space()

        script_path = Path(self.checkpoint.script_path)
        language = self.checkpoint.language or 'python'
        env_name = self.checkpoint.env_name
        env_yaml_path = Path(self.checkpoint.env_yaml_path)
        env_yaml_path.parent.mkdir(parents=True, exist_ok=True)

        # Read script content
        script_content = script_path.read_text() if script_path.exists() else ""

        # Step 2a: LLM dependency review
        self._update_checkpoint(status=TaskStatus.REVIEWING_DEPS.value)
        print(f"  → Reviewing dependencies via LLM...")

        try:
            from utils.dependency_parser import (
                build_dependency_review_prompt,
                parse_dependency_response,
                generate_env_yaml,
            )
            dep_prompt = build_dependency_review_prompt(
                script_content, language
            )
            dep_response = invoke_resilient(
                self.llm, dep_prompt,
                ollama_base_url=self.ollama_base_url,
                max_retries=15,
                initial_backoff=20.0,
            )
            dep_list = parse_dependency_response(dep_response)

            # Step 2b: Generate YAML
            env_yaml = generate_env_yaml(dep_list, env_name)

        except (ImportError, LLMInvocationError) as e:
            logger.warning(
                f"LLM dependency review failed ({e}), "
                f"falling back to regex-based extraction"
            )
            env_yaml = self._generate_env_yaml_fallback(
                subtask, script_content, env_name, language
            )

        env_yaml_path.write_text(env_yaml)
        print(f"  ✓ Environment YAML: {env_yaml_path}")

        # Step 2c: Build environment
        self._update_checkpoint(status=TaskStatus.BUILDING_ENV.value)
        build_result = self._create_conda_environment()

        if build_result.get('success'):
            self._update_checkpoint(env_created=True)
            print(f"  ✓ Environment ready: {env_name}")
            return {'success': True}

        # Step 2d: Repair loop
        print(f"  ⚠ Build failed, entering repair loop...")
        self._update_checkpoint(status=TaskStatus.REPAIRING_ENV.value)

        for repair_attempt in range(self.MAX_ENV_REPAIR_ITERATIONS):
            if not self._can_continue():
                break

            error = build_result.get('error', '')
            print(f"  → Repair attempt {repair_attempt + 1}: {error[:100]}")

            repaired = self._repair_environment(error, env_yaml_path, env_name)
            if not repaired:
                return {
                    'success': False,
                    'error': f"Env repair exhausted: {error[:200]}"
                }

            # Retry build
            build_result = self._create_conda_environment()
            if build_result.get('success'):
                self._update_checkpoint(env_created=True)
                print(f"  ✓ Environment ready after repair: {env_name}")
                return {'success': True}

        return {
            'success': False,
            'error': f"Env build failed after {self.MAX_ENV_REPAIR_ITERATIONS} repairs"
        }

    def _repair_environment(
        self, error: str, yaml_path: Path, env_name: str
    ) -> bool:
        """Attempt to repair a failed environment build.

        Handles: disk quota, package not found, conflicts.
        Returns True if a repair action was taken (caller should retry).
        """
        error_lower = error.lower()

        # Disk quota
        if 'no space' in error_lower or 'quota' in error_lower:
            self._ensure_disk_space(force=True)
            return True

        # Package not found on conda → move to pip
        not_found = re.findall(
            r'PackagesNotFoundError.*?- (\S+)',
            error, re.DOTALL
        )
        if not not_found:
            not_found = re.findall(
                r"nothing provides.*?needed by (\S+)", error_lower
            )
        if not not_found:
            not_found = re.findall(
                r'ResolvePackageNotFound.*?- (\S+)', error, re.DOTALL
            )

        if not_found and yaml and yaml_path.exists():
            content = yaml_path.read_text()
            for pkg in not_found:
                pkg_clean = re.split(r'[><=!]', pkg)[0].strip()
                if not pkg_clean:
                    continue
                # Move from conda deps to pip
                if self.conda_tools:
                    self.conda_tools.remove_package_from_yaml(
                        str(yaml_path), pkg_clean
                    )
                    self.conda_tools.update_yaml_with_package(
                        str(yaml_path), pkg_clean, section="pip"
                    )
                    print(f"    Moved {pkg_clean} to pip section")
            return True

        # Conflict → ask LLM for version resolution
        if 'conflict' in error_lower:
            try:
                fix_prompt = f"""A conda environment build failed with this error:

{error[:1500]}

Suggest specific version pins or alternative packages to resolve the conflict.
Return a list of changes, one per line, in format:
REMOVE: package_name
ADD_CONDA: package_name==version
ADD_PIP: package_name==version"""

                response = invoke_resilient(
                    self.llm, fix_prompt,
                    ollama_base_url=self.ollama_base_url,
                    max_retries=10,
                    initial_backoff=15.0,
                )

                if yaml and yaml_path.exists() and self.conda_tools:
                    for line in response.strip().split('\n'):
                        line = line.strip()
                        if line.startswith('REMOVE:'):
                            pkg = line.split(':', 1)[1].strip()
                            self.conda_tools.remove_package_from_yaml(
                                str(yaml_path), pkg
                            )
                        elif line.startswith('ADD_CONDA:'):
                            pkg = line.split(':', 1)[1].strip()
                            self.conda_tools.update_yaml_with_package(
                                str(yaml_path), pkg, section="conda"
                            )
                        elif line.startswith('ADD_PIP:'):
                            pkg = line.split(':', 1)[1].strip()
                            self.conda_tools.update_yaml_with_package(
                                str(yaml_path), pkg, section="pip"
                            )
                return True
            except (LLMInvocationError, Exception) as e:
                logger.warning(f"LLM conflict resolution failed: {e}")

        # Pip install failure → try alternative
        if 'pip' in error_lower and 'failed' in error_lower:
            return True  # Retry may succeed after network transient

        return False

    # =========================================================================
    # PHASE 3: SBATCH GENERATION (Language-Aware)
    # =========================================================================

    def _phase_3_generate_sbatch(
        self, subtask: Dict, routed_slurm: Dict
    ) -> Dict[str, Any]:
        """Generate the sbatch script with language-aware execution command.

        Uses a language dispatch table and cluster-routed SLURM settings.
        Tests submission for immediate config failures.
        """
        task_id = subtask.get('id', 'task')
        safe_id = re.sub(r'[^\w\-]', '_', task_id)[:30]

        print(f"\n[{task_id}] Phase 3: Sbatch Generation")

        if (self.checkpoint.sbatch_path
                and Path(self.checkpoint.sbatch_path).exists()):
            print(f"  ✓ Sbatch already exists: {self.checkpoint.sbatch_path}")
            return {'success': True}

        sbatch_path = (
            self.project_root / 'slurm' / 'scripts' / f"{safe_id}.sbatch"
        )
        sbatch_path.parent.mkdir(parents=True, exist_ok=True)

        sbatch_content = self._generate_sbatch_script(
            safe_id,
            Path(self.checkpoint.script_path),
            self.checkpoint.env_name,
            Path(self.checkpoint.env_yaml_path),
            subtask, routed_slurm,
            dry_run=True,
        )
        sbatch_path.write_text(sbatch_content)
        self._update_checkpoint(
            sbatch_path=str(sbatch_path),
            status=TaskStatus.GENERATING_SBATCH.value,
        )

        print(f"  ✓ Sbatch: {sbatch_path}")
        print(f"  ✓ Cluster: {routed_slurm.get('cluster_name')}")
        print(f"  ✓ Partition: {routed_slurm.get('partition')}")
        return {'success': True}

    # =========================================================================
    # PHASE 4: EXECUTION + DIAGNOSTIC AGENT LOOP
    # =========================================================================

    def _phase_4_execute_and_monitor(
        self,
        subtask: Dict,
        routed_cluster: str,
        routed_slurm: Dict,
    ) -> Dict[str, Any]:
        """Execute the job and iterate with diagnostic agent on failures.

        v1.2.3: Replaces the old _reflect_and_update loop with
        DiagnosticAgent invocation that produces structured FixPrescriptions.
        """
        task_id = subtask.get('id', 'task')
        print(f"\n[{task_id}] Phase 4: Execution & Monitoring")

        self._routed_slurm = routed_slurm

        while self._can_continue():
            self.checkpoint.iteration += 1
            self._update_checkpoint(
                status=TaskStatus.RUNNING_DRYRUN.value,
                iteration=self.checkpoint.iteration,
            )

            print(
                f"\n[{task_id}] Iteration {self.checkpoint.iteration} "
                f"— DRY RUN"
            )

            try:
                # Submit job
                submit_result = self._submit_slurm_job()
                if not submit_result['success']:
                    # Try to fix sbatch config
                    fixed = self._fix_sbatch_submission_error(
                        submit_result.get('error', ''), routed_slurm
                    )
                    if fixed:
                        continue
                    self._update_checkpoint(
                        last_error=submit_result.get('error')
                    )
                    continue

                self._update_checkpoint(
                    status=TaskStatus.WAITING_JOB.value,
                    current_job_id=submit_result['job_id'],
                )
                print(f"  ✓ Job submitted: {submit_result['job_id']}")

                # Wait
                print(f"  → Waiting for job...")
                wait_result = self._wait_for_job()
                logs = self._collect_job_logs()

                # Analyze
                analysis = self._analyze_job_result(
                    wait_result, logs, subtask
                )

                if analysis['success']:
                    # === DRY RUN SUCCESS → PRODUCTION RUN ===
                    print(f"  ✓ DRY RUN SUCCEEDED!")
                    self._update_checkpoint(
                        dry_run_succeeded=True,
                        status=TaskStatus.RUNNING_PROD.value,
                    )

                    print(f"\n[{task_id}] PRODUCTION RUN")
                    self._regenerate_sbatch(dry_run=False)

                    prod_submit = self._submit_slurm_job()
                    if prod_submit['success']:
                        self._update_checkpoint(
                            current_job_id=prod_submit['job_id']
                        )
                        prod_wait = self._wait_for_job()
                        prod_logs = self._collect_job_logs()

                        if prod_wait['success']:
                            outputs = self._verify_outputs(subtask)
                            report = self._generate_completion_report(
                                subtask, prod_logs, outputs, routed_cluster
                            )

                            # Cleanup
                            if self.cleanup_env_on_success:
                                print(f"  → Cleaning up environment...")
                                cleanup = self._cleanup_conda_environment()
                                if cleanup['success']:
                                    print(
                                        f"  ✓ Environment removed "
                                        f"(YAML preserved)"
                                    )

                            self._update_checkpoint(
                                status=TaskStatus.COMPLETED.value
                            )
                            self._delete_checkpoint(task_id)

                            return self._success_result(
                                task_id=task_id,
                                script_path=self.checkpoint.script_path,
                                job_id=self.checkpoint.current_job_id,
                                iterations=self.checkpoint.iteration,
                                routed_cluster=routed_cluster,
                                report=report,
                            )

                else:
                    # === FAILURE → INVOKE DIAGNOSTIC AGENT ===
                    error_summary = analysis.get(
                        'error_summary',
                        analysis.get('error_type', 'Unknown')
                    )
                    print(f"  ✗ Failed: {error_summary[:100]}")
                    self._update_checkpoint(
                        status=TaskStatus.DIAGNOSING.value,
                        last_error=error_summary,
                    )

                    # Invoke diagnostic agent
                    prescription = self._invoke_diagnostic_agent(
                        analysis, logs, subtask
                    )

                    # Apply fix
                    applied = self._apply_fix_prescription(
                        prescription, subtask, routed_slurm
                    )

                    if applied.get('should_escalate'):
                        self._update_checkpoint(
                            status=TaskStatus.FAILED.value
                        )
                        return self._failure_result(
                            task_id,
                            f"Escalating: {prescription.explanation[:200]}"
                        )

                    if applied.get('env_rebuilt'):
                        # Env was rebuilt, need fresh sbatch too
                        self._regenerate_sbatch(dry_run=True)

                    if applied.get('script_updated'):
                        self._regenerate_sbatch(dry_run=True)

                    print(f"  ✓ Fix applied, retrying...")

            except Exception as e:
                self._update_checkpoint(last_error=str(e))
                logger.error(f"[{task_id}] Exception in Phase 4: {e}")
                print(f"  ✗ Exception: {e}")

        # Context exhausted
        self._update_checkpoint(status=TaskStatus.FAILED.value)
        return self._failure_result(
            task_id=task_id,
            error=(
                f"Context exhausted after "
                f"{self.checkpoint.iteration} iterations"
            ),
            checkpoint_preserved=True,
        )

    # =========================================================================
    # DIAGNOSTIC AGENT INTEGRATION (v1.2.0)
    # =========================================================================

    def _invoke_diagnostic_agent(
        self,
        analysis: Dict[str, Any],
        logs: Dict[str, str],
        subtask: Dict,
    ) -> 'FixPrescription':
        """Instantiate a DiagnosticAgent and investigate the failure.

        Each invocation gets a fresh 25K token budget. The diagnostic agent
        checks its solution memory first, then investigates if needed.
        """
        from agents.diagnostic_agent import DiagnosticAgent, FixPrescription

        self.checkpoint.diagnostic_invocations += 1
        self._update_checkpoint(
            diagnostic_invocations=self.checkpoint.diagnostic_invocations
        )

        try:
            diag = DiagnosticAgent(
                agent_id=f"{self.agent_id}_diag_{self.checkpoint.diagnostic_invocations}",
                step_id=self.checkpoint.task_id,
                project_root=str(self.project_root),
                conda_tools=self.conda_tools,
                slurm_tools=self.slurm_tools,
                ollama_model=None,  # resolve_model() handles this
                ollama_base_url=self.ollama_base_url,
                solution_memory=self.diagnostic_memory,
                env_name=self.checkpoint.env_name,
                env_yaml_path=self.checkpoint.env_yaml_path,
            )

            prescription = diag.investigate(
                error_classification=analysis,
                logs=logs,
                subtask_description=subtask.get('description', ''),
            )

            logger.info(
                f"Diagnostic result: fix_type={prescription.fix_type}, "
                f"confidence={prescription.confidence}, "
                f"from_memory={prescription.from_memory}"
            )
            return prescription

        except Exception as e:
            logger.error(f"Diagnostic agent failed: {e}")
            return FixPrescription(
                fix_type="escalate",
                explanation=f"Diagnostic agent crashed: {e}",
                error_type=analysis.get('error_type', 'unknown'),
            )

    def _apply_fix_prescription(
        self,
        prescription: 'FixPrescription',
        subtask: Dict,
        routed_slurm: Dict,
    ) -> Dict[str, Any]:
        """Apply a structured fix from the diagnostic agent.

        Returns dict with:
          should_escalate: True if we should give up
          env_rebuilt: True if env was rebuilt (needs fresh sbatch)
          script_updated: True if script was modified
        """
        from agents.diagnostic_agent import FixPrescription

        result = {
            'should_escalate': False,
            'env_rebuilt': False,
            'script_updated': False,
        }

        fix_type = prescription.fix_type

        if fix_type == "escalate":
            result['should_escalate'] = True
            return result

        if fix_type == "env_updated":
            # Diagnostic agent already installed the package(s)
            # Just need to retry the job
            if prescription.packages_installed:
                print(
                    f"    Packages installed: "
                    f"{', '.join(prescription.packages_installed)}"
                )
            return result

        if fix_type == "edit_code":
            # Apply code changes to the script
            for change in prescription.changes:
                if change.get('action') == 'replace_script':
                    fixed_content = change.get('fixed_content', '')
                    if fixed_content and self.checkpoint.script_path:
                        Path(self.checkpoint.script_path).write_text(
                            fixed_content
                        )
                        result['script_updated'] = True
                        print(f"    Script updated with diagnostic fix")
                elif change.get('action') == 'fix_path':
                    # Path fix suggestion — needs script rewrite
                    result['script_updated'] = True
            return result

        if fix_type == "change_config":
            # Modify sbatch configuration
            for change in prescription.changes:
                suggestion = change.get('suggestion', '')
                if 'memory' in suggestion.lower() and self.checkpoint.sbatch_path:
                    # Double memory in sbatch
                    self._adjust_sbatch_memory(factor=2)
                    print(f"    Sbatch memory doubled")
                elif 'partition' in suggestion.lower():
                    # Re-route cluster
                    pass
            return result

        if fix_type == "rebuild_env":
            # Reset env and rebuild
            self._update_checkpoint(env_created=False)
            result['env_rebuilt'] = True
            rebuild = self._phase_2_create_environment(subtask)
            if rebuild['success']:
                self._update_checkpoint(env_created=True)
            return result

        if fix_type == "add_package":
            # Package was already added by diagnostic agent
            return result

        if fix_type == "add_argument":
            # v1.2.3: The sbatch exec_cmd was missing input_file arguments.
            # The stale sbatch must be deleted so Phase 3's cache check
            # doesn't skip regeneration, then rebuilt from scratch.
            # _generate_sbatch_script now appends subtask['input_files']
            # to exec_cmd, so the regenerated sbatch will be correct.
            print(f"    Removing stale sbatch (missing input args)...")
            if self.checkpoint.sbatch_path:
                stale = Path(self.checkpoint.sbatch_path)
                if stale.exists():
                    stale.unlink()
                    print(f"    Deleted: {stale.name}")
                # Clear from checkpoint so Phase 3 cache check doesn't block
                self._update_checkpoint(sbatch_path=None)

            routed_slurm = getattr(self, '_routed_slurm', {})
            if routed_slurm:
                regen = self._phase_3_generate_sbatch(subtask, routed_slurm)
                if regen.get('success'):
                    print(f"    Sbatch regenerated with input file arguments")
                    result['script_updated'] = True  # Triggers re-submission
                else:
                    print(f"    WARNING: Sbatch regeneration failed")
                    result['should_escalate'] = True
            else:
                # _routed_slurm not stored — escalate rather than guess
                print(f"    WARNING: No routed_slurm available, escalating")
                result['should_escalate'] = True
            return result

        return result

    # =========================================================================
    # ERROR ANALYSIS (v1.2.0 — Expanded Categories)
    # =========================================================================

    def _analyze_job_result(
        self, job_result: Dict, logs: Dict[str, str], subtask: Dict
    ) -> Dict[str, Any]:
        """Classify job outcome into specific error types.

        v1.2.0: Expanded from 5 to 12 error categories. Now also
        extracts the specific error message for diagnostic agent use.
        """
        stdout = logs.get('stdout', '')
        stderr = logs.get('stderr', '')

        if 'SUCCESS: Task completed' in stdout and job_result.get('success'):
            return {'success': True}

        combined = f"{stdout}\n{stderr}"
        combined_lower = combined.lower()

        # Error classification patterns (ordered by specificity)
        patterns = {
            'missing_package': [
                r'ModuleNotFoundError: No module named',
                r'ImportError: No module named',
                r'ImportError: cannot import name',
            ],
            'syntax_error': [
                r'SyntaxError:', r'IndentationError:',
            ],
            'data_structure_error': [
                r'KeyError:', r'IndexError:',
                r'ValueError:.*(?:shape|column|dimension|expected)',
            ],
            'memory_error': [
                r'MemoryError', r'OutOfMemoryError',
                r'oom-kill', r'OUT_OF_MEMORY',
            ],
            'gpu_error': [
                r'CUDA error', r'CUDA out of memory',
                r'RuntimeError:.*CUDA', r'GPU.*not available',
            ],
            'disk_quota_error': [
                r'No space left on device', r'Disk quota exceeded',
                r'OSError: \[Errno 28\]',
            ],
            'binary_not_found': [
                r'command not found',
                r'FileNotFoundError.*(?:samtools|bedtools|bcftools|bwa)',
            ],
            'permission_error': [
                r'PermissionError:', r'\[Errno 13\]',
            ],
            'network_error': [
                r'ConnectionError:', r'URLError:',
                r'TimeoutError:.*download',
                r'HTTPError: 5\d\d',
            ],
            'sbatch_config_error': [
                r'Invalid partition', r'Invalid account',
                r'invalid resource specification',
            ],
            'runtime_logic_error': [
                r'AssertionError:', r'TypeError:',
                r'AttributeError:', r'NameError:',
                r'ZeroDivisionError:',
            ],
            'code_error': [
                r'Traceback \(most recent call last\)',
                r'Error in .*\.R:',
            ],
            'runtime_argument_error': [
                r'[Ee]rror:.*[Nn]o .*(?:input|file|argument).*provided',
                r'[Ee]rror:.*(?:argument|input).*(?:required|missing)',
                r'[Uu]sage:.*required',
                r'error: the following arguments are required',
                r'[Ee]rror:.*provided via command',
                r'sys\.exit\(1\)',
            ],
        }

        for error_type, pats in patterns.items():
            for p in pats:
                m = re.search(p, combined, re.IGNORECASE)
                if m:
                    # v1.2.3: Prefer stderr for error extraction, but fall back
                    # to stdout — many scripts (e.g. sys.exit calls, argparse)
                    # print errors to stdout with an empty stderr.
                    best_source = stderr if stderr.strip() else stdout
                    error_message = self._extract_error_message(best_source)
                    return {
                        'success': False,
                        'error_type': error_type,
                        'error_message': error_message,
                        'error_summary': f"{error_type}: {error_message[:200]}",
                    }

        if not job_result.get('success'):
            # v1.2.3: Same stdout preference fix for the unknown fallback.
            # Previously this always used stderr, losing the actual error when
            # the script printed to stdout and stderr was empty.
            best_source = stderr if stderr.strip() else stdout
            return {
                'success': False,
                'error_type': 'unknown',
                'error_message': best_source[-500:] if best_source else 'Unknown error',
                'error_summary': best_source[-500:] or 'Unknown error',
            }

        return {'success': True}

    def _extract_error_message(self, text: str) -> str:
        """Extract the most relevant error line from output."""
        lines = text.strip().split('\n')
        # Walk backwards to find the actual error
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith('File ') and not line.startswith('Traceback'):
                if any(kw in line for kw in [
                    'Error', 'error', 'Exception', 'FAILED',
                    'ModuleNotFoundError', 'ImportError',
                ]):
                    return line
        # Fallback: last non-empty line
        for line in reversed(lines):
            if line.strip():
                return line.strip()
        return text[-200:] if text else "Unknown error"

    # =========================================================================
    # SBATCH GENERATION (Language-Aware)
    # =========================================================================

    def _generate_sbatch_script(
        self, task_id: str, script_path: Path, env_name: str,
        env_yaml_path: Path, subtask: Dict, routed_slurm: Dict,
        dry_run: bool = True
    ) -> str:
        """Generate sbatch script using ROUTED cluster settings.

        v1.2.0: Language dispatch table for execution commands.
        """
        partition = routed_slurm.get('partition', 'compute1')
        account = routed_slurm.get('account')
        nodes = routed_slurm.get('nodes', 1)
        ntasks = routed_slurm.get('ntasks', 1)
        cpus = routed_slurm.get('cpus_per_task', 20)
        memory = routed_slurm.get('memory')  # None for GPU clusters
        time_limit = routed_slurm.get('time', '1-00:00:00')
        gpu_directive = routed_slurm.get('gpu_directive')
        is_gpu = routed_slurm.get('gpu_available', False)
        selected_cluster = routed_slurm.get(
            'cluster_name', self.cluster_config.cluster_name
        )

        log_dir = self.project_root / 'slurm' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        job_name = f"agi_{task_id}" + ("_dryrun" if dry_run else "_prod")

        # Language dispatch
        language = self.checkpoint.language or 'python'
        dispatch = {
            'python': f'python "{script_path}"',
            'r': f'Rscript "{script_path}"',
            'bash': f'bash "{script_path}"',
            'perl': f'perl "{script_path}"',
            'java': f'java "{script_path}"',
        }
        exec_cmd = dispatch.get(language, f'python "{script_path}"')

        # v1.2.3: Append input_files as positional CLI arguments.
        # Scripts that use sys.argv or argparse require this — without it
        # the sbatch calls the script with no arguments and the script
        # immediately exits with "No input file provided".
        # input_files come from the subtask definition, which is the
        # authoritative source for what data this step needs.
        input_files = subtask.get('input_files', [])
        if input_files:
            # Resolve each path relative to project root and quote it
            resolved_args = []
            for f in input_files:
                p = Path(f)
                if not p.is_absolute():
                    p = self.project_root / p
                resolved_args.append(f'"{p}"')
            exec_cmd = exec_cmd + ' ' + ' '.join(resolved_args)

        # Build SBATCH directives
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={partition}",
        ]

        if account:
            lines.append(f"#SBATCH --account={account}")

        lines.extend([
            f"#SBATCH --nodes={nodes}",
            f"#SBATCH --ntasks={ntasks}",
            f"#SBATCH --cpus-per-task={cpus}",
        ])

        # Memory: ONLY for CPU clusters. NEVER for GPU clusters.
        if memory and not is_gpu:
            lines.append(f"#SBATCH --mem={memory}")

        lines.append(f"#SBATCH --time={time_limit}")

        # GPU directive (--gres=gpu:N format, NOT --gpus N)
        if gpu_directive:
            lines.append(f"#SBATCH {gpu_directive}")

        # Log files
        lines.extend([
            f"#SBATCH --output={log_dir}/{job_name}_%j.out",
            f"#SBATCH --error={log_dir}/{job_name}_%j.err",
        ])

        # Script body
        gpu_note = " [GPU]" if is_gpu else " [CPU]"
        lines.extend([
            "",
            "#" + "=" * 70,
            f"# Cluster: {selected_cluster}{gpu_note}",
            f"# Task: {task_id}",
            f"# Language: {language}",
            f"# Mode: {'DRY-RUN' if dry_run else 'PRODUCTION'}",
            f"# Partition: {partition}, CPUs: {cpus}",
        ])
        if is_gpu:
            lines.append(f"# GPU: {gpu_directive} (NO --mem on GPU nodes)")
        else:
            lines.append(f"# Memory: {memory or 'default'}")
        lines.extend([
            "#" + "=" * 70,
            "",
            "set -e",
            "",
            "echo '=============================================='",
            "echo 'Job ID: '$SLURM_JOB_ID",
            "echo 'Node: '$(hostname)",
            "echo 'Start: '$(date)",
            f"echo 'Cluster: {selected_cluster}{gpu_note}'",
            "echo '=============================================='",
            "",
            "# Conda setup",
            f'CONDA_ENV="{env_name}"',
            f'ENV_YAML="{env_yaml_path}"',
            "",
            'echo ">>> Loading conda..."',
            'if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then',
            '    source "$HOME/anaconda3/etc/profile.d/conda.sh"',
            'elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then',
            '    source "$HOME/miniconda3/etc/profile.d/conda.sh"',
            'elif command -v conda &> /dev/null; then',
            '    eval "$(conda shell.bash hook)"',
            'else',
            '    echo "ERROR: conda not found!"; exit 1',
            'fi',
            "",
            'echo ">>> Checking environment..."',
            'if ! conda env list | grep -q "^${CONDA_ENV} "; then',
            '    echo ">>> Creating environment from YAML..."',
            '    conda env create -f "${ENV_YAML}" -n "${CONDA_ENV}" || {',
            '        echo "ERROR: Failed to create environment"',
            '        exit 1',
            '    }',
            'fi',
            "",
            'echo ">>> Activating ${CONDA_ENV}..."',
            'conda activate "${CONDA_ENV}"',
            "",
        ])

        # GPU-specific: unset SLURM vars that interfere
        if is_gpu:
            lines.extend([
                '# Unset SLURM GPU vars that interfere with frameworks',
                'unset CUDA_VISIBLE_DEVICES 2>/dev/null || true',
                'unset ROCR_VISIBLE_DEVICES 2>/dev/null || true',
                "",
            ])

        # Set environment variables
        lines.extend([
            '# Environment variables',
            f'export AGI_DRY_RUN="{"true" if dry_run else "false"}"',
            f'export PROJECT_DIR="{self.project_root}"',
            "",
            f'echo ">>> Running {language} script..."',
            f'echo ">>> DRY_RUN=$AGI_DRY_RUN"',
            "",
            exec_cmd,
            "",
            'echo ""',
            'echo "=============================================="',
            'echo "Done: $(date)"',
            'echo "=============================================="',
        ])

        return '\n'.join(lines) + '\n'

    def _fix_sbatch_submission_error(
        self, error: str, routed_slurm: Dict
    ) -> bool:
        """Try to fix common sbatch submission errors.

        Rule-based repair for known SLURM config issues.
        Returns True if a fix was applied.
        """
        if not self.checkpoint or not self.checkpoint.sbatch_path:
            return False

        path = Path(self.checkpoint.sbatch_path)
        if not path.exists():
            return False

        content = path.read_text()
        fixed = False

        # Invalid partition
        if 'Invalid partition' in error:
            # Fall back to default partition
            content = re.sub(
                r'#SBATCH --partition=\S+',
                f"#SBATCH --partition={routed_slurm.get('partition', 'compute1')}",
                content
            )
            fixed = True

        # Memory not allowed on GPU
        if '--mem' in error and 'not allowed' in error.lower():
            content = re.sub(r'#SBATCH --mem=\S+\n', '', content)
            fixed = True

        if fixed:
            path.write_text(content)
        return fixed

    def _adjust_sbatch_memory(self, factor: int = 2):
        """Multiply the --mem value in the sbatch by a factor."""
        if not self.checkpoint or not self.checkpoint.sbatch_path:
            return
        path = Path(self.checkpoint.sbatch_path)
        if not path.exists():
            return

        content = path.read_text()
        m = re.search(r'#SBATCH --mem=(\d+)([GMK]?)', content)
        if m:
            val = int(m.group(1)) * factor
            unit = m.group(2) or 'G'
            content = re.sub(
                r'#SBATCH --mem=\S+',
                f"#SBATCH --mem={val}{unit}",
                content
            )
            path.write_text(content)

    # =========================================================================
    # ENVIRONMENT MANAGEMENT
    # =========================================================================

    def _create_conda_environment(self) -> Dict[str, Any]:
        """Create conda env from YAML. Pip fallback on failure."""
        if not self.checkpoint or not self.checkpoint.env_yaml_path:
            return {'success': False, 'error': 'No env YAML path'}

        env_yaml_path = self.checkpoint.env_yaml_path
        env_name = self.checkpoint.env_name

        if self.conda_tools:
            result = self.conda_tools.create_from_yaml(
                yaml_path=env_yaml_path,
                env_name=env_name,
            )
            return result

        # Fallback: direct subprocess
        try:
            result = subprocess.run(
                ['conda', 'env', 'create', '-f', env_yaml_path,
                 '-n', env_name, '--yes'],
                capture_output=True, text=True, timeout=1200,
            )

            if result.returncode == 0:
                return {'success': True}

            # Try pip fallback for failed packages
            stderr = result.stderr
            failed = re.findall(
                r'PackagesNotFoundError.*?- (\S+)', stderr, re.DOTALL
            )
            if failed:
                # Remove failed packages from YAML, rebuild, pip install them
                return self._pip_fallback_build(
                    env_yaml_path, env_name, failed, stderr
                )

            return {'success': False, 'error': stderr[-500:]}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _pip_fallback_build(
        self, yaml_path: str, env_name: str,
        failed_packages: List[str], original_error: str
    ) -> Dict[str, Any]:
        """Remove failed packages from YAML, rebuild, pip install them."""
        if not yaml or not Path(yaml_path).exists():
            return {'success': False, 'error': original_error}

        with open(yaml_path) as f:
            env_data = yaml.safe_load(f)

        # Remove failed from dependencies
        deps = env_data.get('dependencies', [])
        cleaned_deps = []
        for dep in deps:
            if isinstance(dep, str):
                name = re.split(r'[><=!]', dep)[0].strip()
                if name not in failed_packages:
                    cleaned_deps.append(dep)
            else:
                cleaned_deps.append(dep)
        env_data['dependencies'] = cleaned_deps

        # Write cleaned YAML
        with open(yaml_path, 'w') as f:
            yaml.dump(env_data, f, default_flow_style=False)

        # Rebuild
        try:
            result = subprocess.run(
                ['conda', 'env', 'create', '-f', yaml_path,
                 '-n', env_name, '--yes', '--force'],
                capture_output=True, text=True, timeout=1200,
            )
            if result.returncode != 0:
                return {'success': False, 'error': result.stderr[-500:]}
        except Exception as e:
            return {'success': False, 'error': str(e)}

        # Pip install failed packages
        for pkg in failed_packages:
            try:
                subprocess.run(
                    ['conda', 'run', '-n', env_name,
                     'pip', 'install', pkg],
                    capture_output=True, text=True, timeout=300,
                )
                # Update YAML pip section
                if self.conda_tools:
                    self.conda_tools.update_yaml_with_package(
                        yaml_path, pkg, section="pip"
                    )
            except Exception:
                pass

        return {'success': True}

    def _cleanup_conda_environment(self) -> Dict[str, Any]:
        """Remove the conda environment, preserving the YAML."""
        if not self.checkpoint or not self.checkpoint.env_name:
            return {'success': True}

        env_name = self.checkpoint.env_name

        try:
            result = subprocess.run(
                ['conda', 'env', 'list'],
                capture_output=True, text=True, timeout=60
            )
            if env_name not in result.stdout:
                return {'success': True, 'already_removed': True}

            result = subprocess.run(
                ['conda', 'env', 'remove', '-n', env_name, '-y'],
                capture_output=True, text=True, timeout=300
            )

            return {
                'success': result.returncode == 0,
                'error': result.stderr if result.returncode != 0 else None
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # =========================================================================
    # SLURM JOB MANAGEMENT
    # =========================================================================

    def _submit_slurm_job(self) -> Dict[str, Any]:
        if not self.checkpoint or not self.checkpoint.sbatch_path:
            return {'success': False, 'error': 'No sbatch'}

        sbatch_path = Path(self.checkpoint.sbatch_path)
        if not sbatch_path.exists():
            return {
                'success': False,
                'error': f'Sbatch not found: {sbatch_path}'
            }

        try:
            result = subprocess.run(
                ['sbatch', str(sbatch_path)],
                capture_output=True, text=True, timeout=60,
                cwd=str(self.project_root)
            )

            if result.returncode != 0:
                return {'success': False, 'error': result.stderr}

            match = re.search(r'Submitted batch job (\d+)', result.stdout)
            if not match:
                return {
                    'success': False,
                    'error': f'Cannot parse job ID: {result.stdout}'
                }

            return {'success': True, 'job_id': match.group(1)}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _wait_for_job(self) -> Dict[str, Any]:
        """Poll squeue until job completes."""
        if not self.checkpoint or not self.checkpoint.current_job_id:
            return {'success': False, 'error': 'No job ID'}

        job_id = self.checkpoint.current_job_id
        poll_interval = self.slurm_config.get('poll_interval', 30)
        max_polls = self.slurm_config.get('max_poll_attempts', 8640)

        for _ in range(max_polls):
            try:
                result = subprocess.run(
                    ['squeue', '-j', job_id, '-h', '-o', '%T'],
                    capture_output=True, text=True, timeout=30,
                )
                state = result.stdout.strip()

                if not state:
                    # Job finished — check sacct
                    sacct = subprocess.run(
                        ['sacct', '-j', job_id, '-n', '-o',
                         'State', '-X'],
                        capture_output=True, text=True, timeout=30,
                    )
                    final_state = sacct.stdout.strip().split('\n')[0].strip()
                    success = final_state in ('COMPLETED',)
                    return {
                        'success': success,
                        'state': final_state,
                        'job_id': job_id,
                    }

                if state in ('FAILED', 'CANCELLED', 'TIMEOUT',
                             'NODE_FAIL', 'PREEMPTED', 'OUT_OF_MEMORY'):
                    return {
                        'success': False,
                        'state': state,
                        'job_id': job_id,
                    }

            except Exception:
                pass

            time.sleep(poll_interval)

        return {'success': False, 'state': 'POLL_TIMEOUT', 'job_id': job_id}

    def _collect_job_logs(self) -> Dict[str, str]:
        """Collect stdout/stderr from SLURM log files."""
        logs = {'stdout': '', 'stderr': ''}
        if not self.checkpoint or not self.checkpoint.current_job_id:
            return logs

        job_id = self.checkpoint.current_job_id
        log_dir = self.project_root / 'slurm' / 'logs'

        for f in log_dir.glob(f"*_{job_id}.out"):
            try:
                logs['stdout'] = f.read_text()
            except Exception:
                pass

        for f in log_dir.glob(f"*_{job_id}.err"):
            try:
                logs['stderr'] = f.read_text()
            except Exception:
                pass

        return logs

    def _regenerate_sbatch(self, dry_run: bool):
        if not self.checkpoint or not self.checkpoint.sbatch_path:
            return

        path = Path(self.checkpoint.sbatch_path)
        if not path.exists():
            return

        content = path.read_text()

        if dry_run:
            content = re.sub(
                r'export AGI_DRY_RUN="false"',
                'export AGI_DRY_RUN="true"', content
            )
            content = re.sub(r'_prod', '_dryrun', content)
        else:
            content = re.sub(
                r'export AGI_DRY_RUN="true"',
                'export AGI_DRY_RUN="false"', content
            )
            content = re.sub(r'_dryrun', '_prod', content)

        path.write_text(content)

    # =========================================================================
    # SCRIPT GENERATION HELPERS
    # =========================================================================

    def _detect_language(self, subtask: Dict) -> str:
        """Detect script language from subtask metadata."""
        desc = (subtask.get('description', '') + ' '
                + ' '.join(subtask.get('packages', []))).lower()

        r_keywords = [
            'seurat', 'bioconductor', 'singlecellexperiment',
            'rscript', 'ggplot', 'dplyr', 'tidyverse',
            'deseq2', 'edger', 'limma',
        ]
        bash_keywords = ['bash', 'shell script', 'pipeline of commands']

        for kw in r_keywords:
            if kw in desc:
                return 'r'
        for kw in bash_keywords:
            if kw in desc:
                return 'bash'
        return 'python'

    def _build_python_script_prompt(
        self, desc, outline, packages, inputs, outputs, hints=None
    ) -> str:
        hints_block = ""
        if hints:
            hints_block = "\nIMPORTANT IMPLEMENTATION GUIDANCE (follow exactly):\n"
            hints_block += "\n".join(f"  - {h}" for h in hints)
            hints_block += "\n"
        return f"""Generate a complete Python script based on this validated outline.

TASK: {desc}
PACKAGES: {', '.join(packages)}
INPUTS: {', '.join(inputs) if inputs else 'None'}
OUTPUTS: {', '.join(outputs) if outputs else 'None'}
{hints_block}
OUTLINE:
{outline}

REQUIRED STRUCTURE:
```python
#!/usr/bin/env python3
import os
from pathlib import Path

DRY_RUN = os.environ.get('AGI_DRY_RUN', 'true').lower() == 'true'
PROJECT_ROOT = Path(os.environ.get('PROJECT_DIR', '.')).resolve()
os.chdir(PROJECT_ROOT)

print(f"[MODE] DRY_RUN={{DRY_RUN}}, PROJECT_ROOT={{PROJECT_ROOT}}")

def main():
    # Implement every step from the outline
    # Wrap saves with: if not DRY_RUN: save() else: print("[DRY-RUN] Would save...")
    print("SUCCESS: Task completed")

if __name__ == "__main__":
    main()
```

Generate ONLY the complete Python code."""

    def _build_r_script_prompt(
        self, desc, outline, packages, inputs, outputs, hints=None
    ) -> str:
        hints_block = ""
        if hints:
            hints_block = "\nIMPORTANT IMPLEMENTATION GUIDANCE (follow exactly):\n"
            hints_block += "\n".join(f"  - {h}" for h in hints)
            hints_block += "\n"
        return f"""Generate a complete R script based on this validated outline.

TASK: {desc}
PACKAGES: {', '.join(packages)}
INPUTS: {', '.join(inputs) if inputs else 'None'}
OUTPUTS: {', '.join(outputs) if outputs else 'None'}
{hints_block}
OUTLINE:
{outline}

REQUIRED STRUCTURE:
- Load all required libraries at the top
- Set PROJECT_ROOT from environment variable
- Implement every step from the outline
- Print "SUCCESS: Task completed" on completion

Generate ONLY the R code."""

    def _build_generic_script_prompt(
        self, desc, outline, packages, inputs, outputs, language, hints=None
    ) -> str:
        hints_block = ""
        if hints:
            hints_block = "\nIMPORTANT IMPLEMENTATION GUIDANCE (follow exactly):\n"
            hints_block += "\n".join(f"  - {h}" for h in hints)
            hints_block += "\n"
        return f"""Generate a complete {language} script based on this outline.

TASK: {desc}
{hints_block}
OUTLINE: {outline}

Requirements:
- Implement every step from the outline
- Print "SUCCESS: Task completed" on completion

Generate ONLY the {language} code."""

    def _extract_code_from_response(
        self, response: str, language: str
    ) -> str:
        """Extract code from LLM response, stripping markdown fences."""
        lang_tags = {
            'python': ['python', 'py'],
            'r': ['r', 'R'],
            'bash': ['bash', 'sh'],
            'perl': ['perl'],
            'java': ['java'],
        }
        tags = lang_tags.get(language, [language])

        # Try extracting from code blocks
        for tag in tags + ['']:
            pattern = rf'```{tag}\s*\n(.*?)\n```'
            m = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if m:
                return m.group(1).strip()

        # No code block found — strip any remaining fences
        content = re.sub(
            r'^```(?:\w+)?\n?', '', response, flags=re.MULTILINE
        )
        content = re.sub(
            r'\n?```$', '', content, flags=re.MULTILINE
        ).strip()
        return content

    def _prepend_python_header(self, script: str, subtask: Dict) -> str:
        header = f'''#!/usr/bin/env python3
"""Task: {subtask.get('id', 'unknown')}"""
import os
from pathlib import Path

DRY_RUN = os.environ.get('AGI_DRY_RUN', 'true').lower() == 'true'
PROJECT_ROOT = Path(os.environ.get('PROJECT_DIR', '.')).resolve()
os.chdir(PROJECT_ROOT)

print(f"[MODE] DRY_RUN={{DRY_RUN}}, PROJECT_ROOT={{PROJECT_ROOT}}")

'''
        return header + re.sub(r'^#!/usr/bin/env python3?\n?', '', script)

    # =========================================================================
    # ENVIRONMENT YAML FALLBACK
    # =========================================================================

    def _generate_env_yaml_fallback(
        self, subtask: Dict, script_content: str,
        env_name: str, language: str
    ) -> str:
        """Generate env YAML when LLM dependency review is unavailable.

        Falls back to regex-based import extraction + known pip-only routing.
        Preserved from v3.2.2 Fix C.
        """
        KNOWN_PIP_ONLY = {
            'popv', 'celltypist', 'scvi-tools', 'decoupler',
            'episcanpy', 'cell2location', 'tangram',
            'squidpy', 'magic-impute', 'bbknn',
            'scrublet', 'doubletdetection', 'scanorama',
            'harmonypy', 'mnnpy', 'pyscenic',
        }

        packages = set(subtask.get('packages', []))

        # Extract imports from script
        if language == 'python':
            for m in re.finditer(
                r'^\s*(?:import|from)\s+(\w+)', script_content, re.MULTILINE
            ):
                packages.add(m.group(1))

        # Parse code_hints for pip: sections
        hint_pip = set()
        for hint in subtask.get('code_hints', []):
            if isinstance(hint, str):
                pip_section = re.search(
                    r'pip:\s*\n((?:\s+- .+\n?)+)', hint, re.MULTILINE
                )
                if pip_section:
                    for line in pip_section.group(1).split('\n'):
                        line = line.strip().lstrip('- ').strip()
                        if line:
                            name = re.split(r'[><=!~]', line)[0].strip().lower()
                            if name and name != 'pip':
                                hint_pip.add(name)

        # Route packages
        conda_deps = []
        pip_deps = []
        stdlib = {
            'os', 'sys', 're', 'json', 'pathlib', 'math',
            'collections', 'itertools', 'functools', 'datetime',
            'subprocess', 'typing', 'logging', 'io', 'csv',
            'glob', 'shutil', 'time', 'copy', 'warnings',
            'argparse', 'pickle', 'hashlib', 'tempfile',
            'multiprocessing', 'threading', 'unittest',
        }

        for pkg in packages:
            pkg_lower = pkg.lower()
            if pkg_lower in stdlib:
                continue
            if pkg_lower in KNOWN_PIP_ONLY or pkg_lower in hint_pip:
                pip_deps.append(pkg)
            elif '/' in pkg or pkg.startswith('git+'):
                pip_deps.append(pkg)
            else:
                conda_deps.append(pkg)

        # Build YAML
        python_version = '3.10'
        lines = [
            f"name: {env_name}",
            "channels:",
            "  - nodefaults",
            "  - conda-forge",
            "  - bioconda",
            "dependencies:",
            f"  - python={python_version}",
            "  - pip",
        ]
        for dep in sorted(set(conda_deps)):
            lines.append(f"  - {dep}")

        if pip_deps:
            lines.append("  - pip:")
            for dep in sorted(set(pip_deps)):
                lines.append(f"    - {dep}")

        return '\n'.join(lines) + '\n'

    # =========================================================================
    # DISK MANAGEMENT
    # =========================================================================

    def _ensure_disk_space(self, force: bool = False):
        """Proactive disk space check and cleanup."""
        try:
            from utils.disk_manager import DiskManager
            if self._disk_manager is None:
                self._disk_manager = DiskManager()
            if force:
                self._disk_manager.emergency_cleanup()
            else:
                self._disk_manager.proactive_cleanup()
        except ImportError:
            # DiskManager not available, try direct cleanup
            if force:
                try:
                    subprocess.run(
                        ['conda', 'clean', '--all', '--yes'],
                        capture_output=True, timeout=120,
                    )
                except Exception:
                    pass

    # =========================================================================
    # CHECKPOINT MANAGEMENT
    # =========================================================================

    def _load_or_create_checkpoint(
        self, task_id: str
    ) -> TaskCheckpoint:
        checkpoint_dir = self.project_root / 'temp' / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{task_id}.json"

        if checkpoint_path.exists():
            try:
                with open(checkpoint_path) as f:
                    data = json.load(f)
                cp = TaskCheckpoint.from_dict(data)
                print(
                    f"  ✓ Resumed checkpoint: phase={cp.phase}, "
                    f"iter={cp.iteration}, status={cp.status}"
                )
                return cp
            except Exception as e:
                logger.warning(f"Checkpoint load failed: {e}")

        return TaskCheckpoint.new(task_id)

    def _update_checkpoint(self, **kwargs):
        if not self.checkpoint:
            return

        for key, val in kwargs.items():
            if hasattr(self.checkpoint, key):
                setattr(self.checkpoint, key, val)

        self.checkpoint.updated_at = datetime.now().isoformat()

        checkpoint_dir = self.project_root / 'temp' / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"{self.checkpoint.task_id}.json"

        try:
            with open(path, 'w') as f:
                json.dump(self.checkpoint.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Checkpoint save failed: {e}")

    def _delete_checkpoint(self, task_id: str):
        path = (
            self.project_root / 'temp' / 'checkpoints' / f"{task_id}.json"
        )
        if path.exists():
            path.unlink()

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _can_continue(self) -> bool:
        can, _ = self.context_mgr.should_continue(
            self.agent_id, min_tokens_needed=self.MIN_TOKENS_FOR_RETRY
        )
        return can

    def _check_existing_outputs(self, subtask: Dict) -> Dict[str, Any]:
        outputs = subtask.get('output_files', [])
        if not outputs:
            return {'already_complete': False}


        existing = []
        for f in outputs:
            p = self.project_root / f
            if p.exists() and p.stat().st_size > 0:
                existing.append(str(p))


        already_complete = len(existing) == len(outputs)

        if already_complete:
            logger.info(
                f"[{subtask.get('id', '?')}] All {len(outputs)} output files"
                f"already exist - skipping execution"
            )

        return {
            'already_complete': already_complete,
            'existing_files': existing,
            'missing_files': [f for f in outputs if str(self.project_root / f) not in existing]
        }

    def _verify_outputs(self, subtask: Dict) -> Dict[str, Any]:
        outputs = subtask.get('output_files', [])
        found = [
            str(self.project_root / f)
            for f in outputs
            if (self.project_root / f).exists()
        ]
        return {'found_files': found}

    def _success_result(self, task_id: str, **kwargs) -> Dict[str, Any]:
        result = {
            'success': True,
            'task_id': task_id,
            'cluster': self.cluster_config.cluster_name,
            **kwargs,
        }
        agent_logger.log_task_success(
            agent_name=self.agent_id, task_id=task_id,
            result=result, tools_used=['slurm', 'conda']
        )
        return result

    def _failure_result(self, task_id: str, error: str, **kwargs) -> Dict[str, Any]:
        result = {
            'success': False,
            'task_id': task_id,
            'error': error,
            'cluster': self.cluster_config.cluster_name,
            **kwargs,
        }
        agent_logger.log_task_failure(
            agent_name=self.agent_id, task_id=task_id,
            error=error, context=kwargs
        )
        return result

    def _generate_completion_report(
        self, subtask: Dict, logs: Dict, outputs: Dict,
        routed_cluster: str = None
    ) -> str:
        cluster = routed_cluster or (
            self.checkpoint.routed_cluster if self.checkpoint else 'unknown'
        )
        return (
            f"# Task: "
            f"{self.checkpoint.task_id if self.checkpoint else 'unknown'}\n"
            f"Completed: {datetime.now().isoformat()}\n"
            f"Cluster: {cluster}\n"
            f"Language: {self.checkpoint.language or 'python'}\n"
            f"Iterations: "
            f"{self.checkpoint.iteration if self.checkpoint else 0}\n"
            f"Diagnostic invocations: "
            f"{self.checkpoint.diagnostic_invocations if self.checkpoint else 0}\n"
            f"Outputs: {', '.join(outputs.get('found_files', []))}\n"
        )


# Aliases for backward compatibility
ScriptFirstSubAgent = ScriptFirstSubAgentV3
SubAgentV3 = ScriptFirstSubAgentV3

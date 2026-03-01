"""
Script-First Sub-Agent v1.2.9 — 5-Phase Lifecycle with Diagnostic and Validation Agents

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

    Phase 5: Output Validation (v1.2.9)
      - Runs AFTER Phase 4 production success — validates that outputs
        are scientifically sound, not just that the script exited 0
      - ValidationAgent generates a Python validation script via LLM
        using file-type-specific check hints (FILE_TYPE_DISPATCH)
      - Validation runs as a lightweight SLURM job (2 CPUs, 16G, 30 min)
        with --dependency=afterok:<phase4_job_id>
      - Structured JSON report parsed from stdout (fixed footer contract)
      - On PASS: store success in memory, cleanup env, mark COMPLETED
      - On FAIL: synthesize correction hints via LLM, dual-inject into
        subtask (code_hints + expanded_plan), reset to Phase 1, retry
      - Max 3 validation attempts (each triggers full Phases 1-4 rerun)
      - Memory integration via MemoryClient with "val_" task_id prefix
        prevents identical correction loops across retry attempts
      - Graceful degradation: if ValidationAgent unavailable or no
        output_files defined, step completes as it would in v1.2.8

  v1.2.9 Changes:
    - Phase 5 validation lifecycle after Phase 4 production success
    - ValidationAgent (agents/validation_agent.py) with 4 public methods:
        generate_validation_script(), generate_validation_sbatch(),
        parse_validation_output(), synthesize_correction()
    - Environment cleanup relocated from Phase 4 to _finalize_step_completion()
      (shared by Phase 5 pass and no-validation skip paths)
    - TaskCheckpoint extended with 5 validation fields
    - TaskStatus extended with 4 Phase 5 states
    - Constructor accepts validation_agent and memory_client parameters
    - FailureType enum extended with VALIDATION_FAILURE, VALIDATION_SUCCESS

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
# v1.2.8: Manifest manager for output tracking and cleanup protection
try:
    from utils.manifest import ManifestManager
    _MANIFEST_AVAILABLE = True
except ImportError:
    _MANIFEST_AVAILABLE = False
    logger.warning("ManifestManager not available — output manifest will not be written")

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
    # Phase 5 (v1.2.9)
    GENERATING_VALIDATION = "generating_validation"
    WAITING_VALIDATION_JOB = "waiting_validation_job"
    VALIDATION_PASSED = "validation_passed"
    VALIDATION_FAILED = "validation_failed"
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
    # v1.2.9: Phase 5 validation fields
    validation_script_path: Optional[str] = None
    validation_sbatch_path: Optional[str] = None
    validation_job_id: Optional[str] = None
    validation_passed: bool = False
    validation_attempts: int = 0

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
        # v1.2.9: Phase 5 validation fields
        for key in [
            'validation_script_path', 'validation_sbatch_path',
            'validation_job_id',
        ]:
            if key not in data:
                data[key] = None
        if 'validation_passed' not in data:
            data['validation_passed'] = False
        if 'validation_attempts' not in data:
            data['validation_attempts'] = 0
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
            phase=0, diagnostic_invocations=0,
            validation_script_path=None, validation_sbatch_path=None,
            validation_job_id=None, validation_passed=False,
            validation_attempts=0,
        )

# =============================================================================
# STEP LOGGER v1.2.5
# =============================================================================

class StepLogger:
    """
    Per-step phase logger for Phases 1-3 (master-node execution).

    Phases 1, 2, and 3 run in-process on the GPU master node and produce
    no SLURM log files. This logger writes all phase activity to a
    dedicated per-step file so failures can be diagnosed independently
    of the single combined master job log.

    Design:
      - One file per step: logs/steps/{step_id}_phases.log
      - Append mode: survives job cancellation and resubmission
      - Run boundary header on each open: job ID, node, timestamp
      - Immediate flush on every write: safe against sudden termination
      - Thread-safe: each sub-agent thread owns its own StepLogger instance
    """

    def __init__(self, step_id: str, project_root: Path):
        self.step_id = step_id
        self.log_dir = project_root / 'logs' / 'steps'
        self.log_path = self.log_dir / f"{step_id}_phases.log"
        self._fh = None
        self._open()

    def _open(self):
        """Open log file in append mode and write run boundary header."""
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.log_path, 'a', buffering=1)  # line-buffered

            job_id = os.environ.get('SLURM_JOB_ID', 'local')
            node = os.environ.get('SLURMD_NODENAME', '') or \
                   subprocess.run(
                       ['hostname', '-s'], capture_output=True, text=True
                   ).stdout.strip()

            header = (
                f"\n{'═' * 62}\n"
                f"  RUN   job={job_id}   node={node}\n"
                f"        time={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"{'═' * 62}\n"
            )
            self._fh.write(header)
            self._fh.flush()
        except Exception as e:
            # Logger must never crash the pipeline
            logger.warning(f"[{self.step_id}] StepLogger open failed: {e}")
            self._fh = None

    def log(self, message: str):
        """Write a timestamped line to the step log."""
        if self._fh is None:
            return
        try:
            ts = datetime.now().strftime('%H:%M:%S')
            self._fh.write(f"[{ts}] {message}\n")
            self._fh.flush()
        except Exception:
            pass

    def log_result(self, label: str, result: dict):
        """
        Write the full stdout and stderr from a subprocess result dict.

        This is the key method for capturing complete conda error output —
        the result dict is expected to have 'stdout', 'stderr', and
        optionally 'command' and 'return_code' keys.
        """
        if self._fh is None:
            return
        try:
            rc = result.get('return_code', '?')
            cmd = result.get('command', '')
            self.log(f"{label}  rc={rc}  cmd={cmd}")

            stdout = (result.get('stdout') or '').strip()
            stderr = (result.get('stderr') or '').strip()

            if stdout:
                self._fh.write(f"  --- STDOUT ---\n")
                self._fh.write(stdout + "\n")
            if stderr:
                self._fh.write(f"  --- STDERR ---\n")
                self._fh.write(stderr + "\n")
            if stdout or stderr:
                self._fh.write(f"  --- END ---\n")
            self._fh.flush()
        except Exception:
            pass

    def log_exception(self, e: Exception):
        """Write a full exception with traceback to the step log."""
        if self._fh is None:
            return
        try:
            import traceback
            self._fh.write(f"  --- EXCEPTION ---\n")
            traceback.print_exc(file=self._fh)
            self._fh.write(f"  {type(e).__name__}: {e}\n")
            self._fh.write(f"  --- END EXCEPTION ---\n")
            self._fh.flush()
        except Exception:
            pass

    def close(self):
        """Flush and close the log file handle."""
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass
            self._fh = None

# =============================================================================
# SUB-AGENT v1.2.0
# =============================================================================

class ScriptFirstSubAgentV3:
    """
    Sub-Agent v1.2.9 — 5-phase lifecycle with diagnostic + validation agents.

    Phase 1: Script Generation (validated two-pass)
    Phase 2: Environment Creation (script-informed, LLM-reviewed)
    Phase 3: Sbatch Generation (language-aware, submission-tested)
    Phase 4: Execution + Diagnostic Agent Loop
    Phase 5: Output Validation (ValidationAgent, max 3 correction retries)

    Phase 5 is additive — if ValidationAgent is unavailable or the step
    has no output_files, the step completes identically to v1.2.8.

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
    MAX_VALIDATION_ATTEMPTS = 3

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
        validation_agent=None,
        memory_client=None,
    ):
        self.agent_id = agent_id
        self.sandbox = sandbox
        self.conda_tools = conda_tools
        self.slurm_tools = slurm_tools
        self.use_slurm = use_slurm
        self.slurm_config = slurm_config or {}
        self.cleanup_env_on_success = cleanup_env_on_success
        self.diagnostic_memory = diagnostic_memory
        self.memory_client = memory_client

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

        # Per-step phase logger (v1.2.5) — initialized in execute()
        self.step_logger: Optional[StepLogger] = None

        # v1.2.8: Output manifest manager — initialized lazily on first write
        self._manifest: Optional['ManifestManager'] = None

        # v1.2.9: ValidationAgent — injected by workflow or lazily initialized
        self._validation_agent = validation_agent

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def execute(
        self, subtask: Dict, env_name: str = None
    ) -> Dict[str, Any]:
        """
        Execute a subtask through the 5-phase lifecycle.

        v1.2.9: Phase 5 (output validation) runs after Phase 4 success.
        Phases are sequential with independent error handling.
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

        # v1.2.5: Initialize per-step phase logger before anything else
        self.step_logger = StepLogger(task_id, self.project_root)

        try:
            # Initialize or resume checkpoint
            self.checkpoint = self._load_or_create_checkpoint(task_id)

            # Check if already completed
            if self._check_existing_outputs(subtask).get('already_complete'):
                self._delete_checkpoint(task_id)
                self.step_logger.log(f"Skipped — outputs already exist")
                return self._success_result(
                    task_id, message="Outputs exist", skipped=True
                )

            desc_preview = subtask.get('description', '')[:120].replace('\n', ' ')
            self.step_logger.log(f"Starting execute(): {desc_preview}")
            self.step_logger.log(
                f"Checkpoint phase={self.checkpoint.phase}  "
                f"status={self.checkpoint.status}"
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
            self.step_logger.log(
                f"Routed to cluster={routed_cluster}  "
                f"partition={routed_slurm.get('partition', '?')}"
            )

            try:
                # =============================================================
                # PHASE 1: Script Generation
                # =============================================================
                if self.checkpoint.phase < 1:
                    self.step_logger.log("► Entering Phase 1: Script Generation")
                    result = self._phase_1_generate_script(subtask)
                    if not result['success']:
                        self.step_logger.log(
                            f"✗ Phase 1 FAILED: {result.get('error', '')}"
                        )
                        self._update_checkpoint(status=TaskStatus.FAILED.value)
                        return self._failure_result(
                            task_id, f"Phase 1 failed: {result.get('error')}"
                        )
                    self._update_checkpoint(phase=1)
                    self.step_logger.log(
                        f"✓ Phase 1 complete: {self.checkpoint.script_path}"
                    )

                # =============================================================
                # PHASE 2: Environment Creation
                # =============================================================
                if self.checkpoint.phase < 2:
                    self.step_logger.log("► Entering Phase 2: Environment Creation")
                    result = self._phase_2_create_environment(subtask)
                    if not result['success']:
                        self.step_logger.log(
                            f"✗ Phase 2 FAILED: {result.get('error', '')}"
                        )
                        self._update_checkpoint(status=TaskStatus.FAILED.value)
                        return self._failure_result(
                            task_id, f"Phase 2 failed: {result.get('error')}"
                        )
                    self._update_checkpoint(phase=2)
                    self.step_logger.log(
                        f"✓ Phase 2 complete: env={self.checkpoint.env_name}"
                    )

                # =============================================================
                # PHASE 3: Sbatch Generation
                # =============================================================
                if self.checkpoint.phase < 3:
                    self.step_logger.log("► Entering Phase 3: Sbatch Generation")
                    result = self._phase_3_generate_sbatch(
                        subtask, routed_slurm
                    )
                    if not result['success']:
                        self.step_logger.log(
                            f"✗ Phase 3 FAILED: {result.get('error', '')}"
                        )
                        self._update_checkpoint(status=TaskStatus.FAILED.value)
                        return self._failure_result(
                            task_id, f"Phase 3 failed: {result.get('error')}"
                        )
                    self._update_checkpoint(phase=3)
                    self.step_logger.log(
                        f"✓ Phase 3 complete: {self.checkpoint.sbatch_path}"
                    )

                # =============================================================
                # PHASE 4: Execution + Diagnostic Loop
                # =============================================================
                if self.checkpoint.phase < 4:
                    self.step_logger.log("► Entering Phase 4: Execution")
                    result = self._phase_4_execute_and_monitor(
                        subtask, routed_cluster, routed_slurm
                    )
                    if not result.get('success'):
                        self.step_logger.log(
                            f"✗ Phase 4 complete: FAILED  "
                            f"{result.get('error', '')[:200]}"
                        )
                        return result
                    self.step_logger.log("✓ Phase 4 complete: SUCCESS")
                else:
                    # Resuming after Phase 4 — build minimal result for Phase 5
                    result = {'success': True, 'task_id': task_id}

                # =============================================================
                # PHASE 5: Output Validation (v1.2.9)
                # =============================================================
                self.step_logger.log("► Entering Phase 5: Validation")
                return self._phase_5_validate(
                    subtask, routed_cluster, routed_slurm, result
                )

            except Exception as e:
                logger.error(f"[{task_id}] Unhandled exception: {e}")
                self.step_logger.log(f"✗ Unhandled exception: {type(e).__name__}: {e}")
                self.step_logger.log_exception(e)
                self._update_checkpoint(
                    status=TaskStatus.FAILED.value,
                    last_error=str(e)
                )
                return self._failure_result(task_id, f"Unhandled: {e}")

        finally:
            # Always close the log, even on crash or cancellation
            self.step_logger.close()

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
        sl = self.step_logger  # convenience alias

        # Skip if script already exists and is validated
        if (self.checkpoint.script_path
                and Path(self.checkpoint.script_path).exists()
                and self.checkpoint.script_validated):
            print(f"  ✓ Script already validated: {self.checkpoint.script_path}")
            sl.log(f"Script already validated, skipping Phase 1: "
                   f"{self.checkpoint.script_path}")
            return {'success': True}

        sl.log(f"Language detected: {language}")
        sl.log(f"Script target: {script_path}")

        # Step 1a+1b: Outline generation with validation
        if not self.checkpoint.outline_validated:
            self._update_checkpoint(
                status=TaskStatus.GENERATING_OUTLINE.value
            )
            sl.log("Step 1a: Generating outline via LLM...")
            outline = self._generate_and_validate_outline(subtask, language)
            if not outline:
                sl.log("✗ Outline generation failed after all attempts")
                return {
                    'success': False,
                    'error': 'Could not generate validated outline'
                }
            sl.log(f"✓ Outline validated ({len(outline)} chars)")
            self._update_checkpoint(outline_validated=True)
        else:
            # Reconstruct outline from existing script context
            outline = subtask.get('expanded_plan', desc)
            sl.log("Outline already validated, using expanded_plan")

        # Step 1c+1d: Implementation with validation
        self._update_checkpoint(status=TaskStatus.GENERATING_SCRIPT.value)
        sl.log("Step 1c: Generating full script from outline...")
        script_content = self._generate_and_validate_script(
            subtask, outline, language
        )
        if not script_content:
            sl.log("✗ Script generation failed after all attempts")
            return {
                'success': False,
                'error': 'Could not generate validated script'
            }

        sl.log(f"✓ Script validated ({len(script_content)} chars, "
               f"{script_content.count(chr(10))} lines)")

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
        sl.log(f"✓ Phase 1 done: script={script_path}  env_name={env_name}")
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
                status=TaskStatus.GENERATING_OUTLINE.value
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
        hints = subtask.get('code_hints', [])
    
        content = None
        for attempt in range(self.MAX_IMPLEMENTATION_ATTEMPTS):
            self._update_checkpoint(
                status=TaskStatus.GENERATING_SCRIPT.value
            )

            # v1.2.9: Pass step_id for output path steering
            step_id = subtask.get('id', 'task')
            safe_step_id = re.sub(r'[^\w\-]', '_', step_id)[:30]

            if language == 'python':
                prompt = self._build_python_script_prompt(
                    desc, outline, packages, inputs, outputs, hints=hints,
                    step_id=safe_step_id,
                )
            elif language == 'r':
                prompt = self._build_r_script_prompt(
                    desc, outline, packages, inputs, outputs, hints=hints,
                    step_id=safe_step_id,
                )
            else:
                prompt = self._build_generic_script_prompt(
                    desc, outline, packages, inputs, outputs, language, hints=hints,
                    step_id=safe_step_id,
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
    
            content = self._extract_code_from_response(response, language)
            if not content or len(content) < 50:
                continue
    
            if language == 'python' and 'AGI_DRY_RUN' not in content:
                content = self._prepend_python_header(content, subtask)
    
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

        v1.2.5: All conda subprocess output (stdout + stderr) is written
        to the per-step phase log via self.step_logger so Phase 2 failures
        are fully diagnosable without a SLURM log file.
        """
        task_id = subtask.get('id', 'task')
        sl = self.step_logger  # convenience alias
        print(f"\n[{task_id}] Phase 2: Environment Creation")

        if self.checkpoint.env_created:
            print(f"  ✓ Environment already created: {self.checkpoint.env_name}")
            sl.log(f"Environment already created, skipping Phase 2: "
                   f"{self.checkpoint.env_name}")
            return {'success': True}

        # Proactive disk check
        self._ensure_disk_space()

        script_path = Path(self.checkpoint.script_path)
        language = self.checkpoint.language or 'python'
        env_name = self.checkpoint.env_name
        env_yaml_path = Path(self.checkpoint.env_yaml_path)
        env_yaml_path.parent.mkdir(parents=True, exist_ok=True)

        sl.log(f"env_name={env_name}  yaml={env_yaml_path}  language={language}")

        # Read script content
        script_content = script_path.read_text() if script_path.exists() else ""
        sl.log(f"Script content read: {len(script_content)} chars from {script_path}")

        # Step 2a: LLM dependency review
        self._update_checkpoint(status=TaskStatus.REVIEWING_DEPS.value)
        print(f"  → Reviewing dependencies via LLM...")
        sl.log("Step 2a: LLM dependency review...")

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
            sl.log(
                f"LLM identified: {len(dep_list.conda_packages)} conda, "
                f"{len(dep_list.pip_packages)} pip, "
                f"{len(dep_list.system_binaries)} binaries, "
                f"{len(dep_list.r_packages)} R packages"
            )
            # Step 2b: Generate YAML
            env_yaml = generate_env_yaml(dep_list, env_name)
            sl.log("Step 2b: YAML generated via LLM dependency list")

        except (ImportError, LLMInvocationError) as e:
            logger.warning(
                f"LLM dependency review failed ({e}), "
                f"falling back to regex-based extraction"
            )
            sl.log(f"⚠ LLM dep review failed ({e}), using regex fallback")
            env_yaml = self._generate_env_yaml_fallback(
                subtask, script_content, env_name, language
            )

        env_yaml_path.write_text(env_yaml)
        print(f"  ✓ Environment YAML: {env_yaml_path}")
        sl.log(f"YAML written to {env_yaml_path}:\n{env_yaml}")

        # Step 2c: Build environment
        self._update_checkpoint(status=TaskStatus.BUILDING_ENV.value)
        sl.log("Step 2c: Running conda env create (initial build)...")
        build_result = self._create_conda_environment()
        sl.log_result("Initial conda build", build_result)

        if build_result.get('success'):
            self._update_checkpoint(env_created=True)
            print(f"  ✓ Environment ready: {env_name}")
            sl.log(f"✓ Environment built successfully on first attempt: {env_name}")
            return {'success': True}

        # Step 2d: Repair loop
        print(f"  ⚠ Build failed, entering repair loop...")
        self._update_checkpoint(status=TaskStatus.REPAIRING_ENV.value)
        sl.log(f"✗ Initial build failed. Entering repair loop "
               f"(max {self.MAX_ENV_REPAIR_ITERATIONS} attempts)...")

        # v1.2.6: Track error signatures to detect oscillation patterns
        error_signatures = []

        for repair_attempt in range(self.MAX_ENV_REPAIR_ITERATIONS):
            if not self._can_continue():
                sl.log("Repair loop: _can_continue() returned False, stopping")
                break

            error = build_result.get('error', '')
            print(f"  → Repair attempt {repair_attempt + 1}: {error[:100]}")
            sl.log(f"Repair attempt {repair_attempt + 1}/{self.MAX_ENV_REPAIR_ITERATIONS}: "
                   f"error={error[:300]}")

            # v1.2.6: Classify and track error signature for oscillation detection
            sig = self._classify_repair_error(error)
            error_signatures.append(sig)

            if self._detect_repair_oscillation(error_signatures, sl):
                return {
                    'success': False,
                    'error': f"Env repair oscillation detected: {' ↔ '.join(set(error_signatures[-4:]))}"
                }

            repaired = self._repair_environment(error, env_yaml_path, env_name)
            if not repaired:
                sl.log("✗ _repair_environment() returned False — no repair action possible")
                return {
                    'success': False,
                    'error': f"Env repair exhausted: {error[:200]}"
                }

            sl.log(f"Repair action taken, retrying conda build...")
            build_result = self._create_conda_environment()
            sl.log_result(
                f"conda build after repair {repair_attempt + 1}", build_result
            )

            if build_result.get('success'):
                self._update_checkpoint(env_created=True)
                print(f"  ✓ Environment ready after repair: {env_name}")
                sl.log(f"✓ Environment built after repair attempt "
                       f"{repair_attempt + 1}: {env_name}")
                return {'success': True}

        final_error = build_result.get('error', 'unknown')
        sl.log(f"✗ Phase 2 exhausted all repair attempts. "
               f"Final error: {final_error}")
        return {
            'success': False,
            'error': f"Env build failed after {self.MAX_ENV_REPAIR_ITERATIONS} repairs"
        }

    def _repair_environment(
        self, error: str, yaml_path: Path, env_name: str
    ) -> bool:
        """Attempt to repair a failed environment build.

        Handles: disk quota, package not found, corrupted cache (SafetyError),
        transaction conflicts (ClobberError), stale prefix, solver conflicts.
        Returns True if a repair action was taken (caller should retry).

        v1.2.6: Added SafetyError (corrupted conda cache) and ClobberError
        handlers. SafetyError is checked early because it causes cascading
        ClobberError and prefix-exists failures.
        """
        error_lower = error.lower()

        # ── Disk quota ───────────────────────────────────────────────────
        if 'no space' in error_lower or 'quota' in error_lower:
            self._ensure_disk_space(force=True)
            return True

        # ── Corrupted package cache (SafetyError) ───────────────────────
        # v1.2.6: A corrupted tarball in ~/.conda/pkgs/ causes every build
        # that needs that package to fail. Must be handled BEFORE the
        # prefix-exists check because a corrupted cache leaves a partial
        # prefix on disk, creating an oscillation loop.
        if 'safetyerror' in error_lower or 'appears to be corrupted' in error_lower or 'incorrect size' in error_lower:
            print(f"    Corrupted conda cache detected, cleaning...")
            try:
                # Parse the specific corrupted package path
                # Pattern: "The package for X located at /path/to/pkgs/X-version"
                corrupted_path = None
                path_match = re.search(
                    r'located at\s+(\S+)', error
                )
                if path_match:
                    corrupted_path = Path(path_match.group(1))

                # Remove the specific corrupted package directory
                if corrupted_path and corrupted_path.exists():
                    import shutil
                    shutil.rmtree(str(corrupted_path), ignore_errors=True)
                    print(f"    ✓ Removed corrupted cache: {corrupted_path}")
                    # Also remove the corresponding .tar.bz2 / .conda if present
                    for suffix in ['.tar.bz2', '.conda']:
                        archive = corrupted_path.with_suffix(suffix)
                        if archive.exists():
                            archive.unlink(missing_ok=True)
                            print(f"    ✓ Removed cached archive: {archive}")

                # Run conda clean --packages to verify cache integrity
                subprocess.run(
                    ['conda', 'clean', '--packages', '--yes'],
                    capture_output=True, text=True, timeout=120,
                )
                print(f"    ✓ conda clean --packages completed")

                # Also remove stale env prefix if it exists (the corrupted
                # build likely left one behind)
                self._remove_stale_prefix(env_name)

                return True
            except Exception as e:
                logger.warning(f"Corrupted cache cleanup failed: {e}")
                # Still return True to retry — the clean may have partially worked
                return True

        # ── ClobberError (incompatible package transactions) ─────────────
        # v1.2.6: ClobberError means two packages want to write the same file.
        # Often co-occurs with SafetyError (handled above). Standalone cases
        # need cache cleanup + fresh prefix.
        if 'clobbererror' in error_lower or 'incompatible packages' in error_lower:
            print(f"    ClobberError detected, cleaning cache and prefix...")
            try:
                # Clean the entire package cache to remove any conflicting state
                subprocess.run(
                    ['conda', 'clean', '--packages', '--tarballs', '--yes'],
                    capture_output=True, text=True, timeout=120,
                )
                print(f"    ✓ conda clean --packages --tarballs completed")

                # Remove stale prefix so we get a clean create
                self._remove_stale_prefix(env_name)

                return True
            except Exception as e:
                logger.warning(f"ClobberError cleanup failed: {e}")
                return True

        # ── Package not found on conda → move to pip ─────────────────────
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

        # ── Stale prefix on disk ─────────────────────────────────────────
        if 'prefix already exists' in error_lower or 'condavalueerror' in error_lower:
            return self._remove_stale_prefix(env_name)

        # ── Conflict → ask LLM for version resolution ────────────────────
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

        # ── Pip install failure ───────────────────────────────────────────
        # v1.2.6b: Distinguish permanent pip failures (package doesn't exist)
        # from transient ones (network timeout). Permanent failures need the
        # offending package removed from YAML + partial prefix cleaned up.
        if 'pip' in error_lower and ('failed' in error_lower or 'pip subprocess error' in error_lower):
            # Parse which pip packages failed
            pip_not_found = re.findall(
                r'No matching distribution found for (\S+)', error
            )
            if not pip_not_found:
                pip_not_found = re.findall(
                    r'Could not find a version that satisfies the requirement (\S+)',
                    error
                )

            if pip_not_found and yaml and yaml_path.exists():
                # Permanent failure: remove unfindable packages from YAML
                for pkg in pip_not_found:
                    pkg_clean = re.split(r'[><=!\[]', pkg)[0].strip()
                    if not pkg_clean:
                        continue
                    if self.conda_tools:
                        self.conda_tools.remove_package_from_yaml(
                            str(yaml_path), pkg_clean
                        )
                        print(f"    Removed unfindable pip package: {pkg_clean}")
                # Also clean up the partial prefix left by the failed build
                self._remove_stale_prefix(env_name)
                return True

            # No specific package parsed — treat as transient, but still
            # clean the partial prefix to avoid prefix-exists on retry
            self._remove_stale_prefix(env_name)
            return True


    def _remove_stale_prefix(self, env_name: str) -> bool:
        """Remove a stale conda environment prefix left by a failed build.

        v1.2.6: Extracted from _repair_environment so it can be called as a
        combo action from the SafetyError and ClobberError handlers too.

        Returns True if the prefix was removed (caller should retry).
        """
        print(f"    Removing stale env prefix: {env_name}")
        try:
            remove_result = subprocess.run(
                ['conda', 'env', 'remove', '-n', env_name, '-y', '--quiet'],
                capture_output=True, text=True, timeout=120,
            )
            if remove_result.returncode == 0:
                print(f"    ✓ Stale prefix removed, will retry create")
                return True
            else:
                # conda env remove failed — try direct rm as fallback
                conda_envs_root = subprocess.run(
                    ['conda', 'info', '--json'],
                    capture_output=True, text=True, timeout=30,
                )
                if conda_envs_root.returncode == 0:
                    import json as _json
                    info = _json.loads(conda_envs_root.stdout)
                    for env_dir in info.get('envs_dirs', []):
                        prefix = Path(env_dir) / env_name
                        if prefix.exists():
                            import shutil
                            shutil.rmtree(str(prefix), ignore_errors=True)
                            print(f"    ✓ Removed prefix via shutil: {prefix}")
                            return True
                print(f"    ✗ Could not remove stale prefix")
                return False
        except Exception as e:
            logger.warning(f"Stale prefix removal failed: {e}")
            return False

    @staticmethod
    def _classify_repair_error(error: str) -> str:
        """Classify a conda build error into a short signature string.

        v1.2.6: Used by the repair loop to detect oscillation patterns
        (e.g. alternating between SafetyError and prefix-exists).

        Returns a short, stable string like 'safety_error', 'prefix_exists',
        'packages_not_found', etc.
        """
        el = error.lower()
        if 'safetyerror' in el or 'appears to be corrupted' in el or 'incorrect size' in el:
            return 'safety_error'
        if 'clobbererror' in el or 'incompatible packages' in el:
            return 'clobber_error'
        if 'prefix already exists' in el or 'condavalueerror' in el:
            return 'prefix_exists'
        if 'packagesnotfounderror' in el or 'resolvepackagenotfound' in el:
            return 'packages_not_found'
        if 'no space' in el or 'quota' in el:
            return 'disk_quota'
        if 'conflict' in el:
            return 'conflict'
        if 'pip' in el and 'failed' in el:
            return 'pip_failed'
        return 'unknown'

    @staticmethod
    def _detect_repair_oscillation(
        signatures: list, sl=None
    ) -> bool:
        """Detect if the repair loop is stuck in a repeating error pattern.

        v1.2.6: Prevents burning all 15 iterations on unresolvable cycles.

        Detects:
          - Same error 3+ times → not making progress
          - Same pair alternating 2+ full cycles (4 entries) → oscillation

        Returns True if the loop should bail out.
        """
        if len(signatures) < 3:
            return False

        # Check 1: Same error appearing 3+ times total
        from collections import Counter
        counts = Counter(signatures)
        for sig, count in counts.items():
            if count >= 3:
                msg = (f"✗ Repair oscillation: '{sig}' appeared {count} times "
                       f"across {len(signatures)} attempts — not making progress")
                print(f"    {msg}")
                if sl:
                    sl.log(msg)
                return True

        # Check 2: A-B-A-B alternating pattern (2 full cycles = 4 entries)
        if len(signatures) >= 4:
            last4 = signatures[-4:]
            if (last4[0] == last4[2] and last4[1] == last4[3]
                    and last4[0] != last4[1]):
                msg = (f"✗ Repair oscillation: alternating between "
                       f"'{last4[0]}' and '{last4[1]}' — breaking cycle")
                print(f"    {msg}")
                if sl:
                    sl.log(msg)
                return True

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

        sl = self.step_logger
        print(f"\n[{task_id}] Phase 3: Sbatch Generation")

        if (self.checkpoint.sbatch_path
                and Path(self.checkpoint.sbatch_path).exists()):
            print(f"  ✓ Sbatch already exists: {self.checkpoint.sbatch_path}")
            sl.log(f"Sbatch already exists, skipping Phase 3: "
                   f"{self.checkpoint.sbatch_path}")
            return {'success': True}

        sbatch_path = (
            self.project_root / 'slurm' / 'scripts' / f"{safe_id}.sbatch"
        )
        sbatch_path.parent.mkdir(parents=True, exist_ok=True)
        sl.log(f"Generating sbatch: {sbatch_path}")
        sl.log(f"Cluster={routed_slurm.get('cluster_name')}  "
               f"partition={routed_slurm.get('partition')}")

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

        # Log the generated sbatch content so Phase 3 failures are diagnosable
        sl.log(f"✓ Sbatch written ({len(sbatch_content)} chars)")
        sl.log(f"Sbatch content:\n{sbatch_content}")
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

                            # v1.2.8: Write manifest entry before cleanup
                            # so the protected paths are recorded even if
                            # cleanup raises an exception.
                            self._write_manifest_entry(subtask, outputs)

                            # v1.2.9: Env cleanup and final completion moved
                            # to Phase 5 (validation pass). Set phase=4 so
                            # execute() proceeds to Phase 5 on resume.
                            self._update_checkpoint(phase=4)

                            return self._success_result(
                                task_id=task_id,
                                script_path=self.checkpoint.script_path,
                                job_id=self.checkpoint.current_job_id,
                                iterations=self.checkpoint.iteration,
                                routed_cluster=routed_cluster,
                                report=report,
                                phase4_job_id=prod_submit['job_id'],
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
    # PHASE 5: OUTPUT VALIDATION (v1.2.9)
    # =========================================================================

    def _phase_5_validate(
        self,
        subtask: Dict,
        routed_cluster: str,
        routed_slurm: Dict,
        phase4_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run Phase 5 validation lifecycle with retry loop.

        v1.2.9: Generates a validation script, submits it as a lightweight
        SLURM job, parses the structured JSON report, and either completes
        the step or synthesizes correction guidance and retries from Phase 1.

        Graceful degradation: if ValidationAgent is unavailable or the step
        has no output_files, completes the step as v1.2.8 would.
        """
        task_id = subtask.get('id', 'task')
        sl = self.step_logger

        # ── Graceful degradation ─────────────────────────────────────────
        val_agent = self._get_validation_agent()
        if val_agent is None:
            sl.log("Phase 5 skipped — ValidationAgent not available")
            self._finalize_step_completion(task_id, routed_cluster)
            return phase4_result

        output_files = subtask.get('output_files', [])
        if not output_files:
            sl.log("Phase 5 skipped — no output_files to validate")
            self._finalize_step_completion(task_id, routed_cluster)
            return phase4_result

        print(f"\n[{task_id}] Phase 5: Output Validation")

        # ── Validation retry loop ────────────────────────────────────────
        for attempt in range(self.MAX_VALIDATION_ATTEMPTS):
            self.checkpoint.validation_attempts = attempt + 1
            self._update_checkpoint(
                validation_attempts=attempt + 1,
                status=TaskStatus.GENERATING_VALIDATION.value,
            )
            sl.log(f"Phase 5 attempt {attempt + 1}/{self.MAX_VALIDATION_ATTEMPTS}")

            # ── 5a: Generate validation script ───────────────────────────
            gen_result = val_agent.generate_validation_script(
                subtask=subtask,
                analysis_script_path=self.checkpoint.script_path,
                checkpoint=self.checkpoint,
            )
            if not gen_result.get('success'):
                sl.log(f"✗ 5a failed: {gen_result.get('error', '')}")
                sl.log("Phase 5 skipped — validation script generation failed")
                self._finalize_step_completion(task_id, routed_cluster)
                return phase4_result

            self._update_checkpoint(
                validation_script_path=gen_result['script_path']
            )
            sl.log(f"✓ 5a: {gen_result['script_path']}")

            # ── 5b: Generate sbatch + submit ─────────────────────────────
            phase4_job_id = phase4_result.get('phase4_job_id', '')
            self._update_checkpoint(
                status=TaskStatus.WAITING_VALIDATION_JOB.value
            )

            sbatch_result = val_agent.generate_validation_sbatch(
                subtask=subtask,
                checkpoint=self.checkpoint,
                phase4_job_id=phase4_job_id,
                slurm_config=routed_slurm,
            )
            if not sbatch_result.get('success'):
                sl.log(f"✗ 5b failed: {sbatch_result.get('error', '')}")
                sl.log("Phase 5 skipped — validation sbatch failure")
                self._finalize_step_completion(task_id, routed_cluster)
                return phase4_result

            val_job_id = sbatch_result['job_id']
            self._update_checkpoint(
                validation_sbatch_path=sbatch_result.get('sbatch_path'),
                validation_job_id=val_job_id,
            )
            sl.log(f"✓ 5b: Validation job {val_job_id}")

            # ── 5c: Poll validation job ──────────────────────────────────
            print(f"  → Waiting for validation job {val_job_id}...")
            poll_result = self._poll_validation_job(val_job_id)
            val_report = None
            raw_tail = ""

            if not poll_result.get('success'):
                sl.log(f"✗ 5c: Validation job SLURM failure: "
                       f"{poll_result.get('state', '?')}")
                raw_tail = f"SLURM state: {poll_result.get('state', 'UNKNOWN')}"
            else:
                log_path = val_agent.find_validation_log(task_id, val_job_id)
                if log_path:
                    parse_result = val_agent.parse_validation_output(log_path)
                else:
                    parse_result = {
                        'parsed': False, 'raw_tail': '',
                        'error': 'Validation log not found',
                    }

                if parse_result.get('parsed'):
                    val_report = parse_result['report']
                else:
                    raw_tail = parse_result.get('raw_tail', '')
                    sl.log(f"⚠ Cannot parse report: "
                           f"{parse_result.get('error', '')}")

            # ── Evaluate result ──────────────────────────────────────────
            if val_report and val_report.passed:
                print(f"  ✓ Validation PASSED: {val_report.summary}")
                sl.log(f"✓ PASSED: {val_report.summary}")
                self._update_checkpoint(
                    validation_passed=True,
                    status=TaskStatus.VALIDATION_PASSED.value,
                )
                val_agent.store_validation_success(task_id, val_report)
                self._finalize_step_completion(task_id, routed_cluster)
                return phase4_result

            # ── FAILED ───────────────────────────────────────────────────
            self._update_checkpoint(
                status=TaskStatus.VALIDATION_FAILED.value
            )
            summary = val_report.summary if val_report else raw_tail
            print(f"  ✗ Validation FAILED (attempt {attempt + 1}): "
                  f"{summary[:150]}")
            sl.log(f"✗ FAILED: {summary}")

            # Last attempt — don't synthesize, just fail
            if attempt + 1 >= self.MAX_VALIDATION_ATTEMPTS:
                break

            # ── Synthesize correction + retry from Phase 1 ───────────────
            sl.log("Synthesizing correction guidance...")
            correction = val_agent.synthesize_correction(
                subtask=subtask,
                validation_report=val_report,
                raw_log_tail=raw_tail if not val_report else "",
            )

            # Dual-inject hints (same pattern as INJECT_HINTS.sh)
            hints = correction.get('hints', [])
            subtask['code_hints'] = subtask.get('code_hints', []) + hints

            guidance = correction.get('guidance_block', '')
            if guidance:
                plan = subtask.get('expanded_plan',
                                   subtask.get('description', ''))
                if '[VALIDATION FAILURE' in plan:
                    plan = plan[:plan.index('[VALIDATION FAILURE')]
                subtask['expanded_plan'] = plan.rstrip() + guidance

            sl.log(f"Injected {len(hints)} correction hints, "
                   f"resetting to Phase 1")

            self._reset_checkpoint_to_phase1()

            # ── Re-run Phases 1–4 ────────────────────────────────────────
            sl.log("► Re-running Phases 1–4 with corrected guidance...")

            p1 = self._phase_1_generate_script(subtask)
            if not p1['success']:
                sl.log(f"✗ Phase 1 retry failed: {p1.get('error', '')}")
                continue
            self._update_checkpoint(phase=1)

            p2 = self._phase_2_create_environment(subtask)
            if not p2['success']:
                sl.log(f"✗ Phase 2 retry failed: {p2.get('error', '')}")
                continue
            self._update_checkpoint(phase=2)

            p3 = self._phase_3_generate_sbatch(subtask, routed_slurm)
            if not p3['success']:
                sl.log(f"✗ Phase 3 retry failed: {p3.get('error', '')}")
                continue
            self._update_checkpoint(phase=3)

            p4 = self._phase_4_execute_and_monitor(
                subtask, routed_cluster, routed_slurm
            )
            if not p4['success']:
                sl.log(f"✗ Phase 4 retry failed: {p4.get('error', '')}")
                continue

            phase4_result = p4
            sl.log("✓ Phases 1–4 re-completed, retrying validation...")

        # ── Exhausted all validation attempts ────────────────────────────
        sl.log(f"✗ Validation failed after "
               f"{self.MAX_VALIDATION_ATTEMPTS} attempts")
        self._update_checkpoint(status=TaskStatus.FAILED.value)
        return self._failure_result(
            task_id,
            f"Validation failed after {self.MAX_VALIDATION_ATTEMPTS} attempts",
        )

    def _get_validation_agent(self):
        """Lazily initialize and return the ValidationAgent.

        Returns None if unavailable (graceful degradation — Phase 5 skipped).
        """
        if self._validation_agent is not None:
            return self._validation_agent

        try:
            from agents.validation_agent import ValidationAgent
            self._validation_agent = ValidationAgent(
                agent_id=f"{self.agent_id}_validator",
                project_root=str(self.project_root),
                ollama_model=None,
                ollama_base_url=self.ollama_base_url,
                memory_client=self.memory_client,
            )
            return self._validation_agent
        except ImportError:
            logger.info(
                "ValidationAgent not available — Phase 5 will be skipped"
            )
            return None
        except Exception as e:
            logger.warning(f"ValidationAgent init failed: {e}")
            return None

    def _finalize_step_completion(
        self, task_id: str, routed_cluster: str
    ):
        """Env cleanup + mark completed. Shared by Phase 4 skip and Phase 5 pass.

        v1.2.9: Extracted from the Phase 4 production success block so both
        the 'no validation' path and the 'validation passed' path use
        identical cleanup logic.
        """
        if self.cleanup_env_on_success:
            print(f"  → Cleaning up environment...")
            cleanup = self._cleanup_conda_environment()
            if cleanup.get('success'):
                print(f"  ✓ Environment removed (YAML preserved)")

        self._update_checkpoint(status=TaskStatus.COMPLETED.value)
        self._delete_checkpoint(task_id)

    def _poll_validation_job(self, job_id: str) -> Dict[str, Any]:
        """Poll SLURM until validation job completes.

        Same logic as _wait_for_job() but accepts an explicit job_id
        instead of reading from checkpoint. Validation jobs have a
        30-minute wall time so this will not run indefinitely.
        """
        poll_interval = self.slurm_config.get('poll_interval', 30)

        while True:
            try:
                result = subprocess.run(
                    ['squeue', '-j', job_id, '-h', '-o', '%T'],
                    capture_output=True, text=True, timeout=30,
                )
                state = result.stdout.strip()

                if not state:
                    sacct = subprocess.run(
                        ['sacct', '-j', job_id, '-n', '-o',
                         'State', '-X'],
                        capture_output=True, text=True, timeout=30,
                    )
                    final_state = (
                        sacct.stdout.strip().split('\n')[0].strip()
                    )
                    return {
                        'success': final_state in ('COMPLETED',),
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

    def _reset_checkpoint_to_phase1(self):
        """Reset checkpoint for Phase 1 restart after validation failure.

        Preserves: task_id, env_name, env_yaml_path, validation_attempts,
                   created_at.
        Clears: script, sbatch, job state, all phase markers.
        """
        self._update_checkpoint(
            phase=0,
            script_path=None,
            sbatch_path=None,
            current_job_id=None,
            last_error=None,
            outline_validated=False,
            script_validated=False,
            dry_run_succeeded=False,
            env_created=False,
            validation_script_path=None,
            validation_sbatch_path=None,
            validation_job_id=None,
            validation_passed=False,
            status=TaskStatus.NOT_STARTED.value,
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
        """Create conda env from YAML. Pip fallback on failure.

        v1.2.6: Always returns the full subprocess result dict including
        'stdout', 'stderr', 'return_code', and 'command' so the caller
        (_phase_2_create_environment) can pass it to log_result() for
        complete Phase 2 failure diagnosis.

        v1.2.6b: Pre-remove existing environment on BOTH code paths
        (conda_tools and subprocess fallback) to prevent prefix-exists
        failures when retrying a step from a previous pipeline run.
        """
        if not self.checkpoint or not self.checkpoint.env_yaml_path:
            return {
                'success': False,
                'error': 'No env YAML path',
                'stdout': '',
                'stderr': '',
            }

        env_yaml_path = self.checkpoint.env_yaml_path
        env_name = self.checkpoint.env_name

        # v1.2.6b: Always remove existing env before create, regardless
        # of code path. This prevents prefix-exists failures when a step
        # is retried from a previous pipeline run or after a partial build.
        self._pre_remove_existing_env(env_name)

        if self.conda_tools:
            result = self.conda_tools.create_from_yaml(
                yaml_path=env_yaml_path,
                env_name=env_name,
            )
            # conda_tools already returns stdout/stderr in its result dict
            return result

        # Fallback: direct subprocess
        cmd = ['conda', 'env', 'create', '-f', env_yaml_path,
               '-n', env_name, '--yes']
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True, text=True,
            )

            if proc.returncode == 0:
                return {
                    'success': True,
                    'stdout': proc.stdout,
                    'stderr': proc.stderr,
                    'return_code': proc.returncode,
                    'command': ' '.join(cmd),
                }

            # Try pip fallback for failed packages
            stderr = proc.stderr
            failed = re.findall(
                r'PackagesNotFoundError.*?- (\S+)', stderr, re.DOTALL
            )
            if failed:
                return self._pip_fallback_build(
                    env_yaml_path, env_name, failed, stderr
                )

            return {
                'success': False,
                'error': stderr[-500:],
                'stdout': proc.stdout,
                'stderr': proc.stderr,
                'return_code': proc.returncode,
                'command': ' '.join(cmd),
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'stdout': '',
                'stderr': str(e),
                'return_code': -1,
                'command': ' '.join(cmd),
            }

    def _pre_remove_existing_env(self, env_name: str):
        """Remove an existing conda environment before creating a new one.

        v1.2.6b: Extracted to ensure both the conda_tools and subprocess
        fallback paths in _create_conda_environment() handle pre-existing
        environments consistently. Silently succeeds if the env doesn't exist.
        """
        try:
            existing_check = subprocess.run(
                ['conda', 'env', 'list'],
                capture_output=True, text=True, timeout=30,
            )
            if existing_check.returncode == 0 and env_name in existing_check.stdout:
                logger.info(f"Removing existing env before create: {env_name}")
                subprocess.run(
                    ['conda', 'env', 'remove', '-n', env_name, '-y', '--quiet'],
                    capture_output=True, text=True, timeout=180,
                )
        except Exception as e:
            logger.warning(f"Pre-remove existing env failed: {e}")

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
        """Poll squeue until job completes.

        v1.2.7: No artificial timeout. Polls indefinitely until SLURM
        reports a terminal state. The SLURM job's own wall time limit
        (--time in sbatch) is the only exit condition. This prevents
        premature POLL_TIMEOUT failures when jobs sit in queue on a
        busy cluster.
        """
        if not self.checkpoint or not self.checkpoint.current_job_id:
            return {'success': False, 'error': 'No job ID'}

        job_id = self.checkpoint.current_job_id
        poll_interval = self.slurm_config.get('poll_interval', 30)

        while True:
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
        self, desc, outline, packages, inputs, outputs, hints=None,
        step_id=None
    ) -> str:
        hints_block = ""
        if hints:
            hints_block = "\nIMPORTANT IMPLEMENTATION GUIDANCE (follow exactly):\n"
            hints_block += "\n".join(f"  - {h}" for h in hints)
            hints_block += "\n"

        # v1.2.9: Output path steering — all outputs go under outputs/{step_id}/
        output_dir_id = step_id or "step"
        output_rule = f"""
OUTPUT DIRECTORY RULE (MANDATORY):
  All output files MUST be written under outputs/{output_dir_id}/.
  Create this directory at the top of main() before any processing.
  Example: OUTPUT_DIR = PROJECT_ROOT / "outputs" / "{output_dir_id}"
           OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  Every save/write call must use OUTPUT_DIR / "filename" as the path.
  Do NOT write outputs to the project root or any other directory.
"""

        return f"""Generate a complete Python script based on this validated outline.

TASK: {desc}
PACKAGES: {', '.join(packages)}
INPUTS: {', '.join(inputs) if inputs else 'None'}
OUTPUTS: {', '.join(outputs) if outputs else 'None'}
{hints_block}{output_rule}
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

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "{output_dir_id}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[MODE] DRY_RUN={{DRY_RUN}}, PROJECT_ROOT={{PROJECT_ROOT}}")
print(f"[OUTPUT_DIR] {{OUTPUT_DIR}}")

def main():
    # ALL output files go under OUTPUT_DIR
    # Implement every step from the outline
    # Wrap saves with: if not DRY_RUN: save() else: print("[DRY-RUN] Would save...")
    print("SUCCESS: Task completed")

if __name__ == "__main__":
    main()
```

Generate ONLY the complete Python code."""

    def _build_r_script_prompt(
        self, desc, outline, packages, inputs, outputs, hints=None,
        step_id=None
    ) -> str:
        hints_block = ""
        if hints:
            hints_block = "\nIMPORTANT IMPLEMENTATION GUIDANCE (follow exactly):\n"
            hints_block += "\n".join(f"  - {h}" for h in hints)
            hints_block += "\n"

        # v1.2.9: Output path steering
        output_dir_id = step_id or "step"

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
- Create output directory: OUTPUT_DIR <- file.path(PROJECT_ROOT, "outputs", "{output_dir_id}")
  dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
- ALL output files MUST be written under OUTPUT_DIR (use file.path(OUTPUT_DIR, "filename"))
- Do NOT write outputs to the project root or any other directory
- Implement every step from the outline
- Print "SUCCESS: Task completed" on completion

Generate ONLY the R code."""

    def _build_generic_script_prompt(
        self, desc, outline, packages, inputs, outputs, language, hints=None,
        step_id=None
    ) -> str:
        hints_block = ""
        if hints:
            hints_block = "\nIMPORTANT IMPLEMENTATION GUIDANCE (follow exactly):\n"
            hints_block += "\n".join(f"  - {h}" for h in hints)
            hints_block += "\n"

        # v1.2.9: Output path steering
        output_dir_id = step_id or "step"

        return f"""Generate a complete {language} script based on this outline.

TASK: {desc}
{hints_block}
OUTLINE: {outline}

Requirements:
- Create output directory: outputs/{output_dir_id}/ (mkdir -p or equivalent)
- ALL output files MUST be written under outputs/{output_dir_id}/
- Do NOT write outputs to the project root or any other directory
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
        step_id = subtask.get('id', 'unknown')
        safe_id = re.sub(r'[^\w\-]', '_', step_id)[:30]
        header = f'''#!/usr/bin/env python3
"""Task: {step_id}"""
import os
from pathlib import Path

DRY_RUN = os.environ.get('AGI_DRY_RUN', 'true').lower() == 'true'
PROJECT_ROOT = Path(os.environ.get('PROJECT_DIR', '.')).resolve()
os.chdir(PROJECT_ROOT)

# v1.2.9: All outputs under outputs/{{step_id}}/
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "{safe_id}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[MODE] DRY_RUN={{DRY_RUN}}, PROJECT_ROOT={{PROJECT_ROOT}}")
print(f"[OUTPUT_DIR] {{OUTPUT_DIR}}")

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

    def _write_manifest_entry(
        self, subtask: Dict, outputs: Dict[str, Any]
    ) -> None:
        """Write a manifest entry for a successfully completed step.

        v1.2.8: Called at the end of Phase 4 after production run success,
        before environment cleanup. Records all output paths, artifact paths,
        and env_state so cleanup scripts can protect them correctly.

        The manifest is written even if ManifestManager import failed at
        module load — the try/except ensures this never crashes the pipeline.
        """
        if not _MANIFEST_AVAILABLE:
            return

        try:
            if self._manifest is None:
                self._manifest = ManifestManager(self.project_root)

            task_id = subtask.get('id', 'unknown')

            # Build output entries from verified found files
            found_files = outputs.get('found_files', [])
            output_entries = [
                {'path': f, 'description': f"Output from {task_id}"}
                for f in found_files
            ]

            # Also add any output_files from the subtask definition that
            # exist on disk but weren't in found_files (edge case safety net)
            found_set = set(found_files)
            for rel_path in subtask.get('output_files', []):
                abs_path = str(self.project_root / rel_path)
                if abs_path not in found_set and Path(abs_path).exists():
                    output_entries.append({
                        'path': abs_path,
                        'description': f"Output from {task_id} (subtask spec)",
                    })

            self._manifest.write_step_entry(
                step_id=task_id,
                outputs=output_entries,
                env_yaml=self.checkpoint.env_yaml_path if self.checkpoint else None,
                script=self.checkpoint.script_path if self.checkpoint else None,
                sbatch=self.checkpoint.sbatch_path if self.checkpoint else None,
                env_name=self.checkpoint.env_name if self.checkpoint else None,
                env_state='agent_created',  # default; overridden by INJECT_HINTS if human-modified
                status='completed',
            )
            self.step_logger.log(
                f"✓ Manifest entry written: {len(output_entries)} output(s) recorded"
            )

        except Exception as e:
            # Manifest write must never crash the pipeline
            logger.warning(f"[{subtask.get('id', '?')}] Manifest write failed: {e}")
            if self.step_logger:
                self.step_logger.log(f"⚠ Manifest write failed (non-fatal): {e}")


# Aliases for backward compatibility
ScriptFirstSubAgent = ScriptFirstSubAgentV3
SubAgentV3 = ScriptFirstSubAgentV3

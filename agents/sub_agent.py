"""
Script-First Sub-Agent v3.2 - Complete Implementation

Features:
- DUAL CLUSTER ROUTING: AGI_CLUSTER (CPU) + AGI_GPU_CLUSTER (GPU subtasks)
- GPU-AWARE SBATCH: Auto-routes tasks to GPU/CPU based on package detection
- GPU NODE RULES: No --mem on GPU partitions, --gres=gpu:N format
- CONDA CLEANUP: Removes environment after success (YAML preserved)
- STATE CHECKPOINTING: Resume from where you left off
- TOKEN BUDGET: Sized for qwen3-coder-next:latest 32K context window
- PROPER SLURM: Jobs appear in squeue, uses sbatch correctly
- OPEN-SOURCE CHANNELS: conda-forge, bioconda only (no defaults/main)

Flow:
1. Load cluster config (arc_compute1, arc_gpu1v100, etc.)
2. Route task to GPU or CPU cluster based on packages/metadata
3. Generate artifacts (env.yml, script.py, sbatch with cluster settings)
4. Create conda env
5. Submit via sbatch → appears in squeue
6. Wait → Analyze → Retry or Complete
7. On success: cleanup conda env, delete checkpoint

GPU NODE RULES (ARC):
- Do NOT specify --mem on GPU partitions (causes allocation failures)
- Use --gres=gpu:N format (not --gpus N)
- Standard GPU request: --gres=gpu:1 -N 1 -n 1 -c 80
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import re
import json
import time
import subprocess
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

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


# =============================================================================
# CLUSTER CONFIGURATION (v3.2 - GPU Routing Support)
# =============================================================================
# Changes from v3.1:
#   - Default cluster: arc_compute1 (was zeus_cpu)
#   - Added get_gpu_cluster() for GPU subtask routing
#   - Added get_cluster_for_task() for automatic CPU/GPU routing
#   - Reads AGI_GPU_CLUSTER env var for GPU partition selection
#   - Reads gpu_packages list from cluster_config.yaml for auto-detection
#   - GPU directive format: --gres=gpu:{count} (was --gpus {count})
#   - Memory guard: NEVER set --mem on GPU partitions
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
                    # NO memory key - GPU nodes must NOT specify --mem
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
        self.gpu_cluster_name = os.environ.get('AGI_GPU_CLUSTER', 'arc_gpu1v100')
    
    def _load_config(self) -> Dict:
        """Load cluster config from file."""
        config_path = os.environ.get('AGI_CLUSTER_CONFIG')
        
        if not config_path:
            # Try default locations
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
                print(f"[ClusterConfig] Warning: Failed to load {config_path}: {e}")
        
        return self.DEFAULT_CONFIG
    
    @property
    def slurm(self) -> Dict:
        """SLURM settings for current (CPU) cluster."""
        return self.cluster.get('slurm', {})
    
    @property
    def gpu(self) -> Dict:
        """GPU settings for current cluster."""
        return self.cluster.get('gpu', {})
    
    @property
    def limits(self) -> Dict:
        """Resource limits for current cluster."""
        return self.cluster.get('limits', {})
    
    def get_slurm_value(self, key: str, default: Any = None) -> Any:
        """Get a SLURM setting with env override support."""
        env_key = f'AGI_SUBTASK_{key.upper()}'
        env_val = os.environ.get(env_key)
        if env_val:
            return env_val
        return self.slurm.get(key, default)
    
    def is_gpu_cluster(self) -> bool:
        """Check if current default cluster has GPUs."""
        return self.gpu.get('available', False)
    
    def get_gpu_directive(self, count: int = None) -> str:
        """Get GPU SBATCH directive string (--gres=gpu:N format)."""
        gpu = self.cluster.get('gpu', {})
        if not gpu.get('available', False):
            return ""
        count = count or gpu.get('default_count', 1)
        fmt = gpu.get('directive_format', '--gres=gpu:{count}')
        return fmt.format(count=count)
    
    def get_gpu_cluster(self) -> Optional[Dict]:
        """Load GPU cluster config for tasks requiring GPU resources."""
        gpu_cluster = self.config.get('clusters', {}).get(self.gpu_cluster_name)
        if gpu_cluster:
            return gpu_cluster
        # Fallback: return current cluster if it has GPU
        if self.cluster.get('gpu', {}).get('available', False):
            return self.cluster
        return None
    
    def get_cluster_for_task(self, task_description: str = "", requires_gpu: bool = False) -> Dict:
        """Select appropriate cluster config based on task requirements.
        
        Routes tasks to GPU or CPU partitions:
        - If requires_gpu=True → GPU cluster
        - If task_description mentions GPU packages → GPU cluster
        - Otherwise → default CPU cluster
        
        Returns:
            Cluster config dict with 'slurm' and 'gpu' sections
        """
        # Explicit GPU request
        if requires_gpu:
            gpu_cluster = self.get_gpu_cluster()
            if gpu_cluster:
                return gpu_cluster
        
        # Check task description for GPU package keywords
        if task_description:
            gpu_packages = self.config.get(
                'gpu_packages',
                self.DEFAULT_CONFIG.get('gpu_packages', [])
            )
            task_lower = task_description.lower()
            for pkg in gpu_packages:
                if pkg.lower() in task_lower:
                    gpu_cluster = self.get_gpu_cluster()
                    if gpu_cluster:
                        return gpu_cluster
                    break
        
        # Default: CPU cluster
        return self.cluster
    
    def get_slurm_for_task(self, task_description: str = "", requires_gpu: bool = False) -> Dict:
        """Get complete SLURM settings dict for a task with proper GPU routing.
        
        Returns dict with keys: partition, account, nodes, ntasks, cpus_per_task,
        memory, time, gpu_available, gpu_directive, cluster_name
        
        IMPORTANT: memory is None for GPU clusters (specifying --mem causes
        allocation failures on ARC GPU nodes).
        """
        cluster = self.get_cluster_for_task(task_description, requires_gpu)
        slurm = cluster.get('slurm', {})
        gpu_cfg = cluster.get('gpu', {})
        is_gpu = gpu_cfg.get('available', False)
        
        # Determine which cluster name was selected
        selected_name = self.cluster_name
        if cluster is not self.cluster:
            selected_name = self.gpu_cluster_name
        
        result = {
            'partition': slurm.get('partition', 'compute1'),
            'account': slurm.get('account', 'sdz852'),
            'nodes': slurm.get('nodes', 1),
            'ntasks': slurm.get('ntasks', 1),
            'cpus_per_task': slurm.get('cpus_per_task', 20),
            'memory': slurm.get('memory'),  # None for GPU clusters
            'time': slurm.get('time', '1-00:00:00'),
            'gpu_available': is_gpu,
            'gpu_directive': None,
            'cluster_name': selected_name,
        }
        
        # Apply env var overrides
        for key in ['partition', 'account', 'time']:
            env_key = f"AGI_SUBTASK_{key.upper()}"
            env_val = os.environ.get(env_key)
            if env_val:
                result[key] = env_val
        
        # Memory override - but NEVER apply memory to GPU clusters
        # Specifying --mem on GPU nodes causes allocation failures on ARC
        mem_override = os.environ.get('AGI_SUBTASK_MEMORY')
        if mem_override and not is_gpu:
            result['memory'] = mem_override
        
        cpus_override = os.environ.get('AGI_SUBTASK_CPUS')
        if cpus_override:
            result['cpus_per_task'] = int(cpus_override)
        
        # GPU directive (--gres=gpu:N format)
        if is_gpu:
            count = gpu_cfg.get('default_count', 1)
            gpus_override = os.environ.get('AGI_SUBTASK_GPUS')
            if gpus_override:
                count = int(gpus_override)
            fmt = gpu_cfg.get('directive_format', '--gres=gpu:{count}')
            result['gpu_directive'] = fmt.format(count=count)
        
        return result
    
    def __repr__(self):
        return (
            f"ClusterConfig(cpu={self.cluster_name}, "
            f"gpu={self.gpu_cluster_name}, "
            f"partition={self.slurm.get('partition', '?')})"
        )


# =============================================================================
# CHECKPOINT STATE
# =============================================================================

class TaskStatus(Enum):
    NOT_STARTED = "not_started"
    GENERATING = "generating"
    CREATING_ENV = "creating_env"
    RUNNING_DRYRUN = "running_dryrun"
    WAITING_JOB = "waiting_job"
    REFLECTING = "reflecting"
    RUNNING_PROD = "running_prod"
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
    routed_cluster: Optional[str] = None  # v3.2: which cluster was selected
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TaskCheckpoint':
        # Handle older checkpoints missing routed_cluster
        if 'routed_cluster' not in data:
            data['routed_cluster'] = None
        return cls(**data)
    
    @classmethod
    def new(cls, task_id: str) -> 'TaskCheckpoint':
        now = datetime.now().isoformat()
        return cls(
            task_id=task_id, status=TaskStatus.NOT_STARTED.value,
            iteration=0, env_name=None, env_yaml_path=None,
            script_path=None, sbatch_path=None, current_job_id=None,
            last_error=None, env_created=False, dry_run_succeeded=False,
            created_at=now, updated_at=now, routed_cluster=None
        )


# =============================================================================
# SUB-AGENT
# =============================================================================

class ScriptFirstSubAgentV3:
    """
    Complete SubAgent with dual cluster routing, cleanup, and resume.
    
    Token budget sized for qwen3-coder-next:latest (32K context):
      - MAX_CONTEXT_TOKENS:    25,000  (leaves ~7K for system prompt + response)
      - MAX_TOOL_OUTPUT_TOKENS: 12,000 (fits in 25K budget with history)
      - MIN_TOKENS_FOR_RETRY:   3,000  (at least one more exchange)
    """
    
    # Token limits sized for qwen3-coder-next:latest @ 32K context
    MAX_CONTEXT_TOKENS = 25_000
    MAX_TOOL_OUTPUT_TOKENS = 12_000
    MIN_TOKENS_FOR_RETRY = 3_000
    
    JOB_POLL_INTERVAL = 30
    JOB_TIMEOUT = 14400  # 4 hours
    
    def __init__(
        self,
        agent_id: str,
        sandbox=None,
        conda_tools=None,
        slurm_tools=None,
        ollama_model: str = "qwen3-coder-next:latest",
        ollama_base_url: str = "http://127.0.0.1:11434",
        use_slurm: bool = True,
        slurm_config: Dict[str, Any] = None,
        project_root: str = ".",
        cleanup_env_on_success: bool = True,
        **kwargs
    ):
        self.agent_id = agent_id
        self.llm = OllamaLLM(model=ollama_model, base_url=ollama_base_url)
        
        self.sandbox = sandbox
        self.use_slurm = use_slurm
        self.cleanup_env_on_success = cleanup_env_on_success
        
        # Load cluster configuration (v3.2: dual cluster routing)
        self.cluster_config = ClusterConfig()
        
        # Merge any passed slurm_config (overrides cluster defaults)
        self.slurm_overrides = slurm_config or {}
        
        # Set project root
        if sandbox:
            self.project_root = Path(sandbox.project_dir)
        else:
            self.project_root = Path(project_root)
        
        # Checkpoint directory
        self.checkpoint_dir = self.project_root / 'temp' / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Context management (token budget for qwen3-coder-next:latest 32K)
        self.context_mgr = ContextManager(
            max_context_tokens=self.MAX_CONTEXT_TOKENS,
            max_tool_output_tokens=self.MAX_TOOL_OUTPUT_TOKENS,
            llm_for_summarization=self.llm
        )
        self.context_window = self.context_mgr.create_context_window(agent_id)
        
        self.checkpoint: Optional[TaskCheckpoint] = None
        
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    # =========================================================================
    # CHECKPOINT MANAGEMENT
    # =========================================================================
    
    def _get_checkpoint_path(self, task_id: str) -> Path:
        safe_id = re.sub(r'[^\w\-]', '_', task_id)[:50]
        return self.checkpoint_dir / f"{safe_id}_checkpoint.json"
    
    def _save_checkpoint(self):
        if not self.checkpoint:
            return
        self.checkpoint.updated_at = datetime.now().isoformat()
        path = self._get_checkpoint_path(self.checkpoint.task_id)
        with open(path, 'w') as f:
            json.dump(self.checkpoint.to_dict(), f, indent=2)
    
    def _load_checkpoint(self, task_id: str) -> Optional[TaskCheckpoint]:
        path = self._get_checkpoint_path(task_id)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return TaskCheckpoint.from_dict(json.load(f))
        except Exception:
            return None
    
    def _delete_checkpoint(self, task_id: str):
        path = self._get_checkpoint_path(task_id)
        if path.exists():
            path.unlink()
    
    def _update_checkpoint(self, **kwargs):
        if not self.checkpoint:
            return
        for key, value in kwargs.items():
            if hasattr(self.checkpoint, key):
                setattr(self.checkpoint, key, value)
        self._save_checkpoint()
    
    # =========================================================================
    # MAIN EXECUTION
    # =========================================================================
    
    def execute(
        self,
        subtask: Dict[str, Any],
        env_name: str = None,
        prior_attempts: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        
        task_id = subtask.get('id', 'unknown')
        description = subtask.get('description', subtask.get('title', 'Unknown'))
        requires_gpu = subtask.get('requires_gpu', False)
        
        # v3.2: Route task to appropriate cluster
        routed_slurm = self.cluster_config.get_slurm_for_task(
            task_description=description,
            requires_gpu=requires_gpu
        )
        routed_cluster = routed_slurm.get('cluster_name', self.cluster_config.cluster_name)
        
        print(f"\n[{task_id}] Routed to cluster: {routed_cluster}")
        if routed_slurm.get('gpu_available'):
            print(f"  GPU: {routed_slurm.get('gpu_directive')}")
            print(f"  CPUs: {routed_slurm.get('cpus_per_task')}, No --mem (GPU node)")
        else:
            print(f"  CPUs: {routed_slurm.get('cpus_per_task')}, Memory: {routed_slurm.get('memory')}")
        
        # Check for checkpoint (resume)
        existing = self._load_checkpoint(task_id)
        if existing:
            print(f"\n{'='*60}")
            print(f"[{task_id}] RESUMING FROM CHECKPOINT")
            print(f"  Status: {existing.status}, Iteration: {existing.iteration}")
            print(f"{'='*60}")
            self.checkpoint = existing
            
            if existing.status == TaskStatus.COMPLETED.value:
                return self._success_result(task_id, message="Already completed", resumed=True)
            
            if existing.status == TaskStatus.FAILED.value:
                self.checkpoint.iteration = 0
                self.checkpoint.status = TaskStatus.NOT_STARTED.value
        else:
            self.checkpoint = TaskCheckpoint.new(task_id)
        
        # Store routed cluster in checkpoint
        self._update_checkpoint(routed_cluster=routed_cluster)
        
        agent_logger.log_task_start(
            agent_name=self.agent_id,
            task_id=task_id,
            description=f"v3.2: {description[:100]}",
            attempt=self.checkpoint.iteration + 1
        )
        
        # Check existing outputs
        if self._check_existing_outputs(subtask).get('already_complete'):
            self._delete_checkpoint(task_id)
            return self._success_result(task_id, message="Outputs exist", skipped=True)
        
        # =====================================================================
        # STEP 1: Generate artifacts (uses routed cluster for sbatch)
        # =====================================================================
        if not self.checkpoint.script_path or not Path(self.checkpoint.script_path).exists():
            print(f"\n[{task_id}] Generating artifacts...")
            self._update_checkpoint(status=TaskStatus.GENERATING.value)
            
            artifacts = self._generate_all_artifacts(subtask, routed_slurm)
            if not artifacts['success']:
                self._update_checkpoint(status=TaskStatus.FAILED.value, last_error=artifacts.get('error'))
                return self._failure_result(task_id, f"Artifact generation failed: {artifacts.get('error')}")
            
            self._update_checkpoint(
                env_yaml_path=artifacts['env_yaml_path'],
                script_path=artifacts['script_path'],
                sbatch_path=artifacts['sbatch_path'],
                env_name=artifacts['env_name']
            )
            print(f"  ✓ Cluster: {routed_cluster}")
            print(f"  ✓ Partition: {routed_slurm.get('partition')}")
            print(f"  ✓ Script: {artifacts['script_path']}")
            print(f"  ✓ Sbatch: {artifacts['sbatch_path']}")
        
        # =====================================================================
        # STEP 2: Create conda environment
        # =====================================================================
        if not self.checkpoint.env_created:
            print(f"\n[{task_id}] Creating conda environment...")
            self._update_checkpoint(status=TaskStatus.CREATING_ENV.value)
            
            env_result = self._create_conda_environment()
            if env_result.get('success'):
                self._update_checkpoint(env_created=True)
                print(f"  ✓ Environment ready: {self.checkpoint.env_name}")
            else:
                print(f"  ⚠ Warning: {env_result.get('error')}")
        
        # =====================================================================
        # MAIN LOOP
        # =====================================================================
        while self._can_continue():
            self.checkpoint.iteration += 1
            self._update_checkpoint(status=TaskStatus.RUNNING_DRYRUN.value, iteration=self.checkpoint.iteration)
            
            print(f"\n[{task_id}] Iteration {self.checkpoint.iteration} - DRY RUN")
            
            try:
                # Submit job
                submit_result = self._submit_slurm_job()
                if not submit_result['success']:
                    self._update_checkpoint(last_error=submit_result.get('error'))
                    continue
                
                self._update_checkpoint(status=TaskStatus.WAITING_JOB.value, current_job_id=submit_result['job_id'])
                print(f"  ✓ Job submitted: {submit_result['job_id']}")
                
                # Wait for completion
                print(f"  → Waiting for job...")
                wait_result = self._wait_for_job()
                logs = self._collect_job_logs()
                
                analysis = self._analyze_job_result(wait_result, logs, subtask)
                
                if analysis['success']:
                    print(f"  ✓ DRY RUN SUCCEEDED!")
                    self._update_checkpoint(dry_run_succeeded=True, status=TaskStatus.RUNNING_PROD.value)
                    
                    # Production run
                    print(f"\n[{task_id}] PRODUCTION RUN")
                    self._regenerate_sbatch(dry_run=False)
                    
                    prod_submit = self._submit_slurm_job()
                    if prod_submit['success']:
                        self._update_checkpoint(current_job_id=prod_submit['job_id'])
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
                                    print(f"  ✓ Environment removed (YAML preserved)")
                            
                            self._update_checkpoint(status=TaskStatus.COMPLETED.value)
                            self._delete_checkpoint(task_id)
                            
                            return self._success_result(
                                task_id=task_id,
                                script_path=self.checkpoint.script_path,
                                job_id=self.checkpoint.current_job_id,
                                iterations=self.checkpoint.iteration,
                                routed_cluster=routed_cluster,
                                report=report
                            )
                
                else:
                    # Reflect and update
                    print(f"  ✗ Failed: {analysis.get('error_summary', 'Unknown')[:100]}")
                    self._update_checkpoint(status=TaskStatus.REFLECTING.value, last_error=analysis.get('error_summary'))
                    
                    reflect = self._reflect_and_update(subtask, analysis, logs)
                    if reflect.get('should_escalate'):
                        self._update_checkpoint(status=TaskStatus.FAILED.value)
                        return self._failure_result(task_id, f"Escalating: {reflect.get('reason')}")
                    
                    if reflect['success']:
                        print(f"  ✓ Script updated, retrying...")
                        self._regenerate_sbatch(dry_run=True)
                        
            except Exception as e:
                self._update_checkpoint(last_error=str(e))
                print(f"  ✗ Exception: {e}")
        
        # Context exhausted
        self._update_checkpoint(status=TaskStatus.FAILED.value)
        return self._failure_result(
            task_id=task_id,
            error=f"Context exhausted after {self.checkpoint.iteration} iterations",
            checkpoint_preserved=True
        )
    
    # =========================================================================
    # ARTIFACT GENERATION
    # =========================================================================
    
    def _generate_all_artifacts(
        self, subtask: Dict, routed_slurm: Dict = None
    ) -> Dict[str, Any]:
        """Generate env.yml, script.py, and sbatch with routed cluster settings."""
        task_id = subtask.get('id', 'task')
        safe_id = re.sub(r'[^\w\-]', '_', task_id)[:30]
        
        env_yaml_path = self.project_root / 'envs' / f"{safe_id}.yml"
        script_path = self.project_root / 'scripts' / f"{safe_id}.py"
        sbatch_path = self.project_root / 'slurm' / 'scripts' / f"{safe_id}.sbatch"
        env_name = f"agi_{safe_id}"
        
        for p in [env_yaml_path, script_path, sbatch_path]:
            p.parent.mkdir(parents=True, exist_ok=True)
        
        # 1. Environment YAML (open-source channels only)
        env_yaml = self._generate_env_yaml(subtask, env_name)
        env_yaml_path.write_text(env_yaml)
        
        # 2. Python script
        script_result = self._generate_python_script(subtask)
        if not script_result['success']:
            return script_result
        script_path.write_text(script_result['content'])
        
        # 3. Sbatch script with ROUTED cluster settings
        if not routed_slurm:
            routed_slurm = self.cluster_config.get_slurm_for_task(
                task_description=subtask.get('description', ''),
                requires_gpu=subtask.get('requires_gpu', False)
            )
        
        sbatch = self._generate_sbatch_script(
            safe_id, script_path, env_name, env_yaml_path,
            subtask, routed_slurm, dry_run=True
        )
        sbatch_path.write_text(sbatch)
        
        return {
            'success': True,
            'env_yaml_path': str(env_yaml_path),
            'script_path': str(script_path),
            'sbatch_path': str(sbatch_path),
            'env_name': env_name,
        }
    
    def _generate_env_yaml(self, subtask: Dict, env_name: str) -> str:
        """Generate conda environment YAML with open-source channels only."""
        packages = subtask.get('packages', [])
        deps = ['python>=3.10']
        pip_pkgs = []
        
        for pkg in packages:
            if '/' in pkg or pkg.startswith('git+'):
                pip_pkgs.append(pkg)
            else:
                deps.append(pkg)
        
        # Open-source channels only (no defaults/main - commercial license)
        lines = [
            f"name: {env_name}",
            "channels:",
            "  - conda-forge",
            "  - bioconda",
            "  - nodefaults",
            "dependencies:",
        ]
        for d in deps:
            lines.append(f"  - {d}")
        
        if pip_pkgs:
            lines.extend(["  - pip", "  - pip:"])
            for p in pip_pkgs:
                lines.append(f"    - {p}")
        
        return '\n'.join(lines) + '\n'
    
    def _generate_python_script(self, subtask: Dict) -> Dict[str, Any]:
        desc = subtask.get('description', '')
        packages = subtask.get('packages', [])
        inputs = subtask.get('input_files', [])
        outputs = subtask.get('output_files', [])
        
        prompt = f"""Generate a Python script for this task.

TASK: {desc}
PACKAGES: {', '.join(packages)}
INPUTS: {', '.join(inputs)}
OUTPUTS: {', '.join(outputs)}

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
    # Your code here
    # Wrap saves with: if not DRY_RUN: save() else: print("[DRY-RUN] Would save...")
    print("SUCCESS: Task completed")

if __name__ == "__main__":
    main()
```

Generate ONLY the Python code."""

        try:
            response = self.llm.invoke(prompt)
            content = re.sub(r'^```(?:python)?\n?', '', response, flags=re.MULTILINE)
            content = re.sub(r'\n?```$', '', content, flags=re.MULTILINE).strip()
            
            if 'AGI_DRY_RUN' not in content:
                content = self._prepend_header(content, subtask)
            
            return {'success': True, 'content': content}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _prepend_header(self, script: str, subtask: Dict) -> str:
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
    
    def _generate_sbatch_script(
        self, task_id: str, script_path: Path, env_name: str,
        env_yaml_path: Path, subtask: Dict, routed_slurm: Dict,
        dry_run: bool = True
    ) -> str:
        """Generate sbatch script using ROUTED cluster settings.
        
        Uses routed_slurm dict from get_slurm_for_task() which already has:
        - Correct partition (CPU or GPU)
        - Memory set to None for GPU clusters (prevents --mem)
        - GPU directive in --gres=gpu:N format
        - All env var overrides applied
        """
        partition = routed_slurm.get('partition', 'compute1')
        account = routed_slurm.get('account')
        nodes = routed_slurm.get('nodes', 1)
        ntasks = routed_slurm.get('ntasks', 1)
        cpus = routed_slurm.get('cpus_per_task', 20)
        memory = routed_slurm.get('memory')  # None for GPU clusters
        time_limit = routed_slurm.get('time', '1-00:00:00')
        gpu_directive = routed_slurm.get('gpu_directive')  # e.g. "--gres=gpu:1"
        is_gpu = routed_slurm.get('gpu_available', False)
        selected_cluster = routed_slurm.get('cluster_name', self.cluster_config.cluster_name)
        
        log_dir = self.project_root / 'slurm' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        job_name = f"agi_{task_id}" + ("_dryrun" if dry_run else "_prod")
        
        # Build SBATCH directives
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={partition}",
        ]
        
        # Account (only if specified)
        if account:
            lines.append(f"#SBATCH --account={account}")
        
        lines.extend([
            f"#SBATCH --nodes={nodes}",
            f"#SBATCH --ntasks={ntasks}",
            f"#SBATCH --cpus-per-task={cpus}",
        ])
        
        # Memory: ONLY for CPU clusters. NEVER for GPU clusters.
        # Specifying --mem on ARC GPU nodes causes allocation failures.
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
            "#" + "="*70,
            f"# Cluster: {selected_cluster}{gpu_note}",
            f"# Task: {task_id}",
            f"# Mode: {'DRY-RUN' if dry_run else 'PRODUCTION'}",
            f"# Partition: {partition}, CPUs: {cpus}",
        ])
        if is_gpu:
            lines.append(f"# GPU: {gpu_directive} (NO --mem on GPU nodes)")
        else:
            lines.append(f"# Memory: {memory or 'default'}")
        lines.extend([
            "#" + "="*70,
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
            '    echo ">>> Creating environment: ${CONDA_ENV}"',
            '    conda env create -f "${ENV_YAML}" -n "${CONDA_ENV}" || { echo "ERROR: env create failed"; exit 1; }',
            'fi',
            "",
            'echo ">>> Activating: ${CONDA_ENV}"',
            'conda activate "${CONDA_ENV}"',
            'echo ">>> Python: $(which python)"',
            "",
            "# Environment variables",
            f'export AGI_DRY_RUN="{"true" if dry_run else "false"}"',
            f'export PROJECT_DIR="{self.project_root}"',
            'echo ">>> AGI_DRY_RUN=${AGI_DRY_RUN}"',
            "",
            "# Run script",
            f'cd "{self.project_root}"',
            f'conda run -n "${{CONDA_ENV}}" python "{script_path}"',
            "",
            "EXIT_CODE=$?",
            "",
            "echo '=============================================='",
            "echo 'End: '$(date)",
            "echo 'Exit code: '$EXIT_CODE",
            "echo '=============================================='",
            "",
            f'touch "{log_dir}/{job_name}_${{SLURM_JOB_ID}}.complete"',
            "exit $EXIT_CODE",
        ])
        
        return '\n'.join(lines) + '\n'
    
    # =========================================================================
    # CONDA MANAGEMENT
    # =========================================================================
    
    def _create_conda_environment(self) -> Dict[str, Any]:
        if not self.checkpoint or not self.checkpoint.env_yaml_path:
            return {'success': False, 'error': 'No env yaml'}
        
        env_name = self.checkpoint.env_name
        env_yaml = Path(self.checkpoint.env_yaml_path)
        
        if not env_yaml.exists():
            return {'success': False, 'error': f'YAML not found: {env_yaml}'}
        
        try:
            # Check if exists
            result = subprocess.run(
                ['conda', 'env', 'list'],
                capture_output=True, text=True, timeout=60
            )
            if env_name in result.stdout:
                return {'success': True, 'already_exists': True}
            
            # Create
            result = subprocess.run(
                ['conda', 'env', 'create', '-f', str(env_yaml), '-n', env_name],
                capture_output=True, text=True, timeout=600
            )
            
            if result.returncode == 0 or 'already exists' in result.stderr:
                return {'success': True}
            return {'success': False, 'error': result.stderr[:500]}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _cleanup_conda_environment(self) -> Dict[str, Any]:
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
            return {'success': False, 'error': f'Sbatch not found: {sbatch_path}'}
        
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
                return {'success': False, 'error': f'Cannot parse job ID: {result.stdout}'}
            
            return {'success': True, 'job_id': match.group(1)}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _wait_for_job(self) -> Dict[str, Any]:
        if not self.checkpoint or not self.checkpoint.current_job_id:
            return {'success': False, 'error': 'No job ID'}
        
        job_id = self.checkpoint.current_job_id
        start = time.time()
        
        while time.time() - start < self.JOB_TIMEOUT:
            status = self._get_job_status(job_id)
            
            if status['state'] == 'COMPLETED':
                return {'success': True, 'state': 'COMPLETED'}
            
            if status['state'] in ['FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL']:
                return {'success': False, 'state': status['state']}
            
            if status['state'] == 'UNKNOWN':
                # Check completion marker
                log_dir = self.project_root / 'slurm' / 'logs'
                for marker in log_dir.glob(f"*_{job_id}.complete"):
                    return {'success': True, 'state': 'COMPLETED'}
            
            elapsed = int(time.time() - start)
            if elapsed % 60 == 0:  # Print every minute
                print(f"    ... Job {job_id}: {status['state']} ({elapsed}s)")
            time.sleep(self.JOB_POLL_INTERVAL)
        
        return {'success': False, 'state': 'TIMEOUT'}
    
    def _get_job_status(self, job_id: str) -> Dict[str, Any]:
        try:
            result = subprocess.run(
                ['squeue', '-j', job_id, '-h', '-o', '%T'],
                capture_output=True, text=True, timeout=30
            )
            if result.stdout.strip():
                return {'state': result.stdout.strip()}
            
            result = subprocess.run(
                ['sacct', '-j', job_id, '-n', '-o', 'State', '-P'],
                capture_output=True, text=True, timeout=30
            )
            if result.stdout.strip():
                return {'state': result.stdout.strip().split('\n')[0]}
            
            return {'state': 'UNKNOWN'}
        except Exception:
            return {'state': 'UNKNOWN'}
    
    def _collect_job_logs(self) -> Dict[str, str]:
        logs = {'stdout': '', 'stderr': ''}
        if not self.checkpoint or not self.checkpoint.current_job_id:
            return logs
        
        log_dir = self.project_root / 'slurm' / 'logs'
        job_id = self.checkpoint.current_job_id
        
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
            content = re.sub(r'export AGI_DRY_RUN="false"', 'export AGI_DRY_RUN="true"', content)
            content = re.sub(r'_prod', '_dryrun', content)
        else:
            content = re.sub(r'export AGI_DRY_RUN="true"', 'export AGI_DRY_RUN="false"', content)
            content = re.sub(r'_dryrun', '_prod', content)
        
        path.write_text(content)
    
    # =========================================================================
    # ANALYSIS
    # =========================================================================
    
    def _analyze_job_result(
        self, job_result: Dict, logs: Dict[str, str], subtask: Dict
    ) -> Dict[str, Any]:
        stdout = logs.get('stdout', '')
        stderr = logs.get('stderr', '')
        
        if 'SUCCESS: Task completed' in stdout and job_result.get('success'):
            return {'success': True}
        
        combined = f"{stdout}\n{stderr}".lower()
        
        patterns = {
            'missing_package': [r'modulenotfounderror', r'no module named'],
            'file_not_found': [r'filenotfounderror', r'no such file'],
            'syntax_error': [r'syntaxerror', r'indentationerror'],
            'memory_error': [r'memoryerror', r'out of memory', r'oom'],
            'gpu_error': [r'cuda error', r'cuda out of memory', r'gpu.*not available'],
        }
        
        for error_type, pats in patterns.items():
            for p in pats:
                if re.search(p, combined):
                    return {
                        'success': False,
                        'error_type': error_type,
                        'error_summary': f"{error_type}: see logs",
                    }
        
        if not job_result.get('success'):
            return {'success': False, 'error_summary': stderr[-500:] or 'Unknown error'}
        
        return {'success': True}
    
    def _reflect_and_update(
        self, subtask: Dict, analysis: Dict, logs: Dict[str, str]
    ) -> Dict[str, Any]:
        if not self.checkpoint or not self.checkpoint.script_path:
            return {'success': False, 'error': 'No script'}
        
        script_path = Path(self.checkpoint.script_path)
        if not script_path.exists():
            return {'success': False, 'error': 'Script not found'}
        
        current = script_path.read_text()
        
        prompt = f"""Fix this Python script.

ERROR: {analysis.get('error_type', 'unknown')}

STDERR:
```
{logs.get('stderr', '')[-2000:]}
```

SCRIPT:
```python
{current}
```

Provide the COMPLETE fixed script between ### FIXED ### and ### END ###"""

        try:
            response = self.llm.invoke(prompt)
            
            match = re.search(
                r'### FIXED ###\s*```python\s*(.*?)\s*```\s*### END ###',
                response, re.DOTALL
            )
            if not match:
                match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
            
            if match:
                fixed = match.group(1).strip()
                if fixed and len(fixed) > 50:
                    script_path.write_text(fixed)
                    return {'success': True}
            
            return {'success': False, 'error': 'Could not extract fix'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _generate_completion_report(
        self, subtask: Dict, logs: Dict, outputs: Dict,
        routed_cluster: str = None
    ) -> str:
        cluster = routed_cluster or (self.checkpoint.routed_cluster if self.checkpoint else 'unknown')
        return f"""# Task: {self.checkpoint.task_id if self.checkpoint else 'unknown'}
Completed: {datetime.now().isoformat()}
Cluster: {cluster}
Iterations: {self.checkpoint.iteration if self.checkpoint else 0}
Outputs: {', '.join(outputs.get('found_files', []))}
"""
    
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
        existing = [
            str(self.project_root / f)
            for f in outputs
            if (self.project_root / f).exists()
        ]
        return {'already_complete': len(existing) == len(outputs)}
    
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


# Aliases
ScriptFirstSubAgent = ScriptFirstSubAgentV3
SubAgentV3 = ScriptFirstSubAgentV3

"""
Configuration loader for AGI pipeline v3.2.

Loads settings from config/config.yaml and provides easy access to:
- Context management settings (token limits)
- Agent settings (script-first options)
- SLURM settings
- Cluster configurations
- Resource profiles

v3.2 Updates:
- Token defaults sized for qwen3-coder-next (32K context): 25K/12K/3K
- Default model: qwen3-coder-next
- Default cluster: arc_compute1 (ARC dual-cluster architecture)
- Default GPU cluster: arc_gpu1v100

Usage:
    from utils.config_loader import config, get_context_settings, get_slurm_config

    # Get all settings
    settings = config.get_all()

    # Get specific sections
    context_settings = get_context_settings()
    slurm_config = get_slurm_config()
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class ContextSettings:
    """Token-based context management settings.

    Defaults sized for qwen3-coder-next @ 32K context window:
      25K context budget  (leaves ~7K for system prompt + response)
      12K tool output     (fits in 25K budget with history)
       3K min to continue (at least one more exchange)
    """
    max_tokens_per_task: int = 25000
    max_tool_output_tokens: int = 12000
    min_tokens_to_continue: int = 3000
    summary_target_tokens: int = 1500
    chars_per_token: int = 4
    auto_summarize: bool = True
    recent_history_percent: int = 30


@dataclass
class ScriptFirstSettings:
    """Script-first agent settings"""
    enabled: bool = True
    always_generate_script: bool = True
    scripts_dir: str = "scripts"
    add_script_headers: bool = True
    verify_outputs: bool = True
    max_script_generations: int = 5


@dataclass
class SlurmSettings:
    """SLURM configuration.

    v3.2: Defaults updated for ARC dual-cluster architecture.
    """
    enabled: bool = True
    default_cluster: str = "arc_compute1"
    default_gpu_cluster: str = "arc_gpu1v100"
    poll_interval: int = 10
    max_poll_attempts: int = 720
    job_prefix: str = "agi"
    use_sbatch: bool = True
    wait_for_completion: bool = True


@dataclass
class ResourceProfile:
    """Resource allocation profile"""
    cpus: int = 4
    memory: str = "16G"
    time: str = "04:00:00"
    gpus: int = 0
    gpu_type: Optional[str] = None
    partition: Optional[str] = None
    keywords: List[str] = None


class ConfigLoader:
    """
    Loads and provides access to configuration settings.

    Searches for config in order:
    1. Path specified in AGI_CONFIG environment variable
    2. ./config/config.yaml
    3. ../config/config.yaml
    4. ~/agi/config/config.yaml
    """

    _instance = None
    _config = None
    _config_path = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self._load_config()

    def _find_config_file(self) -> Optional[Path]:
        """Find config file in standard locations"""
        # Check environment variable first
        env_path = os.environ.get('AGI_CONFIG')
        if env_path and Path(env_path).exists():
            return Path(env_path)

        # Check standard locations
        search_paths = [
            Path("config/config.yaml"),
            Path("../config/config.yaml"),
            Path.home() / "agi" / "config" / "config.yaml",
            Path("/etc/agi/config.yaml"),
        ]

        for path in search_paths:
            if path.exists():
                return path

        return None

    def _load_config(self):
        """Load configuration from file"""
        config_path = self._find_config_file()

        if config_path:
            try:
                with open(config_path, 'r') as f:
                    self._config = yaml.safe_load(f)
                self._config_path = config_path
                print(f"Loaded config from: {config_path}")
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
                self._config = {}
        else:
            print("Warning: No config file found, using defaults")
            self._config = {}

    def reload(self, config_path: str = None):
        """Reload configuration, optionally from a specific path"""
        if config_path:
            self._config_path = Path(config_path)
            if self._config_path.exists():
                with open(self._config_path, 'r') as f:
                    self._config = yaml.safe_load(f)
        else:
            self._load_config()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a top-level config value"""
        return self._config.get(key, default)

    def get_nested(self, *keys, default: Any = None) -> Any:
        """Get a nested config value using dot notation or multiple keys"""
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value

    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration"""
        return self._config.copy()

    # ==================== Typed Accessors ====================

    def get_context_settings(self) -> ContextSettings:
        """Get context management settings as typed object.

        Defaults are sized for qwen3-coder-next @ 32K context.
        """
        ctx = self._config.get('context', {})
        return ContextSettings(
            max_tokens_per_task=ctx.get('max_tokens_per_task', 25000),
            max_tool_output_tokens=ctx.get('max_tool_output_tokens', 12000),
            min_tokens_to_continue=ctx.get('min_tokens_to_continue', 3000),
            summary_target_tokens=ctx.get('summary_target_tokens', 1500),
            chars_per_token=ctx.get('chars_per_token', 4),
            auto_summarize=ctx.get('auto_summarize', True),
            recent_history_percent=ctx.get('recent_history_percent', 30)
        )

    def get_script_first_settings(self) -> ScriptFirstSettings:
        """Get script-first agent settings"""
        sf = self._config.get('agents', {}).get('script_first', {})
        return ScriptFirstSettings(
            enabled=sf.get('enabled', True),
            always_generate_script=sf.get('always_generate_script', True),
            scripts_dir=sf.get('scripts_dir', 'scripts'),
            add_script_headers=sf.get('add_script_headers', True),
            verify_outputs=sf.get('verify_outputs', True),
            max_script_generations=sf.get('max_script_generations', 5)
        )

    def get_slurm_settings(self) -> SlurmSettings:
        """Get SLURM settings.

        v3.2: Defaults updated for ARC dual-cluster architecture.
        """
        slurm = self._config.get('slurm', {})
        script_sub = slurm.get('script_submission', {})
        return SlurmSettings(
            enabled=slurm.get('enabled', True),
            default_cluster=slurm.get('default_cluster', 'arc_compute1'),
            default_gpu_cluster=slurm.get('default_gpu_cluster', 'arc_gpu1v100'),
            poll_interval=slurm.get('poll_interval', 10),
            max_poll_attempts=slurm.get('max_poll_attempts', 720),
            job_prefix=slurm.get('job_prefix', 'agi'),
            use_sbatch=script_sub.get('use_sbatch', True),
            wait_for_completion=script_sub.get('wait_for_completion', True)
        )

    def get_ollama_settings(self) -> Dict[str, Any]:
        """Get Ollama LLM settings.

        v3.2: Default model switched to qwen3-coder-next with 32K context.
        """
        ollama = self._config.get('ollama', {})
        return {
            'model': ollama.get('model', 'qwen3-coder-next'),
            'base_url': ollama.get('base_url', 'http://127.0.0.1:11434'),
            'model_context_length': ollama.get('model_context_length', 32768),
        }

    def get_cluster_config(self, cluster_name: str = None) -> Dict[str, Any]:
        """Get configuration for a specific cluster.

        v3.2: Default cluster changed to arc_compute1.
        """
        clusters = self._config.get('clusters', {})

        if cluster_name is None:
            cluster_name = self._config.get('slurm', {}).get(
                'default_cluster', 'arc_compute1'
            )

        return clusters.get(cluster_name, {})

    def get_parallel_settings(self) -> Dict[str, Any]:
        """Get parallel execution settings"""
        parallel = self._config.get('parallel', {})
        return {
            'enabled': parallel.get('enabled', True),
            'max_parallel_jobs': parallel.get('max_parallel_jobs', 10),
            'max_threads': parallel.get('max_threads', 4),
            'max_batch_size': parallel.get('max_batch_size', 5),
            'parallel_strategy': parallel.get('parallel_strategy', 'dependency_based'),
            'min_tasks_for_parallel': parallel.get('min_tasks_for_parallel', 2)
        }

    def get_conda_settings(self) -> Dict[str, Any]:
        """Get conda environment settings"""
        conda = self._config.get('conda', {})
        return {
            'env_prefix': conda.get('env_prefix', 'agi_'),
            'default_python': conda.get('default_python', '3.10'),
            'auto_export_yaml': conda.get('auto_export_yaml', True),
            'channels': conda.get('channels', ['nodefaults', 'conda-forge', 'bioconda']),
            'task_specific_envs': conda.get('task_specific_envs', {}),
            'package_mapping': conda.get('package_mapping', {}),
            'pip_only_packages': conda.get('pip_only_packages', [])
        }

    def get_resource_profile(self, task_description: str = None) -> ResourceProfile:
        """
        Get appropriate resource profile based on task description.
        Matches keywords in task to select profile.
        """
        profiles = self._config.get('resource_profiles', {})
        default_profile = profiles.get('default', {})

        if not task_description:
            return ResourceProfile(
                cpus=default_profile.get('cpus', 4),
                memory=default_profile.get('memory', '16G'),
                time=default_profile.get('time', '04:00:00'),
                gpus=default_profile.get('gpus', 0),
                partition=default_profile.get('partition')
            )

        task_lower = task_description.lower()

        # Check each profile for keyword matches
        for profile_name, profile_config in profiles.items():
            if profile_name == 'default':
                continue

            keywords = profile_config.get('keywords', [])
            for keyword in keywords:
                if keyword.lower() in task_lower:
                    return ResourceProfile(
                        cpus=profile_config.get('cpus', 4),
                        memory=profile_config.get('memory', '16G'),
                        time=profile_config.get('time', '04:00:00'),
                        gpus=profile_config.get('gpus', 0),
                        gpu_type=profile_config.get('gpu_type'),
                        partition=profile_config.get('partition'),
                        keywords=keywords
                    )

        # Return default if no match
        return ResourceProfile(
            cpus=default_profile.get('cpus', 4),
            memory=default_profile.get('memory', '16G'),
            time=default_profile.get('time', '04:00:00'),
            gpus=default_profile.get('gpus', 0),
            partition=default_profile.get('partition')
        )

    def get_failure_patterns(self) -> Dict[str, Dict]:
        """Get failure diagnosis patterns"""
        diagnosis = self._config.get('failure_diagnosis', {})
        return diagnosis.get('patterns', {})

    def get_workflow_settings(self) -> Dict[str, Any]:
        """Get workflow settings"""
        workflow = self._config.get('workflow', {})
        subtask = workflow.get('subtask', {})
        return {
            'enable_checkpointing': workflow.get('enable_checkpointing', True),
            'checkpoint_frequency': workflow.get('checkpoint_frequency', 'per_subtask'),
            'max_execution_time_minutes': workflow.get('max_execution_time_minutes', 480),
            'master_document_enabled': workflow.get('master_document', {}).get('enabled', True),
            'subtask_generate_script': subtask.get('generate_script', True),
            'subtask_submit_via_slurm': subtask.get('submit_via_slurm', True),
            'subtask_verify_outputs': subtask.get('verify_outputs', True),
            'on_context_exhausted': subtask.get('on_context_exhausted', 'skip'),
            'on_unrecoverable_error': subtask.get('on_unrecoverable_error', 'skip')
        }

    def get_sandbox_dirs(self) -> List[str]:
        """Get list of sandbox subdirectories to create"""
        sandbox = self._config.get('sandbox', {})
        return sandbox.get('subdirectories', [
            'data', 'data/inputs', 'data/outputs',
            'scripts', 'logs', 'envs', 'reports', 'temp',
            'slurm', 'slurm/scripts', 'slurm/logs'
        ])


# Global singleton instance
config = ConfigLoader()


# Convenience functions
def get_context_settings() -> ContextSettings:
    """Get context management settings"""
    return config.get_context_settings()


def get_slurm_config() -> SlurmSettings:
    """Get SLURM settings"""
    return config.get_slurm_settings()


def get_ollama_config() -> Dict[str, Any]:
    """Get Ollama settings"""
    return config.get_ollama_settings()


def get_resource_profile(task_description: str = None) -> ResourceProfile:
    """Get resource profile for task"""
    return config.get_resource_profile(task_description)


def reload_config(config_path: str = None):
    """Reload configuration"""
    config.reload(config_path)

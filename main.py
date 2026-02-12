"""
Main entry point for the multi-agent system.
Supports multiple SLURM clusters (CPU and GPU) with flexible configuration.

ARCHITECTURE NOTE:
-----------------
This system maintains separation between:
  - AGI_ROOT: The pipeline code repository (static, shared)
  - PROJECT_DIR: The project-specific directory (logs, data, outputs)

All execution artifacts (logs, outputs, reports) go to PROJECT_DIR.
AGI_ROOT stays clean and only contains the pipeline code.

v3.2 Updates:
- ARC cluster as primary target (GPU + CPU partitions)
- GPU-first architecture: master on GPU node, subtasks route to CPU or GPU
- qwen3-coder-next:latest as default model (32K context on V100)
- Dual cluster routing: AGI_CLUSTER (CPU) + AGI_GPU_CLUSTER (GPU subtasks)
- Cluster configuration via cluster_config.yaml
- Conda cleanup after successful task completion
- State checkpointing for resume capability
- Open-source conda channels only (no defaults/main)

GPU NODE RULES (ARC):
- Do NOT specify --mem on GPU partitions (causes allocation failures)
- Use --gres=gpu:N format (not --gpus N)
- Standard GPU request: --gres=gpu:1 -N 1 -n 1 -c 80

Run with:
    # ARC cluster (default - CPU subtasks to compute1, GPU subtasks to gpu1v100)
    python main.py --prompt-file prompts/my_task.txt --project-dir /path/to/project --slurm

    # Override GPU cluster target
    python main.py --prompt-file prompts/gpu_task.txt --project-dir /path/to/project \\
        --slurm --gpu-cluster arc_gpu1a100

    # Use extended-time CPU partition for subtasks
    python main.py --prompt-file prompts/long_task.txt --project-dir /path/to/project \\
        --slurm --cluster arc_compute2

    # Resume from checkpoints
    python main.py --prompt-file prompts/task.txt --project-dir ./project --resume

    # Clear checkpoints and start fresh
    python main.py --prompt-file prompts/task.txt --project-dir ./project --clear-checkpoints

    # Legacy zeus cluster
    python main.py --prompt-file prompts/task.txt --project-dir ./project \\
        --slurm --cluster zeus_cpu
"""
import argparse
import yaml
from pathlib import Path
from datetime import datetime
import json
import hashlib
import sys
import os
import shutil

# =============================================================================
# Ensure AGI repo root is on Python path regardless of launch context.
# When launched via sbatch, PYTHONPATH may not propagate correctly,
# which causes 'from memory.reflexion_memory import ...' to fail.
# =============================================================================
_agi_root = str(Path(__file__).resolve().parent)
if _agi_root not in sys.path:
    sys.path.insert(0, _agi_root)

from workflows.langgraph_workflow import MultiAgentWorkflow
from tools.sandbox import Sandbox
from tools.slurm_tools import SlurmTools, SlurmConfig

# Import configuration functions for project isolation
from utils.logging_config import configure_logging, agent_logger
from utils.documentation import configure_documentation, doc_generator
from utils.git_tracker import configure_git_tracking, git_tracker


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found at {config_path}, using defaults")
        return {
            "ollama": {
                "model": "qwen3-coder-next:latest",
                "base_url": "http://127.0.0.1:11434",
                "model_context_length": 32768,
            },
            "context": {
                "max_tokens_per_task": 25000,
                "max_tool_output_tokens": 12000,
                "min_tokens_to_continue": 3000,
            },
            "agents": {"max_retries": 12},
            "slurm": {
                "enabled": False,
                "default_cluster": "arc_compute1",
                "default_gpu_cluster": "arc_gpu1v100",
            },
            "parallel": {"enabled": True},
            "clusters": {},
            "reflexion": {"enabled": True},
        }


def load_prompt_file(prompt_path: str) -> dict:
    """
    Load task prompt from a text file.
    
    Supports both simple text files (just the task) and structured markdown
    with sections for task, context, inputs, outputs, etc.
    """
    path = Path(prompt_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    content = path.read_text()
    
    # Parse structured format if present
    result = {
        "task": content,
        "context": {},
        "input_files": [],
        "expected_outputs": [],
        "prompt_file": str(path.absolute())
    }
    
    # Try to parse markdown sections
    if "##" in content or "# " in content:
        lines = content.split('\n')
        current_section = "task"
        section_content = []
        
        for line in lines:
            if line.startswith('## ') or line.startswith('# '):
                # Save previous section
                if section_content:
                    text = '\n'.join(section_content).strip()
                    if current_section == "task" or current_section == "goal":
                        result["task"] = text
                    elif current_section == "context":
                        result["context"]["description"] = text
                    elif current_section == "input" or current_section == "inputs":
                        result["input_files"] = [f.strip().lstrip('- ') for f in text.split('\n') if f.strip()]
                    elif current_section == "output" or current_section == "outputs":
                        result["expected_outputs"] = [f.strip().lstrip('- ') for f in text.split('\n') if f.strip()]
                
                # Start new section
                current_section = line.lstrip('#').strip().lower().replace(' ', '_')
                section_content = []
            else:
                section_content.append(line)
        
        # Don't forget last section
        if section_content:
            text = '\n'.join(section_content).strip()
            if current_section == "task" or current_section == "goal":
                result["task"] = text
    
    return result


def archive_prompt(prompt_data: dict, project_dir: Path, config: dict) -> Path:
    """Archive the prompt file to project directory"""
    prompts_dir = project_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename
    task_hash = hashlib.md5(prompt_data["task"][:100].encode()).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"prompt_{timestamp}_{task_hash}.json"
    
    archive_path = prompts_dir / filename
    prompt_data["archived_at"] = datetime.now().isoformat()
    
    with open(archive_path, 'w') as f:
        json.dump(prompt_data, f, indent=2)
    
    return archive_path


def validate_project_dir(project_dir: str) -> Path:
    """Validate and create project directory"""
    project_path = Path(project_dir).resolve()
    project_path.mkdir(parents=True, exist_ok=True)
    return project_path


def load_cluster_config(config_path: str = None) -> dict:
    """
    Load the full cluster configuration from cluster_config.yaml.
    
    This is the authoritative source for SLURM partition details,
    GPU settings, and resource limits. The clusters section in
    config.yaml is only a summary/fallback.
    """
    if not config_path:
        config_path = os.environ.get('AGI_CLUSTER_CONFIG')
    
    if not config_path:
        for path in [
            Path.cwd() / 'config' / 'cluster_config.yaml',
            Path(__file__).parent / 'config' / 'cluster_config.yaml',
        ]:
            if path.exists():
                config_path = str(path)
                break
    
    if config_path and Path(config_path).exists():
        try:
            with open(config_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Failed to load cluster config from {config_path}: {e}")
    
    return {}


def list_clusters(config: dict, cluster_config: dict = None):
    """List all configured clusters from both config.yaml and cluster_config.yaml"""
    
    # Prefer cluster_config.yaml (full definitions) over config.yaml (summaries)
    if cluster_config and cluster_config.get('clusters'):
        clusters = cluster_config['clusters']
        default_cpu = cluster_config.get('default_cluster', 'arc_compute1')
        default_gpu = cluster_config.get('default_gpu_cluster', 'arc_gpu1v100')
        source = "cluster_config.yaml"
    else:
        clusters = config.get("clusters", {})
        default_cpu = config.get("slurm", {}).get("default_cluster", "arc_compute1")
        default_gpu = config.get("slurm", {}).get("default_gpu_cluster", "arc_gpu1v100")
        source = "config.yaml (summary only)"
    
    print(f"\n{'='*70}")
    print(f"  Available Clusters (source: {source})")
    print(f"{'='*70}")
    print(f"  Default CPU cluster: {default_cpu}")
    print(f"  Default GPU cluster: {default_gpu}")
    
    # Separate GPU and CPU clusters for display
    gpu_clusters = {}
    cpu_clusters = {}
    
    for name, cc in clusters.items():
        gpu_info = cc.get('gpu', {})
        if gpu_info.get('available', False) or cc.get('has_gpu', False):
            gpu_clusters[name] = cc
        else:
            cpu_clusters[name] = cc
    
    if gpu_clusters:
        print(f"\n  --- GPU Partitions ---")
        for name, cc in gpu_clusters.items():
            is_default = " (DEFAULT GPU)" if name == default_gpu else ""
            slurm = cc.get('slurm', {})
            gpu = cc.get('gpu', {})
            desc = cc.get('description', cc.get('name', ''))
            
            print(f"\n  {name}{is_default}")
            print(f"    {desc}")
            print(f"    Partition: {slurm.get('partition', cc.get('default_partition', 'N/A'))}")
            print(f"    CPUs: {slurm.get('cpus_per_task', cc.get('default_cpus', 'N/A'))}")
            print(f"    Time: {slurm.get('time', cc.get('default_time', 'N/A'))}")
            print(f"    GPU: {gpu.get('max_count', gpu.get('default_count', '?'))}× {gpu.get('type', cc.get('gpu_type', '?'))}")
            print(f"    Nodes: {cc.get('limits', {}).get('nodes_total', 'N/A')}")
            print(f"    NOTE: No --mem allowed on GPU partitions")
    
    if cpu_clusters:
        print(f"\n  --- CPU Partitions ---")
        for name, cc in cpu_clusters.items():
            is_default = " (DEFAULT CPU)" if name == default_cpu else ""
            slurm = cc.get('slurm', {})
            desc = cc.get('description', cc.get('name', ''))
            
            print(f"\n  {name}{is_default}")
            print(f"    {desc}")
            print(f"    Partition: {slurm.get('partition', cc.get('default_partition', 'N/A'))}")
            print(f"    CPUs: {slurm.get('cpus_per_task', cc.get('default_cpus', 'N/A'))}")
            print(f"    Memory: {slurm.get('memory', cc.get('default_memory', 'N/A'))}")
            print(f"    Time: {slurm.get('time', cc.get('default_time', 'N/A'))}")
            print(f"    Nodes: {cc.get('limits', {}).get('nodes_total', 'N/A')}")
    
    print(f"\n{'='*70}\n")


def check_cluster_status(sandbox: Sandbox, config: dict, cluster_name: str = None, partition: str = None):
    """Check and display cluster status"""
    slurm_config = SlurmConfig(config_dict=config)
    slurm_tools = SlurmTools(sandbox, config=slurm_config, cluster_name=cluster_name)
    
    if not slurm_tools.slurm_available:
        print("\n✗ SLURM is not available on this system\n")
        return {"available": False}
    
    # Print cluster summary
    print("\n" + "="*70)
    slurm_tools.print_cluster_summary()
    
    # Get live status
    print("\n--- Live Status ---")
    status = slurm_tools.get_cluster_status(partition=partition)
    
    if status["success"]:
        print(f"Total Nodes: {status.get('total_nodes', 'N/A')}")
        print(f"Idle Nodes: {status.get('idle_count', 0)}")
        
        if status.get("gpu_nodes"):
            print(f"GPU Nodes: {status.get('gpu_count', 0)}")
        
        if partition:
            part_info = slurm_tools.get_partition_info(partition)
            if part_info.get("success"):
                print(f"\nPartition '{partition}' Details:")
                cfg = part_info.get("config", {})
                print(f"  Max Time: {cfg.get('max_time')}")
                print(f"  Max CPUs: {cfg.get('max_cpus')}")
                print(f"  Max Memory: {cfg.get('max_memory')}")
                if cfg.get("has_gpu"):
                    print(f"  GPUs: {cfg.get('max_gpus')}x {cfg.get('gpu_type')}")
    
    print("="*70 + "\n")
    return status


def print_banner(task: str, config: dict, project_dir: Path, cluster_info: dict = None,
                 model: str = None, max_iterations: int = None,
                 cluster_for_subtasks: str = None, gpu_cluster_for_subtasks: str = None,
                 cleanup_env: bool = True):
    """Print startup banner"""
    model = model or config['ollama']['model']
    max_iterations = max_iterations or config['agents']['max_retries']
    context_length = config.get('ollama', {}).get('model_context_length', 32768)
    max_task_tokens = config.get('context', {}).get('max_tokens_per_task', 25000)
    
    print(f"\n{'='*70}")
    print(f"  AGI Multi-Agent Pipeline v3.2 (ARC GPU-First)")
    print(f"{'='*70}")
    print(f"  Model:            {model}")
    print(f"  Context Window:   {context_length:,} tokens")
    print(f"  Task Budget:      {max_task_tokens:,} tokens per subtask")
    print(f"  Max Iterations:   {max_iterations}")
    print(f"  Project Dir:      {project_dir}")
    
    if cluster_info:
        print(f"\n  SLURM Configuration:")
        print(f"    Cluster:    {cluster_info.get('cluster', 'N/A')}")
        print(f"    Partition:  {cluster_info.get('partition', 'N/A')}")
        
        if cluster_info.get('gpus'):
            print(f"    GPUs:       {cluster_info['gpus']}× {cluster_info.get('gpu_type', 'generic')}")
        
        if cluster_info.get('nodelist'):
            print(f"    Node(s):    {cluster_info['nodelist']}")
        
        if cluster_info.get('idle_count'):
            print(f"    Available:  {cluster_info['idle_count']} nodes")
    
    # v3.2: Subtask routing info
    if cluster_for_subtasks or gpu_cluster_for_subtasks:
        print(f"\n  Subtask Routing:")
        print(f"    CPU subtasks → {cluster_for_subtasks or 'arc_compute1'}")
        print(f"    GPU subtasks → {gpu_cluster_for_subtasks or 'arc_gpu1v100'}")
        print(f"    Cleanup Env:   {'Enabled' if cleanup_env else 'Disabled'}")
        print(f"    Checkpointing: Enabled")
    
    if config.get("parallel", {}).get("enabled"):
        print(f"\n  Parallel Execution: Enabled")
    
    print(f"{'='*70}")
    print(f"\n  Task:")
    
    # Wrap task text
    words = task.split()
    line = "    "
    for word in words:
        if len(line) + len(word) > 66:
            print(line)
            line = "    " + word + " "
        else:
            line += word + " "
    if line.strip():
        print(line)
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent System for Complex Task Execution (v3.2 ARC GPU-First)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on ARC cluster (default: CPU subtasks → compute1, GPU subtasks → gpu1v100)
  python main.py --prompt-file prompts/analysis.txt --project-dir ./project --slurm

  # Override GPU cluster for subtasks (e.g., use A100s)
  python main.py --prompt-file prompts/ml_task.txt --project-dir ./project \\
      --slurm --gpu-cluster arc_gpu1a100

  # Use extended-time CPU partition for long subtasks
  python main.py --prompt-file prompts/long_task.txt --project-dir ./project \\
      --slurm --cluster arc_compute2

  # Resume from checkpoints
  python main.py --prompt-file prompts/task.txt --project-dir ./project --resume

  # Clear checkpoints and start fresh
  python main.py --prompt-file prompts/task.txt --project-dir ./project --clear-checkpoints

  # Disable conda cleanup (keep environments for debugging)
  python main.py --prompt-file prompts/task.txt --project-dir ./project --no-cleanup-env

  # Override model
  python main.py --prompt-file prompts/task.txt --project-dir ./project \\
      --model llama3.1:70b

  # Legacy zeus cluster
  python main.py --prompt-file prompts/task.txt --project-dir ./project \\
      --slurm --cluster zeus_cpu

  # List available clusters
  python main.py --list-clusters --project-dir ./test
        """
    )
    
    # Task input (mutually exclusive)
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument("--task", type=str, help="High-level task description (inline)")
    task_group.add_argument("--prompt-file", type=str, help="Path to prompt file containing task")
    task_group.add_argument("--list-clusters", action="store_true", help="List all configured clusters")
    task_group.add_argument("--cluster-status", action="store_true", help="Check cluster/partition status")
    
    # Project directory (required for most operations)
    parser.add_argument("--project-dir", type=str, required=True, help="Project directory for all files")
    
    # Model and execution options
    model_group = parser.add_argument_group('Model Options')
    model_group.add_argument(
        "--model", "-m",
        type=str,
        help="Ollama model to use (overrides config). Default: qwen3-coder-next:latest"
    )
    model_group.add_argument(
        "--max-iterations", "--max-retries",
        type=int,
        help="Maximum iterations per subtask (overrides config, max 12)"
    )
    model_group.add_argument(
        "--ollama-url",
        type=str,
        help="Ollama server URL (default: http://127.0.0.1:11434)"
    )
    
    # Cluster selection
    cluster_group = parser.add_argument_group('Cluster Options')
    cluster_group.add_argument(
        "--cluster",
        type=str,
        default=os.environ.get('AGI_CLUSTER', 'arc_compute1'),
        help="CPU cluster for subtask SLURM settings (default: arc_compute1 or AGI_CLUSTER env)"
    )
    cluster_group.add_argument(
        "--gpu-cluster",
        type=str,
        default=os.environ.get('AGI_GPU_CLUSTER', 'arc_gpu1v100'),
        help="GPU cluster for subtasks requiring GPU (default: arc_gpu1v100 or AGI_GPU_CLUSTER env)"
    )
    cluster_group.add_argument(
        "--cluster-config",
        type=str,
        default=os.environ.get('AGI_CLUSTER_CONFIG'),
        help="Path to cluster_config.yaml"
    )
    cluster_group.add_argument(
        "--partition", "-p",
        type=str,
        help="SLURM partition override for master context (e.g., 'compute1', 'gpu1v100')"
    )
    cluster_group.add_argument(
        "--nodelist", "-w",
        type=str,
        help="Specific node(s) to run on"
    )
    cluster_group.add_argument(
        "--exclude", "-x",
        type=str,
        help="Node(s) to exclude"
    )
    
    # SLURM options
    slurm_group = parser.add_argument_group('SLURM Options')
    slurm_group.add_argument("--slurm", action="store_true", help="Enable SLURM job submission")
    slurm_group.add_argument("--no-slurm", action="store_true", help="Disable SLURM (force interactive)")
    slurm_group.add_argument("--cpus", "-c", type=int, help="CPUs per job")
    slurm_group.add_argument(
        "--memory", "--mem", type=str,
        help="Memory per job (e.g., '64G'). WARNING: Do NOT use for GPU partitions"
    )
    slurm_group.add_argument("--time", "-t", type=str, help="Time limit (e.g., '1-00:00:00')")
    
    # GPU options
    gpu_group = parser.add_argument_group('GPU Options')
    gpu_group.add_argument(
        "--gpus", "-G", type=int, default=0,
        help="Number of GPUs to request (uses --gres=gpu:N format)"
    )
    gpu_group.add_argument("--gpu-type", type=str, help="GPU type (e.g., 'v100', 'a100')")
    
    # Sub-agent options (v3.2)
    subagent_group = parser.add_argument_group('Sub-Agent Options (v3.2)')
    subagent_group.add_argument(
        "--cleanup-env",
        action="store_true",
        default=True,
        help="Clean up conda environments after successful task completion (default)"
    )
    subagent_group.add_argument(
        "--no-cleanup-env",
        action="store_true",
        help="Keep conda environments after task completion"
    )
    subagent_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoints if available"
    )
    subagent_group.add_argument(
        "--clear-checkpoints",
        action="store_true",
        help="Clear all checkpoints before starting"
    )
    
    # Parallel execution
    parser.add_argument("--parallel", action="store_true", help="Enable parallel subtask execution")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel execution")
    parser.add_argument("--max-parallel", type=int, help="Maximum parallel jobs")
    
    # Other options
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--dry-run", action="store_true", help="Print configuration without executing")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--thread-id", type=str, help="Thread ID for workflow persistence")
    parser.add_argument("--context", type=str, help="Additional context as JSON string")
    
    args = parser.parse_args()
    
    # =========================================================================
    # LOAD CONFIGURATION
    # =========================================================================
    config = load_config(args.config)
    
    # Apply command-line overrides
    if args.model:
        config['ollama']['model'] = args.model
    if args.max_iterations:
        config['agents']['max_retries'] = min(args.max_iterations, 12)
    if args.ollama_url:
        config['ollama']['base_url'] = args.ollama_url
    
    # Load full cluster definitions from cluster_config.yaml
    full_cluster_config = load_cluster_config(args.cluster_config)
    
    # Validate and setup project directory
    project_dir = validate_project_dir(args.project_dir)
    
    # =========================================================================
    # CONFIGURE PROJECT-SPECIFIC LOGGING AND TRACKING
    # =========================================================================
    configure_logging(project_dir)
    configure_documentation(project_dir)
    configure_git_tracking(project_dir)
    
    agent_logger.log_workflow_event("project_configured", {
        "project_dir": str(project_dir),
        "model": config['ollama']['model'],
        "max_retries": config['agents']['max_retries'],
        "cluster": args.cluster,
        "gpu_cluster": args.gpu_cluster,
    })
    # =========================================================================
    
    # Initialize sandbox
    sandbox = Sandbox(project_dir)
    
    # Handle --list-clusters (now uses full cluster_config.yaml)
    if args.list_clusters:
        list_clusters(config, full_cluster_config)
        sys.exit(0)
    
    # Handle --cluster-status
    if args.cluster_status:
        check_cluster_status(sandbox, config, args.cluster, args.partition)
        sys.exit(0)
    
    # Require task input if not listing/checking
    if not args.task and not args.prompt_file:
        parser.error("Either --task or --prompt-file is required")
    
    # =========================================================================
    # SET CLUSTER ENVIRONMENT VARIABLES FOR SUB-AGENT (v3.2)
    # =========================================================================
    # The sub-agent reads these environment variables directly to determine
    # cluster settings for sbatch generation. Two clusters are set:
    #   AGI_CLUSTER     → default target for CPU subtasks
    #   AGI_GPU_CLUSTER → target for subtasks that need GPU resources
    
    cluster_for_subtasks = args.cluster  # Already defaults to arc_compute1
    gpu_cluster_for_subtasks = args.gpu_cluster  # Already defaults to arc_gpu1v100
    
    os.environ['AGI_CLUSTER'] = cluster_for_subtasks
    os.environ['AGI_GPU_CLUSTER'] = gpu_cluster_for_subtasks
    
    # Set cluster config path
    cluster_config_path = args.cluster_config
    if not cluster_config_path:
        for p in [
            Path.cwd() / 'config' / 'cluster_config.yaml',
            Path(__file__).parent / 'config' / 'cluster_config.yaml',
        ]:
            if p.exists():
                cluster_config_path = str(p)
                break
    
    if cluster_config_path:
        os.environ['AGI_CLUSTER_CONFIG'] = cluster_config_path
    
    print(f"  Subtask CPU cluster:  {cluster_for_subtasks}")
    print(f"  Subtask GPU cluster:  {gpu_cluster_for_subtasks}")
    if cluster_config_path:
        print(f"  Cluster config:       {cluster_config_path}")
    # =========================================================================
    
    # Resolve cluster info for the master job context
    # Try full cluster_config.yaml first, fall back to config.yaml summaries
    cluster_name = cluster_for_subtasks
    if full_cluster_config and full_cluster_config.get('clusters', {}).get(cluster_name):
        cc = full_cluster_config['clusters'][cluster_name]
        cluster_config_entry = {
            "name": cc.get('name', cluster_name),
            "description": cc.get('description', ''),
            "default_partition": cc.get('slurm', {}).get('partition', 'compute1'),
            "default_cpus": cc.get('slurm', {}).get('cpus_per_task', 20),
            "default_memory": cc.get('slurm', {}).get('memory', '64G'),
            "default_time": cc.get('slurm', {}).get('time', '1-00:00:00'),
            "has_gpu": cc.get('gpu', {}).get('available', False),
            "gpu_type": cc.get('gpu', {}).get('type'),
        }
    else:
        cluster_config_entry = config.get("clusters", {}).get(cluster_name, {})
    
    # Determine partition
    partition = args.partition or cluster_config_entry.get("default_partition", "compute1")
    
    # Determine SLURM usage
    use_slurm = False
    if args.slurm:
        use_slurm = True
    elif args.no_slurm:
        use_slurm = False
    elif config.get("slurm", {}).get("enabled"):
        use_slurm = True
    
    # Check SLURM availability
    slurm_status = None
    if use_slurm:
        slurm_config_obj = SlurmConfig(config_dict=config)
        slurm_tools = SlurmTools(sandbox, config=slurm_config_obj, cluster_name=cluster_name)
        
        if not slurm_tools.slurm_available:
            print(f"WARNING: SLURM requested but not available. Falling back to interactive mode.")
            use_slurm = False
        else:
            slurm_status = slurm_tools.get_cluster_status(partition=partition)
    
    # Determine parallel execution
    parallel_enabled = True
    if args.no_parallel:
        parallel_enabled = False
    elif args.parallel:
        parallel_enabled = True
    elif config.get("parallel", {}).get("enabled") is not None:
        parallel_enabled = config["parallel"]["enabled"]
    
    # Determine cleanup behavior (v3.2)
    cleanup_env = True
    if args.no_cleanup_env:
        cleanup_env = False
    
    # =========================================================================
    # BUILD SLURM JOB CONFIG
    # =========================================================================
    # Check if the target cluster is a GPU cluster (no --mem allowed)
    is_gpu_cluster = cluster_config_entry.get("has_gpu", False)
    
    # Memory: NEVER set for GPU clusters (causes allocation failures on ARC)
    if is_gpu_cluster:
        job_memory = None
        if args.memory:
            print(f"WARNING: --memory ignored for GPU cluster '{cluster_name}' (causes allocation failures)")
    else:
        job_memory = args.memory or cluster_config_entry.get("default_memory", "64G")
    
    slurm_job_config = {
        "cluster": cluster_name,
        "partition": partition,
        "cpus": args.cpus or cluster_config_entry.get("default_cpus", 20),
        "memory": job_memory,
        "time": args.time or cluster_config_entry.get("default_time", "1-00:00:00"),
        "gpus": args.gpus,
        "gpu_type": args.gpu_type or cluster_config_entry.get("gpu_type"),
        "nodelist": args.nodelist,
        "exclude_nodes": args.exclude,
        "max_parallel_jobs": args.max_parallel or config.get("parallel", {}).get("max_parallel_jobs", 10),
        "poll_interval": config.get("slurm", {}).get("poll_interval", 10),
        "max_poll_attempts": config.get("slurm", {}).get("max_poll_attempts", 720),
        # v3.2: Dual cluster routing + sub-agent options
        "cluster_for_subtasks": cluster_for_subtasks,
        "gpu_cluster_for_subtasks": gpu_cluster_for_subtasks,
        "cleanup_env_on_success": cleanup_env,
        "enable_checkpoints": True,
    }
    
    # =========================================================================
    # HANDLE CHECKPOINT MANAGEMENT (v3.2)
    # =========================================================================
    checkpoint_dir = project_dir / 'temp' / 'checkpoints'
    
    if args.clear_checkpoints:
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
            print(f"  Cleared checkpoints: {checkpoint_dir}")
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if args.resume and checkpoint_dir.exists():
        checkpoint_files = list(checkpoint_dir.glob('*_checkpoint.json'))
        if checkpoint_files:
            print(f"  Found {len(checkpoint_files)} checkpoint(s) - will resume")
            for cp in checkpoint_files[:5]:
                print(f"    - {cp.name}")
            if len(checkpoint_files) > 5:
                print(f"    ... and {len(checkpoint_files) - 5} more")
    # =========================================================================
    
    # Load task
    if args.prompt_file:
        try:
            prompt_data = load_prompt_file(args.prompt_file)
            main_task = prompt_data["task"]
            context = prompt_data.get("context", {})
            context["input_files"] = prompt_data.get("input_files", [])
            context["expected_outputs"] = prompt_data.get("expected_outputs", [])
            context["prompt_file"] = prompt_data.get("prompt_file", "")
            
            archive_path = archive_prompt(prompt_data, project_dir, config)
            print(f"Prompt archived to: {archive_path}")
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error parsing prompt file: {e}")
            sys.exit(1)
    else:
        main_task = args.task
        context = {}
        
        if args.context:
            try:
                context = json.loads(args.context)
            except json.JSONDecodeError:
                print("Warning: Could not parse context JSON")
    
    # Add execution context
    context["project_dir"] = str(project_dir)
    context["use_slurm"] = use_slurm
    context["cluster"] = cluster_name
    context["partition"] = partition
    context["parallel_enabled"] = parallel_enabled
    context["cluster_for_subtasks"] = cluster_for_subtasks
    context["gpu_cluster_for_subtasks"] = gpu_cluster_for_subtasks
    
    # Prepare cluster info for banner
    cluster_info = {
        "cluster": cluster_name,
        "partition": partition,
        "gpus": args.gpus or 0,
        "gpu_type": args.gpu_type or cluster_config_entry.get("gpu_type"),
        "nodelist": args.nodelist,
        "idle_count": slurm_status.get("idle_count") if slurm_status else None,
    }
    
    # Print banner
    print_banner(
        main_task, config, project_dir,
        cluster_info if use_slurm else None,
        model=args.model,
        max_iterations=args.max_iterations,
        cluster_for_subtasks=cluster_for_subtasks,
        gpu_cluster_for_subtasks=gpu_cluster_for_subtasks,
        cleanup_env=cleanup_env,
    )
    
    # Dry run
    if args.dry_run:
        print("DRY RUN MODE - No execution will occur\n")
        print("Task:", main_task)
        print("\nContext:", json.dumps(context, indent=2))
        print("\nExecution Mode:", "SLURM" if use_slurm else "Interactive")
        print("Model:", config['ollama']['model'])
        print("Context Window:", config.get('ollama', {}).get('model_context_length', 32768))
        print("Task Token Budget:", config.get('context', {}).get('max_tokens_per_task', 25000))
        print("Max Iterations:", config['agents']['max_retries'])
        print("Cluster:", cluster_name)
        print("Partition:", partition)
        if args.gpus:
            print("GPUs:", args.gpus, f"(--gres=gpu:{args.gpus})", args.gpu_type or "")
        print("Parallel:", "Enabled" if parallel_enabled else "Disabled")
        print("\n--- v3.2 Subtask Routing ---")
        print("CPU Subtasks →", cluster_for_subtasks)
        print("GPU Subtasks →", gpu_cluster_for_subtasks)
        print("Cleanup Env:", "Enabled" if cleanup_env else "Disabled")
        print("Checkpointing:", "Enabled")
        print("Resume Mode:", "Yes" if args.resume else "No")
        print("\nSLURM Config:", json.dumps(slurm_job_config, indent=2, default=str))
        print("\nProject structure:")
        print(sandbox.get_directory_tree())
        print("\nProject Isolation:")
        print(f"  Logs will go to:      {project_dir}/logs/")
        print(f"  Reports will go to:   {project_dir}/reports/")
        print(f"  Checkpoints in:       {project_dir}/temp/checkpoints/")
        print(f"  SLURM scripts in:     {project_dir}/slurm/scripts/")
        print(f"  SLURM logs in:        {project_dir}/slurm/logs/")
        sys.exit(0)
    
    # Initialize workflow
    try:
        workflow = MultiAgentWorkflow(
            ollama_model=config['ollama']['model'],
            ollama_base_url=config['ollama'].get('base_url', 'http://127.0.0.1:11434'),
            max_retries=config['agents']['max_retries'],
            project_dir=project_dir,
            use_slurm=use_slurm,
            parallel_enabled=parallel_enabled,
            slurm_config=slurm_job_config,
            use_reflexion_memory=config.get('reflexion', {}).get('enabled', True),
            cleanup_env_on_success=cleanup_env,
        )
    except Exception as e:
        print(f"Error initializing workflow: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    # Execute
    try:
        result = workflow.run(
            main_task=main_task,
            context=context,
            thread_id=args.thread_id,
        )
        
        print(f"\n{'='*70}")
        print(f"  Execution Complete!")
        print(f"{'='*70}")
        print(f"  Status:    {result.get('status', 'unknown')}")
        print(f"  Completed: {len(result.get('completed_subtasks', []))} subtasks")
        print(f"  Failed:    {len(result.get('failed_subtasks', []))} subtasks")
        print(f"\n  Report saved to: {project_dir}/reports/")
        print(f"{'='*70}\n")
        
        # Exit with appropriate code
        if result.get('status') == 'completed' and not result.get('failed_subtasks'):
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Cleaning up...")
        print(f"  Checkpoints preserved in: {checkpoint_dir}")
        print("  Resume with: --resume flag")
        sys.exit(130)
    except Exception as e:
        print(f"\nError during execution: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

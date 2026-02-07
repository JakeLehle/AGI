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
- Cluster configuration via AGI_CLUSTER environment variable
- Conda cleanup after successful task completion
- State checkpointing for resume capability
- Open-source conda channels only (no defaults/main)

Run with:
    # Zeus cluster (CPU, default)
    python main.py --prompt-file prompts/my_task.txt --project-dir /path/to/project --slurm

    # GPU cluster with specific partition
    python main.py --prompt-file prompts/gpu_task.txt --project-dir /path/to/project \\
        --slurm --cluster gpu_v100

    # Resume from checkpoints
    python main.py --prompt-file prompts/task.txt --project-dir ./project --resume

    # Clear checkpoints and start fresh
    python main.py --prompt-file prompts/task.txt --project-dir ./project --clear-checkpoints
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
            "ollama": {"model": "llama3.1:70b", "base_url": "http://127.0.0.1:11434"},
            "agents": {"max_retries": 12},
            "slurm": {"enabled": False, "default_cluster": "zeus_cpu"},
            "parallel": {"enabled": True},
            "clusters": {},
            "reflexion": {"enabled": True}
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


def list_clusters(config: dict):
    """List all configured clusters"""
    clusters = config.get("clusters", {})
    
    print("\n" + "="*70)
    print("  Available Clusters")
    print("="*70)
    
    default = config.get("slurm", {}).get("default_cluster", "zeus_cpu")
    
    for name, cluster_config in clusters.items():
        is_default = " (DEFAULT)" if name == default else ""
        has_gpu = cluster_config.get("has_gpu", False)
        gpu_str = " [GPU]" if has_gpu else " [CPU]"
        
        print(f"\n  {name}{gpu_str}{is_default}")
        print(f"    {cluster_config.get('description', 'No description')}")
        print(f"    Cores/Node: {cluster_config.get('cores_per_node', 'N/A')}")
        print(f"    Memory/Node: {cluster_config.get('memory_per_node', 'N/A')}")
        print(f"    Default Partition: {cluster_config.get('default_partition', 'N/A')}")
        
        # List partitions
        partitions = cluster_config.get("partitions", {})
        if partitions:
            print(f"    Partitions:")
            for pname, pconfig in partitions.items():
                gpu_info = ""
                if pconfig.get("has_gpu"):
                    gpu_info = f" (GPU: {pconfig.get('max_gpus', '?')}x {pconfig.get('gpu_type', '?')})"
                print(f"      - {pname}: {pconfig.get('description', '')[:40]}{gpu_info}")
    
    print("\n" + "="*70 + "\n")


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
                 model: str = None, max_iterations: int = None, cluster_for_subtasks: str = None,
                 cleanup_env: bool = True):
    """Print startup banner"""
    model = model or config['ollama']['model']
    max_iterations = max_iterations or config['agents']['max_retries']
    
    print(f"\n{'='*70}")
    print(f"  AGI Multi-Agent Pipeline v3.2")
    print(f"{'='*70}")
    print(f"  Model:           {model}")
    print(f"  Max Iterations:  {max_iterations}")
    print(f"  Project Dir:     {project_dir}")
    
    if cluster_info:
        print(f"\n  SLURM Configuration:")
        print(f"    Cluster:   {cluster_info.get('cluster', 'N/A')}")
        print(f"    Partition: {cluster_info.get('partition', 'N/A')}")
        
        if cluster_info.get('gpus'):
            print(f"    GPUs:      {cluster_info['gpus']}x {cluster_info.get('gpu_type', 'generic')}")
        
        if cluster_info.get('nodelist'):
            print(f"    Node(s):   {cluster_info['nodelist']}")
        
        if cluster_info.get('idle_count'):
            print(f"    Available: {cluster_info['idle_count']} nodes")
    
    # v3.2 specific info
    if cluster_for_subtasks:
        print(f"\n  Subtask Configuration:")
        print(f"    Cluster:       {cluster_for_subtasks}")
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
        description="Multi-Agent System for Complex Task Execution (v3.2 with Cluster Config)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on zeus CPU cluster (default)
  python main.py --prompt-file prompts/analysis.txt --project-dir ./project --slurm

  # Run on GPU cluster
  python main.py --prompt-file prompts/ml_task.txt --project-dir ./project \\
      --slurm --cluster gpu_v100

  # Resume from checkpoints
  python main.py --prompt-file prompts/task.txt --project-dir ./project --resume

  # Clear checkpoints and start fresh
  python main.py --prompt-file prompts/task.txt --project-dir ./project --clear-checkpoints

  # Disable conda cleanup (keep environments)
  python main.py --prompt-file prompts/task.txt --project-dir ./project --no-cleanup-env

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
        help="Ollama model to use (overrides config)"
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
        default=os.environ.get('AGI_CLUSTER', 'zeus_cpu'),
        help="Cluster for subtask SLURM settings (default: zeus_cpu or AGI_CLUSTER env)"
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
        help="SLURM partition to use (e.g., 'normal', 'gpu1v100')"
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
    slurm_group.add_argument("--memory", "--mem", type=str, help="Memory per job (e.g., '16G')")
    slurm_group.add_argument("--time", "-t", type=str, help="Time limit (e.g., '04:00:00')")
    
    # GPU options
    gpu_group = parser.add_argument_group('GPU Options')
    gpu_group.add_argument("--gpus", "-G", type=int, default=0, help="Number of GPUs to request")
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
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply command-line overrides
    if args.model:
        config['ollama']['model'] = args.model
    if args.max_iterations:
        config['agents']['max_retries'] = min(args.max_iterations, 12)
    if args.ollama_url:
        config['ollama']['base_url'] = args.ollama_url
    
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
        "cluster": args.cluster
    })
    # =========================================================================
    
    # Initialize sandbox
    sandbox = Sandbox(project_dir)
    
    # Handle --list-clusters
    if args.list_clusters:
        list_clusters(config)
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
    # cluster settings for sbatch generation.
    
    cluster_for_subtasks = args.cluster or os.environ.get('AGI_CLUSTER', 'zeus_cpu')
    os.environ['AGI_CLUSTER'] = cluster_for_subtasks
    
    # Set cluster config path
    cluster_config_path = args.cluster_config
    if not cluster_config_path:
        # Try to find config file
        possible_paths = [
            Path.cwd() / 'config' / 'cluster_config.yaml',
            Path(__file__).parent / 'config' / 'cluster_config.yaml',
        ]
        for p in possible_paths:
            if p.exists():
                cluster_config_path = str(p)
                break
    
    if cluster_config_path:
        os.environ['AGI_CLUSTER_CONFIG'] = cluster_config_path
    
    print(f"  Cluster for subtasks: {cluster_for_subtasks}")
    if cluster_config_path:
        print(f"  Cluster config: {cluster_config_path}")
    # =========================================================================
    
    # Determine cluster to use (for master job context)
    cluster_name = args.cluster or config.get("slurm", {}).get("default_cluster", "zeus_cpu")
    
    # Get cluster config
    cluster_config = config.get("clusters", {}).get(cluster_name, {})
    
    # Determine partition
    partition = args.partition or cluster_config.get("default_partition", "normal")
    
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
    
    # Build SLURM config
    slurm_job_config = {
        "cluster": cluster_name,
        "partition": partition,
        "cpus": args.cpus or cluster_config.get("default_cpus", 4),
        "memory": args.memory or cluster_config.get("default_memory", "16G"),
        "time": args.time or cluster_config.get("default_time", "04:00:00"),
        "gpus": args.gpus,
        "gpu_type": args.gpu_type,
        "nodelist": args.nodelist,
        "exclude_nodes": args.exclude,
        "max_parallel_jobs": args.max_parallel or config.get("parallel", {}).get("max_parallel_jobs", 10),
        "poll_interval": config.get("slurm", {}).get("poll_interval", 10),
        "max_poll_attempts": config.get("slurm", {}).get("max_poll_attempts", 720),
        # v3.2: Sub-agent options
        "cleanup_env_on_success": cleanup_env,
        "enable_checkpoints": True,
        "cluster_for_subtasks": cluster_for_subtasks,
    }
    
    # Validate GPU request
    if args.gpus and args.gpus > 0:
        partition_config = cluster_config.get("partitions", {}).get(partition, {})
        if not partition_config.get("has_gpu"):
            gpu_partitions = [p for p, c in cluster_config.get("partitions", {}).items() if c.get("has_gpu")]
            print(f"ERROR: Partition '{partition}' does not have GPUs.")
            if gpu_partitions:
                print(f"Available GPU partitions: {', '.join(gpu_partitions)}")
            sys.exit(1)
    
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
            for cp in checkpoint_files[:5]:  # Show first 5
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
    context["cluster_for_subtasks"] = cluster_for_subtasks  # v3.2
    
    # Prepare cluster info for banner
    cluster_info = {
        "cluster": cluster_name,
        "partition": partition,
        "gpus": args.gpus or 0,
        "gpu_type": args.gpu_type or cluster_config.get("partitions", {}).get(partition, {}).get("gpu_type"),
        "nodelist": args.nodelist,
        "idle_count": slurm_status.get("idle_count") if slurm_status else None
    }
    
    # Print banner
    print_banner(
        main_task, config, project_dir, 
        cluster_info if use_slurm else None,
        model=args.model,
        max_iterations=args.max_iterations,
        cluster_for_subtasks=cluster_for_subtasks,
        cleanup_env=cleanup_env
    )
    
    # Dry run
    if args.dry_run:
        print("DRY RUN MODE - No execution will occur\n")
        print("Task:", main_task)
        print("\nContext:", json.dumps(context, indent=2))
        print("\nExecution Mode:", "SLURM" if use_slurm else "Interactive")
        print("Model:", config['ollama']['model'])
        print("Max Iterations:", config['agents']['max_retries'])
        print("Cluster:", cluster_name)
        print("Partition:", partition)
        if args.gpus:
            print("GPUs:", args.gpus, args.gpu_type or "")
        print("Parallel:", "Enabled" if parallel_enabled else "Disabled")
        print("\n--- v3.2 Settings ---")
        print("Cluster for Subtasks:", cluster_for_subtasks)
        print("Cleanup Env:", "Enabled" if cleanup_env else "Disabled")
        print("Checkpointing:", "Enabled")
        print("Resume Mode:", "Yes" if args.resume else "No")
        print("\nSLURM Config:", json.dumps(slurm_job_config, indent=2))
        print("\nProject structure:")
        print(sandbox.get_directory_tree())
        print("\nProject Isolation:")
        print(f"  Logs will go to: {project_dir}/logs/")
        print(f"  Reports will go to: {project_dir}/reports/")
        print(f"  Checkpoints in: {project_dir}/temp/checkpoints/")
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
            cleanup_env_on_success=cleanup_env,  # v3.2
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
            thread_id=args.thread_id
        )
        
        print(f"\n{'='*70}")
        print(f"  Execution Complete!")
        print(f"{'='*70}")
        print(f"  Status: {result.get('status', 'unknown')}")
        print(f"  Completed: {len(result.get('completed_subtasks', []))} subtasks")
        print(f"  Failed: {len(result.get('failed_subtasks', []))} subtasks")
        print(f"\n  Report saved to: {project_dir}/reports/")
        print(f"{'='*70}\n")
        
        # Exit with appropriate code
        if result.get('status') == 'completed' and not result.get('failed_subtasks'):
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Cleaning up...")
        # Checkpoints are preserved automatically by sub-agents
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

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

Run with:
    # Zeus cluster (CPU, default)
    python main.py --prompt-file prompts/my_task.txt --project-dir /path/to/project --slurm

    # GPU cluster with specific partition
    python main.py --prompt-file prompts/gpu_task.txt --project-dir /path/to/project \\
        --slurm --cluster gpu_cluster --partition gpu1v100 --gpus 2

    # Override model and max iterations
    python main.py --prompt-file prompts/task.txt --project-dir ./project \\
        --model llama3.1:8b --max-iterations 6

    # Check available clusters and partitions
    python main.py --list-clusters --project-dir ./test
    python main.py --cluster-status --cluster gpu_cluster --project-dir ./test
"""

import argparse
import yaml
from pathlib import Path
from datetime import datetime
import json
import hashlib
import sys

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
            "slurm": {"enabled": False, "default_cluster": "zeus"},
            "parallel": {"enabled": True},
            "clusters": {}
        }


def load_prompt_file(prompt_path: str) -> dict:
    """
    Load task prompt from a text file.
    
    Expected format:
    ---
    # Task Description
    The main task description here...
    
    # Input Files
    - path/to/file1.csv
    - path/to/file2.json
    
    # Expected Outputs
    - reports/analysis.md
    - data/outputs/results.csv
    
    # Context
    Any additional context or constraints...
    ---
    
    Returns:
        Dict with 'task', 'input_files', 'expected_outputs', 'context'
    """
    prompt_path = Path(prompt_path)
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    content = prompt_path.read_text()
    
    result = {
        "task": "",
        "input_files": [],
        "expected_outputs": [],
        "context": {},
        "raw_prompt": content,
        "prompt_file": str(prompt_path)
    }
    
    current_section = "task"
    section_content = []
    
    for line in content.split('\n'):
        line_stripped = line.strip()
        
        if line_stripped.startswith('# Task') or line_stripped.startswith('# Description'):
            if section_content:
                result[current_section] = '\n'.join(section_content).strip()
            current_section = "task"
            section_content = []
        elif line_stripped.startswith('# Input'):
            if section_content:
                result[current_section] = '\n'.join(section_content).strip()
            current_section = "input_files"
            section_content = []
        elif line_stripped.startswith('# Expected') or line_stripped.startswith('# Output'):
            if section_content:
                result[current_section] = '\n'.join(section_content).strip()
            current_section = "expected_outputs"
            section_content = []
        elif line_stripped.startswith('# Context') or line_stripped.startswith('# Additional'):
            if section_content:
                result[current_section] = '\n'.join(section_content).strip()
            current_section = "context"
            section_content = []
        else:
            section_content.append(line)
    
    if section_content:
        result[current_section] = '\n'.join(section_content).strip()
    
    # Parse list sections
    if isinstance(result.get("input_files"), str):
        result["input_files"] = [
            line.strip().lstrip('- ').strip()
            for line in result["input_files"].split('\n')
            if line.strip() and not line.strip().startswith('#')
        ]
    
    if isinstance(result.get("expected_outputs"), str):
        result["expected_outputs"] = [
            line.strip().lstrip('- ').strip()
            for line in result["expected_outputs"].split('\n')
            if line.strip() and not line.strip().startswith('#')
        ]
    
    if isinstance(result.get("context"), str):
        result["context"] = {"notes": result["context"]}
    
    return result


def archive_prompt(prompt_data: dict, project_dir: Path, config: dict):
    """Save prompt to archive for record-keeping"""
    prompts_dir = project_dir / config.get("prompts", {}).get("prompts_dir", "prompts")
    prompts_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_hash = hashlib.md5(prompt_data.get("task", "")[:100].encode()).hexdigest()[:8]
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
    
    default = config.get("slurm", {}).get("default_cluster", "zeus")
    
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
        
        # Show some idle nodes
        idle_nodes = status.get("idle_nodes", [])
        if idle_nodes:
            print(f"\nIdle nodes: {', '.join(idle_nodes[:10])}")
            if len(idle_nodes) > 10:
                print(f"  ... and {len(idle_nodes) - 10} more")
    else:
        print(f"Error getting status: {status.get('error')}")
    
    print("="*70 + "\n")
    
    return {
        "available": True,
        "idle_count": status.get("idle_count", 0),
        "cluster": cluster_name or slurm_config.default_cluster
    }


def print_banner(task: str, config: dict, project_dir: Path, cluster_info: dict = None, 
                 model: str = None, max_iterations: int = None):
    """Print startup banner"""
    print(f"\n{'='*70}")
    print(f"  Multi-Agent System - Task Executor")
    print(f"{'='*70}")
    print(f"  Project Directory: {project_dir}")
    print(f"  Model: {model or config['ollama']['model']}")
    print(f"  Max Iterations: {max_iterations or config['agents']['max_retries']}")
    
    if cluster_info:
        cluster_name = cluster_info.get('cluster', 'N/A')
        partition = cluster_info.get('partition', 'default')
        print(f"  Cluster: {cluster_name}")
        print(f"  Partition: {partition}")
        
        if cluster_info.get('gpus', 0) > 0:
            gpu_type = cluster_info.get('gpu_type', 'GPU')
            print(f"  GPUs: {cluster_info['gpus']}x {gpu_type}")
        
        if cluster_info.get('nodelist'):
            print(f"  Node(s): {cluster_info['nodelist']}")
        
        if cluster_info.get('idle_count'):
            print(f"  Available Nodes: {cluster_info['idle_count']}")
    
    if config.get("parallel", {}).get("enabled"):
        print(f"  Parallel Execution: Enabled")
    
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
        description="Multi-Agent System for Complex Task Execution (Multi-Cluster Support)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on zeus cluster (CPU, default)
  python main.py --prompt-file prompts/analysis.txt --project-dir ./project --slurm

  # Run on GPU cluster with V100s
  python main.py --prompt-file prompts/ml_task.txt --project-dir ./ml_project \\
      --slurm --cluster gpu_cluster --partition gpu1v100 --gpus 2

  # Run on GPU cluster with A100s (high memory)
  python main.py --task "Train large model" --project-dir ./training \\
      --slurm --cluster gpu_cluster --partition gpu1a100 --gpus 4 --memory 256G

  # Override model and max iterations from command line
  python main.py --prompt-file prompts/task.txt --project-dir ./project \\
      --model llama3.1:8b --max-iterations 6

  # Run on specific node
  python main.py --task "Debug job" --project-dir ./debug \\
      --slurm --cluster gpu_cluster --partition gpu1v100 --nodelist gpu004

  # List available clusters
  python main.py --list-clusters --project-dir ./test

  # Check cluster status
  python main.py --cluster-status --cluster gpu_cluster --project-dir ./test

  # Check specific partition status
  python main.py --cluster-status --cluster gpu_cluster --partition gpu1a100 --project-dir ./test
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
        help="Ollama model to use (overrides config). E.g., 'llama3.1:70b', 'llama3.1:8b'"
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
        help="Cluster to use (e.g., 'zeus', 'gpu_cluster'). Default from config."
    )
    cluster_group.add_argument(
        "--partition", "-p",
        type=str,
        help="SLURM partition to use (e.g., 'normal', 'gpu1v100', 'dgxa100')"
    )
    cluster_group.add_argument(
        "--nodelist", "-w",
        type=str,
        help="Specific node(s) to run on (e.g., 'gpu001' or 'gpu[001-003]')"
    )
    cluster_group.add_argument(
        "--exclude", "-x",
        type=str,
        help="Node(s) to exclude (e.g., 'gpu002')"
    )
    
    # SLURM options
    slurm_group = parser.add_argument_group('SLURM Options')
    slurm_group.add_argument("--slurm", action="store_true", help="Enable SLURM job submission")
    slurm_group.add_argument("--no-slurm", action="store_true", help="Disable SLURM (force interactive)")
    slurm_group.add_argument("--cpus", "-c", type=int, help="CPUs per job")
    slurm_group.add_argument("--memory", "--mem", type=str, help="Memory per job (e.g., '16G', '256G')")
    slurm_group.add_argument("--time", "-t", type=str, help="Time limit (e.g., '04:00:00', '3-00:00:00')")
    
    # GPU options
    gpu_group = parser.add_argument_group('GPU Options')
    gpu_group.add_argument("--gpus", "-G", type=int, default=0, help="Number of GPUs to request")
    gpu_group.add_argument("--gpu-type", type=str, help="GPU type (e.g., 'v100', 'a100')")
    
    # Parallel execution
    parser.add_argument("--parallel", action="store_true", help="Enable parallel subtask execution")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel execution")
    parser.add_argument("--max-parallel", type=int, help="Maximum parallel jobs")
    
    # Other options
    parser.add_argument("--context", type=str, help="Additional context as JSON string")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--thread-id", type=str, help="Thread ID for resuming previous execution")
    parser.add_argument("--resume", action="store_true", help="Resume previous execution")
    parser.add_argument("--dry-run", action="store_true", help="Parse and validate without executing")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply command-line overrides to config
    if args.model:
        config['ollama']['model'] = args.model
    
    if args.max_iterations:
        # Enforce the hard limit of 12
        config['agents']['max_retries'] = min(args.max_iterations, 12)
    
    if args.ollama_url:
        config['ollama']['base_url'] = args.ollama_url
    
    # Validate and setup project directory
    project_dir = validate_project_dir(args.project_dir)
    
    # =========================================================================
    # CRITICAL: Configure utilities for project isolation
    # This ensures all logs, docs, and git operations happen in project_dir
    # NOT in the AGI root directory where the code lives
    # =========================================================================
    configure_logging(str(project_dir))
    configure_documentation(str(project_dir))
    configure_git_tracking(str(project_dir))
    
    # Log that we've configured for this project
    agent_logger.log_workflow_event("project_configured", {
        "project_dir": str(project_dir),
        "model": config['ollama']['model'],
        "max_retries": config['agents']['max_retries']
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
    
    # Determine cluster to use
    cluster_name = args.cluster or config.get("slurm", {}).get("default_cluster", "zeus")
    
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
        slurm_config = SlurmConfig(config_dict=config)
        slurm_tools = SlurmTools(sandbox, config=slurm_config, cluster_name=cluster_name)
        
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
        max_iterations=args.max_iterations
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
        print("\nSLURM Config:", json.dumps(slurm_job_config, indent=2))
        print("\nProject structure:")
        print(sandbox.get_directory_tree())
        print("\nProject Isolation:")
        print(f"  Logs will go to: {project_dir}/logs/")
        print(f"  Reports will go to: {project_dir}/reports/")
        print(f"  Git operations in: {project_dir}/.git/")
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
            slurm_config=slurm_job_config
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
        print(f"  Status: {result['status']}")
        print(f"  Completed Subtasks: {len(result.get('completed_subtasks', []))}")
        print(f"  Failed Subtasks: {len(result.get('failed_subtasks', []))}")
        print(f"  Model Used: {config['ollama']['model']}")
        print(f"  Max Iterations: {config['agents']['max_retries']}")
        print(f"  Cluster: {cluster_name}")
        print(f"  Execution Mode: {'SLURM' if use_slurm else 'Interactive'}")
        
        # Print attempt statistics if available
        if result.get('subtask_attempts'):
            print(f"\n  Attempt Statistics:")
            for task_id, attempts in result.get('subtask_attempts', {}).items():
                status_icon = "✓" if attempts < config['agents']['max_retries'] else "✗"
                print(f"    {status_icon} {task_id}: {attempts}/{config['agents']['max_retries']} attempts")
        
        print(f"\n  Final Report:")
        print(f"  {'-'*66}")
        
        report = result.get('final_report', 'No report generated')
        for line in report.split('\n')[:30]:
            print(f"  {line}")
        
        if len(report.split('\n')) > 30:
            print(f"  ... [truncated, see full report in {project_dir}/reports/]")
        
        print(f"  {'-'*66}")
        
        print(f"\n  Output Locations:")
        print(f"    - Documentation: {project_dir}/README.md")
        print(f"    - Logs: {project_dir}/logs/")
        print(f"    - Reports: {project_dir}/reports/")
        print(f"    - Outputs: {project_dir}/data/outputs/")
        print(f"    - Environment YAMLs: {project_dir}/envs/")
        if use_slurm:
            print(f"    - SLURM Scripts: {project_dir}/slurm/scripts/")
            print(f"    - SLURM Logs: {project_dir}/slurm/logs/")
        
        print(f"\n{'='*70}\n")
        
        # Save execution report to project directory
        try:
            report_path = doc_generator.save_execution_report()
            print(f"  Execution report saved to: {report_path}")
        except Exception as e:
            if args.verbose:
                print(f"  Warning: Could not save execution report: {e}")
        
        if result['status'] == 'completed':
            sys.exit(0)
        elif result['status'] == 'escalated':
            sys.exit(2)
        elif result['status'] == 'emergency_stopped':
            sys.exit(3)
        else:
            sys.exit(1)
        
    except KeyboardInterrupt:
        print(f"\n\n{'='*70}")
        print(f"  Execution interrupted by user")
        print(f"  State saved - resume with --thread-id")
        
        if use_slurm:
            print(f"  Cancelling SLURM jobs...")
            try:
                slurm_tools.cancel_all_jobs()
            except:
                pass
        
        print(f"{'='*70}\n")
        sys.exit(130)
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"  ERROR: Execution failed")
        print(f"{'='*70}")
        print(f"  Error: {str(e)}")
        print(f"\n  Check {project_dir}/logs/ directory for details")
        print(f"{'='*70}\n")
        
        if args.verbose:
            import traceback
            traceback.print_exc()
        
        sys.exit(1)


if __name__ == "__main__":
    main()

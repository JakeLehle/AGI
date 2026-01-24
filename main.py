"""
Main entry point for the multi-agent system.
Supports SLURM job submission and parallel execution on zeus cluster.

Run with:
    python main.py --prompt-file prompts/my_task.txt --project-dir /path/to/project

With SLURM:
    python main.py --prompt-file prompts/my_task.txt --project-dir /path/to/project --slurm

Or with inline task:
    python main.py --task "Your task description" --project-dir /path/to/project
"""

import argparse
import yaml
from pathlib import Path
from datetime import datetime
import json
import hashlib
import sys

from workflows.langgraph_workflow import MultiAgentWorkflow
from utils.logging_config import agent_logger
from tools.sandbox import Sandbox
from tools.slurm_tools import SlurmTools


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
            "slurm": {"enabled": False},
            "parallel": {"enabled": True}
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
    
    # Parse sections
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
        
        # Check for section headers
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
    
    # Don't forget last section
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
    
    # Parse context if it's a string
    if isinstance(result.get("context"), str):
        result["context"] = {"notes": result["context"]}
    
    return result


def archive_prompt(prompt_data: dict, project_dir: Path, config: dict):
    """Save prompt to archive for record-keeping"""
    prompts_dir = project_dir / config.get("prompts", {}).get("prompts_dir", "prompts")
    prompts_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_hash = hashlib.md5(prompt_data.get("task", "")[:100].encode()).hexdigest()[:8]
    filename = f"prompt_{timestamp}_{task_hash}.json"
    
    archive_path = prompts_dir / filename
    
    # Add metadata
    prompt_data["archived_at"] = datetime.now().isoformat()
    
    with open(archive_path, 'w') as f:
        json.dump(prompt_data, f, indent=2)
    
    return archive_path


def validate_project_dir(project_dir: str) -> Path:
    """Validate and create project directory"""
    project_path = Path(project_dir).resolve()
    
    # Create if doesn't exist
    project_path.mkdir(parents=True, exist_ok=True)
    
    return project_path


def check_slurm_status(sandbox: Sandbox) -> dict:
    """Check SLURM availability and cluster status"""
    slurm_tools = SlurmTools(sandbox)
    
    if not slurm_tools.slurm_available:
        return {
            "available": False,
            "message": "SLURM commands not found"
        }
    
    status = slurm_tools.get_cluster_status()
    
    if status["success"]:
        return {
            "available": True,
            "idle_nodes": status.get("idle_count", 0),
            "total_nodes": status.get("total_nodes", 0),
            "message": f"{status.get('idle_count', 0)} idle nodes available"
        }
    else:
        return {
            "available": False,
            "message": status.get("error", "Unknown error")
        }


def print_banner(task: str, config: dict, project_dir: Path, slurm_status: dict = None):
    """Print startup banner"""
    print(f"\n{'='*70}")
    print(f"  Multi-Agent System - Task Executor")
    print(f"{'='*70}")
    print(f"  Project Directory: {project_dir}")
    print(f"  Model: {config['ollama']['model']}")
    print(f"  Max Iterations: {config['agents']['max_retries']}")
    
    if slurm_status:
        if slurm_status.get("available"):
            print(f"  SLURM: Enabled ({slurm_status.get('message', 'Available')})")
        else:
            print(f"  SLURM: Disabled ({slurm_status.get('message', 'Not available')})")
    
    if config.get("parallel", {}).get("enabled"):
        print(f"  Parallel Execution: Enabled (max {config.get('slurm', {}).get('max_parallel_jobs', 5)} concurrent)")
    
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
        description="Multi-Agent System for Complex Task Execution (zeus cluster)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with a prompt file (interactive mode)
  python main.py --prompt-file prompts/biotech_analysis.txt --project-dir ./biotech_project

  # Run with SLURM job submission
  python main.py --prompt-file prompts/biotech_analysis.txt --project-dir ./biotech_project --slurm

  # Run with inline task and parallel execution
  python main.py --task "Analyze CSV files" --project-dir ./analysis --parallel

  # Specify resources for SLURM jobs
  python main.py --prompt-file prompts/task.txt --project-dir ./project --slurm --cpus 8 --memory 32G

  # Check cluster status
  python main.py --cluster-status --project-dir ./test

  # Resume previous execution
  python main.py --project-dir ./myproject --thread-id abc123 --resume
        """
    )
    
    # Task input (mutually exclusive)
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument(
        "--task",
        type=str,
        help="High-level task description (inline)"
    )
    task_group.add_argument(
        "--prompt-file",
        type=str,
        help="Path to prompt file containing task description"
    )
    task_group.add_argument(
        "--cluster-status",
        action="store_true",
        help="Just check and display cluster status"
    )
    
    # Required project directory
    parser.add_argument(
        "--project-dir",
        type=str,
        required=True,
        help="Project directory for all files and outputs"
    )
    
    # SLURM options
    slurm_group = parser.add_argument_group('SLURM Options')
    slurm_group.add_argument(
        "--slurm",
        action="store_true",
        help="Enable SLURM job submission (default: interactive)"
    )
    slurm_group.add_argument(
        "--no-slurm",
        action="store_true",
        help="Disable SLURM even if configured (force interactive)"
    )
    slurm_group.add_argument(
        "--partition",
        type=str,
        default="normal",
        help="SLURM partition to use (default: normal)"
    )
    slurm_group.add_argument(
        "--cpus",
        type=int,
        help="CPUs per job (default: from config)"
    )
    slurm_group.add_argument(
        "--memory",
        type=str,
        help="Memory per job, e.g., '16G' (default: from config)"
    )
    slurm_group.add_argument(
        "--time",
        type=str,
        help="Time limit per job, e.g., '04:00:00' (default: from config)"
    )
    
    # Parallel execution
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel execution of independent subtasks"
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel execution (sequential only)"
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        help="Maximum parallel jobs (default: from config)"
    )
    
    # Other options
    parser.add_argument(
        "--context",
        type=str,
        help="Additional context as JSON string"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        help="Thread ID for resuming previous execution"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume previous execution (requires --thread-id)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate inputs without executing"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Validate and setup project directory
    project_dir = validate_project_dir(args.project_dir)
    
    # Initialize sandbox
    sandbox = Sandbox(project_dir)
    
    # Check cluster status if requested
    if args.cluster_status:
        print("\nChecking cluster status...")
        status = check_slurm_status(sandbox)
        
        if status["available"]:
            print(f"\n✓ SLURM is available")
            print(f"  Idle nodes: {status.get('idle_nodes', 'unknown')}")
            print(f"  Total nodes: {status.get('total_nodes', 'unknown')}")
            
            # Get detailed status
            slurm_tools = SlurmTools(sandbox)
            detailed = slurm_tools.get_cluster_status()
            if detailed["success"]:
                print(f"\nNode Status:")
                for node in detailed.get("nodes", [])[:15]:  # First 15 nodes
                    print(f"  {node['name']}: {node['state']}")
        else:
            print(f"\n✗ SLURM is not available: {status.get('message')}")
        
        sys.exit(0)
    
    # Require task input if not checking status
    if not args.task and not args.prompt_file:
        parser.error("Either --task or --prompt-file is required")
    
    # Determine SLURM usage
    use_slurm = False
    if args.slurm:
        use_slurm = True
    elif args.no_slurm:
        use_slurm = False
    elif config.get("slurm", {}).get("enabled"):
        use_slurm = True
    
    # Check SLURM availability if requested
    slurm_status = None
    if use_slurm:
        slurm_status = check_slurm_status(sandbox)
        if not slurm_status["available"]:
            print(f"WARNING: SLURM requested but not available: {slurm_status.get('message')}")
            print("Falling back to interactive mode")
            use_slurm = False
    
    # Determine parallel execution
    parallel_enabled = True
    if args.no_parallel:
        parallel_enabled = False
    elif args.parallel:
        parallel_enabled = True
    elif config.get("parallel", {}).get("enabled") is not None:
        parallel_enabled = config["parallel"]["enabled"]
    
    # Build SLURM config from args and config file
    slurm_config = {
        "partition": args.partition or config.get("slurm", {}).get("partition", "normal"),
        "cpus": args.cpus or config.get("slurm", {}).get("default_cpus", 4),
        "memory": args.memory or config.get("slurm", {}).get("default_memory", "16G"),
        "time": args.time or config.get("slurm", {}).get("default_time", "04:00:00"),
        "max_parallel_jobs": args.max_parallel or config.get("slurm", {}).get("max_parallel_jobs", 5),
        "poll_interval": config.get("slurm", {}).get("poll_interval", 10),
        "max_poll_attempts": config.get("slurm", {}).get("max_poll_attempts", 720),
    }
    
    # Load task from prompt file or inline
    if args.prompt_file:
        try:
            prompt_data = load_prompt_file(args.prompt_file)
            main_task = prompt_data["task"]
            context = prompt_data.get("context", {})
            
            # Add input files and expected outputs to context
            context["input_files"] = prompt_data.get("input_files", [])
            context["expected_outputs"] = prompt_data.get("expected_outputs", [])
            context["prompt_file"] = prompt_data.get("prompt_file", "")
            
            # Archive the prompt
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
        
        # Parse additional context if provided
        if args.context:
            try:
                context = json.loads(args.context)
            except json.JSONDecodeError:
                print("Warning: Could not parse context JSON, proceeding without context")
    
    # Add project directory to context
    context["project_dir"] = str(project_dir)
    context["use_slurm"] = use_slurm
    context["parallel_enabled"] = parallel_enabled
    
    # Print startup banner
    print_banner(main_task, config, project_dir, slurm_status)
    
    # Dry run - just validate and show what would happen
    if args.dry_run:
        print("DRY RUN MODE - No execution will occur\n")
        print("Task:", main_task)
        print("\nContext:", json.dumps(context, indent=2))
        print("\nExecution Mode:", "SLURM" if use_slurm else "Interactive")
        print("Parallel Execution:", "Enabled" if parallel_enabled else "Disabled")
        print("\nSLURM Config:", json.dumps(slurm_config, indent=2))
        print("\nProject structure will be created at:", project_dir)
        print("\nDirectory tree:")
        print(sandbox.get_directory_tree())
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
            slurm_config=slurm_config
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
        print(f"  Execution Mode: {'SLURM' if use_slurm else 'Interactive'}")
        
        print(f"\n  Final Report:")
        print(f"  {'-'*66}")
        
        # Print report with wrapping
        report = result.get('final_report', 'No report generated')
        for line in report.split('\n')[:30]:  # First 30 lines
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
            print(f"    - SLURM Logs: {project_dir}/slurm/logs/")
        
        print(f"\n  Git commands:")
        print(f"    - View history: cd {project_dir} && git log")
        print(f"    - View failures: cd {project_dir} && git tag -l 'failure-*'")
        
        print(f"\n{'='*70}\n")
        
        # Return appropriate exit code
        if result['status'] == 'completed':
            sys.exit(0)
        elif result['status'] == 'escalated':
            sys.exit(2)
        else:
            sys.exit(1)
        
    except KeyboardInterrupt:
        print(f"\n\n{'='*70}")
        print(f"  Execution interrupted by user")
        print(f"  State saved - resume with --thread-id")
        
        # Cancel any running SLURM jobs
        if use_slurm:
            print(f"  Cancelling SLURM jobs...")
            try:
                from tools.slurm_tools import SlurmTools
                slurm_tools = SlurmTools(sandbox)
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

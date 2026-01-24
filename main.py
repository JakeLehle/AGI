"""
Main entry point for the multi-agent system.
Run with: python main.py --task "Your task description"
"""

import argparse
import yaml
from pathlib import Path
from workflows.langgraph_workflow import MultiAgentWorkflow
from utils.logging_config import agent_logger
import json

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found at {config_path}, using defaults")
        return {
            "ollama": {"model": "llama3.1:70b"},
            "agents": {"max_retries": 3}
        }

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent System for Complex Task Execution"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="High-level task description"
    )
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
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Parse context if provided
    context = {}
    if args.context:
        try:
            context = json.loads(args.context)
        except json.JSONDecodeError:
            print("Warning: Could not parse context JSON, proceeding without context")
    
    print(f"\n{'='*60}")
    print(f"Multi-Agent System Starting")
    print(f"{'='*60}")
    print(f"Task: {args.task}")
    print(f"Model: {config['ollama']['model']}")
    print(f"Max Retries: {config['agents']['max_retries']}")
    print(f"{'='*60}\n")
    
    # Initialize workflow
    workflow = MultiAgentWorkflow(
        ollama_model=config['ollama']['model'],
        max_retries=config['agents']['max_retries']
    )
    
    # Execute
    try:
        result = workflow.run(
            main_task=args.task,
            context=context,
            thread_id=args.thread_id
        )
        
        print(f"\n{'='*60}")
        print(f"Execution Complete!")
        print(f"{'='*60}")
        print(f"Status: {result['status']}")
        print(f"Completed Subtasks: {len(result.get('completed_subtasks', []))}")
        print(f"Failed Subtasks: {len(result.get('failed_subtasks', []))}")
        print(f"\nFinal Report:")
        print(f"{'-'*60}")
        print(result.get('final_report', 'No report generated'))
        print(f"{'-'*60}")
        print(f"\nDocumentation saved to: README.md")
        print(f"Logs saved to: logs/")
        print(f"Git history: git log")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: Execution failed")
        print(f"{'='*60}")
        print(f"Error: {str(e)}")
        print(f"\nCheck logs/ directory for details")
        print(f"{'='*60}\n")
        raise

if __name__ == "__main__":
    main()

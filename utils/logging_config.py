"""
Logging configuration for the multi-agent system.
Provides structured JSON logging with automatic context.
"""

from loguru import logger
import sys
from pathlib import Path
from datetime import datetime
import json

class AgentLogger:
    """Structured logger for agent operations"""
    
    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path)
        self.logs_dir = self.project_path / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # Remove default logger
        logger.remove()
        
        # Add console logger (human-readable)
        logger.add(
            sys.stdout,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[agent]}</cyan> | {message}",
            level="INFO",
            colorize=True
        )
        
        # Add file logger (JSON structured)
        logger.add(
            self.logs_dir / f"agent_{datetime.now():%Y%m%d_%H%M%S}.jsonl",
            format="{message}",
            level="DEBUG",
            serialize=True,  # JSON format
            rotation="500 MB",
            compression="zip"
        )
        
        # Add error-only logger
        logger.add(
            self.logs_dir / "errors.jsonl",
            format="{message}",
            level="ERROR",
            serialize=True,
            rotation="100 MB"
        )
    
    def log_task_start(self, agent_name: str, task_id: str, description: str, attempt: int = 1):
        """Log task initiation"""
        logger.bind(agent=agent_name).info(
            "Task started",
            extra={
                "event": "task_started",
                "task_id": task_id,
                "description": description,
                "attempt": attempt,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def log_task_success(self, agent_name: str, task_id: str, result: dict, tools_used: list):
        """Log successful task completion"""
        logger.bind(agent=agent_name).success(
            f"Task {task_id} completed successfully",
            extra={
                "event": "task_completed",
                "task_id": task_id,
                "tools_used": tools_used,
                "result_preview": str(result)[:200],
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def log_task_failure(self, agent_name: str, task_id: str, error: str, context: dict):
        """Log task failure with context"""
        logger.bind(agent=agent_name).error(
            f"Task {task_id} failed",
            extra={
                "event": "task_failed",
                "task_id": task_id,
                "error": error,
                "context": context,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def log_reflection(self, agent_name: str, task_id: str, reflection: str):
        """Log agent reflection/analysis"""
        logger.bind(agent=agent_name).info(
            "Reflection generated",
            extra={
                "event": "reflection",
                "task_id": task_id,
                "reflection": reflection,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def log_tool_creation(self, agent_name: str, tool_name: str, reason: str, code: str):
        """Log dynamic tool creation"""
        logger.bind(agent=agent_name).warning(
            f"New tool created: {tool_name}",
            extra={
                "event": "tool_created",
                "tool_name": tool_name,
                "reason": reason,
                "code": code,
                "timestamp": datetime.now().isoformat()
            }
        )

# Global logger instance
agent_logger = AgentLogger()

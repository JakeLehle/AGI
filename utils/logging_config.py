"""
Logging configuration for the multi-agent system.
Provides structured JSON logging with automatic context.

Supports project-scoped logging where each project has its own log directory.
"""

from loguru import logger
import sys
from pathlib import Path
from datetime import datetime
import json
from typing import Optional


class AgentLogger:
    """
    Structured logger for agent operations.
    
    Can be initialized with a specific project path to ensure all logs
    go to the project's logs/ directory rather than the AGI root.
    """
    
    _instance: Optional['AgentLogger'] = None
    _initialized_path: Optional[Path] = None
    
    def __init__(self, project_path: str = None):
        """
        Initialize logger for a specific project.
        
        Args:
            project_path: Path to the project directory. If None, logging
                         is deferred until configure() is called.
        """
        self.project_path = Path(project_path) if project_path else None
        self.logs_dir = None
        self._configured = False
        
        if self.project_path:
            self._setup_logging()
    
    def configure(self, project_path: str):
        """
        Configure or reconfigure logging for a specific project directory.
        
        This should be called early in main.py after project_dir is known.
        
        Args:
            project_path: Absolute path to the project directory
        """
        self.project_path = Path(project_path).resolve()
        self._setup_logging()
        
        # Update class-level tracking
        AgentLogger._initialized_path = self.project_path
        
        logger.info(f"Logger configured for project: {self.project_path}")
    
    def _setup_logging(self):
        """Set up loguru handlers for the project"""
        if not self.project_path:
            return
            
        self.logs_dir = self.project_path / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Remove any existing handlers to avoid duplicates
        logger.remove()
        
        # Console logger (human-readable)
        logger.add(
            sys.stdout,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[agent]}</cyan> | {message}",
            level="INFO",
            colorize=True,
            filter=lambda record: "agent" in record["extra"]
        )
        
        # Fallback console logger for messages without agent context
        logger.add(
            sys.stdout,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
            level="INFO",
            colorize=True,
            filter=lambda record: "agent" not in record["extra"]
        )
        
        # File logger (JSON structured) - project-specific
        log_filename = f"agent_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
        logger.add(
            self.logs_dir / log_filename,
            format="{message}",
            level="DEBUG",
            serialize=True,
            rotation="500 MB",
            compression="zip"
        )
        
        # Error-only logger - project-specific
        logger.add(
            self.logs_dir / "errors.jsonl",
            format="{message}",
            level="ERROR",
            serialize=True,
            rotation="100 MB"
        )
        
        self._configured = True
    
    def _ensure_configured(self):
        """Ensure logger is configured before use"""
        if not self._configured:
            # Fallback to current directory if not configured
            # This allows basic functionality during imports
            logger.warning("AgentLogger used before configure() - logs may go to wrong location")
    
    def log_task_start(self, agent_name: str, task_id: str, description: str, attempt: int = 1):
        """Log task initiation"""
        self._ensure_configured()
        logger.bind(agent=agent_name).info(
            f"Task {task_id} started (attempt {attempt})",
            extra={
                "event": "task_started",
                "task_id": task_id,
                "description": description,
                "attempt": attempt,
                "timestamp": datetime.now().isoformat(),
                "project_path": str(self.project_path) if self.project_path else None
            }
        )
    
    def log_task_success(self, agent_name: str, task_id: str, result: dict, tools_used: list):
        """Log successful task completion"""
        self._ensure_configured()
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
        self._ensure_configured()
        logger.bind(agent=agent_name).error(
            f"Task {task_id} failed: {error[:100]}",
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
        self._ensure_configured()
        logger.bind(agent=agent_name).info(
            f"Reflection for {task_id}",
            extra={
                "event": "reflection",
                "task_id": task_id,
                "reflection": reflection,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def log_tool_creation(self, agent_name: str, tool_name: str, reason: str, code: str):
        """Log dynamic tool creation"""
        self._ensure_configured()
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
    
    def log_slurm_job(self, agent_name: str, job_id: str, status: str, details: dict = None):
        """Log SLURM job events"""
        self._ensure_configured()
        logger.bind(agent=agent_name).info(
            f"SLURM job {job_id}: {status}",
            extra={
                "event": "slurm_job",
                "job_id": job_id,
                "status": status,
                "details": details or {},
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def log_workflow_event(self, event_type: str, details: dict):
        """Log workflow-level events"""
        self._ensure_configured()
        logger.bind(agent="workflow").info(
            f"Workflow event: {event_type}",
            extra={
                "event": event_type,
                "details": details,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def get_log_path(self) -> Optional[Path]:
        """Return the current logs directory path"""
        return self.logs_dir


# Global instance - will be configured by main.py
# This allows imports to work, but logging is deferred until configure() is called
agent_logger = AgentLogger()


def configure_logging(project_path: str):
    """
    Convenience function to configure the global logger.
    
    Call this from main.py after project_dir is known:
        from utils.logging_config import configure_logging
        configure_logging(project_dir)
    """
    agent_logger.configure(project_path)

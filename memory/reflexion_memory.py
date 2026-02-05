"""
ReflexionMemory: Mem0 wrapper for AGI pipeline loop prevention.

This module provides the core memory interface for the Reflexion Engine,
enabling the pipeline to:
1. Track failed approaches to avoid repeating them
2. Store successful solutions for reuse
3. Use semantic similarity to detect "same approach, different words"
4. Maintain persistent memory across pipeline invocations

The key method is `check_if_tried()` which uses vector similarity to detect
when a proposed approach is too similar to something already attempted,
even if worded differently. This breaks the infinite retry loop.

Example:
    memory = ReflexionMemory()
    
    # After a failure, store what was tried
    memory.store_failure(
        task_id="task_001",
        error_type=FailureType.MISSING_PACKAGE,
        error_message="ModuleNotFoundError: No module named 'pandas'",
        approach_tried="Added import pandas without installing the package"
    )
    
    # Before next retry, check if proposed approach was already tried
    result = memory.check_if_tried(
        task_id="task_001",
        proposed_approach="Import pandas at the top of the file"
    )
    # Returns: {"tried": True, "similarity": 0.89, ...}
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Data Types
# =============================================================================

class FailureType(Enum):
    """
    Classification of failure types for routing to specialist agents.
    
    In v3 architecture, each type maps to a specialist:
    - CODE_BUG -> Developer Agent
    - MISSING_PACKAGE -> Deployer Agent
    - MISSING_FILE -> Developer/PM
    - OUT_OF_MEMORY -> Architect (redesign needed)
    - GPU_ERROR -> Deployer Agent
    - TIMEOUT -> Architect (optimization needed)
    - DESIGN_FLAW -> Architect
    - DEPENDENCY_ISSUE -> PM (task ordering)
    - DATA_ISSUE -> Analyst
    - UNKNOWN -> Run diagnostics first
    """
    CODE_BUG = "code_bug"
    MISSING_PACKAGE = "missing_package"
    MISSING_FILE = "missing_file"
    OUT_OF_MEMORY = "out_of_memory"
    GPU_ERROR = "gpu_error"
    TIMEOUT = "timeout"
    DESIGN_FLAW = "design_flaw"
    DEPENDENCY_ISSUE = "dependency_issue"
    DATA_ISSUE = "data_issue"
    PERMISSION_ERROR = "permission_error"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    SLURM_ERROR = "slurm_error"
    CONDA_ERROR = "conda_error"
    UNKNOWN = "unknown"


@dataclass
class FailureRecord:
    """Record of a failed attempt for storage in memory."""
    task_id: str
    error_type: Union[FailureType, str]
    error_message: str
    approach_tried: str
    script_path: Optional[str] = None
    diagnostic_results: Optional[Dict] = None
    slurm_job_id: Optional[str] = None
    attempt_number: int = 1
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    resolved: bool = False
    resolution: Optional[str] = None
    
    def __post_init__(self):
        if isinstance(self.error_type, str):
            try:
                self.error_type = FailureType(self.error_type)
            except ValueError:
                self.error_type = FailureType.UNKNOWN


@dataclass
class SolutionRecord:
    """Record of a successful solution for reuse."""
    task_id: str
    problem_pattern: str
    error_type: Union[FailureType, str]
    solution: str
    context: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    reuse_count: int = 0
    
    def __post_init__(self):
        if isinstance(self.error_type, str):
            try:
                self.error_type = FailureType(self.error_type)
            except ValueError:
                self.error_type = FailureType.UNKNOWN


# =============================================================================
# Main Memory Class
# =============================================================================

class ReflexionMemory:
    """
    Memory layer for the Reflexion Engine using Mem0 + embedded Qdrant.
    
    Provides semantic memory for:
    - Tracking failed approaches (to avoid repeating)
    - Storing successful solutions (for reuse)
    - Checking if proposed approaches are too similar to tried ones
    
    Memory persists across pipeline invocations via embedded Qdrant storage.
    
    Attributes:
        SIMILARITY_THRESHOLD: Approaches above this threshold are considered "same"
        FAILURES_USER: Mem0 namespace for failure records
        SOLUTIONS_USER: Mem0 namespace for solution records
    """
    
    # Similarity threshold - above this, approaches are considered duplicates
    SIMILARITY_THRESHOLD = 0.85
    
    # Mem0 user IDs for namespacing
    FAILURES_USER = "agi_failures"
    SOLUTIONS_USER = "agi_solutions"
    
    def __init__(
        self, 
        config_path: Optional[str] = None, 
        config: Optional[Dict] = None
    ):
        """
        Initialize ReflexionMemory with Mem0 backend.
        
        Args:
            config_path: Path to mem0_config.yaml file
            config: Direct config dict (overrides config_path)
        """
        # Import here for graceful failure if not installed
        try:
            from mem0 import Memory
        except ImportError:
            raise ImportError(
                "mem0ai package not installed. "
                "Install with: pip install mem0ai"
            )
        
        # Load configuration
        if config is not None:
            mem0_config = config
            self._custom_prompt = None
        else:
            from .config import get_mem0_config
            config_data = get_mem0_config(config_path)
            mem0_config = config_data["config"]
            self._custom_prompt = config_data.get("custom_fact_extraction_prompt")
        
        # Initialize Mem0 with embedded Qdrant
        logger.info("Initializing Mem0 with embedded Qdrant...")
        self.memory = Memory.from_config(mem0_config)
        logger.info("ReflexionMemory initialized successfully")
    
    # =========================================================================
    # FAILURE TRACKING
    # =========================================================================
    
    def store_failure(
        self,
        task_id: str,
        error_type: FailureType,
        error_message: str,
        approach_tried: str,
        script_path: Optional[str] = None,
        diagnostic_results: Optional[Dict] = None,
        slurm_job_id: Optional[str] = None,
        attempt_number: int = 1,
    ) -> Dict[str, Any]:
        """
        Store a failed approach attempt in memory.
        
        Args:
            task_id: Unique identifier for the task
            error_type: Classification of the error
            error_message: The actual error/traceback
            approach_tried: Description of what was attempted
            script_path: Path to the script that failed
            diagnostic_results: Results from diagnostic scripts
            slurm_job_id: SLURM job ID if applicable
            attempt_number: Which attempt this was
            
        Returns:
            Mem0 result with memory_id
        """
        record = FailureRecord(
            task_id=task_id,
            error_type=error_type,
            error_message=error_message,
            approach_tried=approach_tried,
            script_path=script_path,
            diagnostic_results=diagnostic_results,
            slurm_job_id=slurm_job_id,
            attempt_number=attempt_number,
        )
        
        # Create searchable text
        memory_text = self._failure_to_text(record)
        
        # Store with metadata for filtering
        # infer=False skips LLM fact extraction (fast, ~100ms vs 30-60s)
        # We don't need extraction since we're storing structured records
        result = self.memory.add(
            memory_text,
            user_id=self.FAILURES_USER,
            infer=False,  # Skip LLM extraction for speed
            metadata={
                "task_id": task_id,
                "error_type": error_type.value if isinstance(error_type, FailureType) else error_type,
                "record_type": "failure",
                "attempt_number": attempt_number,
                "timestamp": record.timestamp,
                "has_diagnostics": diagnostic_results is not None,
            }
        )
        
        logger.info(
            f"Stored failure for task {task_id}: "
            f"{error_type.value if isinstance(error_type, FailureType) else error_type} "
            f"(attempt #{attempt_number})"
        )
        return result
    
    def _failure_to_text(self, record: FailureRecord) -> str:
        """Convert failure record to searchable text."""
        error_type_str = (
            record.error_type.value 
            if isinstance(record.error_type, FailureType) 
            else record.error_type
        )
        
        parts = [
            f"Task ID: {record.task_id}",
            f"Error Type: {error_type_str}",
            f"Error Message: {record.error_message[:500]}",
            f"Approach Tried: {record.approach_tried}",
            f"Attempt Number: {record.attempt_number}",
        ]
        
        if record.script_path:
            parts.append(f"Script Path: {record.script_path}")
        if record.slurm_job_id:
            parts.append(f"SLURM Job: {record.slurm_job_id}")
        if record.diagnostic_results:
            diag_summary = json.dumps(record.diagnostic_results, indent=2)[:500]
            parts.append(f"Diagnostics: {diag_summary}")
        
        return "\n".join(parts)
    
    # =========================================================================
    # SOLUTION TRACKING
    # =========================================================================
    
    def store_solution(
        self,
        task_id: str,
        problem_pattern: str,
        error_type: FailureType,
        solution: str,
        context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Store a successful solution for future reuse.
        
        Args:
            task_id: Task where this solution was discovered
            problem_pattern: Searchable description of the problem
            error_type: Type of error this solution addresses
            solution: The fix that worked
            context: Additional context (environment, config, etc.)
            
        Returns:
            Mem0 result with memory_id
        """
        record = SolutionRecord(
            task_id=task_id,
            problem_pattern=problem_pattern,
            error_type=error_type,
            solution=solution,
            context=context or {},
        )
        
        memory_text = self._solution_to_text(record)
        
        result = self.memory.add(
            memory_text,
            user_id=self.SOLUTIONS_USER,
            infer=False,  # Skip LLM extraction for speed
            metadata={
                "task_id": task_id,
                "error_type": (
                    error_type.value 
                    if isinstance(error_type, FailureType) 
                    else error_type
                ),
                "record_type": "solution",
                "timestamp": record.timestamp,
            }
        )
        
        logger.info(
            f"Stored solution for "
            f"{error_type.value if isinstance(error_type, FailureType) else error_type}: "
            f"{problem_pattern[:50]}..."
        )
        return result
    
    def _solution_to_text(self, record: SolutionRecord) -> str:
        """Convert solution record to searchable text."""
        error_type_str = (
            record.error_type.value 
            if isinstance(record.error_type, FailureType) 
            else record.error_type
        )
        
        parts = [
            f"Problem Pattern: {record.problem_pattern}",
            f"Error Type: {error_type_str}",
            f"Solution: {record.solution}",
        ]
        
        if record.context:
            context_str = json.dumps(record.context, indent=2)[:300]
            parts.append(f"Context: {context_str}")
        
        return "\n".join(parts)
    
    # =========================================================================
    # SIMILARITY CHECKING - Core Loop Prevention
    # =========================================================================
    
    def check_if_tried(
        self,
        task_id: str,
        proposed_approach: str,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Check if a similar approach was already tried for this task.
        
        *** THIS IS THE KEY METHOD FOR PREVENTING INFINITE LOOPS ***
        
        Uses semantic similarity to detect when a "new" approach is actually
        a rewording of something already attempted.
        
        Args:
            task_id: The task to check
            proposed_approach: The approach being considered
            threshold: Similarity threshold (default: 0.85)
            
        Returns:
            {
                "tried": bool,           # True if similar approach found
                "similar_approach": str, # The matching approach
                "similarity": float,     # Similarity score (0-1)
                "memory_id": str,        # ID of matching memory
                "attempt_count": int,    # Total approaches tried
            }
        """
        threshold = threshold or self.SIMILARITY_THRESHOLD
        
        # Build search query
        search_query = f"Task ID: {task_id}\nApproach Tried: {proposed_approach}"
        
        try:
            results = self.memory.search(
                search_query,
                user_id=self.FAILURES_USER,
                limit=10,
            )
        except Exception as e:
            logger.warning(f"Memory search failed: {e}")
            return {
                "tried": False,
                "similar_approach": None,
                "similarity": 0.0,
                "memory_id": None,
                "attempt_count": 0,
            }
        
        # Filter to this task and find best match
        task_attempts = []
        best_match = None
        best_score = 0.0
        
        for result in results.get("results", []):
            metadata = result.get("metadata", {})
            
            # Only consider this task's failures
            if metadata.get("task_id") != task_id:
                continue
            
            task_attempts.append(result)
            score = result.get("score", 0)
            
            if score > best_score:
                best_score = score
                best_match = result
        
        # Check threshold
        if best_match and best_score >= threshold:
            return {
                "tried": True,
                "similar_approach": best_match.get("memory", ""),
                "similarity": best_score,
                "memory_id": best_match.get("id"),
                "attempt_count": len(task_attempts),
            }
        
        return {
            "tried": False,
            "similar_approach": best_match.get("memory") if best_match else None,
            "similarity": best_score,
            "memory_id": None,
            "attempt_count": len(task_attempts),
        }
    
    # =========================================================================
    # RETRIEVAL METHODS
    # =========================================================================
    
    def get_tried_approaches(
        self,
        task_id: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get all approaches tried for a specific task.
        
        Args:
            task_id: The task to query
            limit: Maximum results
            
        Returns:
            List of failure records for this task
        """
        try:
            results = self.memory.search(
                f"Task ID: {task_id}",
                user_id=self.FAILURES_USER,
                limit=limit,
            )
        except Exception as e:
            logger.warning(f"Failed to get tried approaches: {e}")
            return []
        
        task_failures = []
        for result in results.get("results", []):
            metadata = result.get("metadata", {})
            if metadata.get("task_id") == task_id:
                task_failures.append({
                    "approach": result.get("memory", ""),
                    "error_type": metadata.get("error_type"),
                    "attempt_number": metadata.get("attempt_number", 0),
                    "timestamp": metadata.get("timestamp"),
                    "memory_id": result.get("id"),
                    "score": result.get("score", 0),
                })
        
        # Sort by attempt number
        task_failures.sort(key=lambda x: x.get("attempt_number", 0))
        return task_failures
    
    def count_attempts_by_type(self, task_id: str) -> Dict[str, int]:
        """
        Count attempts by error type for escalation decisions.
        
        Args:
            task_id: The task to analyze
            
        Returns:
            Dict mapping error_type -> count
        """
        attempts = self.get_tried_approaches(task_id)
        counts = {}
        for attempt in attempts:
            error_type = attempt.get("error_type", "unknown")
            counts[error_type] = counts.get(error_type, 0) + 1
        return counts
    
    def search_similar_failures(
        self,
        error_message: str,
        error_type: Optional[FailureType] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find similar failures across ALL tasks.
        
        Args:
            error_message: The error to search for
            error_type: Optional filter
            limit: Maximum results
            
        Returns:
            List of similar failure records
        """
        query = f"Error Message: {error_message}"
        if error_type:
            error_type_str = (
                error_type.value 
                if isinstance(error_type, FailureType) 
                else error_type
            )
            query += f"\nError Type: {error_type_str}"
        
        try:
            results = self.memory.search(
                query,
                user_id=self.FAILURES_USER,
                limit=limit,
            )
        except Exception as e:
            logger.warning(f"Failed to search failures: {e}")
            return []
        
        return [
            {
                "memory": result.get("memory", ""),
                "score": result.get("score", 0),
                "task_id": result.get("metadata", {}).get("task_id"),
                "error_type": result.get("metadata", {}).get("error_type"),
                "memory_id": result.get("id"),
            }
            for result in results.get("results", [])
        ]
    
    def get_working_solutions(
        self,
        problem_description: str,
        error_type: Optional[FailureType] = None,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Find solutions that worked for similar problems.
        
        Args:
            problem_description: Current problem description
            error_type: Optional filter
            limit: Maximum results
            
        Returns:
            List of potentially applicable solutions
        """
        query = f"Problem Pattern: {problem_description}"
        if error_type:
            error_type_str = (
                error_type.value 
                if isinstance(error_type, FailureType) 
                else error_type
            )
            query += f"\nError Type: {error_type_str}"
        
        try:
            results = self.memory.search(
                query,
                user_id=self.SOLUTIONS_USER,
                limit=limit,
            )
        except Exception as e:
            logger.warning(f"Failed to search solutions: {e}")
            return []
        
        return [
            {
                "solution": result.get("memory", ""),
                "score": result.get("score", 0),
                "task_id": result.get("metadata", {}).get("task_id"),
                "error_type": result.get("metadata", {}).get("error_type"),
                "memory_id": result.get("id"),
            }
            for result in results.get("results", [])
        ]
    
    # =========================================================================
    # RESOLUTION & LIFECYCLE
    # =========================================================================
    
    def mark_resolved(
        self,
        task_id: str,
        resolution_description: str,
        successful_approach: str,
    ) -> Dict[str, Any]:
        """
        Mark a task as resolved and store the winning solution.
        
        Args:
            task_id: The resolved task
            resolution_description: What fixed it
            successful_approach: The approach that worked
            
        Returns:
            Result of storing the solution
        """
        attempts = self.get_tried_approaches(task_id, limit=1)
        
        if attempts:
            error_type = FailureType(attempts[0].get("error_type", "unknown"))
        else:
            error_type = FailureType.UNKNOWN
        
        result = self.store_solution(
            task_id=task_id,
            problem_pattern=f"Task {task_id} - {resolution_description}",
            error_type=error_type,
            solution=successful_approach,
            context={
                "resolution": resolution_description,
                "total_attempts": len(self.get_tried_approaches(task_id)),
            },
        )
        
        logger.info(f"Marked task {task_id} as resolved")
        return result
    
    # =========================================================================
    # STATISTICS & DEBUGGING
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics for monitoring."""
        try:
            failures = self.memory.get_all(user_id=self.FAILURES_USER)
            solutions = self.memory.get_all(user_id=self.SOLUTIONS_USER)
        except Exception as e:
            logger.warning(f"Failed to get stats: {e}")
            return {"error": str(e)}
        
        failure_results = failures.get("results", [])
        solution_results = solutions.get("results", [])
        
        return {
            "total_failures": len(failure_results),
            "total_solutions": len(solution_results),
            "failure_types": self._count_by_field(failure_results, "error_type"),
            "solution_types": self._count_by_field(solution_results, "error_type"),
            "unique_tasks": len(set(
                r.get("metadata", {}).get("task_id")
                for r in failure_results
                if r.get("metadata", {}).get("task_id")
            )),
        }
    
    def _count_by_field(self, results: List[Dict], field: str) -> Dict[str, int]:
        """Count records by a metadata field."""
        counts = {}
        for result in results:
            value = result.get("metadata", {}).get(field, "unknown")
            counts[value] = counts.get(value, 0) + 1
        return counts
    
    def clear_task_memory(self, task_id: str) -> int:
        """
        Clear all memory for a specific task.
        
        Args:
            task_id: Task to clear
            
        Returns:
            Number of memories deleted
        """
        failures = self.get_tried_approaches(task_id, limit=100)
        deleted = 0
        
        for failure in failures:
            if memory_id := failure.get("memory_id"):
                try:
                    self.memory.delete(memory_id)
                    deleted += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {memory_id}: {e}")
        
        logger.warning(f"Cleared {deleted} memories for task {task_id}")
        return deleted
    
    def reset_all(self, confirm: bool = False) -> bool:
        """
        Reset ALL memory. DESTRUCTIVE.
        
        Args:
            confirm: Must be True to proceed
            
        Returns:
            True if reset succeeded
        """
        if not confirm:
            logger.warning("Reset not confirmed. Pass confirm=True")
            return False
        
        try:
            self.memory.reset()
            logger.warning("ALL MEMORY RESET")
            return True
        except Exception as e:
            logger.error(f"Reset failed: {e}")
            return False

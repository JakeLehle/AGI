"""
Reflexion Engine: Core loop-prevention logic for AGI Pipeline v3.

The Reflexion Engine sits between the Master Agent and Sub-Agents, providing:
1. Failure classification (what type of error occurred)
2. Approach deduplication (have we tried this before?)
3. Escalation decisions (when to give up or route differently)
4. Solution lookup (do we already know how to fix this?)

The engine uses semantic memory (Mem0) to detect when agents are
attempting the same fix with different wording - breaking infinite loops.

Usage:
    from engines import ReflexionEngine, ReflexionAction
    
    engine = ReflexionEngine()
    
    # When a task fails, get a decision
    decision = engine.reflect_on_failure(
        task_id="task_001",
        error_message="ModuleNotFoundError: No module named 'pandas'",
        proposed_approach="Add import statement for pandas",
        context={"script_path": "/path/to/script.py"}
    )
    
    if decision.action == ReflexionAction.REJECT_DUPLICATE:
        # Agent tried this before, need genuinely different approach
        pass
    elif decision.action == ReflexionAction.RETRY:
        # OK to try this approach
        pass
    elif decision.action == ReflexionAction.ESCALATE:
        # Too many attempts, escalate to human or different agent
        pass
    elif decision.action == ReflexionAction.APPLY_SOLUTION:
        # We have a known solution for this problem
        pass
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging
import re

# Use absolute imports instead of relative
from memory import ReflexionMemory, FailureType

logger = logging.getLogger(__name__)


# =============================================================================
# Decision Types
# =============================================================================

class ReflexionAction(Enum):
    """Possible actions the engine can recommend."""
    RETRY = "retry"                    # OK to try this approach
    REJECT_DUPLICATE = "reject_duplicate"  # Too similar to previous attempt
    ESCALATE = "escalate"              # Give up, route to human/different agent
    APPLY_SOLUTION = "apply_solution"  # Use known solution
    RUN_DIAGNOSTICS = "run_diagnostics"  # Need more info before deciding


@dataclass
class ReflexionDecision:
    """Decision returned by the Reflexion Engine."""
    action: ReflexionAction
    reason: str
    failure_type: Optional[FailureType] = None
    
    # For REJECT_DUPLICATE
    similar_approach: Optional[str] = None
    similarity_score: float = 0.0
    
    # For APPLY_SOLUTION
    known_solution: Optional[str] = None
    solution_confidence: float = 0.0
    
    # For ESCALATE
    attempt_count: int = 0
    escalation_target: Optional[str] = None  # "human", "architect", etc.
    
    # For RUN_DIAGNOSTICS
    recommended_diagnostics: List[str] = field(default_factory=list)
    
    # Context
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# Error Classification
# =============================================================================

# Patterns for classifying errors
ERROR_PATTERNS = {
    FailureType.MISSING_PACKAGE: [
        r"ModuleNotFoundError: No module named '(\w+)'",
        r"ImportError: cannot import name",
        r"ImportError: No module named",
        r"ModuleNotFoundError:",
    ],
    FailureType.MISSING_FILE: [
        r"FileNotFoundError:",
        r"No such file or directory",
        r"IOError:.*not found",
        r"cannot find.*file",
    ],
    FailureType.SYNTAX_ERROR: [
        r"SyntaxError:",
        r"IndentationError:",
        r"TabError:",
    ],
    FailureType.OUT_OF_MEMORY: [
        r"MemoryError:",
        r"CUDA out of memory",
        r"OOM",
        r"Cannot allocate memory",
        r"std::bad_alloc",
    ],
    FailureType.GPU_ERROR: [
        r"CUDA error:",
        r"cudaError",
        r"RuntimeError:.*CUDA",
        r"GPU.*not available",
        r"No GPU",
    ],
    FailureType.SLURM_ERROR: [
        r"slurmstepd:",
        r"CANCELLED",
        r"srun:",
        r"sbatch:",
        r"SLURM",
    ],
    FailureType.TIMEOUT: [
        r"TimeoutError:",
        r"Timeout",
        r"exceeded.*time",
        r"TIMEOUT",
        r"DUE TO TIME LIMIT",
    ],
    FailureType.PERMISSION_ERROR: [
        r"PermissionError:",
        r"Permission denied",
        r"Access denied",
        r"EACCES",
    ],
    FailureType.CONDA_ERROR: [
        r"conda.*error",
        r"CondaError",
        r"PackagesNotFoundError",
        r"ResolvePackageNotFound",
    ],
    FailureType.DATA_ISSUE: [
        r"ValueError:.*shape",
        r"KeyError:",
        r"IndexError:",
        r"pandas.*error",
        r"DataFrame",
    ],
    FailureType.RUNTIME_ERROR: [
        r"RuntimeError:",
        r"Exception:",
        r"Error:",
    ],
}

# Escalation thresholds per failure type
ESCALATION_THRESHOLDS = {
    FailureType.CODE_BUG: 3,
    FailureType.MISSING_PACKAGE: 2,
    FailureType.MISSING_FILE: 2,
    FailureType.OUT_OF_MEMORY: 2,  # Likely needs architectural change
    FailureType.GPU_ERROR: 2,
    FailureType.SLURM_ERROR: 2,
    FailureType.TIMEOUT: 2,
    FailureType.DESIGN_FLAW: 1,  # Immediate escalation
    FailureType.DEPENDENCY_ISSUE: 2,
    FailureType.DATA_ISSUE: 3,
    FailureType.PERMISSION_ERROR: 2,
    FailureType.SYNTAX_ERROR: 3,
    FailureType.RUNTIME_ERROR: 3,
    FailureType.CONDA_ERROR: 2,
    FailureType.UNKNOWN: 3,
}

# Which agent type should handle each failure type
AGENT_ROUTING = {
    FailureType.CODE_BUG: "developer",
    FailureType.MISSING_PACKAGE: "deployer",
    FailureType.MISSING_FILE: "developer",
    FailureType.OUT_OF_MEMORY: "architect",
    FailureType.GPU_ERROR: "deployer",
    FailureType.SLURM_ERROR: "deployer",
    FailureType.TIMEOUT: "architect",
    FailureType.DESIGN_FLAW: "architect",
    FailureType.DEPENDENCY_ISSUE: "pm",
    FailureType.DATA_ISSUE: "analyst",
    FailureType.PERMISSION_ERROR: "deployer",
    FailureType.SYNTAX_ERROR: "developer",
    FailureType.RUNTIME_ERROR: "developer",
    FailureType.CONDA_ERROR: "deployer",
    FailureType.UNKNOWN: "developer",
}


# =============================================================================
# Reflexion Engine
# =============================================================================

class ReflexionEngine:
    """
    Core engine for preventing infinite retry loops.
    
    The engine intercepts failure handling and makes decisions about:
    - Whether a proposed fix is too similar to something already tried
    - When to escalate vs. retry
    - Whether we have a known solution to apply
    
    Configuration:
        similarity_threshold: How similar approaches must be to be "duplicates" (0.85)
        max_total_attempts: Absolute cap on attempts per task (10)
        enable_diagnostics: Whether to recommend diagnostic scripts (True)
    """
    
    def __init__(
        self,
        memory: Optional[ReflexionMemory] = None,
        similarity_threshold: float = 0.85,
        max_total_attempts: int = 10,
        enable_diagnostics: bool = True,
    ):
        """
        Initialize the Reflexion Engine.
        
        Args:
            memory: ReflexionMemory instance (creates new one if None)
            similarity_threshold: Threshold for duplicate detection
            max_total_attempts: Maximum attempts before forced escalation
            enable_diagnostics: Enable diagnostic script recommendations
        """
        self.memory = memory or ReflexionMemory()
        self.similarity_threshold = similarity_threshold
        self.max_total_attempts = max_total_attempts
        self.enable_diagnostics = enable_diagnostics
        
        logger.info(
            f"ReflexionEngine initialized: "
            f"threshold={similarity_threshold}, max_attempts={max_total_attempts}"
        )
    
    # =========================================================================
    # Main Decision Method
    # =========================================================================
    
    def reflect_on_failure(
        self,
        task_id: str,
        error_message: str,
        proposed_approach: Optional[str] = None,
        context: Optional[Dict] = None,
    ) -> ReflexionDecision:
        """
        Analyze a failure and decide how to proceed.
        
        This is the main entry point for the Reflexion Engine.
        
        Args:
            task_id: Unique task identifier
            error_message: The error that occurred
            proposed_approach: What the agent wants to try next (optional)
            context: Additional context (script_path, etc.)
            
        Returns:
            ReflexionDecision with recommended action
        """
        context = context or {}
        
        # Step 1: Classify the error
        failure_type = self.classify_error(error_message)
        logger.info(f"Classified error as: {failure_type.value}")
        
        # Step 2: Get attempt history
        attempt_counts = self.memory.count_attempts_by_type(task_id)
        total_attempts = sum(attempt_counts.values())
        type_attempts = attempt_counts.get(failure_type.value, 0)
        
        logger.info(
            f"Task {task_id}: {total_attempts} total attempts, "
            f"{type_attempts} for {failure_type.value}"
        )
        
        # Step 3: Check for absolute limit
        if total_attempts >= self.max_total_attempts:
            return ReflexionDecision(
                action=ReflexionAction.ESCALATE,
                reason=f"Reached maximum attempts ({self.max_total_attempts})",
                failure_type=failure_type,
                attempt_count=total_attempts,
                escalation_target="human",
            )
        
        # Step 4: Check for type-specific escalation threshold
        threshold = ESCALATION_THRESHOLDS.get(failure_type, 3)
        if type_attempts >= threshold:
            target = AGENT_ROUTING.get(failure_type, "human")
            return ReflexionDecision(
                action=ReflexionAction.ESCALATE,
                reason=f"Exhausted {threshold} attempts for {failure_type.value}",
                failure_type=failure_type,
                attempt_count=type_attempts,
                escalation_target=target if target != AGENT_ROUTING.get(failure_type) else "human",
            )
        
        # Step 5: Look for known solutions
        solutions = self.memory.get_working_solutions(
            error_message,
            failure_type,
            limit=3,
        )
        
        if solutions and solutions[0].get("score", 0) > 0.8:
            return ReflexionDecision(
                action=ReflexionAction.APPLY_SOLUTION,
                reason="Found high-confidence solution from past successes",
                failure_type=failure_type,
                known_solution=solutions[0].get("solution", ""),
                solution_confidence=solutions[0].get("score", 0),
            )
        
        # Step 6: If no proposed approach, recommend diagnostics or generic retry
        if not proposed_approach:
            if self.enable_diagnostics and failure_type == FailureType.UNKNOWN:
                return ReflexionDecision(
                    action=ReflexionAction.RUN_DIAGNOSTICS,
                    reason="Unknown error type, need diagnostics",
                    failure_type=failure_type,
                    recommended_diagnostics=self._get_recommended_diagnostics(error_message),
                )
            
            return ReflexionDecision(
                action=ReflexionAction.RETRY,
                reason="No proposed approach provided, allowing retry",
                failure_type=failure_type,
                attempt_count=total_attempts,
            )
        
        # Step 7: Check if proposed approach is a duplicate
        check_result = self.memory.check_if_tried(
            task_id,
            proposed_approach,
            threshold=self.similarity_threshold,
        )
        
        if check_result["tried"]:
            return ReflexionDecision(
                action=ReflexionAction.REJECT_DUPLICATE,
                reason="Proposed approach too similar to previous attempt",
                failure_type=failure_type,
                similar_approach=check_result["similar_approach"],
                similarity_score=check_result["similarity"],
                attempt_count=check_result["attempt_count"],
            )
        
        # Step 8: Approach is novel enough, allow retry
        return ReflexionDecision(
            action=ReflexionAction.RETRY,
            reason="Approach is sufficiently different from previous attempts",
            failure_type=failure_type,
            attempt_count=total_attempts,
            similarity_score=check_result["similarity"],
        )
    
    # =========================================================================
    # Error Classification
    # =========================================================================
    
    def classify_error(self, error_message: str) -> FailureType:
        """
        Classify an error message into a FailureType.
        
        Uses regex patterns to identify error types.
        
        Args:
            error_message: The error text to classify
            
        Returns:
            FailureType enum value
        """
        error_lower = error_message.lower()
        
        for failure_type, patterns in ERROR_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, error_message, re.IGNORECASE):
                    return failure_type
        
        return FailureType.UNKNOWN
    
    # =========================================================================
    # Recording Methods
    # =========================================================================
    
    def record_attempt(
        self,
        task_id: str,
        error_message: str,
        approach_tried: str,
        script_path: Optional[str] = None,
        slurm_job_id: Optional[str] = None,
        diagnostic_results: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Record an attempt in memory after it's been executed.
        
        Call this AFTER attempting a fix, regardless of outcome.
        
        Args:
            task_id: Task identifier
            error_message: The error that was being fixed
            approach_tried: What was attempted
            script_path: Path to the script
            slurm_job_id: SLURM job ID if applicable
            diagnostic_results: Any diagnostic output
            
        Returns:
            Memory storage result
        """
        failure_type = self.classify_error(error_message)
        
        # Get current attempt count for this type
        counts = self.memory.count_attempts_by_type(task_id)
        attempt_number = counts.get(failure_type.value, 0) + 1
        
        return self.memory.store_failure(
            task_id=task_id,
            error_type=failure_type,
            error_message=error_message,
            approach_tried=approach_tried,
            script_path=script_path,
            slurm_job_id=slurm_job_id,
            diagnostic_results=diagnostic_results,
            attempt_number=attempt_number,
        )
    
    def record_success(
        self,
        task_id: str,
        problem_description: str,
        solution: str,
        context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Record a successful solution for future reuse.
        
        Call this when a task succeeds after failures.
        
        Args:
            task_id: Task identifier
            problem_description: What was wrong
            solution: What fixed it
            context: Additional context
            
        Returns:
            Memory storage result
        """
        return self.memory.mark_resolved(
            task_id=task_id,
            resolution_description=problem_description,
            successful_approach=solution,
        )
    
    # =========================================================================
    # Diagnostic Recommendations
    # =========================================================================
    
    def _get_recommended_diagnostics(self, error_message: str) -> List[str]:
        """Get recommended diagnostic scripts based on error patterns."""
        diagnostics = []
        
        error_lower = error_message.lower()
        
        # File-related
        if any(x in error_lower for x in ["file", "path", "directory", "not found"]):
            diagnostics.append("check_file_exists")
        
        # Module/package related
        if any(x in error_lower for x in ["module", "import", "package"]):
            diagnostics.append("check_module_installed")
            diagnostics.append("check_conda_env")
        
        # Memory related
        if any(x in error_lower for x in ["memory", "oom", "alloc"]):
            diagnostics.append("check_memory")
        
        # GPU related
        if any(x in error_lower for x in ["cuda", "gpu", "device"]):
            diagnostics.append("check_gpu")
        
        # Disk related
        if any(x in error_lower for x in ["disk", "space", "quota"]):
            diagnostics.append("check_disk_space")
        
        # Default: run basic checks
        if not diagnostics:
            diagnostics = ["check_file_exists", "check_memory", "check_disk_space"]
        
        return diagnostics
    
    # =========================================================================
    # Query Methods
    # =========================================================================
    
    def get_task_summary(self, task_id: str) -> Dict[str, Any]:
        """
        Get a summary of attempts for a task.
        
        Args:
            task_id: Task to summarize
            
        Returns:
            Summary dict with counts, approaches, etc.
        """
        approaches = self.memory.get_tried_approaches(task_id)
        counts = self.memory.count_attempts_by_type(task_id)
        
        return {
            "task_id": task_id,
            "total_attempts": len(approaches),
            "attempts_by_type": counts,
            "approaches": [
                {
                    "attempt": a.get("attempt_number"),
                    "type": a.get("error_type"),
                    "summary": a.get("approach", "")[:100] + "...",
                }
                for a in approaches
            ],
            "at_escalation_threshold": any(
                counts.get(ft.value, 0) >= ESCALATION_THRESHOLDS.get(ft, 3)
                for ft in FailureType
            ),
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall engine statistics."""
        memory_stats = self.memory.get_stats()
        return {
            "memory": memory_stats,
            "config": {
                "similarity_threshold": self.similarity_threshold,
                "max_total_attempts": self.max_total_attempts,
                "enable_diagnostics": self.enable_diagnostics,
            },
            "escalation_thresholds": {
                ft.value: thresh 
                for ft, thresh in ESCALATION_THRESHOLDS.items()
            },
        }

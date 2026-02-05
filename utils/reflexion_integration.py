"""
LangGraph Integration for Reflexion Memory.

This module provides helpers for integrating the Reflexion Memory system
into LangGraph workflows. It wraps the memory client with workflow-friendly
interfaces and provides decorators for automatic failure tracking.

Usage in LangGraph workflow:

    from utils.reflexion_integration import (
        ReflexionState,
        with_reflexion,
        check_before_retry,
        handle_failure,
    )
    
    # Add to your state
    class AgentState(TypedDict):
        task_id: str
        messages: List[BaseMessage]
        reflexion: ReflexionState  # Add this
    
    # Wrap your agent node
    @with_reflexion
    def sub_agent_node(state: AgentState) -> AgentState:
        # Your agent logic
        ...

Integration Points:
    1. Before retry: check_before_retry() to see if approach was tried
    2. After failure: handle_failure() to record and get decision
    3. After success: record_solution() to save for future
"""

import logging
from typing import Dict, List, Optional, Any, TypedDict, Callable
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps

# Import memory client
from mcp_server.client import MemoryClient

logger = logging.getLogger(__name__)


# =============================================================================
# State Types for LangGraph
# =============================================================================

class ReflexionState(TypedDict, total=False):
    """
    Reflexion state to add to your LangGraph AgentState.
    
    Add this to your state:
        class AgentState(TypedDict):
            task_id: str
            reflexion: ReflexionState
    """
    # Current failure info
    last_error: Optional[str]
    last_error_type: Optional[str]
    last_approach: Optional[str]
    
    # Decision from engine
    action: Optional[str]  # retry, reject_duplicate, escalate, apply_solution
    reason: Optional[str]
    
    # Tracking
    attempt_count: int
    tried_approaches: List[str]
    
    # Solution if found
    known_solution: Optional[str]
    solution_confidence: float
    
    # Escalation target if needed
    escalation_target: Optional[str]
    
    # Flags
    should_escalate: bool
    should_apply_solution: bool
    is_duplicate: bool


def create_initial_reflexion_state() -> ReflexionState:
    """Create initial reflexion state for a new task."""
    return {
        "last_error": None,
        "last_error_type": None,
        "last_approach": None,
        "action": None,
        "reason": None,
        "attempt_count": 0,
        "tried_approaches": [],
        "known_solution": None,
        "solution_confidence": 0.0,
        "escalation_target": None,
        "should_escalate": False,
        "should_apply_solution": False,
        "is_duplicate": False,
    }


# =============================================================================
# Integration Functions
# =============================================================================

# Global client instance (lazy initialized)
_client: Optional[MemoryClient] = None


def get_memory_client() -> MemoryClient:
    """Get or create the memory client singleton."""
    global _client
    if _client is None:
        _client = MemoryClient(use_direct=True)
    return _client


def check_before_retry(
    task_id: str,
    proposed_approach: str,
    threshold: float = 0.85,
) -> Dict[str, Any]:
    """
    Check if a proposed approach was already tried.
    
    Call this BEFORE attempting a retry to avoid infinite loops.
    
    Args:
        task_id: Unique task identifier
        proposed_approach: What the agent wants to try
        threshold: Similarity threshold (0.0 - 1.0)
        
    Returns:
        {
            "allowed": bool,        # True if OK to proceed
            "tried": bool,          # True if similar approach was tried
            "similarity": float,    # How similar to previous attempt
            "similar_approach": str # What was tried before
        }
    
    Example:
        check = check_before_retry("task_001", "Install pandas via pip")
        if not check["allowed"]:
            # Need a different approach
            pass
    """
    client = get_memory_client()
    result = client.check_if_tried(task_id, proposed_approach, threshold)
    
    return {
        "allowed": not result["tried"],
        "tried": result["tried"],
        "similarity": result["similarity"],
        "similar_approach": result.get("similar_approach"),
        "attempt_count": result.get("attempt_count", 0),
    }


def handle_failure(
    task_id: str,
    error_message: str,
    approach_tried: str,
    proposed_next_approach: Optional[str] = None,
    script_path: Optional[str] = None,
    slurm_job_id: Optional[str] = None,
) -> ReflexionState:
    """
    Handle a task failure: record it and get a decision.
    
    This is the main integration point. Call this when a task fails.
    
    Args:
        task_id: Unique task identifier
        error_message: The error that occurred
        approach_tried: What was just attempted
        proposed_next_approach: What to try next (optional)
        script_path: Path to the script
        slurm_job_id: SLURM job ID
        
    Returns:
        ReflexionState with decision and next steps
    
    Example:
        state = handle_failure(
            task_id="task_001",
            error_message="ModuleNotFoundError: No module named 'pandas'",
            approach_tried="Added import statement",
            proposed_next_approach="pip install pandas"
        )
        
        if state["should_escalate"]:
            # Route to human or different agent
            pass
        elif state["should_apply_solution"]:
            # Use the known solution
            solution = state["known_solution"]
        elif state["is_duplicate"]:
            # Need a genuinely different approach
            pass
        else:
            # OK to proceed with proposed approach
            pass
    """
    client = get_memory_client()
    
    # Classify the error
    classification = client.classify_error(error_message)
    error_type = classification["error_type"]
    
    # Store the failure
    client.store_failure(
        task_id=task_id,
        error_type=error_type,
        error_message=error_message,
        approach_tried=approach_tried,
        script_path=script_path,
        slurm_job_id=slurm_job_id,
    )
    
    # Get decision from reflexion engine
    decision = client.reflect_on_failure(
        task_id=task_id,
        error_message=error_message,
        proposed_approach=proposed_next_approach,
    )
    
    # Build state
    state: ReflexionState = {
        "last_error": error_message,
        "last_error_type": error_type,
        "last_approach": approach_tried,
        "action": decision["action"],
        "reason": decision["reason"],
        "attempt_count": decision.get("attempt_count", 0),
        "tried_approaches": [],  # Could populate from get_tried_approaches
        "known_solution": decision.get("known_solution"),
        "solution_confidence": decision.get("solution_confidence", 0.0),
        "escalation_target": decision.get("escalation_target"),
        "should_escalate": decision["action"] == "escalate",
        "should_apply_solution": decision["action"] == "apply_solution",
        "is_duplicate": decision["action"] == "reject_duplicate",
    }
    
    logger.info(
        f"Reflexion decision for {task_id}: {decision['action']} - {decision['reason']}"
    )
    
    return state


def record_solution(
    task_id: str,
    problem_pattern: str,
    error_type: str,
    solution: str,
) -> None:
    """
    Record a successful solution for future reuse.
    
    Call this when a task succeeds after failures.
    
    Args:
        task_id: Task that was solved
        problem_pattern: Description of the problem
        error_type: Type of error that was fixed
        solution: What fixed it
    
    Example:
        record_solution(
            task_id="task_001",
            problem_pattern="Missing pandas package in conda environment",
            error_type="missing_package",
            solution="pip install pandas --break-system-packages"
        )
    """
    client = get_memory_client()
    client.store_solution(
        task_id=task_id,
        problem_pattern=problem_pattern,
        error_type=error_type,
        solution=solution,
    )
    logger.info(f"Recorded solution for {task_id}: {solution[:50]}...")


def get_task_history(task_id: str) -> Dict[str, Any]:
    """
    Get the full history of attempts for a task.
    
    Useful for debugging or displaying progress.
    
    Args:
        task_id: Task to query
        
    Returns:
        Summary with attempts, counts, and approaches
    """
    client = get_memory_client()
    return client.get_task_summary(task_id)


def find_similar_solutions(
    error_message: str,
    error_type: Optional[str] = None,
    limit: int = 3,
) -> List[Dict]:
    """
    Search for solutions to similar problems.
    
    Call this before attempting a fix to see if we already know the answer.
    
    Args:
        error_message: The error to find solutions for
        error_type: Type of error (optional filter)
        limit: Maximum results
        
    Returns:
        List of solutions with scores
    """
    client = get_memory_client()
    return client.get_working_solutions(error_message, error_type, limit)


# =============================================================================
# LangGraph Conditional Edges
# =============================================================================

def should_retry_or_escalate(state: Dict) -> str:
    """
    Conditional edge function for LangGraph.
    
    Use this as a conditional edge to route based on reflexion decision.
    
    Returns:
        "retry" - Proceed with retry
        "escalate" - Route to escalation handler
        "apply_solution" - Apply known solution
        "reject" - Need different approach
    
    Example in LangGraph:
        workflow.add_conditional_edges(
            "handle_failure",
            should_retry_or_escalate,
            {
                "retry": "sub_agent",
                "escalate": "escalation_handler",
                "apply_solution": "apply_solution_node",
                "reject": "generate_new_approach",
            }
        )
    """
    reflexion = state.get("reflexion", {})
    action = reflexion.get("action", "retry")
    
    if action == "escalate":
        return "escalate"
    elif action == "apply_solution":
        return "apply_solution"
    elif action == "reject_duplicate":
        return "reject"
    else:
        return "retry"


def is_at_escalation_threshold(state: Dict) -> bool:
    """
    Check if we've hit the escalation threshold.
    
    Use in conditional edges or guards.
    """
    reflexion = state.get("reflexion", {})
    return reflexion.get("should_escalate", False)


# =============================================================================
# Decorator for Automatic Tracking
# =============================================================================

def with_reflexion(func: Callable) -> Callable:
    """
    Decorator to add automatic reflexion tracking to a node.
    
    Wraps exceptions and records failures automatically.
    
    Example:
        @with_reflexion
        def sub_agent_node(state: AgentState) -> AgentState:
            # Your logic here
            result = run_task(state)
            return state
    """
    @wraps(func)
    def wrapper(state: Dict) -> Dict:
        task_id = state.get("task_id", "unknown")
        
        try:
            # Initialize reflexion state if not present
            if "reflexion" not in state:
                state["reflexion"] = create_initial_reflexion_state()
            
            # Run the node
            result = func(state)
            
            # On success, clear failure state
            if result.get("success", True):
                result["reflexion"]["last_error"] = None
                result["reflexion"]["is_duplicate"] = False
            
            return result
            
        except Exception as e:
            # Record the failure
            error_msg = str(e)
            approach = state.get("current_approach", "Unknown approach")
            
            reflexion_state = handle_failure(
                task_id=task_id,
                error_message=error_msg,
                approach_tried=approach,
            )
            
            state["reflexion"] = reflexion_state
            state["error"] = error_msg
            
            # Re-raise or return based on your error handling
            return state
    
    return wrapper


# =============================================================================
# Example Node Implementations
# =============================================================================

def example_failure_handler_node(state: Dict) -> Dict:
    """
    Example node for handling failures in LangGraph.
    
    Add this after your execution node to process failures.
    """
    # Get error info from state
    error = state.get("error")
    if not error:
        return state
    
    task_id = state.get("task_id", "unknown")
    approach = state.get("current_approach", "Unknown")
    next_approach = state.get("proposed_approach")
    script_path = state.get("script_path")
    slurm_job_id = state.get("slurm_job_id")
    
    # Process failure through reflexion engine
    reflexion_state = handle_failure(
        task_id=task_id,
        error_message=error,
        approach_tried=approach,
        proposed_next_approach=next_approach,
        script_path=script_path,
        slurm_job_id=slurm_job_id,
    )
    
    state["reflexion"] = reflexion_state
    
    return state


def example_pre_retry_check_node(state: Dict) -> Dict:
    """
    Example node to check before retrying.
    
    Add this before your retry logic to prevent duplicates.
    """
    task_id = state.get("task_id", "unknown")
    proposed = state.get("proposed_approach")
    
    if not proposed:
        return state
    
    check = check_before_retry(task_id, proposed)
    
    if not check["allowed"]:
        state["reflexion"] = state.get("reflexion", {})
        state["reflexion"]["is_duplicate"] = True
        state["reflexion"]["similar_approach"] = check["similar_approach"]
        state["reflexion"]["action"] = "reject_duplicate"
        logger.warning(
            f"Approach rejected as duplicate (similarity: {check['similarity']:.2f})"
        )
    
    return state

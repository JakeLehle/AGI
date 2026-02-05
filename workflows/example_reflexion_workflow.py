"""
Example LangGraph Workflow with Reflexion Memory Integration.

This shows how to integrate the reflexion memory system into a LangGraph
workflow for automatic loop prevention and failure tracking.

Key Integration Points:
1. State includes ReflexionState
2. Failure handler node processes errors through reflexion engine
3. Conditional edges route based on reflexion decisions
4. Pre-retry check prevents duplicate approaches

Copy and adapt the relevant parts to your existing workflow.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, TypedDict, Annotated
from datetime import datetime
import operator
import logging

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import reflexion integration
from utils.reflexion_integration import (
    ReflexionState,
    create_initial_reflexion_state,
    handle_failure,
    check_before_retry,
    record_solution,
    should_retry_or_escalate,
    find_similar_solutions,
)

logger = logging.getLogger(__name__)


# =============================================================================
# State Definition
# =============================================================================

class AgentState(TypedDict):
    """
    LangGraph state with reflexion integration.
    """
    # Task identification
    task_id: str
    task_description: str
    
    # Execution state
    current_approach: Optional[str]
    proposed_approach: Optional[str]
    script_path: Optional[str]
    slurm_job_id: Optional[str]
    
    # Results
    output: Optional[str]
    error: Optional[str]
    success: bool
    
    # Reflexion state - ADD THIS TO YOUR STATE
    reflexion: ReflexionState
    
    # Iteration tracking
    iteration: int
    max_iterations: int


def create_initial_state(
    task_id: str,
    task_description: str,
    max_iterations: int = 10,
) -> AgentState:
    """Create initial state for a new task."""
    return {
        "task_id": task_id,
        "task_description": task_description,
        "current_approach": None,
        "proposed_approach": None,
        "script_path": None,
        "slurm_job_id": None,
        "output": None,
        "error": None,
        "success": False,
        "reflexion": create_initial_reflexion_state(),
        "iteration": 0,
        "max_iterations": max_iterations,
    }


# =============================================================================
# Node Implementations
# =============================================================================

def plan_approach_node(state: AgentState) -> AgentState:
    """
    Plan the next approach to try.
    
    This node should:
    1. Check for known solutions first
    2. Generate a new approach if none found
    3. Verify the approach wasn't tried before
    """
    task_id = state["task_id"]
    error = state.get("error")
    
    # If there was an error, check for known solutions first
    if error:
        solutions = find_similar_solutions(error)
        if solutions and solutions[0].get("score", 0) > 0.8:
            logger.info(f"Found known solution with score {solutions[0]['score']:.2f}")
            state["proposed_approach"] = solutions[0].get("solution", "")
            state["reflexion"]["known_solution"] = solutions[0].get("solution")
            state["reflexion"]["should_apply_solution"] = True
            return state
    
    # Generate a new approach (your LLM logic here)
    # This is a placeholder - replace with your actual approach generation
    if state["iteration"] == 0:
        approach = f"Initial approach for: {state['task_description']}"
    else:
        approach = f"Alternative approach #{state['iteration']} for: {state['task_description']}"
    
    # Check if this approach was tried before
    check = check_before_retry(task_id, approach)
    if not check["allowed"]:
        logger.warning(f"Approach rejected as duplicate (similarity: {check['similarity']:.2f})")
        # Modify the approach or try something different
        approach = f"Modified approach (avoiding: {check['similar_approach'][:50]}...)"
    
    state["proposed_approach"] = approach
    state["current_approach"] = approach
    
    return state


def execute_task_node(state: AgentState) -> AgentState:
    """
    Execute the planned approach.
    
    This node should:
    1. Generate/modify the script
    2. Submit to SLURM
    3. Wait for completion
    4. Capture output/errors
    """
    # Placeholder - replace with your actual execution logic
    approach = state["current_approach"]
    
    logger.info(f"Executing approach: {approach}")
    
    # Simulate execution
    # In reality, this would:
    # - Generate a script
    # - Submit to SLURM
    # - Wait for completion
    # - Parse results
    
    # For demo, simulate success on 3rd iteration
    if state["iteration"] >= 2:
        state["success"] = True
        state["output"] = "Task completed successfully!"
        state["error"] = None
    else:
        state["success"] = False
        state["error"] = "ModuleNotFoundError: No module named 'pandas'"
        state["output"] = None
    
    state["iteration"] += 1
    
    return state


def handle_failure_node(state: AgentState) -> AgentState:
    """
    Process a failure through the reflexion engine.
    
    THIS IS THE KEY INTEGRATION POINT.
    
    This node:
    1. Records the failure in memory
    2. Gets a decision from the reflexion engine
    3. Updates state with next steps
    """
    error = state.get("error")
    if not error:
        return state
    
    task_id = state["task_id"]
    approach = state.get("current_approach", "Unknown approach")
    script_path = state.get("script_path")
    slurm_job_id = state.get("slurm_job_id")
    
    # Process through reflexion engine
    reflexion_state = handle_failure(
        task_id=task_id,
        error_message=error,
        approach_tried=approach,
        proposed_next_approach=state.get("proposed_approach"),
        script_path=script_path,
        slurm_job_id=slurm_job_id,
    )
    
    state["reflexion"] = reflexion_state
    
    logger.info(
        f"Reflexion decision: {reflexion_state['action']} - {reflexion_state['reason']}"
    )
    
    return state


def handle_success_node(state: AgentState) -> AgentState:
    """
    Process a successful task completion.
    
    Records the solution for future reuse.
    """
    if not state.get("success"):
        return state
    
    # If we succeeded after failures, record the solution
    if state["reflexion"].get("last_error"):
        record_solution(
            task_id=state["task_id"],
            problem_pattern=state["reflexion"]["last_error"],
            error_type=state["reflexion"].get("last_error_type", "unknown"),
            solution=state["current_approach"],
        )
        logger.info("Recorded successful solution for future reuse")
    
    return state


def escalation_node(state: AgentState) -> AgentState:
    """
    Handle escalation when we've exhausted retries.
    
    This could:
    - Notify a human
    - Route to a different agent type
    - Log for manual review
    """
    target = state["reflexion"].get("escalation_target", "human")
    
    logger.warning(
        f"Escalating task {state['task_id']} to {target}: "
        f"{state['reflexion'].get('reason')}"
    )
    
    # Add escalation logic here
    # e.g., send notification, create ticket, etc.
    
    return state


def apply_solution_node(state: AgentState) -> AgentState:
    """
    Apply a known solution from memory.
    """
    solution = state["reflexion"].get("known_solution")
    if solution:
        logger.info(f"Applying known solution: {solution[:100]}...")
        state["current_approach"] = solution
        state["proposed_approach"] = solution
    
    return state


# =============================================================================
# Conditional Edge Functions
# =============================================================================

def check_execution_result(state: AgentState) -> str:
    """Route based on execution result."""
    if state.get("success"):
        return "success"
    else:
        return "failure"


def check_reflexion_decision(state: AgentState) -> str:
    """
    Route based on reflexion engine decision.
    
    This implements the core loop-breaking logic.
    """
    reflexion = state.get("reflexion", {})
    action = reflexion.get("action", "retry")
    
    # Check iteration limit as fallback
    if state["iteration"] >= state["max_iterations"]:
        return "escalate"
    
    if action == "escalate":
        return "escalate"
    elif action == "apply_solution":
        return "apply_solution"
    elif action == "reject_duplicate":
        # Need a different approach - go back to planning
        return "plan_new"
    else:
        return "retry"


def should_continue(state: AgentState) -> str:
    """Check if we should continue iterating."""
    if state.get("success"):
        return "done"
    if state["iteration"] >= state["max_iterations"]:
        return "escalate"
    return "continue"


# =============================================================================
# Build the Workflow
# =============================================================================

def create_workflow() -> StateGraph:
    """
    Create the LangGraph workflow with reflexion integration.
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("plan", plan_approach_node)
    workflow.add_node("execute", execute_task_node)
    workflow.add_node("handle_failure", handle_failure_node)
    workflow.add_node("handle_success", handle_success_node)
    workflow.add_node("escalate", escalation_node)
    workflow.add_node("apply_solution", apply_solution_node)
    
    # Set entry point
    workflow.set_entry_point("plan")
    
    # Add edges
    workflow.add_edge("plan", "execute")
    
    # After execution, check result
    workflow.add_conditional_edges(
        "execute",
        check_execution_result,
        {
            "success": "handle_success",
            "failure": "handle_failure",
        }
    )
    
    # Success leads to end
    workflow.add_edge("handle_success", END)
    
    # After failure handling, check reflexion decision
    workflow.add_conditional_edges(
        "handle_failure",
        check_reflexion_decision,
        {
            "retry": "plan",           # Try again with new approach
            "escalate": "escalate",    # Give up and escalate
            "apply_solution": "apply_solution",  # Use known fix
            "plan_new": "plan",        # Duplicate detected, plan differently
        }
    )
    
    # After applying solution, execute it
    workflow.add_edge("apply_solution", "execute")
    
    # Escalation leads to end
    workflow.add_edge("escalate", END)
    
    return workflow


# =============================================================================
# Main
# =============================================================================

def run_task(task_id: str, task_description: str) -> Dict[str, Any]:
    """
    Run a task through the workflow.
    
    Args:
        task_id: Unique task identifier
        task_description: What to accomplish
        
    Returns:
        Final state after workflow completion
    """
    # Create workflow
    workflow = create_workflow()
    
    # Compile with checkpointer
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    
    # Create initial state
    initial_state = create_initial_state(
        task_id=task_id,
        task_description=task_description,
    )
    
    # Run workflow
    config = {"configurable": {"thread_id": task_id}}
    
    logger.info(f"Starting task: {task_id}")
    
    final_state = None
    for state in app.stream(initial_state, config):
        # Log progress
        node_name = list(state.keys())[0]
        node_state = state[node_name]
        logger.debug(f"Node {node_name}: iteration={node_state.get('iteration')}")
        final_state = node_state
    
    logger.info(f"Task {task_id} completed: success={final_state.get('success')}")
    
    return final_state


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example: Run a task
    result = run_task(
        task_id="example_task_001",
        task_description="Analyze gene expression data using pandas and create a visualization"
    )
    
    print("\n" + "=" * 50)
    print("Final Result:")
    print(f"  Success: {result.get('success')}")
    print(f"  Iterations: {result.get('iteration')}")
    print(f"  Reflexion Action: {result.get('reflexion', {}).get('action')}")
    if result.get('output'):
        print(f"  Output: {result.get('output')}")
    if result.get('error'):
        print(f"  Error: {result.get('error')}")

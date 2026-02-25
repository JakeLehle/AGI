"""
LangGraph Workflow v1.2.2 - Script-First Architecture with Diagnostic Agent

Orchestrates multi-agent system with:
- Token-based context limits (configurable, default 25K) instead of iteration counts
- Each subtask gets its own persistent context window
- Parallel SLURM job submission for independent tasks
- Script generation and execution paradigm
- Living document master prompt management
- **Reflexion Memory for loop prevention**
- **Diagnostic Agent with cross-task solution memory** (v1.2.0)

v1.2.2 Updates:
- FIX F: submit_parallel_jobs now uses ThreadPoolExecutor for true concurrent
  task execution. Each task runs its full 4-phase lifecycle in its own thread.
  max_parallel_agents (default 4) controls thread pool size. Individual thread
  crashes are isolated — one task failing cannot affect sibling tasks.
- FIX G: Progress-first routing in route_after_execution. Priority order:
  (1) dispatch ready tasks, (2) review failures, (3) exit. ESCALATE from
  master_review and reflexion_check no longer terminates the pipeline — it
  marks the task as permanently failed and returns to the main loop. The
  pipeline only exits when all tasks are completed, permanently failed, or
  blocked by failed dependencies.
- FIX H: Tasks with failed dependencies get status 'blocked' instead of
  having their deps stripped. _break_deadlock Strategy 1 (strip failed deps)
  removed. New 'blocked' status provides clear traceability in the final
  report showing exactly which failed dependency prevents each blocked task.

v1.2.0 Updates:
- DiagnosticMemory initialized in workflow __init__() and passed through
  to every ScriptFirstSubAgent instance (in both submit_parallel_jobs and
  execute_sequential). The diagnostic agent is invoked from within the
  sub-agent's Phase 4, NOT from the workflow — so the workflow changes are
  purely wiring.
- Version bump in log messages and docstrings.

v3.2.1 Updates:
- Model selection is now fully modular via utils.model_config.resolve_model().
  No hardcoded model names in workflow code. Resolution priority:
    1. Explicit parameter (from main.py CLI --model)
    2. OLLAMA_MODEL environment variable (set in RUN scripts)
    3. config.yaml → ollama.model
    4. Single fallback constant in utils/model_config.py
- All downstream agents (MasterAgent, ScriptFirstSubAgent) also use
  resolve_model() internally, so passing None is safe at every level.

v3.2 Updates:
- Cluster configuration via AGI_CLUSTER environment variable
- Conda cleanup after successful task completion
- State checkpointing for resume capability
- cleanup_env_on_success parameter support
- Token budget: 25K/12K/3K (context/tool output/min continue)
- Transition logging at every routing decision
- GPU-aware parallel batching (separate GPU/CPU tasks)

Key v3 architecture principles:
- SubAgents generate scripts, submit SLURM jobs, monitor completion
- Context window exhaustion (not iteration count) determines retry limits
- Each subtask maintains its own context across ALL retries
- Reflexion Engine prevents repeating semantically similar approaches
- Master document tracks pipeline state across invocations

Environment Variables:
- AGI_MAX_CONTEXT_TOKENS: Max tokens per subtask (default: 25000)
- AGI_MAX_TOOL_OUTPUT_TOKENS: Max tool output before summarization (default: 12000)
- AGI_MIN_TOKENS_TO_CONTINUE: Min tokens to continue (default: 3000)
- AGI_CLUSTER: Cluster profile for subtask SLURM settings
- AGI_CLUSTER_CONFIG: Path to cluster_config.yaml
- OLLAMA_MODEL: Model override (used by resolve_model)

v3.2.2 Updates:
- FIX D: Deadlock detection and recovery in route_execution_mode().
  When identify_parallel_tasks() returns status='blocked' (no tasks ready,
  but pending tasks remain), the router now calls _break_deadlock() which:
    1. Strips deps pointing to failed/non-existent task IDs
    2. Force-completes non-executable tasks (documentation sections that
       slipped past Fix A's filter in master_agent.py)
    3. Re-evaluates which tasks are now ready
    4. If tasks were unblocked, routes to sequential/parallel execution
    5. If still deadlocked after recovery, exits with status='deadlocked'
       (not 'completed') so the report clearly shows the failure mode
  This is a safety net — Fixes A+B in master_agent.py prevent the root
  cause (bad deps / non-executable steps), but Fix D catches edge cases.
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from pathlib import Path
import operator
import os
import re
import json
import uuid
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import logging

from langgraph.graph import StateGraph, END
try:
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver as MemorySaver
    except ImportError:
        MemorySaver = None

from agents.master_agent import MasterAgent
# Import sub-agent - try v1.2.0 class first, fall back to v3.2 then base
try:
    from agents.sub_agent import ScriptFirstSubAgentV3 as ScriptFirstSubAgent
except ImportError:
    from agents.sub_agent import ScriptFirstSubAgent

from tools.sandbox import Sandbox
from tools.conda_tools import CondaTools
from tools.slurm_tools import SlurmTools, SlurmConfig
from utils.logging_config import agent_logger
from utils.git_tracker import git_tracker
from utils.documentation import doc_generator
from utils.context_manager import ContextManager
from utils.model_config import resolve_model, resolve_base_url

# ============================================================================
# REFLEXION MEMORY INTEGRATION
# ============================================================================
try:
    from utils.reflexion_integration import (
        ReflexionState,
        create_initial_reflexion_state,
        handle_failure as reflexion_handle_failure,
        check_before_retry,
        record_solution,
        find_similar_solutions,
        get_memory_client,
    )
    REFLEXION_AVAILABLE = True
except ImportError:
    REFLEXION_AVAILABLE = False
    # Provide fallback types
    ReflexionState = Dict[str, Any]
    def create_initial_reflexion_state():
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
    def get_memory_client():
        return None

# ============================================================================
# DIAGNOSTIC MEMORY INTEGRATION (v1.2.0)
# ============================================================================
try:
    from memory.diagnostic_memory import DiagnosticMemory
    DIAGNOSTIC_MEMORY_AVAILABLE = True
except ImportError:
    DIAGNOSTIC_MEMORY_AVAILABLE = False
    DiagnosticMemory = None


logger = logging.getLogger(__name__)


# ============================================================================
# WORKFLOW STATE
# ============================================================================

class WorkflowState(TypedDict):
    """State that flows through the workflow"""
    # Input
    main_task: str
    context: Dict[str, Any]
    project_dir: str

    # Task management
    subtasks: List[Dict[str, Any]]
    current_subtask_idx: int
    current_subtask: Dict[str, Any]

    # Parallel execution
    parallel_batch: List[Dict[str, Any]]
    parallel_results: List[Dict[str, Any]]
    running_jobs: Dict[str, str]  # task_id -> slurm_job_id

    # Context tracking (replaces iteration counting)
    agent_context_status: Dict[str, Dict]  # agent_id -> context status

    # Environment
    env_name: str

    # Results
    completed_subtasks: Annotated[List[Dict], operator.add]
    failed_subtasks: Annotated[List[Dict], operator.add]

    # Configuration
    use_slurm: bool
    parallel_enabled: bool

    # Output
    final_report: str
    status: str
    master_decision: Dict[str, Any]

    # Reflexion Memory State
    reflexion: ReflexionState
    task_attempt_counts: Dict[str, int]  # task_id -> attempt count

    # v3.2: Additional state
    cleanup_env_on_success: bool
    checkpoint_info: Dict[str, Any]


# ============================================================================
# MAIN WORKFLOW CLASS
# ============================================================================

class MultiAgentWorkflow:
    """
    LangGraph workflow v1.2.2 with script-first architecture, reflexion memory,
    and diagnostic agent integration.

    Key features:
    - Token-based context limits (configurable via environment)
    - Each subtask gets its own persistent context window
    - Parallel SLURM job submission
    - Script generation → SLURM submit → monitor paradigm
    - Living document master prompt
    - Reflexion Memory for semantic duplicate detection
    - v3.2: Cluster configuration, conda cleanup, GPU-aware batching
    - v3.2.1: Modular model config via resolve_model() — no hardcoded names
    - v3.2.2: Deadlock detection/recovery in router (Fix D safety net)
    - v1.2.0: DiagnosticMemory wired through to sub-agents
    - v1.2.2: True parallel execution (Fix F), progress-first routing (Fix G),
              blocked-by-failure status (Fix H)
    """

    # Token limits — read from environment or use defaults matching config.yaml
    MAX_CONTEXT_TOKENS = int(os.environ.get('AGI_MAX_CONTEXT_TOKENS', 25000))
    MAX_TOOL_OUTPUT_TOKENS = int(os.environ.get('AGI_MAX_TOOL_OUTPUT_TOKENS', 12000))
    MIN_TOKENS_TO_CONTINUE = int(os.environ.get('AGI_MIN_TOKENS_TO_CONTINUE', 3000))

    def __init__(
        self,
        ollama_model: str = None,
        ollama_base_url: str = None,
        max_retries: int = 12,  # Kept for backward compat, but token-based now
        project_dir: str = None,
        use_slurm: bool = True,
        parallel_enabled: bool = True,
        slurm_config: Dict[str, Any] = None,
        use_reflexion_memory: bool = True,
        cleanup_env_on_success: bool = True,  # v3.2: Cleanup conda envs after success
        max_parallel_agents: int = 4,          # v1.2.2: Thread pool size
    ):
        # v3.2.1: Resolve model and URL via centralized config (no hardcoded names)
        self.ollama_model = resolve_model(ollama_model)
        self.ollama_base_url = resolve_base_url(ollama_base_url)

        self.project_dir = Path(project_dir) if project_dir else Path.cwd()
        self.use_slurm = use_slurm
        self.parallel_enabled = parallel_enabled
        self.slurm_config = slurm_config or {}
        self.use_reflexion_memory = use_reflexion_memory and REFLEXION_AVAILABLE
        self.cleanup_env_on_success = cleanup_env_on_success  # v3.2
        self.max_parallel_agents = max_parallel_agents  # v1.2.2

        # Initialize sandbox
        self.sandbox = Sandbox(self.project_dir)

        # Initialize conda tools
        self.conda_tools = CondaTools(self.project_dir, self.sandbox.get_envs_dir())

        # Initialize SLURM tools
        self.slurm_tools = None
        if use_slurm:
            cluster_name = self.slurm_config.get("cluster")
            self.slurm_tools = SlurmTools(
                self.sandbox,
                cluster_name=cluster_name
            )
            if not self.slurm_tools.slurm_available:
                print("WARNING: SLURM not available, falling back to local execution")
                self.use_slurm = False

        # Initialize master agent — passes resolved model, but MasterAgent
        # also calls resolve_model() internally so None would be safe too
        self.master = MasterAgent(
            sandbox=self.sandbox,
            ollama_model=self.ollama_model,
            ollama_base_url=self.ollama_base_url,
            project_dir=self.project_dir
        )

        # Context manager for overall workflow
        self.context_mgr = ContextManager(
            max_context_tokens=self.MAX_CONTEXT_TOKENS,
            max_tool_output_tokens=self.MAX_TOOL_OUTPUT_TOKENS
        )

        # Thread lock for parallel execution
        self._lock = threading.Lock()

        # Reflexion Memory Client
        self.memory_client = None
        if self.use_reflexion_memory:
            try:
                self.memory_client = get_memory_client()
                logger.info("Reflexion memory enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize reflexion memory: {e}")
                self.use_reflexion_memory = False

        # ── v1.2.0: Diagnostic Memory ────────────────────────────────
        # Shared across all sub-agents and all tasks. Solutions learned
        # fixing one step's errors are immediately available to every
        # other step in the pipeline (and future pipeline runs).
        self.diagnostic_memory = None
        if DIAGNOSTIC_MEMORY_AVAILABLE:
            try:
                self.diagnostic_memory = DiagnosticMemory()
                logger.info("Diagnostic memory enabled (cross-task solution cache)")
            except Exception as e:
                logger.warning(f"Failed to initialize diagnostic memory: {e}")
                self.diagnostic_memory = None

        # Log configuration
        logger.info(
            f"MultiAgentWorkflow v1.2.2 initialized: "
            f"model={self.ollama_model}, "
            f"tokens={self.MAX_CONTEXT_TOKENS}/{self.MAX_TOOL_OUTPUT_TOKENS}/{self.MIN_TOKENS_TO_CONTINUE}, "
            f"slurm={self.use_slurm}, "
            f"parallel={self.parallel_enabled}, "
            f"max_parallel_agents={self.max_parallel_agents}, "
            f"reflexion={self.use_reflexion_memory}, "
            f"diagnostic_memory={self.diagnostic_memory is not None}, "
            f"cleanup_env={self.cleanup_env_on_success}, "
            f"cluster={os.environ.get('AGI_CLUSTER', 'default')}"
        )

        # Build workflow
        self.workflow = self._build_workflow()

        # Add persistence
        if MemorySaver is not None:
            self.memory = MemorySaver()
            self.app = self.workflow.compile(checkpointer=self.memory)
        else:
            self.memory = None
            self.app = self.workflow.compile()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(WorkflowState)

        # Add nodes
        workflow.add_node("setup", self.setup_environment)
        workflow.add_node("decompose", self.master_decompose)
        workflow.add_node("identify_parallel", self.identify_parallel_tasks)
        workflow.add_node("submit_parallel", self.submit_parallel_jobs)
        workflow.add_node("execute_sequential", self.execute_sequential)
        workflow.add_node("wait_for_jobs", self.wait_for_parallel_jobs)
        workflow.add_node("handle_results", self.handle_results)
        workflow.add_node("reflexion_check", self.reflexion_check)
        workflow.add_node("master_review", self.master_review)
        workflow.add_node("generate_report", self.generate_final_report)
        workflow.add_node("cleanup", self.cleanup)

        # Define flow
        workflow.set_entry_point("setup")
        workflow.add_edge("setup", "decompose")
        workflow.add_edge("decompose", "identify_parallel")

        # Route based on parallel availability
        workflow.add_conditional_edges(
            "identify_parallel",
            self.route_execution_mode,
            {
                "parallel": "submit_parallel",
                "sequential": "execute_sequential",
                "complete": "generate_report"
            }
        )

        # After parallel submission, wait for completion
        workflow.add_edge("submit_parallel", "wait_for_jobs")
        workflow.add_edge("wait_for_jobs", "handle_results")

        # After sequential execution
        workflow.add_edge("execute_sequential", "handle_results")

        # After handling results - go through reflexion check first
        workflow.add_conditional_edges(
            "handle_results",
            self.route_after_execution,
            {
                "next_batch": "identify_parallel",
                "review": "reflexion_check",
                "complete": "generate_report"
            }
        )

        # v1.2.2: escalate no longer exits — marks failed, returns to main loop
        workflow.add_conditional_edges(
            "reflexion_check",
            self.route_after_reflexion,
            {
                "apply_solution": "execute_sequential",
                "escalate": "identify_parallel",       # ← FIX G: was generate_report
                "master_review": "master_review",
            }
        )

        # v1.2.2: all paths return to main loop — no early exit
        workflow.add_conditional_edges(
            "master_review",
            self.route_after_review,
            {
                "retry": "identify_parallel",
                "skip": "identify_parallel",
                # "escalate" removed — route_after_review maps it to "skip"
            }
        )

        workflow.add_edge("generate_report", "cleanup")
        workflow.add_edge("cleanup", END)

        return workflow

    # ==================== Node Implementations ====================

    def setup_environment(self, state: WorkflowState) -> Dict:
        """Setup base environment"""
        env_result = self.conda_tools.create_environment(
            env_name=f"project_{datetime.now().strftime('%Y%m%d')}",
            python_version="3.10",
            packages=["pandas", "numpy"],
            description=f"Base env for: {state['main_task'][:50]}"
        )

        env_name = env_result.get("env_name", "agi_project")

        agent_logger.log_workflow_event("setup_complete", {
            "env_name": env_name,
            "use_slurm": self.use_slurm,
            "parallel_enabled": self.parallel_enabled,
            "max_parallel_agents": self.max_parallel_agents,
            "reflexion_enabled": self.use_reflexion_memory,
            "diagnostic_memory_enabled": self.diagnostic_memory is not None,
            "cleanup_env_on_success": self.cleanup_env_on_success,  # v3.2
            "cluster": os.environ.get('AGI_CLUSTER', 'unknown'),  # v3.2
            "model": self.ollama_model,
            "token_budget": f"{self.MAX_CONTEXT_TOKENS}/{self.MAX_TOOL_OUTPUT_TOKENS}/{self.MIN_TOKENS_TO_CONTINUE}",
        })

        return {
            "env_name": env_name,
            "project_dir": str(self.project_dir),
            "use_slurm": self.use_slurm,
            "parallel_enabled": self.parallel_enabled,
            "agent_context_status": {},
            "running_jobs": {},
            "reflexion": create_initial_reflexion_state(),
            "task_attempt_counts": {},
            "cleanup_env_on_success": self.cleanup_env_on_success,  # v3.2
            "checkpoint_info": {},  # v3.2
        }

    def master_decompose(self, state: WorkflowState) -> Dict:
        """Decompose main task into subtasks"""
        subtasks = self.master.decompose_task(
            main_task=state['main_task'],
            context=state.get('context', {})
        )

        # v3.2: Check for existing checkpoints
        checkpoint_info = self._check_existing_checkpoints(subtasks)
        if checkpoint_info['has_checkpoints']:
            agent_logger.log_workflow_event("checkpoints_found", {
                "count": len(checkpoint_info['checkpoints']),
                "tasks": list(checkpoint_info['checkpoints'].keys())
            })
            print(f"  Found {len(checkpoint_info['checkpoints'])} checkpoint(s) from previous run")

        # v3.2: Log GPU/CPU breakdown from master agent's requires_gpu flag
        gpu_tasks = [s for s in subtasks if s.get('requires_gpu')]
        cpu_tasks = [s for s in subtasks if not s.get('requires_gpu')]
        agent_logger.log_workflow_event("decomposition_complete", {
            "total_subtasks": len(subtasks),
            "gpu_tasks": len(gpu_tasks),
            "cpu_tasks": len(cpu_tasks),
            "gpu_task_ids": [s['id'] for s in gpu_tasks],
        })

        return {
            "subtasks": subtasks,
            "current_subtask_idx": 0,
            "current_subtask": None,
            "parallel_batch": [],
            "parallel_results": [],
            "checkpoint_info": checkpoint_info,  # v3.2
        }

    def _check_existing_checkpoints(self, subtasks: List[Dict]) -> Dict[str, Any]:
        """Check for existing checkpoints from previous runs (v3.2)"""
        checkpoint_dir = self.project_dir / 'temp' / 'checkpoints'

        if not checkpoint_dir.exists():
            return {"has_checkpoints": False, "checkpoints": {}}

        checkpoints = {}
        for subtask in subtasks:
            task_id = subtask.get('id', '')
            safe_id = re.sub(r'[^\w\-]', '_', task_id)[:50]
            checkpoint_path = checkpoint_dir / f"{safe_id}_checkpoint.json"

            if checkpoint_path.exists():
                try:
                    with open(checkpoint_path) as f:
                        cp = json.load(f)
                    checkpoints[task_id] = {
                        "status": cp.get('status'),
                        "iteration": cp.get('iteration'),
                        "path": str(checkpoint_path)
                    }
                except Exception:
                    pass

        return {
            "has_checkpoints": len(checkpoints) > 0,
            "checkpoints": checkpoints
        }

    def identify_parallel_tasks(self, state: WorkflowState) -> Dict:
        """
        Identify tasks that can run in parallel.
    
        v3.2: GPU-aware batching — GPU and CPU tasks are batched separately
        to avoid saturating the GPU partition with too many concurrent jobs.
    
        v1.2.4 FIX: Deadlock recovery moved INTO this node (from route_execution_mode).
        route_execution_mode is a conditional edge — LangGraph discards any state
        mutations made there. Recovery must happen here where return values ARE
        applied to state. Sets status to 'parallel_ready' or 'sequential_ready'
        so the router can dispatch without touching state itself.
        """
        subtasks = state['subtasks']
    
        # Find pending tasks
        pending = [st for st in subtasks if st.get('status', 'pending') == 'pending']
        if not pending:
            return {
                "parallel_batch": [],
                "current_subtask": None,
                "status": "all_tasks_processed",
            }
    
        # Find tasks with satisfied dependencies
        completed_ids = {st['id'] for st in subtasks if st.get('status') == 'completed'}
        ready_tasks = []
        for st in pending:
            deps = set(st.get('dependencies', []))
            if deps.issubset(completed_ids):
                ready_tasks.append(st)
    
        if not ready_tasks:
            # No tasks are ready — attempt deadlock recovery before giving up.
            # Pass a copy of state with updated subtasks list so _break_deadlock
            # sees the current in-memory status (not stale state from LangGraph).
            recovery_state = dict(state)
            recovery_state['subtasks'] = subtasks
    
            recovery = self._break_deadlock(recovery_state)
    
            if recovery['unblocked']:
                ready = recovery['ready_tasks']
                print(
                    f"\n⚠ DEADLOCK DETECTED — recovered: "
                    f"{len(ready)} task(s) unblocked"
                )
                agent_logger.log_workflow_event("deadlock_recovered", {
                    "unblocked_tasks": [t['id'] for t in ready],
                    "actions_taken": recovery.get('actions', []),
                })
    
                # Batch or sequential — same logic as the normal ready path below
                if self.parallel_enabled and self.use_slurm and len(ready) > 1:
                    max_batch = self.slurm_config.get("max_parallel_jobs", 10)
                    gpu_ready = [t for t in ready if t.get('requires_gpu')]
                    cpu_ready = [t for t in ready if not t.get('requires_gpu')]
                    max_gpu_batch = min(
                        self.slurm_config.get("max_parallel_gpu_jobs", 4), max_batch
                    )
                    batch = (
                        gpu_ready[:max_gpu_batch]
                        + cpu_ready[:max_batch - min(len(gpu_ready), max_gpu_batch)]
                    )
                    batch = batch[:max_batch]
                    agent_logger.log_workflow_event("parallel_batch", {
                        "batch_size": len(batch),
                        "gpu_tasks": len([t for t in batch if t.get('requires_gpu')]),
                        "cpu_tasks": len([t for t in batch if not t.get('requires_gpu')]),
                        "tasks": [t['id'] for t in batch],
                        "after_deadlock_recovery": True,
                    })
                    return {
                        "subtasks": subtasks,  # updated by _break_deadlock (phantom deps stripped)
                        "parallel_batch": batch,
                        "current_subtask": None,
                        "status": "parallel_ready",
                    }
                else:
                    return {
                        "subtasks": subtasks,
                        "parallel_batch": [],
                        "current_subtask": ready[0],
                        "status": "sequential_ready",
                    }
    
            # Truly unrecoverable — log and let router exit cleanly
            print("\n⚠ DEADLOCK DETECTED — unrecoverable, exiting pipeline")
            agent_logger.log_workflow_event("deadlock_unrecoverable", {
                "actions_attempted": recovery.get('actions', []),
                "remaining_blocked": recovery.get('remaining_blocked', []),
            })
            return {
                "subtasks": subtasks,
                "parallel_batch": [],
                "current_subtask": None,
                "status": "blocked",
            }
    
        # Normal path — tasks are ready, batch them
        if self.parallel_enabled and self.use_slurm and len(ready_tasks) > 1:
            max_batch = self.slurm_config.get("max_parallel_jobs", 10)
            # v3.2: Separate GPU and CPU tasks to avoid GPU partition saturation
            gpu_ready = [t for t in ready_tasks if t.get('requires_gpu')]
            cpu_ready = [t for t in ready_tasks if not t.get('requires_gpu')]
            max_gpu_batch = min(
                self.slurm_config.get("max_parallel_gpu_jobs", 4), max_batch
            )
            batch = (
                gpu_ready[:max_gpu_batch]
                + cpu_ready[:max_batch - min(len(gpu_ready), max_gpu_batch)]
            )
            batch = batch[:max_batch]
            agent_logger.log_workflow_event("parallel_batch", {
                "batch_size": len(batch),
                "gpu_tasks": len([t for t in batch if t.get('requires_gpu')]),
                "cpu_tasks": len([t for t in batch if not t.get('requires_gpu')]),
                "tasks": [t['id'] for t in batch],
            })
            if len(batch) == 1:
                return {
                    "parallel_batch": [],
                    "current_subtask": batch[0],
                }
            return {
                "parallel_batch": batch,
                "current_subtask": None,
            }
        else:
            # Sequential mode
            return {
                "parallel_batch": [],
                "current_subtask": ready_tasks[0],
            }

def submit_parallel_jobs(self, state: WorkflowState) -> Dict:
        """Submit parallel tasks with true concurrent execution.

        v1.2.2 FIX F: Uses ThreadPoolExecutor to run all batch tasks
        simultaneously. Each task gets its own thread running the full
        4-phase sub-agent lifecycle. The method blocks until ALL tasks
        in the batch have resolved (success, failure, or crash).

        v1.2.7 FIX: All results treated as immediate. The sub-agent runs
        the full lifecycle (script → env → sbatch → submit → monitor →
        diagnostic loop) within its thread. By the time the future resolves,
        the task is terminal. The old job_id-based split routed successful
        tasks to running_jobs → wait_for_parallel_jobs for redundant
        re-polling of already-completed SLURM jobs, causing tasks to
        ghost-stall at "running" when the poll failed to resolve.
        running_jobs is now always empty; wait_for_parallel_jobs is a
        passthrough.

        Thread safety:
        - Each thread creates its own ScriptFirstSubAgent instance
        - Master document writes protected by self._lock
        - Ollama handles concurrent requests via OLLAMA_NUM_PARALLEL
        - Checkpoint files are per-task (no conflicts)
        - DiagnosticMemory is append-only (safe for concurrent reads/writes)
        """
        batch = state['parallel_batch']
        env_name = state['env_name']
        immediate_results = []
        # Guard: empty batch should never reach here but handle it safely
        # rather than crashing ThreadPoolExecutor with max_workers=0
        if not batch:
            logger.warning("submit_parallel_jobs called with empty batch — skipping")
            return {
                "running_jobs": {},
                "parallel_results": [],
            }

        max_workers = min(self.max_parallel_agents, len(batch))

        def _execute_task(subtask):
            """Execute a single task in its own thread."""
            task_id = subtask['id']
            try:
                # Reflexion pre-check
                if self.use_reflexion_memory and subtask.get('context', {}).get('retry_reason'):
                    proposed = subtask.get('context', {}).get('retry_reason', '')
                    check = check_before_retry(task_id, proposed)
                    if not check["allowed"]:
                        logger.warning(
                            f"Task {task_id}: duplicate approach rejected "
                            f"(similarity: {check['similarity']:.2f})"
                        )
                        subtask.setdefault('context', {})
                        subtask['context']['avoid_approach'] = check['similar_approach']
                        subtask['context']['duplicate_warning'] = True

                # Mark running (thread-safe via lock)
                with self._lock:
                    self.master.master_document.mark_running(task_id)

                # Create agent (each thread gets its own instance)
                agent = ScriptFirstSubAgent(
                    agent_id=f"agent_{task_id}",
                    sandbox=self.sandbox,
                    conda_tools=self.conda_tools,
                    slurm_tools=self.slurm_tools,
                    ollama_model=self.ollama_model,
                    ollama_base_url=self.ollama_base_url,
                    use_slurm=True,
                    slurm_config=self.slurm_config,
                    project_root=str(self.project_dir),
                    cleanup_env_on_success=self.cleanup_env_on_success,
                    diagnostic_memory=self.diagnostic_memory,
                )

                result = agent.execute(subtask, env_name=env_name)
                return (subtask, result)

            except Exception as e:
                logger.error(f"Task {task_id}: thread crashed: {e}")
                return (subtask, {
                    'success': False,
                    'error': f"Agent thread crashed: {e}",
                })

        # Log batch launch
        agent_logger.log_workflow_event("parallel_launch", {
            "batch_size": len(batch),
            "max_workers": max_workers,
            "tasks": [s['id'] for s in batch],
        })

        # Launch all tasks concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_execute_task, subtask): subtask['id']
                for subtask in batch
            }

            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    subtask, result = future.result()
                except Exception as e:
                    # future.result() itself threw — should not happen given
                    # the try/except inside _execute_task, but safety net
                    logger.error(f"Task {task_id}: future exception: {e}")
                    subtask = next(s for s in batch if s['id'] == task_id)
                    result = {'success': False, 'error': f"Future exception: {e}"}

                # v1.2.7: ALL results are immediate. The sub-agent runs the
                # full 4-phase lifecycle in its thread — by the time the
                # future resolves, the task is terminal (success or failure).
                # The old code split on result.get('job_id'), routing
                # successful tasks to running_jobs → wait_for_parallel_jobs
                # for redundant re-polling. This caused tasks to ghost-stall
                # at "running" status because wait_for_parallel_jobs polled
                # already-completed SLURM jobs and failed to resolve them.
                immediate_results.append({
                    "subtask": subtask,
                    "result": result,
                })
                # Update master doc (thread-safe via lock)
                with self._lock:
                    if result.get('success'):
                        logger.info(
                            f"Task {task_id} completed "
                            f"(skipped={result.get('skipped', False)})"
                        )
                        self.master.mark_subtask_complete(
                            task_id,
                            result.get('outputs', {}),
                            result.get('report', ''))
                    else:
                        logger.warning(
                            f"Task {task_id} failed: "
                            f"{result.get('error', 'unknown')[:200]}")
                        self.master.master_document.mark_failed(
                            task_id,
                            error_summary=result.get(
                                'error', 'Unknown error'
                            ),
                            attempts=1,
                            script_path=result.get('script_path'),
                        )

        # Log batch completion
        n_success = sum(1 for r in immediate_results if r['result'].get('success'))
        n_fail = len(immediate_results) - n_success
        agent_logger.log_workflow_event("parallel_complete", {
            "total": len(immediate_results),
            "success": n_success,
            "failed": n_fail,
            "tasks": [r['subtask']['id'] for r in immediate_results],
        })

        # v1.2.7: running_jobs always empty — no second poll needed.
        # wait_for_parallel_jobs becomes a passthrough.
        return {
            "running_jobs": {},
            "parallel_results": immediate_results,
        }

    def execute_sequential(self, state: WorkflowState) -> Dict:
        """Execute a single task sequentially with sub-agent v1.2.0 features"""
        subtask = state['current_subtask']
        env_name = state['env_name']
        task_id = subtask['id']

        # Check for known solutions (reflexion memory)
        if self.use_reflexion_memory:
            previous_error = subtask.get('context', {}).get('previous_error')
            if previous_error:
                try:
                    solutions = find_similar_solutions(previous_error)
                    if solutions and solutions[0].get('score', 0) > 0.8:
                        logger.info(
                            f"Task {task_id}: Found known solution "
                            f"(score: {solutions[0]['score']:.2f})"
                        )
                        subtask['context'] = subtask.get('context', {})
                        subtask['context']['suggested_solution'] = solutions[0].get('solution')
                        subtask['context']['solution_confidence'] = solutions[0].get('score')
                except Exception as e:
                    logger.warning(f"Error checking for solutions: {e}")

        # Mark as running
        self.master.master_document.mark_running(task_id)

        # Create sub-agent — passes resolved model
        # v1.2.0: diagnostic_memory passed through for cross-task solutions
        agent = ScriptFirstSubAgent(
            agent_id=f"agent_{task_id}",
            sandbox=self.sandbox,
            conda_tools=self.conda_tools,
            slurm_tools=self.slurm_tools,
            ollama_model=self.ollama_model,
            ollama_base_url=self.ollama_base_url,
            use_slurm=self.use_slurm,
            slurm_config=self.slurm_config,
            project_root=str(self.project_dir),
            cleanup_env_on_success=self.cleanup_env_on_success,  # v3.2
            diagnostic_memory=self.diagnostic_memory,  # v1.2.0
        )

        # Execute - sub-agent handles checkpointing internally
        result = agent.execute(subtask, env_name=env_name)

        return {
            "parallel_results": [{
                "subtask": subtask,
                "result": result
            }],
            "agent_context_status": {
                **state.get('agent_context_status', {}),
                task_id: result.get('context_status', {})
            }
        }

    def wait_for_parallel_jobs(self, state: WorkflowState) -> Dict:
        """Wait for all parallel SLURM jobs to complete"""
        running_jobs = state.get('running_jobs', {})

        existing_results = state.get('parallel_results', [])

        if not running_jobs:
            return {"parallel_results": existing_results}

        results = list(existing_results)
        poll_interval = self.slurm_config.get("poll_interval", 10)
        max_attempts = self.slurm_config.get("max_poll_attempts", 720)

        # Create mapping of job_id -> subtask
        job_to_subtask = {}
        for subtask in state['parallel_batch']:
            task_id = subtask['id']
            if task_id in running_jobs:
                job_to_subtask[running_jobs[task_id]] = subtask

        # Poll until all jobs complete
        completed_jobs = set()
        attempts = 0

        while len(completed_jobs) < len(running_jobs) and attempts < max_attempts:
            for task_id, job_id in running_jobs.items():
                if job_id in completed_jobs:
                    continue

                status = self.slurm_tools.get_job_status(job_id)

                if status.get('state') in ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT']:
                    completed_jobs.add(job_id)
                    subtask = job_to_subtask.get(job_id)

                    if subtask:
                        result = {
                            "success": status.get('state') == 'COMPLETED',
                            "job_id": job_id,
                            "state": status.get('state'),
                            "outputs": self._collect_outputs(subtask, job_id),
                            "error": status.get('error') if status.get('state') != 'COMPLETED' else None
                        }
                        results.append({
                            "subtask": subtask,
                            "result": result
                        })

            if len(completed_jobs) < len(running_jobs):
                time.sleep(poll_interval)
                attempts += 1

        # Handle timeout
        for task_id, job_id in running_jobs.items():
            if job_id not in completed_jobs:
                subtask = job_to_subtask.get(job_id)
                if subtask:
                    results.append({
                        "subtask": subtask,
                        "result": {
                            "success": False,
                            "job_id": job_id,
                            "state": "TIMEOUT",
                            "error": "Job monitoring timed out"
                        }
                    })

        return {
            "parallel_results": results,
            "running_jobs": {}
        }

    def _collect_outputs(self, subtask: Dict, job_id: str) -> Dict:
        """Collect outputs from completed SLURM job"""
        outputs = {}

        expected_outputs = subtask.get('output_files', [])
        for output_file in expected_outputs:
            output_path = self.project_dir / output_file
            if output_path.exists():
                outputs[output_file] = str(output_path)

        # Also check SLURM output logs
        log_dir = self.project_dir / 'slurm' / 'logs'
        stdout_files = list(log_dir.glob(f"*_{job_id}.out"))
        stderr_files = list(log_dir.glob(f"*_{job_id}.err"))

        if stdout_files:
            outputs['stdout'] = str(stdout_files[0])
        if stderr_files:
            outputs['stderr'] = str(stderr_files[0])

        return outputs

    def handle_results(self, state: WorkflowState) -> Dict:
        """Process results from execution"""
        results = state.get('parallel_results', [])
        completed = []
        failed = []
        task_attempt_counts = state.get('task_attempt_counts', {})

        for item in results:
            subtask = item['subtask']
            result = item['result']
            task_id = subtask['id']

            # Update attempt count
            task_attempt_counts[task_id] = task_attempt_counts.get(task_id, 0) + 1

            # Update subtask status
            subtask['last_result'] = result

            if result.get('success'):
                subtask['status'] = 'completed'
                completed.append(subtask)

                # v3.2: Log if this was a resumed task
                if result.get('resumed'):
                    agent_logger.log_workflow_event("task_resumed_completed", {
                        "task_id": task_id,
                        "was_cached": result.get('skipped', False)
                    })

                # Update master document
                self.master.mark_subtask_complete(
                    task_id,
                    result.get('outputs', {}),
                    result.get('report', '')
                )

                # v3.2: Log cleanup status
                if result.get('env_cleaned'):
                    agent_logger.log_workflow_event("env_cleaned", {
                        "task_id": task_id,
                        "env_name": result.get('env_name')
                    })

                # Record solution if reflexion memory enabled
                if self.use_reflexion_memory:
                    previous_error = subtask.get('context', {}).get('previous_error')
                    approach = subtask.get('context', {}).get('approach_tried')
                    if previous_error and approach:
                        try:
                            record_solution(task_id, previous_error, approach)
                        except Exception as e:
                            logger.warning(f"Failed to record solution: {e}")

            else:
                subtask['status'] = 'failed'
                subtask['last_result'] = result
                failed.append(subtask)

                # v1.2.3: Safety-net master document sync on failure.
                # Patch 4A handles the parallel thread path. This handles
                # all other paths (sequential, wait_for_parallel_jobs TIMEOUT,
                # any future result sources) through this single convergence
                # point. Guard checks current master doc status first — if
                # Patch 4A already wrote "failed", mark_failed is idempotent
                # and the second call is harmless. If status is still "running"
                # (the ghost state that caused the v1.2.2 deadlock), this
                # corrects it unconditionally.
                current_status = (
                    self.master.master_document.steps
                    .get(task_id, {})
                    .get('status')
                )
                if current_status != 'completed':
                    self.master.master_document.mark_failed(
                        task_id,
                        error_summary=result.get('error', 'Unknown error'),
                        attempts=task_attempt_counts.get(task_id, 1),
                        script_path=result.get('script_path'),
                    )

                agent_logger.log_workflow_event("task_failed", {
                    "task_id": task_id,
                    "error": result.get('error', 'unknown')[:200],
                    "attempts": task_attempt_counts.get(task_id, 1),
                    "was_ghost_running": current_status == 'running',
                })

        return {
            "completed_subtasks": completed,
            "failed_subtasks": failed,
            "task_attempt_counts": task_attempt_counts,
            "parallel_results": [],
            "running_jobs": {},
        }

    def reflexion_check(self, state: WorkflowState) -> Dict:
        """Check reflexion memory for failed tasks"""
        failed = state.get('failed_subtasks', [])
        reflexion_state = state.get('reflexion', create_initial_reflexion_state())

        if not failed or not self.use_reflexion_memory:
            return {"reflexion": reflexion_state}

        # Process the first failed task
        failed_task = failed[0]
        task_id = failed_task['id']
        error = failed_task.get('last_result', {}).get('error', '')
        approach = failed_task.get('context', {}).get('approach_tried', '')

        try:
            # Use reflexion engine to analyze
            result = reflexion_handle_failure(
                task_id=task_id,
                error_message=error,
                approach_tried=approach
            )

            reflexion_state.update(result)

            agent_logger.log_workflow_event("reflexion_check", {
                "task_id": task_id,
                "action": result.get('action'),
                "should_escalate": result.get('should_escalate'),
                "should_apply_solution": result.get('should_apply_solution'),
                "is_duplicate": result.get('is_duplicate')
            })

        except Exception as e:
            logger.warning(f"Reflexion check failed: {e}")
            reflexion_state['action'] = 'retry'

        return {"reflexion": reflexion_state}

    def master_review(self, state: WorkflowState) -> Dict:
        """Master agent reviews failed tasks"""
        failed = state.get('failed_subtasks', [])

        if not failed:
            return {"master_decision": {"decision": "SKIP"}}

        # Get master's decision on first failed task
        failed_task = failed[0]

        decision = self.master.review_failure(
            subtask=failed_task,
            error=failed_task.get('last_result', {}).get('error', ''),
            context=failed_task.get('context', {})
        )

        agent_logger.log_workflow_event("master_review", {
            "task_id": failed_task['id'],
            "decision": decision.get('decision'),
            "reason": decision.get('reasoning', '')
        })

        return {"master_decision": decision}

    def generate_final_report(self, state: WorkflowState) -> Dict:
        """Generate final execution report.

        v3.2.2 Fix D: Now reports 'deadlocked' status when the pipeline
        exits due to unrecoverable dependency cycles, instead of the
        misleading 'completed' status from v3.2.1.

        v1.2.2 Fix H: Reports tasks with 'blocked' status separately,
        showing which failed dependency prevents each blocked task.
        """
        completed = state.get('completed_subtasks', [])
        failed = state.get('failed_subtasks', [])
        is_deadlocked = state.get('status') == 'deadlocked'

        # v3.2: Collect GPU/CPU routing stats
        all_subtasks = state.get('subtasks', [])
        gpu_tasks = [s for s in all_subtasks if s.get('requires_gpu')]
        cpu_tasks = [s for s in all_subtasks if not s.get('requires_gpu')]

        # v1.2.2: Include both 'pending' and 'blocked' statuses
        blocked_tasks = [s for s in all_subtasks
                         if s.get('status') in ('pending', 'blocked')]

        # v1.2.2: Count explicitly blocked (by failed deps)
        explicitly_blocked = [s for s in all_subtasks if s.get('status') == 'blocked']

        # Determine final status string
        if is_deadlocked:
            final_status = "DEADLOCKED"
            status_emoji = "⚠"
        elif failed:
            final_status = "PARTIAL"
            status_emoji = "⚠"
        elif len(completed) == len(all_subtasks):
            final_status = "SUCCESS"
            status_emoji = "✓"
        else:
            final_status = "COMPLETED"
            status_emoji = "✓"

        # Build report
        report = f"""
# AGI Pipeline Execution Report

Generated: {datetime.now().isoformat()}
Status: {status_emoji} {final_status}
Model: {self.ollama_model}
Cluster: {os.environ.get('AGI_CLUSTER', 'unknown')}
Token Budget: {self.MAX_CONTEXT_TOKENS}/{self.MAX_TOOL_OUTPUT_TOKENS}/{self.MIN_TOKENS_TO_CONTINUE} (context/tool/min)
Diagnostic Memory: {'enabled' if self.diagnostic_memory else 'disabled'}
Parallel Agents: {self.max_parallel_agents}

## Summary

- **Total Subtasks**: {len(all_subtasks)}
- **Completed**: {len(completed)}
- **Failed**: {len(failed)}
- **Blocked**: {len(blocked_tasks)}
- **Blocked by Failed Deps**: {len(explicitly_blocked)}
- **Success Rate**: {len(completed) / max(len(all_subtasks), 1) * 100:.1f}%
- **GPU Tasks**: {len(gpu_tasks)} ({len([s for s in gpu_tasks if s.get('status') == 'completed'])} completed)
- **CPU Tasks**: {len(cpu_tasks)} ({len([s for s in cpu_tasks if s.get('status') == 'completed'])} completed)

"""

        # v3.2.2: Deadlock section
        if is_deadlocked:
            report += """## ⚠ DEADLOCK DETECTED

The pipeline exited because all remaining tasks have unsatisfied
dependencies that cannot be resolved. This typically means:
1. Circular dependencies in the task graph (Fix B should prevent this)
2. Non-executable documentation sections blocking real tasks (Fix A should prevent this)
3. A dependency references a task that was never created

### Blocked Tasks

"""
            for st in blocked_tasks:
                deps = st.get('dependencies', [])
                report += f"- **{st['id']}**: deps={deps}, title: {st.get('title', 'N/A')[:80]}\n"

            report += "\n### Recommended Actions\n\n"
            report += "1. Check `reports/master_prompt_state.json` for the dependency graph\n"
            report += "2. Verify Fix A filtered non-executable sections correctly\n"
            report += "3. Verify Fix B sanitized the dependency DAG\n"
            report += "4. Re-run with `--verbose` to see dependency chain at decomposition\n\n"

        # v1.2.2: Blocked-by-failure section (Fix H)
        if explicitly_blocked:
            report += "## ⊘ Tasks Blocked by Failed Dependencies\n\n"
            report += "These tasks were not attempted because a required dependency "
            report += "permanently failed.\n\n"
            for st in explicitly_blocked:
                blocked_by = st.get('blocked_by', [])
                report += f"- **{st['id']}** ({st.get('title', 'N/A')[:60]})\n"
                report += f"  Blocked by: {', '.join(blocked_by)}\n"
            report += "\n"

        report += "## Task Details\n\n"

        for subtask in completed:
            gpu_flag = " [GPU]" if subtask.get('requires_gpu') else ""
            note = ""
            if subtask.get('completion_note') == 'force_completed_by_deadlock_recovery':
                note = " *(force-completed by deadlock recovery)*"
            report += f"### ✓ {subtask['id']}{gpu_flag}{note}\n"
            report += f"{subtask.get('description', 'No description')[:200]}\n\n"

        for subtask in failed:
            gpu_flag = " [GPU]" if subtask.get('requires_gpu') else ""
            report += f"### ✗ {subtask['id']}{gpu_flag}\n"
            report += f"{subtask.get('description', 'No description')[:200]}\n"
            report += f"**Error**: {subtask.get('last_result', {}).get('error', 'Unknown')[:500]}\n\n"

        for subtask in blocked_tasks:
            gpu_flag = " [GPU]" if subtask.get('requires_gpu') else ""
            blocked_by = subtask.get('blocked_by', [])
            blocked_label = f" (blocked by: {', '.join(blocked_by)})" if blocked_by else ""
            report += f"### ⊘ {subtask['id']}{gpu_flag} (BLOCKED{blocked_label})\n"
            report += f"{subtask.get('description', 'No description')[:200]}\n"
            report += f"**Deps**: {subtask.get('dependencies', [])}\n\n"

        # Add reflexion memory summary
        if self.use_reflexion_memory:
            try:
                task_attempt_counts = state.get('task_attempt_counts', {})
                memory_summary = "\n## Reflexion Memory Summary\n\n"
                memory_summary += f"Tasks with retries:\n"
                for task_id, count in task_attempt_counts.items():
                    if count > 1:
                        memory_summary += f"- {task_id}: {count} attempt(s)\n"

                client = get_memory_client()
                if client and hasattr(client, '_engine'):
                    stats = client._engine.get_stats()
                    if stats:
                        memory_summary += f"\nTotal memories stored: {stats.get('memory', {}).get('total', 'N/A')}\n"

                report += memory_summary
            except Exception as e:
                logger.warning(f"Failed to add memory summary: {e}")

        # v1.2.0: Add diagnostic memory summary
        if self.diagnostic_memory:
            try:
                diag_summary = "\n## Diagnostic Memory Summary\n\n"
                diag_stats = self.diagnostic_memory.get_stats()
                diag_summary += f"- Solutions stored: {diag_stats.get('total_solutions', 0)}\n"
                diag_summary += f"- Solutions reused: {diag_stats.get('solutions_reused', 0)}\n"
                report += diag_summary
            except Exception as e:
                logger.warning(f"Failed to add diagnostic memory summary: {e}")

        # Save report
        reports_dir = self.project_dir / 'reports'
        reports_dir.mkdir(parents=True, exist_ok=True)

        report_path = reports_dir / f"final_report_{datetime.now():%Y%m%d_%H%M%S}.md"
        report_path.write_text(report)

        # Save documentation
        doc_generator.save_readme()

        # Final git commit
        git_tracker.commit_task_attempt(
            task_id="final",
            agent_name="master",
            description="Pipeline completion",
            status="success",
            files_modified=["reports/final_report.md"],
            tools_used=["documentation_generator"],
            result=report[:500]
        )

        return {
            "final_report": report,
            "status": "deadlocked" if is_deadlocked else "completed"
        }

    def cleanup(self, state: WorkflowState) -> Dict:
        """Cleanup temporary files"""
        self.sandbox.cleanup_temp()

        # Cancel any lingering jobs
        if self.slurm_tools:
            try:
                self.slurm_tools.cancel_all_jobs()
            except Exception as e:
                logger.warning(f"Error cancelling jobs: {e}")

        return {}

    # ==================== Routing Functions ====================

    def route_execution_mode(self, state: WorkflowState) -> str:
        """Route based on execution mode.
    
        Reads status set by identify_parallel_tasks and routes accordingly.
        Does NOT mutate state — conditional edges in LangGraph discard any
        state changes made here. All recovery logic lives in identify_parallel_tasks.
    
        v1.2.4: Removed deadlock recovery from this function. It was silently
        discarding state mutations (current_subtask assignments), causing the
        NoneType crash in execute_sequential. Recovery now happens in
        identify_parallel_tasks which is a proper node whose return dict IS
        applied to state.
        """
        batch = state.get('parallel_batch', [])
        current = state.get('current_subtask')
        status = state.get('status', '')
    
        if status == 'all_tasks_processed':
            decision = "complete"
    
        elif status == 'parallel_ready':
            # Deadlock was recovered inside identify_parallel_tasks, batch is set
            decision = "parallel"
    
        elif status == 'sequential_ready':
            # Deadlock was recovered inside identify_parallel_tasks, current is set
            decision = "sequential"
    
        elif status == 'blocked':
            # identify_parallel_tasks already attempted recovery and failed.
            # Exit cleanly with deadlocked status for the report.
            agent_logger.log_workflow_event("deadlock_unrecoverable_exit", {
                "completed": len([s for s in state.get('subtasks', [])
                                  if s.get('status') == 'completed']),
                "failed": len([s for s in state.get('subtasks', [])
                               if s.get('status') == 'failed']),
                "blocked": len([s for s in state.get('subtasks', [])
                                if s.get('status') == 'blocked']),
            })
            # Note: cannot mutate state here — report node checks status field
            # which will still be 'blocked'; generate_report handles that case.
            decision = "complete"
    
        elif batch and len(batch) > 1:
            decision = "parallel"
    
        elif current or (batch and len(batch) == 1):
            if batch and len(batch) == 1:
                # Single-item batch collapses to sequential — but we cannot set
                # current_subtask here. identify_parallel_tasks should have set
                # current_subtask directly for single-task batches.
                decision = "sequential"
            else:
                decision = "sequential"
    
        else:
            decision = "complete"
    
        agent_logger.log_workflow_event("transition", {
            "from": "identify_parallel",
            "to": decision,
            "batch_size": len(batch),
            "has_current": current is not None,
            "status": status,
        })
    
        return decision

    def _break_deadlock(self, state: WorkflowState) -> Dict[str, Any]:
        """Attempt to recover from a dependency deadlock.

        v3.2.2 Fix D: Safety net for when identify_parallel_tasks() returns
        status='blocked'. This should rarely trigger if Fix A (filter
        non-executable steps) and Fix B (sanitize dependencies) in
        master_agent.py are working correctly.

        v1.2.2 FIX H: Strategy 1 (strip failed deps) REMOVED. Tasks with
        failed dependencies stay blocked to maintain dependency integrity
        and provide clear error traceability. Only true circular dependency
        deadlocks are broken (Strategies 2-4).

        Recovery strategies (applied in order):
        1. (REMOVED in v1.2.2 — failed-dep tasks now get 'blocked' status)
        2. Strip deps pointing to non-existent task IDs (bad LLM output)
        3. Force-complete non-executable tasks (doc sections that slipped
           through Fix A's filter — detected by: no code_hints, no packages,
           title matches documentation patterns)
        4. Last resort: force first pending task to have no deps
        """
        subtasks = state.get('subtasks', [])
        actions = []

        all_ids = {st['id'] for st in subtasks}
        completed_ids = {st['id'] for st in subtasks
                        if st.get('status') == 'completed'}
        failed_ids = {st['id'] for st in subtasks
                      if st.get('status') == 'failed'}
        pending = [st for st in subtasks
                   if st.get('status', 'pending') == 'pending']

        # ── v1.2.2 FIX H: Separate "blocked by failure" from "true deadlock"
        blocked_by_failure = []
        potentially_deadlocked = []
        for st in pending:
            deps = set(st.get('dependencies', []))
            if deps.intersection(failed_ids):
                blocked_by_failure.append(st)
            else:
                potentially_deadlocked.append(st)

        # Mark failure-blocked tasks explicitly (they stay blocked, not stripped)
        if blocked_by_failure:
            for st in blocked_by_failure:
                st['status'] = 'blocked'
                st['blocked_by'] = [d for d in st.get('dependencies', [])
                                    if d in failed_ids]
                print(f"    → {st['id']}: blocked by failed deps: {st['blocked_by']}")
            actions.append(f"marked_{len(blocked_by_failure)}_tasks_blocked_by_failures")

        # If ALL pending tasks were blocked by failures, not a circular deadlock
        if not potentially_deadlocked:
            ready = self._find_ready_tasks(subtasks)
            if ready:
                return {'unblocked': True, 'ready_tasks': ready,
                        'actions': actions, 'remaining_blocked': []}
            return {
                'unblocked': False, 'ready_tasks': [],
                'actions': actions,
                'remaining_blocked': [st['id'] for st in blocked_by_failure],
            }

        # ── Strategy 2: Strip deps on non-existent task IDs
        for st in potentially_deadlocked:
            deps = st.get('dependencies', [])
            phantom_deps = [d for d in deps if d not in all_ids]
            if phantom_deps:
                st['dependencies'] = [d for d in deps if d in all_ids]
                actions.append(f"stripped_phantom_deps_{st['id']}_{phantom_deps}")
                print(f"    → {st['id']}: removed phantom deps: {phantom_deps}")

        ready = self._find_ready_tasks(subtasks)
        if ready:
            return {
                'unblocked': True,
                'ready_tasks': ready,
                'actions': actions,
                'remaining_blocked': [],
            }

        # ── Strategy 3: Force-complete non-executable tasks
        # These have no code to execute — mark them completed so they
        # stop blocking downstream tasks.
        DOC_PATTERNS = [
            r'(?i)expected\s*output', r'(?i)success\s*criteria',
            r'(?i)notes?\b', r'(?i)dependencies\b',
            r'(?i)anndata\s*structure', r'(?i)environment\b',
            r'(?i)input\s*files?\b', r'(?i)output\s*files?\b',
            r'(?i)important\s*considerations',
        ]

        force_completed = 0
        for st in list(potentially_deadlocked):  # copy list since we modify status
            title = st.get('title', '')
            code_hints = st.get('code_hints', [])
            packages = st.get('packages', [])
            description = st.get('description', '')

            # Heuristic: non-executable if no code hints, no packages,
            # and title matches a documentation pattern
            is_doc = False
            if not code_hints and not packages:
                for pattern in DOC_PATTERNS:
                    if re.search(pattern, title):
                        is_doc = True
                        break
                # Also catch very short descriptions with no procedural content
                if not is_doc and len(description) < 100:
                    procedural_verbs = [
                        'run', 'execute', 'compute', 'generate', 'create',
                        'load', 'save', 'write', 'read', 'process', 'train',
                        'cluster', 'annotate', 'plot', 'visualize', 'filter',
                    ]
                    has_verb = any(v in description.lower()
                                  for v in procedural_verbs)
                    if not has_verb:
                        is_doc = True

            if is_doc:
                st['status'] = 'completed'
                st['completion_note'] = 'force_completed_by_deadlock_recovery'
                completed_ids.add(st['id'])
                force_completed += 1
                print(f"    → {st['id']}: force-completed "
                      f"(non-executable: '{title[:50]}')")

        if force_completed > 0:
            actions.append(f"force_completed_{force_completed}_doc_tasks")

        # Re-check after force-completing
        ready = self._find_ready_tasks(subtasks)
        if ready:
            return {
                'unblocked': True,
                'ready_tasks': ready,
                'actions': actions,
                'remaining_blocked': [],
            }

        # ── Strategy 4: Last resort — force first pending task to have no deps
        # This breaks the cycle at an arbitrary point, letting at least one
        # task attempt execution. Only used if all else fails.
        still_pending = [st for st in subtasks
                         if st.get('status', 'pending') == 'pending']
        if still_pending:
            victim = still_pending[0]
            old_deps = victim.get('dependencies', [])
            victim['dependencies'] = []
            actions.append(f"forced_root_{victim['id']}_cleared_{len(old_deps)}_deps")
            print(f"    → {victim['id']}: forced to root (cleared deps: {old_deps})")

            ready = self._find_ready_tasks(subtasks)
            if ready:
                return {
                    'unblocked': True,
                    'ready_tasks': ready,
                    'actions': actions,
                    'remaining_blocked': [
                        st['id'] for st in subtasks
                        if st.get('status', 'pending') == 'pending'
                        and st['id'] not in {r['id'] for r in ready}
                    ],
                }

        # Truly unrecoverable
        return {
            'unblocked': False,
            'ready_tasks': [],
            'actions': actions,
            'remaining_blocked': [st['id'] for st in subtasks
                                  if st.get('status', 'pending') == 'pending'],
        }

    def _find_ready_tasks(self, subtasks: List[Dict]) -> List[Dict]:
        """Find pending tasks whose dependencies are all satisfied.

        Helper for _break_deadlock() — same logic as identify_parallel_tasks()
        but without modifying state or batching.
        """
        completed_ids = {st['id'] for st in subtasks
                        if st.get('status') == 'completed'}
        ready = []
        for st in subtasks:
            if st.get('status', 'pending') != 'pending':
                continue
            deps = set(st.get('dependencies', []))
            if deps.issubset(completed_ids):
                ready.append(st)
        return ready

    def route_after_execution(self, state: WorkflowState) -> str:
        """Route after execution with progress-first priority.

        v1.2.2 FIX G: Three-tier priority routing:
          1. Ready tasks exist → "next_batch" (keep making progress)
          2. Reviewable failures exist → "review" (attempt recovery)
          3. Nothing actionable → "complete" (exit pipeline)

        This ensures independent tasks always get attempted before the
        pipeline spends time reviewing failures. The pipeline only exits
        when ALL tasks are completed, permanently failed, or blocked.
        """
        failed = state.get('failed_subtasks', [])
        subtasks = state.get('subtasks', [])

        # Compute current state
        completed_ids = {st['id'] for st in subtasks
                         if st.get('status') == 'completed'}
        failed_ids = {st['id'] for st in subtasks
                      if st.get('status') == 'failed'}
        pending = [st for st in subtasks
                   if st.get('status', 'pending') == 'pending']

        # Find tasks whose deps are all satisfied
        ready = [
            st for st in pending
            if set(st.get('dependencies', [])).issubset(completed_ids)
        ]

        # PRIORITY 1: More work available → dispatch it
        if ready:
            decision = "next_batch"
            agent_logger.log_workflow_event("transition", {
                "from": "handle_results",
                "to": decision,
                "reason": f"{len(ready)} tasks ready",
                "ready_tasks": [t['id'] for t in ready[:10]],
                "pending_count": len(pending),
                "failed_count": len(failed),
                "completed_count": len(completed_ids),
            })
            return decision

        # PRIORITY 2: No ready tasks but reviewable failures exist
        reviewable = [
            f for f in failed
            if not f.get('_permanently_failed')
            and not f.get('_unrecoverable')
        ]

        if reviewable:
            decision = "review"
            agent_logger.log_workflow_event("transition", {
                "from": "handle_results",
                "to": decision,
                "reason": f"{len(reviewable)} reviewable failure(s), 0 ready tasks",
                "reviewable_tasks": [f['id'] for f in reviewable[:5]],
            })
            return decision

        # PRIORITY 3: Nothing actionable
        # Mark tasks blocked by failed deps before exiting
        for st in pending:
            blocking_deps = [
                dep for dep in st.get('dependencies', [])
                if dep in failed_ids
            ]
            if blocking_deps:
                st['status'] = 'blocked'
                st['blocked_by'] = blocking_deps

        decision = "complete"
        blocked_count = len([st for st in subtasks if st.get('status') == 'blocked'])
        agent_logger.log_workflow_event("transition", {
            "from": "handle_results",
            "to": decision,
            "reason": "no ready tasks, no reviewable failures",
            "completed": len(completed_ids),
            "failed": len(failed_ids),
            "blocked": blocked_count,
        })
        return decision

    def route_after_reflexion(self, state: WorkflowState) -> str:
        """Route after reflexion check.

        v1.2.2: escalate marks task as permanently failed and returns to
        main loop instead of exiting pipeline.
        """
        reflexion = state.get('reflexion', create_initial_reflexion_state())

        if reflexion.get('should_escalate'):
            # v1.2.2: Mark permanently failed, don't exit pipeline
            failed = state.get('failed_subtasks', [])
            if failed:
                failed[0]['_permanently_failed'] = True
                failed[0]['_escalated'] = True
            decision = "escalate"     # → identify_parallel (graph edge changed)
        elif reflexion.get('should_apply_solution'):
            decision = "apply_solution"
        else:
            decision = "master_review"

        agent_logger.log_workflow_event("transition", {
            "from": "reflexion_check",
            "to": decision,
            "action": reflexion.get('action'),
            "is_duplicate": reflexion.get('is_duplicate', False),
        })

        return decision

    def route_after_review(self, state: WorkflowState) -> str:
        """Route after master review.

        v1.2.2: ESCALATE no longer exits the pipeline. All three decisions
        return to identify_parallel:
          RETRY    → reset task to pending, re-attempt in next batch
          SKIP     → mark permanently failed, continue pipeline
          ESCALATE → mark permanently failed, continue pipeline (same as SKIP)
        """
        decision = state.get('master_decision', {})
        decision_type = decision.get('decision', 'SKIP')

        if decision_type == 'RETRY':
            failed = state.get('failed_subtasks', [])
            if failed:
                failed[0]['status'] = 'pending'
                failed[0].pop('last_result', None)
                failed[0].pop('_permanently_failed', None)
            route = "retry"
        else:
            # SKIP and ESCALATE both mark permanently failed
            failed = state.get('failed_subtasks', [])
            if failed:
                failed[0]['_permanently_failed'] = True
                if decision_type == 'ESCALATE':
                    failed[0]['_escalated'] = True
            route = "skip"

        agent_logger.log_workflow_event("transition", {
            "from": "master_review",
            "to": route,
            "master_decision": decision_type,
            "reasoning": decision.get('reasoning', '')[:200],
        })

        return route

    # ==================== Main Entry Point ====================

    def run(
        self,
        main_task: str,
        context: Dict[str, Any] = None,
        thread_id: str = None
    ) -> Dict:
        """
        Execute the workflow.

        Args:
            main_task: High-level task description
            context: Additional context
            thread_id: Thread ID for persistence

        Returns:
            Final state
        """
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        initial_state = {
            "main_task": main_task,
            "context": context or {},
            "project_dir": str(self.project_dir),
            "subtasks": [],
            "current_subtask_idx": 0,
            "current_subtask": None,
            "parallel_batch": [],
            "parallel_results": [],
            "running_jobs": {},
            "agent_context_status": {},
            "env_name": "",
            "completed_subtasks": [],
            "failed_subtasks": [],
            "use_slurm": self.use_slurm,
            "parallel_enabled": self.parallel_enabled,
            "final_report": "",
            "status": "started",
            "master_decision": {},
            "reflexion": create_initial_reflexion_state(),
            "task_attempt_counts": {},
            "cleanup_env_on_success": self.cleanup_env_on_success,  # v3.2
            "checkpoint_info": {},  # v3.2
        }

        final_state = self.app.invoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}}
        )

        return final_state


# Need this import for wait_for_parallel_jobs
import time

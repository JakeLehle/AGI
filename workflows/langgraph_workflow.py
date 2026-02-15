"""
LangGraph Workflow v3.2.1 - Script-First Architecture with Reflexion Memory

Orchestrates multi-agent system with:
- Token-based context limits (configurable, default 25K) instead of iteration counts
- Each subtask gets its own persistent context window
- Parallel SLURM job submission for independent tasks
- Script generation and execution paradigm
- Living document master prompt management
- **Reflexion Memory for loop prevention**

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
# Import sub-agent - try v3.2 class first, fall back to base class
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
    LangGraph workflow v3.2.2 with script-first architecture and reflexion memory.

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

        # Log configuration
        logger.info(
            f"MultiAgentWorkflow v3.2.2 initialized: "
            f"model={self.ollama_model}, "
            f"tokens={self.MAX_CONTEXT_TOKENS}/{self.MAX_TOOL_OUTPUT_TOKENS}/{self.MIN_TOKENS_TO_CONTINUE}, "
            f"slurm={self.use_slurm}, "
            f"parallel={self.parallel_enabled}, "
            f"reflexion={self.use_reflexion_memory}, "
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

        # After reflexion check
        workflow.add_conditional_edges(
            "reflexion_check",
            self.route_after_reflexion,
            {
                "apply_solution": "execute_sequential",
                "escalate": "generate_report",
                "master_review": "master_review",
            }
        )

        # After master review
        workflow.add_conditional_edges(
            "master_review",
            self.route_after_review,
            {
                "retry": "identify_parallel",
                "skip": "identify_parallel",
                "escalate": "generate_report"
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
            "reflexion_enabled": self.use_reflexion_memory,
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
        """
        subtasks = state['subtasks']

        # Find pending tasks
        pending = [st for st in subtasks if st.get('status', 'pending') == 'pending']

        if not pending:
            return {
                "parallel_batch": [],
                "status": "all_tasks_processed"
            }

        # Find tasks with satisfied dependencies
        completed_ids = {st['id'] for st in subtasks
                        if st.get('status') == 'completed'}

        ready_tasks = []
        for st in pending:
            deps = set(st.get('dependencies', []))
            if deps.issubset(completed_ids):
                ready_tasks.append(st)

        if not ready_tasks:
            # Check if we're blocked
            if pending:
                return {"parallel_batch": [], "status": "blocked"}
            return {"parallel_batch": [], "status": "all_tasks_processed"}

        # Batch for parallel execution
        if self.parallel_enabled and self.use_slurm and len(ready_tasks) > 1:
            max_batch = self.slurm_config.get("max_parallel_jobs", 10)

            # v3.2: Separate GPU and CPU tasks to avoid GPU partition saturation
            gpu_ready = [t for t in ready_tasks if t.get('requires_gpu')]
            cpu_ready = [t for t in ready_tasks if not t.get('requires_gpu')]

            # Prioritize: run GPU tasks first (usually fewer, longer-running)
            max_gpu_batch = min(self.slurm_config.get("max_parallel_gpu_jobs", 4), max_batch)
            batch = gpu_ready[:max_gpu_batch] + cpu_ready[:max_batch - min(len(gpu_ready), max_gpu_batch)]
            batch = batch[:max_batch]

            agent_logger.log_workflow_event("parallel_batch", {
                "batch_size": len(batch),
                "gpu_tasks": len([t for t in batch if t.get('requires_gpu')]),
                "cpu_tasks": len([t for t in batch if not t.get('requires_gpu')]),
                "tasks": [t['id'] for t in batch]
            })

            return {
                "parallel_batch": batch,
                "current_subtask": None
            }
        else:
            # Sequential mode
            return {
                "parallel_batch": [],
                "current_subtask": ready_tasks[0]
            }

    def submit_parallel_jobs(self, state: WorkflowState) -> Dict:
        """Submit parallel SLURM jobs (non-blocking)"""
        batch = state['parallel_batch']
        env_name = state['env_name']
        running_jobs = {}

        for subtask in batch:
            task_id = subtask['id']

            # Check reflexion memory before retry
            if self.use_reflexion_memory and subtask.get('context', {}).get('retry_reason'):
                proposed_approach = subtask.get('context', {}).get('retry_reason', '')
                check = check_before_retry(task_id, proposed_approach)

                if not check["allowed"]:
                    logger.warning(
                        f"Task {task_id}: Proposed approach rejected as duplicate "
                        f"(similarity: {check['similarity']:.2f})"
                    )
                    subtask['context'] = subtask.get('context', {})
                    subtask['context']['avoid_approach'] = check['similar_approach']
                    subtask['context']['duplicate_warning'] = True

            # Mark as running in master document
            self.master.master_document.mark_running(task_id)

            # Create sub-agent — passes resolved model, but sub-agent
            # also calls resolve_model() internally so None would be safe too
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
                cleanup_env_on_success=self.cleanup_env_on_success,  # v3.2
            )

            # Execute - handles its own SLURM submission via sbatch
            result = agent.execute(subtask, env_name=env_name)

            if result.get('job_id'):
                running_jobs[task_id] = result['job_id']
            elif result.get('success'):
                # Task completed without needing SLURM (e.g., resumed from checkpoint)
                logger.info(f"Task {task_id} completed immediately (possibly resumed)")

        return {
            "running_jobs": running_jobs,
            "parallel_results": []
        }

    def execute_sequential(self, state: WorkflowState) -> Dict:
        """Execute a single task sequentially with sub-agent v3.2.1 features"""
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

        if not running_jobs:
            return {"parallel_results": []}

        results = []
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

                # Record solution if reflexion enabled
                if self.use_reflexion_memory:
                    try:
                        approach = subtask.get('context', {}).get('approach_tried', '')
                        if approach:
                            record_solution(
                                task_id=task_id,
                                error_message=subtask.get('context', {}).get('previous_error', ''),
                                solution=approach
                            )
                    except Exception as e:
                        logger.warning(f"Failed to record solution: {e}")

            else:
                # Check context status
                context_status = result.get('context_status', {})
                remaining_tokens = context_status.get('remaining_tokens', self.MAX_CONTEXT_TOKENS)

                if remaining_tokens < self.MIN_TOKENS_TO_CONTINUE:
                    subtask['status'] = 'failed'
                    subtask['failure_reason'] = 'context_exhausted'
                    failed.append(subtask)
                    self.master.mark_subtask_failed(
                        task_id,
                        "Context window exhausted",
                        result.get('error', '')
                    )
                else:
                    # Mark for retry
                    subtask['status'] = 'pending'
                    subtask['context'] = subtask.get('context', {})
                    subtask['context']['previous_error'] = result.get('error')
                    subtask['context']['approach_tried'] = result.get('approach', '')

                # v3.2: Check if checkpoint was preserved
                if result.get('checkpoint_preserved'):
                    agent_logger.log_workflow_event("checkpoint_preserved", {
                        "task_id": task_id,
                        "reason": result.get('error', 'unknown')
                    })

        # Update subtasks in state
        updated_subtasks = state['subtasks'].copy()
        for subtask in updated_subtasks:
            for item in results:
                if item['subtask']['id'] == subtask['id']:
                    subtask.update(item['subtask'])

        return {
            "subtasks": updated_subtasks,
            "completed_subtasks": completed,
            "failed_subtasks": failed,
            "task_attempt_counts": task_attempt_counts,
            "parallel_results": []
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
        """
        completed = state.get('completed_subtasks', [])
        failed = state.get('failed_subtasks', [])
        is_deadlocked = state.get('status') == 'deadlocked'

        # v3.2: Collect GPU/CPU routing stats
        all_subtasks = state.get('subtasks', [])
        gpu_tasks = [s for s in all_subtasks if s.get('requires_gpu')]
        cpu_tasks = [s for s in all_subtasks if not s.get('requires_gpu')]

        # v3.2.2: Count blocked tasks (pending with unsatisfied deps)
        blocked_tasks = [s for s in all_subtasks
                         if s.get('status', 'pending') == 'pending']

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

## Summary

- **Total Subtasks**: {len(all_subtasks)}
- **Completed**: {len(completed)}
- **Failed**: {len(failed)}
- **Blocked**: {len(blocked_tasks)}
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
            report += f"### ⊘ {subtask['id']}{gpu_flag} (BLOCKED)\n"
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
        """Route based on execution mode with deadlock detection.

        v3.2.2 Fix D: When status='blocked' (all pending tasks have unsatisfied
        deps), the old code fell through to 'complete' — silently exiting with
        0 completed, 0 failed. Now we:
        1. Attempt deadlock recovery via _break_deadlock()
        2. If recovery unblocks tasks, route to sequential/parallel
        3. If truly deadlocked, report 'deadlocked' status (not 'completed')
        """
        batch = state.get('parallel_batch', [])
        current = state.get('current_subtask')
        status = state.get('status', '')

        if status == 'all_tasks_processed':
            decision = "complete"

        elif status == 'blocked':
            # ── FIX D: Deadlock detection and recovery ────────────────
            print("\n⚠ DEADLOCK DETECTED: All pending tasks have unsatisfied dependencies")
            agent_logger.log_workflow_event("deadlock_detected", {
                "pending_count": len([s for s in state.get('subtasks', [])
                                      if s.get('status', 'pending') == 'pending']),
                "completed_count": len([s for s in state.get('subtasks', [])
                                        if s.get('status') == 'completed']),
                "failed_count": len([s for s in state.get('subtasks', [])
                                     if s.get('status') == 'failed']),
            })

            recovery = self._break_deadlock(state)

            if recovery['unblocked']:
                # Recovery succeeded — route the unblocked tasks
                unblocked = recovery['ready_tasks']
                print(f"  ✓ Deadlock broken: {len(unblocked)} task(s) unblocked")
                agent_logger.log_workflow_event("deadlock_recovered", {
                    "unblocked_tasks": [t['id'] for t in unblocked],
                    "actions_taken": recovery.get('actions', []),
                })

                if self.parallel_enabled and self.use_slurm and len(unblocked) > 1:
                    state['parallel_batch'] = unblocked
                    state['current_subtask'] = None
                    decision = "parallel"
                else:
                    state['current_subtask'] = unblocked[0]
                    state['parallel_batch'] = []
                    decision = "sequential"
            else:
                # True deadlock — cannot recover
                print("  ✗ UNRECOVERABLE DEADLOCK: Cannot break dependency cycle")
                print(f"    Actions attempted: {recovery.get('actions', [])}")
                agent_logger.log_workflow_event("deadlock_unrecoverable", {
                    "actions_attempted": recovery.get('actions', []),
                    "remaining_tasks": recovery.get('remaining_blocked', []),
                })
                # Set status so generate_report knows this was a deadlock
                state['status'] = 'deadlocked'
                decision = "complete"

        elif batch and len(batch) > 1:
            decision = "parallel"
        elif current or (batch and len(batch) == 1):
            if batch and len(batch) == 1:
                state['current_subtask'] = batch[0]
                state['parallel_batch'] = []
            decision = "sequential"
        else:
            decision = "complete"

        # v3.2: Transition logging
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

        Recovery strategies (applied in order):
        1. Strip deps pointing to failed task IDs (those tasks won't complete)
        2. Strip deps pointing to non-existent task IDs (bad LLM output)
        3. Force-complete non-executable tasks (doc sections that slipped
           through Fix A's filter — detected by: no code_hints, no packages,
           title matches documentation patterns)
        4. After each strategy, re-check if any tasks are now ready

        Returns:
            {
                'unblocked': bool,
                'ready_tasks': List[Dict],  # tasks now ready to execute
                'actions': List[str],       # what recovery actions were taken
                'remaining_blocked': List[str],  # task IDs still blocked
            }
        """
        subtasks = state.get('subtasks', [])
        actions = []

        # Build lookup sets
        all_task_ids = {st['id'] for st in subtasks}
        completed_ids = {st['id'] for st in subtasks
                        if st.get('status') == 'completed'}
        failed_ids = {st['id'] for st in subtasks
                      if st.get('status') == 'failed'}
        pending = [st for st in subtasks
                   if st.get('status', 'pending') == 'pending']

        if not pending:
            return {
                'unblocked': False,
                'ready_tasks': [],
                'actions': ['no_pending_tasks'],
                'remaining_blocked': [],
            }

        # ── Strategy 1: Strip deps on failed tasks ────────────────────
        # If a dependency has already failed, waiting on it is pointless.
        # Remove it so downstream tasks can attempt execution.
        stripped_failed = 0
        for st in pending:
            deps = st.get('dependencies', [])
            original_len = len(deps)
            cleaned = [d for d in deps if d not in failed_ids]
            if len(cleaned) < original_len:
                st['dependencies'] = cleaned
                stripped_failed += (original_len - len(cleaned))
                print(f"    → {st['id']}: stripped {original_len - len(cleaned)} "
                      f"failed dep(s): {set(deps) & failed_ids}")

        if stripped_failed > 0:
            actions.append(f"stripped_{stripped_failed}_failed_deps")

        # ── Strategy 2: Strip deps on non-existent task IDs ───────────
        # LLM may have generated deps like "step_15" when only step_1..7 exist.
        stripped_invalid = 0
        for st in pending:
            deps = st.get('dependencies', [])
            original_len = len(deps)
            cleaned = [d for d in deps if d in all_task_ids]
            if len(cleaned) < original_len:
                invalid = set(deps) - all_task_ids
                st['dependencies'] = cleaned
                stripped_invalid += (original_len - len(cleaned))
                print(f"    → {st['id']}: stripped {original_len - len(cleaned)} "
                      f"invalid dep(s): {invalid}")

        if stripped_invalid > 0:
            actions.append(f"stripped_{stripped_invalid}_invalid_deps")

        # Check if strategies 1+2 unblocked anything
        ready = self._find_ready_tasks(subtasks)
        if ready:
            return {
                'unblocked': True,
                'ready_tasks': ready,
                'actions': actions,
                'remaining_blocked': [],
            }

        # ── Strategy 3: Force-complete non-executable tasks ───────────
        # Documentation sections that slipped past Fix A's filter.
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
        for st in list(pending):  # copy list since we modify status
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
        """Route after execution"""
        failed = state.get('failed_subtasks', [])
        subtasks = state.get('subtasks', [])

        # Check for reviewable failures
        for f in failed:
            context_status = f.get('last_result', {}).get('context_status', {})
            if context_status.get('remaining_tokens', 10000) >= self.MIN_TOKENS_TO_CONTINUE:
                decision = "review"
                agent_logger.log_workflow_event("transition", {
                    "from": "handle_results",
                    "to": decision,
                    "reason": f"reviewable failure: {f.get('id')}",
                    "remaining_tokens": context_status.get('remaining_tokens'),
                })
                return decision

        # Check for more pending
        pending = [st for st in subtasks if st.get('status') == 'pending']
        if pending:
            decision = "next_batch"
        else:
            decision = "complete"

        agent_logger.log_workflow_event("transition", {
            "from": "handle_results",
            "to": decision,
            "pending_count": len(pending),
            "failed_count": len(failed),
            "completed_count": len(state.get('completed_subtasks', [])),
        })

        return decision

    def route_after_reflexion(self, state: WorkflowState) -> str:
        """Route after reflexion check"""
        reflexion = state.get('reflexion', {})

        if reflexion.get('should_escalate'):
            decision = "escalate"
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
        """Route after master review"""
        decision = state.get('master_decision', {})
        decision_type = decision.get('decision', 'SKIP')

        if decision_type == 'RETRY':
            route = "retry"
        elif decision_type == 'SKIP':
            route = "skip"
        else:
            route = "escalate"

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

"""
LangGraph workflow that orchestrates the multi-agent system.
Supports parallel subtask execution via SLURM and state management.
Enforces strict attempt limits across workflow invocations.
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from pathlib import Path
import operator
import uuid
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from langgraph.graph import StateGraph, END
try:
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    # Fallback for older langgraph versions
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver as MemorySaver
    except ImportError:
        MemorySaver = None

from agents.master_agent import MasterAgent
from agents.sub_agent import SubAgent
from tools.sandbox import Sandbox
from tools.conda_tools import CondaTools
from tools.slurm_tools import SlurmTools, SlurmConfig
from utils.logging_config import agent_logger
from utils.git_tracker import git_tracker
from utils.documentation import doc_generator


# GLOBAL LIMITS - These are enforced at the workflow level
MAX_ATTEMPTS_PER_SUBTASK = 12  # Hard limit across all invocations
MAX_WORKFLOW_ITERATIONS = 100  # Safety limit to prevent runaway workflows


# Define the state that flows through the workflow
class WorkflowState(TypedDict):
    # Input
    main_task: str
    context: Dict[str, Any]
    project_dir: str
    
    # Task management
    subtasks: List[Dict[str, Any]]
    current_subtask_idx: int
    current_subtask: Dict[str, Any]
    
    # Parallel execution tracking
    parallel_batch: List[Dict[str, Any]]
    parallel_results: List[Dict[str, Any]]
    
    # ATTEMPT TRACKING - Key addition to enforce limits
    subtask_attempts: Dict[str, int]  # {subtask_id: total_attempts}
    workflow_iterations: int  # Total workflow loop iterations
    
    # Environment
    env_name: str
    
    # Results tracking
    completed_subtasks: Annotated[List[Dict], operator.add]
    failed_subtasks: Annotated[List[Dict], operator.add]
    
    # Configuration
    max_retries: int
    use_slurm: bool
    parallel_enabled: bool
    
    # Final output
    final_report: str
    status: str
    
    # Master decisions
    master_decision: Dict[str, Any]


class MultiAgentWorkflow:
    """
    LangGraph workflow for multi-agent task execution.
    Supports both sequential and parallel execution via SLURM.
    Enforces strict attempt limits to prevent infinite loops.
    """
    
    def __init__(
        self,
        ollama_model: str = "llama3.1:70b",
        ollama_base_url: str = "http://127.0.0.1:11434",
        max_retries: int = 12,
        project_dir: str = None,
        use_slurm: bool = False,
        parallel_enabled: bool = True,
        slurm_config: Dict[str, Any] = None
    ):
        """
        Initialize the workflow.
        
        Args:
            ollama_model: Ollama model to use
            ollama_base_url: Ollama server URL
            max_retries: Maximum iterations per subtask (capped at 12)
            project_dir: Project directory for all files
            use_slurm: Whether to use SLURM for job submission
            parallel_enabled: Whether to run independent subtasks in parallel
            slurm_config: SLURM configuration overrides
        """
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        # Enforce the hard limit
        self.max_retries = min(max_retries, MAX_ATTEMPTS_PER_SUBTASK)
        self.project_dir = Path(project_dir) if project_dir else Path.cwd()
        self.use_slurm = use_slurm
        self.parallel_enabled = parallel_enabled
        self.slurm_config = slurm_config or {}
        
        # Initialize sandbox
        self.sandbox = Sandbox(self.project_dir)
        
        # Initialize conda tools
        self.conda_tools = CondaTools(self.project_dir, self.sandbox.get_envs_dir())
        
        # Initialize SLURM tools if enabled
        self.slurm_tools = None
        if use_slurm:
            cluster_name = self.slurm_config.get("cluster")
            self.slurm_tools = SlurmTools(
                self.sandbox,
                cluster_name=cluster_name
            )
            if not self.slurm_tools.slurm_available:
                print("WARNING: SLURM not available, falling back to interactive mode")
                self.use_slurm = False
            else:
                # Set partition if specified
                partition = self.slurm_config.get("partition")
                if partition:
                    print(f"Using cluster: {self.slurm_tools.cluster_name}, partition: {partition}")
        
        # Initialize master agent
        self.master = MasterAgent(
            sandbox=self.sandbox,
            ollama_model=ollama_model,
            ollama_base_url=ollama_base_url
        )
        
        # Thread lock for parallel execution
        self._lock = threading.Lock()
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
        # Add persistence (if available)
        if MemorySaver is not None:
            self.memory = MemorySaver()
            self.app = self.workflow.compile(checkpointer=self.memory)
        else:
            # No checkpointing available
            self.memory = None
            self.app = self.workflow.compile()
    
    def _build_workflow(self) -> StateGraph:
        """Construct the LangGraph workflow"""
        
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("setup_environment", self.setup_environment)
        workflow.add_node("decompose", self.master_decompose)
        workflow.add_node("check_limits", self.check_workflow_limits)
        workflow.add_node("identify_parallel", self.identify_parallel_tasks)
        workflow.add_node("execute_parallel", self.execute_parallel_batch)
        workflow.add_node("execute_sequential", self.execute_sequential)
        workflow.add_node("handle_results", self.handle_results)
        workflow.add_node("master_review", self.master_review)
        workflow.add_node("generate_report", self.generate_final_report)
        workflow.add_node("cleanup", self.cleanup)
        workflow.add_node("emergency_stop", self.emergency_stop)
        
        # Define flow
        workflow.set_entry_point("setup_environment")
        
        # Setup -> Decompose
        workflow.add_edge("setup_environment", "decompose")
        
        # Decompose -> Check limits
        workflow.add_edge("decompose", "check_limits")
        
        # Check limits routes to continue or emergency stop
        workflow.add_conditional_edges(
            "check_limits",
            self.route_after_limit_check,
            {
                "continue": "identify_parallel",
                "stop": "emergency_stop"
            }
        )
        
        # After identifying parallel tasks, route based on parallelism
        workflow.add_conditional_edges(
            "identify_parallel",
            self.route_execution_mode,
            {
                "parallel": "execute_parallel",
                "sequential": "execute_sequential",
                "complete": "generate_report"
            }
        )
        
        # After parallel execution, handle results
        workflow.add_edge("execute_parallel", "handle_results")
        
        # After sequential execution, handle results
        workflow.add_edge("execute_sequential", "handle_results")
        
        # Handle results routes to next action
        workflow.add_conditional_edges(
            "handle_results",
            self.route_after_execution,
            {
                "next_batch": "check_limits",  # Check limits before continuing
                "review": "master_review",
                "complete": "generate_report"
            }
        )
        
        # After master review
        workflow.add_conditional_edges(
            "master_review",
            self.route_after_review,
            {
                "retry": "check_limits",  # Check limits before retrying
                "skip": "check_limits",
                "escalate": "generate_report"
            }
        )
        
        # Generate report then cleanup
        workflow.add_edge("generate_report", "cleanup")
        
        # Cleanup is the end
        workflow.add_edge("cleanup", END)
        
        # Emergency stop also ends
        workflow.add_edge("emergency_stop", END)
        
        return workflow
    
    # ==================== Node Implementations ====================
    
    def setup_environment(self, state: WorkflowState) -> Dict:
        """Setup conda environment for the project"""
        
        # Create project environment
        env_result = self.conda_tools.create_environment(
            env_name=f"project_{datetime.now().strftime('%Y%m%d')}",
            python_version="3.10",
            packages=["pandas", "numpy", "requests"],
            description=f"Environment for: {state['main_task'][:50]}"
        )
        
        env_name = env_result.get("env_name", "agi_project")
        
        # Log cluster status if using SLURM
        cluster_info = ""
        if self.use_slurm and self.slurm_tools:
            status = self.slurm_tools.get_cluster_status()
            if status["success"]:
                cluster_info = f", Cluster: {status.get('idle_count', 0)} idle nodes available"
        
        agent_logger.log_task_start(
            agent_name="setup",
            task_id="environment_setup",
            description=f"Created environment: {env_name}{cluster_info}",
            attempt=1
        )
        
        return {
            "env_name": env_name,
            "project_dir": str(self.project_dir),
            "use_slurm": self.use_slurm,
            "parallel_enabled": self.parallel_enabled,
            "subtask_attempts": {},  # Initialize attempt tracking
            "workflow_iterations": 0
        }
    
    def master_decompose(self, state: WorkflowState) -> Dict:
        """Master agent decomposes the main task"""
        
        subtasks = self.master.decompose_task(
            main_task=state['main_task'],
            context=state.get('context', {})
        )
        
        # Initialize attempt tracking for each subtask
        subtask_attempts = {st['id']: 0 for st in subtasks}
        
        return {
            "subtasks": subtasks,
            "current_subtask_idx": 0,
            "current_subtask": None,
            "parallel_batch": [],
            "parallel_results": [],
            "max_retries": self.max_retries,
            "subtask_attempts": subtask_attempts
        }
    
    def check_workflow_limits(self, state: WorkflowState) -> Dict:
        """Check if workflow has exceeded safety limits"""
        
        workflow_iterations = state.get('workflow_iterations', 0) + 1
        
        agent_logger.log_reflection(
            agent_name="workflow_monitor",
            task_id="limit_check",
            reflection=f"Workflow iteration {workflow_iterations}/{MAX_WORKFLOW_ITERATIONS}"
        )
        
        return {
            "workflow_iterations": workflow_iterations
        }
    
    def identify_parallel_tasks(self, state: WorkflowState) -> Dict:
        """Identify tasks that can be run in parallel"""
        
        subtasks = state['subtasks']
        subtask_attempts = state.get('subtask_attempts', {})
        pending = [st for st in subtasks if st['status'] == 'pending']
        
        if not pending:
            return {
                "parallel_batch": [],
                "status": "all_tasks_processed"
            }
        
        # Find tasks with satisfied dependencies AND within attempt limits
        ready_tasks = []
        for st in pending:
            task_id = st['id']
            
            # Check attempt limit FIRST
            attempts = subtask_attempts.get(task_id, 0)
            if attempts >= self.max_retries:
                # Mark as failed due to max attempts
                st['status'] = 'max_attempts_exceeded'
                agent_logger.log_task_failure(
                    agent_name="workflow",
                    task_id=task_id,
                    error=f"Skipping {task_id}: exceeded max attempts ({attempts}/{self.max_retries})",
                    context={"attempts": attempts}
                )
                continue
            
            # Check dependencies
            deps_satisfied = True
            for dep_id in st.get('dependencies', []):
                dep_task = next((t for t in subtasks if t['id'] == dep_id), None)
                if dep_task and dep_task['status'] not in ['completed', 'skipped', 'max_attempts_exceeded']:
                    deps_satisfied = False
                    break
            
            if deps_satisfied:
                ready_tasks.append(st)
        
        if not ready_tasks:
            # Check if we're blocked or actually done
            still_pending = [st for st in subtasks if st['status'] == 'pending']
            if not still_pending:
                return {
                    "parallel_batch": [],
                    "status": "all_tasks_processed"
                }
            return {
                "parallel_batch": [],
                "status": "blocked"
            }
        
        # If parallel is enabled and we have multiple ready tasks, batch them
        if self.parallel_enabled and len(ready_tasks) > 1:
            max_batch = self.slurm_config.get("max_parallel_jobs", 5)
            batch = ready_tasks[:max_batch]
            
            agent_logger.log_task_start(
                agent_name="workflow",
                task_id="parallel_batch",
                description=f"Executing {len(batch)} tasks in parallel",
                attempt=1
            )
            
            return {
                "subtasks": subtasks,  # Update with any status changes
                "parallel_batch": batch,
                "current_subtask": None
            }
        else:
            return {
                "subtasks": subtasks,
                "parallel_batch": [],
                "current_subtask": ready_tasks[0]
            }
    
    def execute_parallel_batch(self, state: WorkflowState) -> Dict:
        """Execute a batch of tasks in parallel"""
        
        batch = state['parallel_batch']
        env_name = state['env_name']
        subtask_attempts = state.get('subtask_attempts', {})
        results = []
        
        if self.use_slurm and self.slurm_tools:
            results = self._execute_batch_slurm(batch, env_name, subtask_attempts)
        else:
            results = self._execute_batch_threads(batch, env_name, subtask_attempts)
        
        return {
            "parallel_results": results
        }
    
    def _execute_batch_slurm(self, batch: List[Dict], env_name: str, subtask_attempts: Dict) -> List[Dict]:
        """Execute batch using SLURM job submissions"""
        
        results = []
        
        for subtask in batch:
            task_id = subtask['id']
            prior_attempts = subtask_attempts.get(task_id, 0)
            
            agent = SubAgent(
                agent_id=f"agent_{task_id}",
                sandbox=self.sandbox,
                conda_tools=self.conda_tools,
                slurm_tools=self.slurm_tools,
                ollama_model=self.ollama_model,
                ollama_base_url=self.ollama_base_url,
                max_iterations=self.max_retries,
                use_slurm=True,
                slurm_config=self.slurm_config
            )
            
            # Pass prior_attempts to enforce cross-invocation limit
            result = agent.execute(subtask, env_name=env_name, prior_attempts=prior_attempts)
            results.append({
                "subtask": subtask,
                "result": result
            })
        
        return results
    
    def _execute_batch_threads(self, batch: List[Dict], env_name: str, subtask_attempts: Dict) -> List[Dict]:
        """Execute batch using thread pool (for non-SLURM parallel execution)"""
        
        results = []
        max_workers = min(len(batch), self.slurm_config.get("max_parallel_jobs", 4))
        
        def execute_subtask(subtask):
            task_id = subtask['id']
            prior_attempts = subtask_attempts.get(task_id, 0)
            
            agent = SubAgent(
                agent_id=f"agent_{task_id}",
                sandbox=self.sandbox,
                conda_tools=self.conda_tools,
                slurm_tools=self.slurm_tools,
                ollama_model=self.ollama_model,
                ollama_base_url=self.ollama_base_url,
                max_iterations=self.max_retries,
                use_slurm=False,
                slurm_config=self.slurm_config
            )
            return {
                "subtask": subtask,
                "result": agent.execute(subtask, env_name=env_name, prior_attempts=prior_attempts)
            }
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(execute_subtask, st): st for st in batch}
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    subtask = futures[future]
                    results.append({
                        "subtask": subtask,
                        "result": {
                            "success": False,
                            "error": str(e),
                            "task_id": subtask['id'],
                            "total_attempts": subtask_attempts.get(subtask['id'], 0)
                        }
                    })
        
        return results
    
    def execute_sequential(self, state: WorkflowState) -> Dict:
        """Execute a single task sequentially"""
        
        subtask = state['current_subtask']
        env_name = state['env_name']
        subtask_attempts = state.get('subtask_attempts', {})
        
        if not subtask:
            return {"parallel_results": []}
        
        task_id = subtask['id']
        prior_attempts = subtask_attempts.get(task_id, 0)
        
        # Install required packages for this subtask
        required_packages = subtask.get("required_packages", [])
        if required_packages:
            self.conda_tools.install_packages(
                env_name=env_name,
                packages=required_packages
            )
        
        # Create sub-agent
        agent = SubAgent(
            agent_id=f"agent_{task_id}",
            sandbox=self.sandbox,
            conda_tools=self.conda_tools,
            slurm_tools=self.slurm_tools,
            ollama_model=self.ollama_model,
            ollama_base_url=self.ollama_base_url,
            max_iterations=self.max_retries,
            use_slurm=self.use_slurm,
            slurm_config=self.slurm_config
        )
        
        # Pass prior_attempts
        result = agent.execute(subtask, env_name=env_name, prior_attempts=prior_attempts)
        
        return {
            "parallel_results": [{
                "subtask": subtask,
                "result": result
            }]
        }
    
    def handle_results(self, state: WorkflowState) -> Dict:
        """Process results from parallel or sequential execution"""
        
        results = state.get('parallel_results', [])
        subtask_attempts = state.get('subtask_attempts', {}).copy()
        completed = []
        failed = []
        
        for item in results:
            subtask = item['subtask']
            result = item['result']
            task_id = subtask['id']
            
            # Update attempt count
            new_attempts = result.get('total_attempts', subtask_attempts.get(task_id, 0) + 1)
            subtask_attempts[task_id] = new_attempts
            
            # Update subtask status
            subtask['last_result'] = result
            subtask['attempts'] = new_attempts
            
            # Check for should_skip flag (agent exceeded max attempts)
            if result.get('should_skip'):
                subtask['status'] = 'max_attempts_exceeded'
                failed.append(subtask)
                
                agent_logger.log_task_failure(
                    agent_name=f"agent_{task_id}",
                    task_id=task_id,
                    error=f"Max attempts exceeded ({new_attempts})",
                    context={"result": result}
                )
            elif result.get('success'):
                subtask['status'] = 'completed'
                completed.append(subtask)
            else:
                subtask['status'] = 'failed'
                failed.append(subtask)
            
            # Git commit
            git_tracker.commit_task_attempt(
                task_id=task_id,
                agent_name=f"agent_{task_id}",
                description=subtask['description'],
                status="success" if result.get('success') else "failure",
                files_modified=result.get('files_created', []),
                tools_used=result.get('tools_used', []),
                result=str(result.get('report', {}).get('summary', ''))[:500],
                error=str(result.get('errors', [])) if not result.get('success') else None
            )
            
            # Log to documentation
            doc_generator.log_change({
                "task_id": task_id,
                "description": subtask['description'],
                "status": "success" if result.get('success') else "failure",
                "agent": f"agent_{task_id}",
                "tools_used": result.get('tools_used', []),
                "result": result.get('report', {}),
                "files_modified": result.get('files_created', []),
                "attempts": new_attempts,
                "execution_mode": result.get('execution_mode', 'interactive')
            })
            
            # Track in master
            if result.get('success'):
                self.master.mark_subtask_complete(task_id, result.get('report', {}))
            else:
                self.master.mark_subtask_failed(task_id, result.get('report', {}))
        
        # Update subtasks list with new statuses
        updated_subtasks = []
        for st in state['subtasks']:
            matching = next((r for r in results if r['subtask']['id'] == st['id']), None)
            if matching:
                updated_subtasks.append(matching['subtask'])
            else:
                updated_subtasks.append(st)
        
        return {
            "subtasks": updated_subtasks,
            "subtask_attempts": subtask_attempts,
            "completed_subtasks": completed,
            "failed_subtasks": failed,
            "parallel_results": []
        }
    
    def master_review(self, state: WorkflowState) -> Dict:
        """Master agent reviews failed subtasks and decides next steps"""
        
        failed = state.get('failed_subtasks', [])
        subtask_attempts = state.get('subtask_attempts', {})
        
        if not failed:
            return {"master_decision": {"decision": "CONTINUE"}}
        
        # Review the most recent failure
        subtask = failed[-1]
        task_id = subtask['id']
        failure_info = subtask.get('last_result', {})
        total_attempts = subtask_attempts.get(task_id, 0)
        
        # ENFORCE LIMIT: If max attempts reached, force SKIP (not REFORMULATE)
        if total_attempts >= self.max_retries:
            agent_logger.log_reflection(
                agent_name="master",
                task_id=task_id,
                reflection=f"Forcing SKIP: {task_id} exceeded max attempts ({total_attempts}/{self.max_retries})"
            )
            
            for st in state['subtasks']:
                if st['id'] == task_id:
                    st['status'] = 'skipped'
                    break
            
            return {
                "subtasks": state['subtasks'],
                "master_decision": {
                    "decision": "SKIP",
                    "reasoning": f"Maximum attempts ({self.max_retries}) exceeded - cannot reformulate"
                }
            }
        
        # Normal master review
        decision = self.master.review_failure(subtask, failure_info)
        
        # Override REFORMULATE if close to limit
        if decision['decision'] == 'REFORMULATE' and total_attempts >= self.max_retries - 2:
            agent_logger.log_reflection(
                agent_name="master",
                task_id=task_id,
                reflection=f"Overriding REFORMULATE to SKIP: only {self.max_retries - total_attempts} attempts remaining"
            )
            decision = {
                "decision": "SKIP",
                "reasoning": f"Only {self.max_retries - total_attempts} attempts remaining, skipping to prevent infinite loop"
            }
        
        # Handle different decisions
        if decision['decision'] == 'REFORMULATE':
            for st in state['subtasks']:
                if st['id'] == task_id:
                    st['description'] = decision.get('new_approach', st['description'])
                    st['context'] = {
                        **st.get('context', {}),
                        'reformulated': True,
                        'original_description': subtask['description'],
                        'prior_attempts': total_attempts
                    }
                    st['status'] = 'pending'
                    # DO NOT reset attempts - they carry over
                    break
        
        elif decision['decision'] == 'SKIP':
            for st in state['subtasks']:
                if st['id'] == task_id:
                    st['status'] = 'skipped'
                    break
        
        return {
            "subtasks": state['subtasks'],
            "master_decision": decision
        }
    
    def generate_final_report(self, state: WorkflowState) -> Dict:
        """Generate final report and documentation"""
        
        # Generate report from master agent
        report = self.master.generate_final_report(
            main_task=state['main_task'],
            subtask_results=state.get('completed_subtasks', [])
        )
        
        # Add attempt statistics
        subtask_attempts = state.get('subtask_attempts', {})
        attempt_stats = "\n\n## Attempt Statistics\n\n"
        for task_id, attempts in subtask_attempts.items():
            attempt_stats += f"- {task_id}: {attempts}/{self.max_retries} attempts\n"
        report += attempt_stats
        
        # Save README
        doc_generator.save_readme()
        
        # Export final environment state
        self.conda_tools.export_environment(state['env_name'])
        
        # Final commit
        git_tracker.commit_task_attempt(
            task_id="final",
            agent_name="master",
            description="Project completion",
            status="success",
            files_modified=["README.md", "reports/final_report.md"],
            tools_used=["documentation_generator"],
            result=report[:500]
        )
        
        return {
            "final_report": report,
            "status": "completed"
        }
    
    def cleanup(self, state: WorkflowState) -> Dict:
        """Cleanup temporary files and optionally environments"""
        
        # Clean up temp directory
        self.sandbox.cleanup_temp()
        
        # Cancel any remaining SLURM jobs
        if self.slurm_tools:
            self.slurm_tools.cancel_all_jobs()
        
        return {}
    
    def emergency_stop(self, state: WorkflowState) -> Dict:
        """Emergency stop when workflow limits exceeded"""
        
        workflow_iterations = state.get('workflow_iterations', 0)
        
        agent_logger.log_task_failure(
            agent_name="workflow",
            task_id="emergency_stop",
            error=f"Workflow exceeded safety limit: {workflow_iterations} iterations",
            context={
                "subtask_attempts": state.get('subtask_attempts', {}),
                "completed": len(state.get('completed_subtasks', [])),
                "failed": len(state.get('failed_subtasks', []))
            }
        )
        
        # Generate emergency report
        report = f"""# Emergency Stop Report

**Reason**: Workflow exceeded maximum iterations ({workflow_iterations}/{MAX_WORKFLOW_ITERATIONS})

## Summary
- Completed subtasks: {len(state.get('completed_subtasks', []))}
- Failed subtasks: {len(state.get('failed_subtasks', []))}
- Total workflow iterations: {workflow_iterations}

## Attempt Counts
"""
        for task_id, attempts in state.get('subtask_attempts', {}).items():
            report += f"- {task_id}: {attempts} attempts\n"
        
        report += "\n## Recommendation\nReview the workflow logic and subtask definitions to prevent infinite loops."
        
        # Save report
        report_path = self.sandbox.get_reports_dir() / "emergency_stop_report.md"
        report_path.write_text(report)
        
        return {
            "status": "emergency_stopped",
            "final_report": report
        }
    
    # ==================== Routing Functions ====================
    
    def route_after_limit_check(self, state: WorkflowState) -> str:
        """Route based on workflow iteration limit"""
        
        workflow_iterations = state.get('workflow_iterations', 0)
        
        if workflow_iterations >= MAX_WORKFLOW_ITERATIONS:
            return "stop"
        
        return "continue"
    
    def route_execution_mode(self, state: WorkflowState) -> str:
        """Decide whether to execute parallel or sequential"""
        
        batch = state.get('parallel_batch', [])
        current = state.get('current_subtask')
        status = state.get('status', '')
        
        if status == 'all_tasks_processed':
            return "complete"
        
        if batch and len(batch) > 1:
            return "parallel"
        elif current or (batch and len(batch) == 1):
            if batch and len(batch) == 1:
                state['current_subtask'] = batch[0]
                state['parallel_batch'] = []
            return "sequential"
        else:
            return "complete"
    
    def route_after_execution(self, state: WorkflowState) -> str:
        """Decide what to do after execution"""
        
        failed = state.get('failed_subtasks', [])
        subtasks = state.get('subtasks', [])
        subtask_attempts = state.get('subtask_attempts', {})
        
        # Check if any failures need review (and haven't exceeded max attempts)
        for f in failed:
            task_id = f['id']
            attempts = subtask_attempts.get(task_id, 0)
            
            # Only review if under max attempts and task blocks others
            if attempts < self.max_retries:
                dependents = [st for st in subtasks 
                            if task_id in st.get('dependencies', []) 
                            and st['status'] == 'pending']
                if dependents:
                    return "review"
        
        # Check if there are more pending tasks
        pending = [st for st in subtasks if st['status'] == 'pending']
        if pending:
            return "next_batch"
        
        return "complete"
    
    def route_after_review(self, state: WorkflowState) -> str:
        """Route based on master's decision"""
        
        decision = state.get('master_decision', {})
        decision_type = decision.get('decision', 'ESCALATE')
        
        if decision_type in ['REFORMULATE', 'SPLIT', 'CONTINUE']:
            return "retry"
        elif decision_type == 'SKIP':
            return "skip"
        else:
            return "escalate"
    
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
            context: Additional context (input_files, expected_outputs, etc.)
            thread_id: Unique ID for this execution (for state persistence)
        
        Returns:
            Final state with results
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
            "subtask_attempts": {},
            "workflow_iterations": 0,
            "env_name": "",
            "completed_subtasks": [],
            "failed_subtasks": [],
            "max_retries": self.max_retries,
            "use_slurm": self.use_slurm,
            "parallel_enabled": self.parallel_enabled,
            "final_report": "",
            "status": "started",
            "master_decision": {}
        }
        
        # Run workflow with state persistence
        final_state = self.app.invoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}}
        )
        
        return final_state

"""
LangGraph Workflow - Script-First Architecture

Orchestrates multi-agent system with:
- Token-based context limits (70K per agent) instead of iteration counts
- Parallel SLURM job submission for independent tasks
- Script generation and execution paradigm
- Living document master prompt management

Key changes from iteration-based:
- SubAgents generate scripts, submit SLURM jobs, monitor completion
- Context window exhaustion (not iteration count) determines retry limits
- Master document tracks pipeline state across invocations
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
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver as MemorySaver
    except ImportError:
        MemorySaver = None

from agents.master_agent import MasterAgent
from agents.sub_agent import ScriptFirstSubAgent
from tools.sandbox import Sandbox
from tools.conda_tools import CondaTools
from tools.slurm_tools import SlurmTools, SlurmConfig
from utils.logging_config import agent_logger
from utils.git_tracker import git_tracker
from utils.documentation import doc_generator
from utils.context_manager import ContextManager


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


class MultiAgentWorkflow:
    """
    LangGraph workflow with script-first architecture.
    
    Key features:
    - Token-based context limits (70K) instead of iteration counts
    - Parallel SLURM job submission
    - Script generation → SLURM submit → monitor paradigm
    - Living document master prompt
    """
    
    # Token limits
    MAX_CONTEXT_TOKENS = 70_000
    MAX_TOOL_OUTPUT_TOKENS = 25_000
    
    def __init__(
        self,
        ollama_model: str = "llama3.1:70b",
        ollama_base_url: str = "http://127.0.0.1:11434",
        max_retries: int = 12,  # Kept for backward compat, but token-based now
        project_dir: str = None,
        use_slurm: bool = True,
        parallel_enabled: bool = True,
        slurm_config: Dict[str, Any] = None
    ):
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        self.project_dir = Path(project_dir) if project_dir else Path.cwd()
        self.use_slurm = use_slurm
        self.parallel_enabled = parallel_enabled
        self.slurm_config = slurm_config or {}
        
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
        
        # Initialize master agent
        self.master = MasterAgent(
            sandbox=self.sandbox,
            ollama_model=ollama_model,
            ollama_base_url=ollama_base_url,
            project_dir=self.project_dir
        )
        
        # Context manager for overall workflow
        self.context_mgr = ContextManager(
            max_context_tokens=self.MAX_CONTEXT_TOKENS,
            max_tool_output_tokens=self.MAX_TOOL_OUTPUT_TOKENS
        )
        
        # Thread lock for parallel execution
        self._lock = threading.Lock()
        
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
        
        # After handling results
        workflow.add_conditional_edges(
            "handle_results",
            self.route_after_execution,
            {
                "next_batch": "identify_parallel",
                "review": "master_review",
                "complete": "generate_report"
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
            "parallel_enabled": self.parallel_enabled
        })
        
        return {
            "env_name": env_name,
            "project_dir": str(self.project_dir),
            "use_slurm": self.use_slurm,
            "parallel_enabled": self.parallel_enabled,
            "agent_context_status": {},
            "running_jobs": {}
        }
    
    def master_decompose(self, state: WorkflowState) -> Dict:
        """Decompose main task into subtasks"""
        subtasks = self.master.decompose_task(
            main_task=state['main_task'],
            context=state.get('context', {})
        )
        
        return {
            "subtasks": subtasks,
            "current_subtask_idx": 0,
            "current_subtask": None,
            "parallel_batch": [],
            "parallel_results": []
        }
    
    def identify_parallel_tasks(self, state: WorkflowState) -> Dict:
        """Identify tasks that can run in parallel"""
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
            batch = ready_tasks[:max_batch]
            
            agent_logger.log_workflow_event("parallel_batch", {
                "batch_size": len(batch),
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
            
            # Mark as running in master document
            self.master.master_document.mark_running(task_id)
            
            # Create sub-agent
            agent = ScriptFirstSubAgent(
                agent_id=f"agent_{task_id}",
                sandbox=self.sandbox,
                conda_tools=self.conda_tools,
                slurm_tools=self.slurm_tools,
                ollama_model=self.ollama_model,
                ollama_base_url=self.ollama_base_url,
                use_slurm=True,
                slurm_config=self.slurm_config,
                project_root=str(self.project_dir)
            )
            
            # Generate script and submit (don't wait)
            # This is a simplified version - full implementation would
            # separate script generation from job submission
            result = agent.execute(subtask, env_name=env_name)
            
            if result.get('job_id'):
                running_jobs[task_id] = result['job_id']
            
            state['parallel_results'] = state.get('parallel_results', [])
            state['parallel_results'].append({
                'subtask': subtask,
                'result': result
            })
        
        return {
            "running_jobs": running_jobs
        }
    
    def wait_for_parallel_jobs(self, state: WorkflowState) -> Dict:
        """Wait for submitted SLURM jobs to complete"""
        running_jobs = state.get('running_jobs', {})
        
        if not running_jobs or not self.slurm_tools:
            return {}
        
        # Wait for all jobs
        job_ids = list(running_jobs.values())
        wait_result = self.slurm_tools.wait_for_jobs(
            job_ids=job_ids,
            poll_interval=self.slurm_config.get('poll_interval', 10),
            max_attempts=self.slurm_config.get('max_poll_attempts', 720)
        )
        
        # Update results with job status
        parallel_results = state.get('parallel_results', [])
        for item in parallel_results:
            task_id = item['subtask']['id']
            if task_id in running_jobs:
                job_id = running_jobs[task_id]
                job_status = wait_result.get('jobs', {}).get(job_id, {})
                
                # Get job output
                output = self.slurm_tools.get_job_output(job_id)
                
                item['result']['job_status'] = job_status
                item['result']['stdout'] = output.get('stdout', '')
                item['result']['stderr'] = output.get('stderr', '')
                
                # Update success based on final status
                if job_status.get('status') == 'COMPLETED' and job_status.get('exit_code') == '0':
                    item['result']['success'] = True
        
        return {
            "parallel_results": parallel_results,
            "running_jobs": {}
        }
    
    def execute_sequential(self, state: WorkflowState) -> Dict:
        """Execute single task sequentially"""
        subtask = state['current_subtask']
        env_name = state['env_name']
        
        if not subtask:
            return {"parallel_results": []}
        
        task_id = subtask['id']
        
        # Mark as running
        self.master.master_document.mark_running(task_id)
        
        # Create sub-agent
        agent = ScriptFirstSubAgent(
            agent_id=f"agent_{task_id}",
            sandbox=self.sandbox,
            conda_tools=self.conda_tools,
            slurm_tools=self.slurm_tools,
            ollama_model=self.ollama_model,
            ollama_base_url=self.ollama_base_url,
            use_slurm=self.use_slurm,
            slurm_config=self.slurm_config,
            project_root=str(self.project_dir)
        )
        
        # Execute
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
    
    def handle_results(self, state: WorkflowState) -> Dict:
        """Process results from execution"""
        results = state.get('parallel_results', [])
        completed = []
        failed = []
        
        for item in results:
            subtask = item['subtask']
            result = item['result']
            task_id = subtask['id']
            
            # Update subtask status
            subtask['last_result'] = result
            
            if result.get('success'):
                subtask['status'] = 'completed'
                completed.append(subtask)
                
                # Update master document
                self.master.mark_subtask_complete(task_id, result)
                
                # Git commit
                git_tracker.commit_task_attempt(
                    task_id=task_id,
                    agent_name=f"agent_{task_id}",
                    description=subtask.get('description', '')[:100],
                    status="success",
                    files_modified=[result.get('script_path', '')] + result.get('output_files', []),
                    tools_used=['script_generator', 'slurm' if self.use_slurm else 'local'],
                    result=f"Script: {result.get('script_path')}"
                )
                
                # Log to documentation
                doc_generator.log_change({
                    "task_id": task_id,
                    "description": subtask.get('description', ''),
                    "status": "success",
                    "script_path": result.get('script_path'),
                    "output_files": result.get('output_files', []),
                    "conda_env_yaml": result.get('env_yaml')
                })
            else:
                subtask['status'] = 'failed'
                failed.append(subtask)
                
                # Update master document
                self.master.mark_subtask_failed(task_id, result)
                
                # Log to documentation
                doc_generator.log_change({
                    "task_id": task_id,
                    "description": subtask.get('description', ''),
                    "status": "failure",
                    "error": result.get('error', 'Unknown'),
                    "script_path": result.get('script_path'),
                    "context_status": result.get('context_status', {})
                })
        
        # Update subtasks list
        updated_subtasks = []
        for st in state['subtasks']:
            matching = next((r for r in results if r['subtask']['id'] == st['id']), None)
            if matching:
                updated_subtasks.append(matching['subtask'])
            else:
                updated_subtasks.append(st)
        
        return {
            "subtasks": updated_subtasks,
            "completed_subtasks": completed,
            "failed_subtasks": failed,
            "parallel_results": []
        }
    
    def master_review(self, state: WorkflowState) -> Dict:
        """Master reviews failures"""
        failed = state.get('failed_subtasks', [])
        
        if not failed:
            return {"master_decision": {"decision": "CONTINUE"}}
        
        subtask = failed[-1]
        result = subtask.get('last_result', {})
        
        # Check context status - if exhausted, force skip
        context_status = result.get('context_status', {})
        remaining_tokens = context_status.get('remaining_tokens', 10000)
        
        if remaining_tokens < 5000:
            agent_logger.log_reflection(
                agent_name="master",
                task_id=subtask['id'],
                reflection=f"Forcing SKIP: context exhausted ({remaining_tokens} tokens remaining)"
            )
            
            subtask['status'] = 'skipped'
            for st in state['subtasks']:
                if st['id'] == subtask['id']:
                    st['status'] = 'skipped'
            
            return {
                "subtasks": state['subtasks'],
                "master_decision": {
                    "decision": "SKIP",
                    "reasoning": f"Context window exhausted ({remaining_tokens} tokens)"
                }
            }
        
        # Normal master review
        decision = self.master.review_failure(subtask, result)
        
        if decision['decision'] == 'RETRY':
            # Update subtask for retry
            for st in state['subtasks']:
                if st['id'] == subtask['id']:
                    st['status'] = 'pending'
                    st['context'] = {
                        **st.get('context', {}),
                        'retry_reason': decision.get('modification', ''),
                        'previous_error': result.get('error', '')
                    }
        
        elif decision['decision'] == 'SKIP':
            for st in state['subtasks']:
                if st['id'] == subtask['id']:
                    st['status'] = 'skipped'
        
        return {
            "subtasks": state['subtasks'],
            "master_decision": decision
        }
    
    def generate_final_report(self, state: WorkflowState) -> Dict:
        """Generate final report"""
        report = self.master.generate_final_report(
            main_task=state['main_task'],
            subtask_results=state.get('completed_subtasks', [])
        )
        
        # Add context usage summary
        context_summary = "\n\n## Context Usage Summary\n\n"
        for agent_id, status in state.get('agent_context_status', {}).items():
            if status:
                usage_pct = status.get('usage_percent', 0)
                context_summary += f"- {agent_id}: {usage_pct:.1f}% context used\n"
        
        report += context_summary
        
        # Save documentation
        doc_generator.save_readme()
        
        # Final git commit
        git_tracker.commit_task_attempt(
            task_id="final",
            agent_name="master",
            description="Pipeline completion",
            status="success",
            files_modified=["reports/final_report.md", "reports/pipeline_status.md"],
            tools_used=["documentation_generator"],
            result=report[:500]
        )
        
        return {
            "final_report": report,
            "status": "completed"
        }
    
    def cleanup(self, state: WorkflowState) -> Dict:
        """Cleanup temporary files"""
        self.sandbox.cleanup_temp()
        
        # Cancel any lingering jobs
        if self.slurm_tools:
            self.slurm_tools.cancel_all_jobs()
        
        return {}
    
    # ==================== Routing Functions ====================
    
    def route_execution_mode(self, state: WorkflowState) -> str:
        """Route based on execution mode"""
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
        """Route after execution"""
        failed = state.get('failed_subtasks', [])
        subtasks = state.get('subtasks', [])
        
        # Check for reviewable failures
        for f in failed:
            # Only review if context not exhausted
            context_status = f.get('last_result', {}).get('context_status', {})
            if context_status.get('remaining_tokens', 10000) >= 5000:
                return "review"
        
        # Check for more pending
        pending = [st for st in subtasks if st.get('status') == 'pending']
        if pending:
            return "next_batch"
        
        return "complete"
    
    def route_after_review(self, state: WorkflowState) -> str:
        """Route after master review"""
        decision = state.get('master_decision', {})
        decision_type = decision.get('decision', 'SKIP')
        
        if decision_type == 'RETRY':
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
            "master_decision": {}
        }
        
        final_state = self.app.invoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}}
        )
        
        return final_state

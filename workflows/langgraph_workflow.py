"""
LangGraph workflow that orchestrates the multi-agent system.
Implements Ralph loops, state management, and conditional routing.
"""

from typing import TypedDict, Annotated, List, Dict, Any
import operator
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from agents.master_agent import MasterAgent
from agents.sub_agent import SubAgent
from utils.logging_config import agent_logger
from utils.git_tracker import git_tracker
from utils.documentation import doc_generator

# Define the state that flows through the workflow
class WorkflowState(TypedDict):
    # Input
    main_task: str
    context: Dict[str, Any]
    
    # Task management
    subtasks: List[Dict[str, Any]]
    current_subtask_idx: int
    current_subtask: Dict[str, Any]
    
    # Results tracking
    completed_subtasks: Annotated[List[Dict], operator.add]
    failed_subtasks: Annotated[List[Dict], operator.add]
    
    # Configuration
    max_retries: int
    current_retry: int
    
    # Final output
    final_report: str
    status: str

class MultiAgentWorkflow:
    """LangGraph workflow for multi-agent task execution"""
    
    def __init__(self, ollama_model: str = "llama3.1:70b", max_retries: int = 3):
        self.master = MasterAgent(ollama_model=ollama_model)
        self.max_retries = max_retries
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
        # Add persistence
        self.memory = SqliteSaver.from_conn_string("workflow_state.db")
        self.app = self.workflow.compile(checkpointer=self.memory)
    
    def _build_workflow(self) -> StateGraph:
        """Construct the LangGraph workflow"""
        
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("decompose", self.master_decompose)
        workflow.add_node("execute_subtask", self.execute_subtask)
        workflow.add_node("reflect_and_retry", self.reflect_and_retry)
        workflow.add_node("master_review", self.master_review)
        workflow.add_node("generate_report", self.generate_final_report)
        workflow.add_node("commit_changes", self.commit_to_git)
        
        # Define flow
        workflow.set_entry_point("decompose")
        
        # After decomposition, start executing subtasks
        workflow.add_edge("decompose", "execute_subtask")
        
        # After execution, route based on success/failure
        workflow.add_conditional_edges(
            "execute_subtask",
            self.route_after_execution,
            {
                "success": "commit_changes",
                "retry": "reflect_and_retry",
                "review": "master_review",
                "next_task": "execute_subtask",
                "complete": "generate_report"
            }
        )
        
        # After reflection, try again
        workflow.add_edge("reflect_and_retry", "execute_subtask")
        
        # After master review, route based on decision
        workflow.add_conditional_edges(
            "master_review",
            self.route_after_review,
            {
                "retry": "execute_subtask",
                "skip": "execute_subtask",
                "escalate": "generate_report"
            }
        )
        
        # After successful commit, continue
        workflow.add_conditional_edges(
            "commit_changes",
            self.check_more_tasks,
            {
                "continue": "execute_subtask",
                "done": "generate_report"
            }
        )
        
        # Report generation is the end
        workflow.add_edge("generate_report", END)
        
        return workflow
    
    # Node implementations
    
    def master_decompose(self, state: WorkflowState) -> Dict:
        """Master agent decomposes the main task"""
        
        subtasks = self.master.decompose_task(
            main_task=state['main_task'],
            context=state.get('context', {})
        )
        
        return {
            "subtasks": subtasks,
            "current_subtask_idx": 0,
            "current_subtask": subtasks[0] if subtasks else None,
            "max_retries": self.max_retries,
            "current_retry": 0
        }
    
    def execute_subtask(self, state: WorkflowState) -> Dict:
        """Execute current subtask with a sub-agent"""
        
        subtask = state['current_subtask']
        if not subtask:
            return {"status": "no_subtask"}
        
        # Create sub-agent for this task
        agent = SubAgent(
            agent_id=f"agent_{subtask['id']}",
            ollama_model="llama3.1:70b"
        )
        
        # Execute
        result = agent.execute(subtask, attempt=state['current_retry'] + 1)
        
        # Update subtask with result
        updated_subtask = {
            **subtask,
            "attempts": state['current_retry'] + 1,
            "last_result": result,
            "status": "completed" if result['success'] else "failed"
        }
        
        return {
            "current_subtask": updated_subtask,
            "current_retry": state['current_retry'] + 1 if not result['success'] else 0
        }
    
    def reflect_and_retry(self, state: WorkflowState) -> Dict:
        """Analyze failure and prepare for retry"""
        
        subtask = state['current_subtask']
        
        # Log reflection
        reflection = subtask.get('last_result', {}).get('reflection', {})
        
        agent_logger.log_reflection(
            agent_name="reflection_agent",
            task_id=subtask['id'],
            reflection=reflection.get('improvement_strategy', 'No strategy provided')
        )
        
        # Apply improvement strategy to subtask
        improved_subtask = {
            **subtask,
            "context": {
                **subtask.get('context', {}),
                "previous_attempt": subtask.get('last_result'),
                "improvement_strategy": reflection.get('improvement_strategy', '')
            }
        }
        
        return {
            "current_subtask": improved_subtask
        }
    
    def master_review(self, state: WorkflowState) -> Dict:
        """Master agent reviews failed subtask and decides next steps"""
        
        subtask = state['current_subtask']
        failure_info = subtask.get('last_result', {})
        
        decision = self.master.review_failure(subtask, failure_info)
        
        # Handle different decisions
        if decision['decision'] == 'REFORMULATE':
            # Update subtask with new approach
            reformulated = {
                **subtask,
                "description": decision.get('new_approach', subtask['description']),
                "context": {
                    **subtask.get('context', {}),
                    "reformulated": True,
                    "original_description": subtask['description']
                },
                "attempts": 0  # Reset attempts
            }
            return {
                "current_subtask": reformulated,
                "current_retry": 0,
                "master_decision": decision
            }
        
        elif decision['decision'] == 'SPLIT':
            # Would need to implement subtask splitting
            # For now, treat as reformulate
            return {
                "master_decision": decision,
                "current_retry": 0
            }
        
        elif decision['decision'] == 'SKIP':
            # Mark as skipped and move to next
            return {
                "failed_subtasks": [{**subtask, "status": "skipped"}],
                "master_decision": decision,
                "current_retry": 0
            }
        
        else:  # ESCALATE
            return {
                "failed_subtasks": [{**subtask, "status": "escalated"}],
                "master_decision": decision,
                "status": "escalated"
            }
    
    def commit_to_git(self, state: WorkflowState) -> Dict:
        """Commit successful subtask to Git"""
        
        subtask = state['current_subtask']
        result = subtask.get('last_result', {})
        
        # Commit to git
        git_tracker.commit_task_attempt(
            task_id=subtask['id'],
            agent_name=f"agent_{subtask['id']}",
            description=subtask['description'],
            status="success",
            files_modified=[],  # Would track actual files
            tools_used=result.get('tools_used', []),
            result=str(result.get('result', {})),
            reasoning=result.get('reflection', {}).get('analysis', '')
        )
        
        # Log to documentation
        doc_generator.log_change({
            "task_id": subtask['id'],
            "description": subtask['description'],
            "status": "success",
            "agent": f"agent_{subtask['id']}",
            "tools_used": result.get('tools_used', []),
            "result": result.get('result', {}),
            "files_modified": []
        })
        
        return {
            "completed_subtasks": [subtask]
        }
    
    def generate_final_report(self, state: WorkflowState) -> Dict:
        """Generate final report and documentation"""
        
        # Generate report from master agent
        report = self.master.generate_final_report(
            main_task=state['main_task'],
            subtask_results=state.get('completed_subtasks', [])
        )
        
        # Save README
        doc_generator.save_readme()
        
        # Final commit
        git_tracker.commit_task_attempt(
            task_id="final",
            agent_name="master",
            description="Project completion",
            status="success",
            files_modified=["README.md"],
            tools_used=["documentation_generator"],
            result=report
        )
        
        return {
            "final_report": report,
            "status": "completed"
        }
    
    # Routing functions
    
    def route_after_execution(self, state: WorkflowState) -> str:
        """Decide what to do after subtask execution"""
        
        subtask = state['current_subtask']
        result = subtask.get('last_result', {})
        
        if result.get('success'):
            return "success"
        
        # Failed - check if we should retry
        if state['current_retry'] < state['max_retries']:
            if result.get('should_retry', True):
                return "retry"
        
        # Max retries reached - escalate to master
        return "review"
    
    def route_after_review(self, state: WorkflowState) -> str:
        """Route based on master's decision"""
        
        decision = state.get('master_decision', {})
        decision_type = decision.get('decision', 'ESCALATE')
        
        if decision_type in ['REFORMULATE', 'SPLIT']:
            return "retry"
        elif decision_type == 'SKIP':
            return "skip"
        else:
            return "escalate"
    
    def check_more_tasks(self, state: WorkflowState) -> str:
        """Check if there are more subtasks to execute"""
        
        current_idx = state['current_subtask_idx']
        total_subtasks = len(state['subtasks'])
        
        if current_idx + 1 < total_subtasks:
            # Update to next subtask
            state['current_subtask_idx'] = current_idx + 1
            state['current_subtask'] = state['subtasks'][current_idx + 1]
            state['current_retry'] = 0
            return "continue"
        else:
            return "done"
    
    def run(self, main_task: str, context: Dict[str, Any] = None, thread_id: str = None) -> Dict:
        """
        Execute the workflow
        
        Args:
            main_task: High-level task description
            context: Additional context
            thread_id: Unique ID for this execution (for state persistence)
        
        Returns:
            Final state with results
        """
        
        if thread_id is None:
            import uuid
            thread_id = str(uuid.uuid4())
        
        initial_state = {
            "main_task": main_task,
            "context": context or {},
            "subtasks": [],
            "current_subtask_idx": 0,
            "current_subtask": None,
            "completed_subtasks": [],
            "failed_subtasks": [],
            "max_retries": self.max_retries,
            "current_retry": 0,
            "final_report": "",
            "status": "started"
        }
        
        # Run workflow with state persistence
        final_state = self.app.invoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}}
        )
        
        return final_state

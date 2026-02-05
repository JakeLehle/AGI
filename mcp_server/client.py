"""
Memory Client for AGI Agents.

A simple synchronous client for interacting with the MCP Memory Server.
This can be used directly by agents in the LangGraph workflow.

Usage:
    from mcp_server.client import MemoryClient
    
    client = MemoryClient()
    
    # Check if an approach was tried
    result = client.check_if_tried("task_001", "Install pandas via pip")
    if result["tried"]:
        print(f"Already tried something similar: {result['similar_approach']}")
    
    # Store a failure
    client.store_failure(
        task_id="task_001",
        error_type="missing_package",
        error_message="ModuleNotFoundError: No module named 'pandas'",
        approach_tried="Added import without installing"
    )
    
    # Get a decision
    decision = client.reflect_on_failure(
        task_id="task_001",
        error_message="ModuleNotFoundError: No module named 'pandas'",
        proposed_approach="pip install pandas"
    )
    print(f"Action: {decision['action']}")
"""

import json
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class MemoryClient:
    """
    Synchronous client for the MCP Memory Server.
    
    Can operate in two modes:
    1. HTTP mode: Connects to running MCP server
    2. Direct mode: Uses ReflexionMemory directly (no server needed)
    """
    
    def __init__(
        self,
        server_url: Optional[str] = None,
        use_direct: bool = True,
    ):
        """
        Initialize the memory client.
        
        Args:
            server_url: URL of MCP server (e.g., http://127.0.0.1:8765)
            use_direct: If True and no server_url, use memory directly
        """
        self.server_url = server_url
        self.use_direct = use_direct and server_url is None
        
        if self.use_direct:
            # Import and use memory directly
            from memory import ReflexionMemory
            from engines import ReflexionEngine
            
            self._memory = ReflexionMemory()
            self._engine = ReflexionEngine(memory=self._memory)
            logger.info("MemoryClient initialized in direct mode")
        else:
            self._memory = None
            self._engine = None
            logger.info(f"MemoryClient initialized for server: {server_url}")
    
    def _call_server(self, tool_name: str, arguments: Dict) -> Dict[str, Any]:
        """Call a tool on the MCP server."""
        import requests
        
        request = {
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }
        
        response = requests.post(f"{self.server_url}/mcp", json=request)
        response.raise_for_status()
        
        data = response.json()
        if "error" in data:
            raise RuntimeError(f"MCP error: {data['error']}")
        
        # Parse the result
        content_text = data["result"]["content"][0]["text"]
        return json.loads(content_text)
    
    # =========================================================================
    # Main Methods
    # =========================================================================
    
    def check_if_tried(
        self,
        task_id: str,
        proposed_approach: str,
        threshold: float = 0.85,
    ) -> Dict[str, Any]:
        """
        Check if a proposed approach was already attempted.
        
        Args:
            task_id: Unique task identifier
            proposed_approach: The approach to check
            threshold: Similarity threshold
            
        Returns:
            {"tried": bool, "similarity": float, "similar_approach": str, ...}
        """
        if self.use_direct:
            return self._memory.check_if_tried(
                task_id=task_id,
                proposed_approach=proposed_approach,
                threshold=threshold,
            )
        else:
            return self._call_server("check_if_tried", {
                "task_id": task_id,
                "proposed_approach": proposed_approach,
                "threshold": threshold,
            })
    
    def store_failure(
        self,
        task_id: str,
        error_type: str,
        error_message: str,
        approach_tried: str,
        script_path: Optional[str] = None,
        slurm_job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Record a failed attempt.
        
        Args:
            task_id: Task identifier
            error_type: Type of error (e.g., "missing_package")
            error_message: The error that occurred
            approach_tried: What was attempted
            script_path: Path to script (optional)
            slurm_job_id: SLURM job ID (optional)
            
        Returns:
            Storage result
        """
        if self.use_direct:
            from memory import FailureType
            try:
                ft = FailureType(error_type)
            except ValueError:
                ft = FailureType.UNKNOWN
            
            return self._memory.store_failure(
                task_id=task_id,
                error_type=ft,
                error_message=error_message,
                approach_tried=approach_tried,
                script_path=script_path,
                slurm_job_id=slurm_job_id,
            )
        else:
            return self._call_server("store_failure", {
                "task_id": task_id,
                "error_type": error_type,
                "error_message": error_message,
                "approach_tried": approach_tried,
                "script_path": script_path,
                "slurm_job_id": slurm_job_id,
            })
    
    def store_solution(
        self,
        task_id: str,
        problem_pattern: str,
        error_type: str,
        solution: str,
    ) -> Dict[str, Any]:
        """
        Record a working solution.
        
        Args:
            task_id: Task that was solved
            problem_pattern: Description of the problem
            error_type: Type of error
            solution: What fixed it
            
        Returns:
            Storage result
        """
        if self.use_direct:
            from memory import FailureType
            try:
                ft = FailureType(error_type)
            except ValueError:
                ft = FailureType.UNKNOWN
            
            return self._memory.store_solution(
                task_id=task_id,
                problem_pattern=problem_pattern,
                error_type=ft,
                solution=solution,
            )
        else:
            return self._call_server("store_solution", {
                "task_id": task_id,
                "problem_pattern": problem_pattern,
                "error_type": error_type,
                "solution": solution,
            })
    
    def reflect_on_failure(
        self,
        task_id: str,
        error_message: str,
        proposed_approach: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get a decision from the Reflexion Engine.
        
        Args:
            task_id: Task that failed
            error_message: The error that occurred
            proposed_approach: What to try next (optional)
            
        Returns:
            Decision with action, reason, etc.
        """
        if self.use_direct:
            decision = self._engine.reflect_on_failure(
                task_id=task_id,
                error_message=error_message,
                proposed_approach=proposed_approach,
            )
            return {
                "action": decision.action.value,
                "reason": decision.reason,
                "failure_type": decision.failure_type.value if decision.failure_type else None,
                "similar_approach": decision.similar_approach,
                "similarity_score": decision.similarity_score,
                "known_solution": decision.known_solution,
                "attempt_count": decision.attempt_count,
                "escalation_target": decision.escalation_target,
            }
        else:
            return self._call_server("reflect_on_failure", {
                "task_id": task_id,
                "error_message": error_message,
                "proposed_approach": proposed_approach,
            })
    
    def get_tried_approaches(self, task_id: str) -> List[Dict]:
        """Get all approaches tried for a task."""
        if self.use_direct:
            return self._memory.get_tried_approaches(task_id)
        else:
            result = self._call_server("get_tried_approaches", {"task_id": task_id})
            return result.get("approaches", [])
    
    def get_working_solutions(
        self,
        error_message: str,
        error_type: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict]:
        """Find solutions for similar problems."""
        if self.use_direct:
            from memory import FailureType
            ft = None
            if error_type:
                try:
                    ft = FailureType(error_type)
                except ValueError:
                    pass
            
            # Note: ReflexionMemory uses 'problem_description' as the parameter name
            return self._memory.get_working_solutions(
                problem_description=error_message,
                error_type=ft,
                limit=limit,
            )
        else:
            result = self._call_server("get_working_solutions", {
                "error_message": error_message,
                "error_type": error_type,
                "limit": limit,
            })
            return result.get("solutions", [])
    
    def classify_error(self, error_message: str) -> Dict[str, Any]:
        """Classify an error message."""
        if self.use_direct:
            from engines import AGENT_ROUTING, ESCALATION_THRESHOLDS
            
            failure_type = self._engine.classify_error(error_message)
            return {
                "error_type": failure_type.value,
                "recommended_agent": AGENT_ROUTING.get(failure_type, "developer"),
                "escalation_threshold": ESCALATION_THRESHOLDS.get(failure_type, 3),
            }
        else:
            return self._call_server("classify_error", {"error_message": error_message})
    
    def get_task_summary(self, task_id: str) -> Dict[str, Any]:
        """Get summary of attempts for a task."""
        if self.use_direct:
            return self._engine.get_task_summary(task_id)
        else:
            return self._call_server("get_task_summary", {"task_id": task_id})

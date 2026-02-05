#!/usr/bin/env python3
"""
MCP Server for AGI Reflexion Memory.

This server exposes the Reflexion Memory system via the Model Context Protocol (MCP),
allowing agents to query past failures, check if approaches were tried, and retrieve
working solutions during task execution.

Available Tools:
    - check_if_tried: Check if a proposed approach was already attempted
    - store_failure: Record a failed attempt
    - store_solution: Record a working solution
    - get_tried_approaches: List all approaches tried for a task
    - get_working_solutions: Find solutions for similar problems
    - get_task_summary: Get summary of attempts for a task
    - classify_error: Classify an error message into a failure type

Usage:
    # Start the server
    python -m mcp_server.memory_server
    
    # Or run directly
    python mcp_server/memory_server.py

Configuration:
    Environment variables:
    - OLLAMA_BASE_URL: Ollama server URL (default: http://localhost:11434)
    - AGI_DATA_DIR: Data directory for Qdrant storage (default: ~/agi_data)
    - MCP_SERVER_PORT: Port to run server on (default: 8765)
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory import ReflexionMemory, FailureType
from engines import ReflexionEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MemoryMCPServer:
    """
    MCP Server wrapper for the Reflexion Memory system.
    
    Provides a JSON-RPC style interface for memory operations.
    """
    
    def __init__(self):
        """Initialize the MCP server with memory and engine."""
        logger.info("Initializing Memory MCP Server...")
        
        self.memory = ReflexionMemory()
        self.engine = ReflexionEngine(memory=self.memory)
        
        # Register available tools
        self.tools = {
            "check_if_tried": self._check_if_tried,
            "store_failure": self._store_failure,
            "store_solution": self._store_solution,
            "get_tried_approaches": self._get_tried_approaches,
            "get_working_solutions": self._get_working_solutions,
            "get_task_summary": self._get_task_summary,
            "classify_error": self._classify_error,
            "reflect_on_failure": self._reflect_on_failure,
            "get_memory_stats": self._get_memory_stats,
        }
        
        logger.info(f"Registered {len(self.tools)} tools")
    
    # =========================================================================
    # Tool Implementations
    # =========================================================================
    
    async def _check_if_tried(
        self,
        task_id: str,
        proposed_approach: str,
        threshold: float = 0.85,
    ) -> Dict[str, Any]:
        """
        Check if a proposed approach was already attempted.
        
        This is the KEY method for preventing infinite loops.
        
        Args:
            task_id: Unique task identifier
            proposed_approach: The approach the agent wants to try
            threshold: Similarity threshold (0.0 - 1.0)
            
        Returns:
            {
                "tried": bool,
                "similar_approach": str or None,
                "similarity": float,
                "attempt_count": int
            }
        """
        result = self.memory.check_if_tried(
            task_id=task_id,
            proposed_approach=proposed_approach,
            threshold=threshold,
        )
        return result
    
    async def _store_failure(
        self,
        task_id: str,
        error_type: str,
        error_message: str,
        approach_tried: str,
        script_path: Optional[str] = None,
        slurm_job_id: Optional[str] = None,
        diagnostic_results: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Record a failed attempt in memory.
        
        Args:
            task_id: Unique task identifier
            error_type: Type of error (from FailureType enum)
            error_message: The error that occurred
            approach_tried: What was attempted
            script_path: Path to the script (optional)
            slurm_job_id: SLURM job ID (optional)
            diagnostic_results: Diagnostic output (optional)
            
        Returns:
            Storage result with memory ID
        """
        # Convert string to FailureType
        try:
            failure_type = FailureType(error_type)
        except ValueError:
            failure_type = FailureType.UNKNOWN
        
        result = self.memory.store_failure(
            task_id=task_id,
            error_type=failure_type,
            error_message=error_message,
            approach_tried=approach_tried,
            script_path=script_path,
            slurm_job_id=slurm_job_id,
            diagnostic_results=diagnostic_results,
        )
        
        return {"success": True, "result": result}
    
    async def _store_solution(
        self,
        task_id: str,
        problem_pattern: str,
        error_type: str,
        solution: str,
        context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Record a working solution for future reuse.
        
        Args:
            task_id: Task that was solved
            problem_pattern: Description of the problem
            error_type: Type of error that was solved
            solution: What fixed it
            context: Additional context (optional)
            
        Returns:
            Storage result with memory ID
        """
        try:
            failure_type = FailureType(error_type)
        except ValueError:
            failure_type = FailureType.UNKNOWN
        
        result = self.memory.store_solution(
            task_id=task_id,
            problem_pattern=problem_pattern,
            error_type=failure_type,
            solution=solution,
            context=context,
        )
        
        return {"success": True, "result": result}
    
    async def _get_tried_approaches(
        self,
        task_id: str,
    ) -> Dict[str, Any]:
        """
        Get all approaches tried for a task.
        
        Args:
            task_id: Task to query
            
        Returns:
            List of approach records
        """
        approaches = self.memory.get_tried_approaches(task_id)
        return {
            "task_id": task_id,
            "count": len(approaches),
            "approaches": approaches,
        }
    
    async def _get_working_solutions(
        self,
        error_message: str,
        error_type: Optional[str] = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """
        Find working solutions for similar problems.
        
        Args:
            error_message: The error to find solutions for
            error_type: Type of error (optional, for filtering)
            limit: Maximum results to return
            
        Returns:
            List of solution records with similarity scores
        """
        failure_type = None
        if error_type:
            try:
                failure_type = FailureType(error_type)
            except ValueError:
                pass
        
        solutions = self.memory.get_working_solutions(
            error_message=error_message,
            error_type=failure_type,
            limit=limit,
        )
        
        return {
            "count": len(solutions),
            "solutions": solutions,
        }
    
    async def _get_task_summary(
        self,
        task_id: str,
    ) -> Dict[str, Any]:
        """
        Get a summary of all attempts for a task.
        
        Args:
            task_id: Task to summarize
            
        Returns:
            Summary with counts, types, and approaches
        """
        return self.engine.get_task_summary(task_id)
    
    async def _classify_error(
        self,
        error_message: str,
    ) -> Dict[str, Any]:
        """
        Classify an error message into a failure type.
        
        Args:
            error_message: The error to classify
            
        Returns:
            Classification result with type and routing info
        """
        failure_type = self.engine.classify_error(error_message)
        
        # Get routing info
        from engines import AGENT_ROUTING, ESCALATION_THRESHOLDS
        
        return {
            "error_type": failure_type.value,
            "recommended_agent": AGENT_ROUTING.get(failure_type, "developer"),
            "escalation_threshold": ESCALATION_THRESHOLDS.get(failure_type, 3),
        }
    
    async def _reflect_on_failure(
        self,
        task_id: str,
        error_message: str,
        proposed_approach: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get a decision from the Reflexion Engine.
        
        This combines classification, history lookup, and decision making.
        
        Args:
            task_id: Task that failed
            error_message: The error that occurred
            proposed_approach: What the agent wants to try (optional)
            
        Returns:
            ReflexionDecision as dict
        """
        decision = self.engine.reflect_on_failure(
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
            "solution_confidence": decision.solution_confidence,
            "attempt_count": decision.attempt_count,
            "escalation_target": decision.escalation_target,
            "recommended_diagnostics": decision.recommended_diagnostics,
        }
    
    async def _get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory system statistics.
        
        Returns:
            Stats including counts, config, thresholds
        """
        return self.engine.get_stats()
    
    # =========================================================================
    # Server Methods
    # =========================================================================
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Get MCP tool definitions for registration.
        
        Returns list of tool schemas for the MCP protocol.
        """
        return [
            {
                "name": "check_if_tried",
                "description": "Check if a proposed approach was already attempted for a task. KEY for preventing infinite loops.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "Unique task identifier"},
                        "proposed_approach": {"type": "string", "description": "The approach to check"},
                        "threshold": {"type": "number", "description": "Similarity threshold (0-1)", "default": 0.85},
                    },
                    "required": ["task_id", "proposed_approach"],
                },
            },
            {
                "name": "store_failure",
                "description": "Record a failed attempt in memory for future reference.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "error_type": {"type": "string", "enum": [t.value for t in FailureType]},
                        "error_message": {"type": "string"},
                        "approach_tried": {"type": "string"},
                        "script_path": {"type": "string"},
                        "slurm_job_id": {"type": "string"},
                    },
                    "required": ["task_id", "error_type", "error_message", "approach_tried"],
                },
            },
            {
                "name": "store_solution",
                "description": "Record a working solution for future reuse by other tasks.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "problem_pattern": {"type": "string"},
                        "error_type": {"type": "string", "enum": [t.value for t in FailureType]},
                        "solution": {"type": "string"},
                    },
                    "required": ["task_id", "problem_pattern", "error_type", "solution"],
                },
            },
            {
                "name": "get_tried_approaches",
                "description": "List all approaches that have been tried for a task.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                    },
                    "required": ["task_id"],
                },
            },
            {
                "name": "get_working_solutions",
                "description": "Find solutions that worked for similar problems in the past.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "error_message": {"type": "string"},
                        "error_type": {"type": "string", "enum": [t.value for t in FailureType]},
                        "limit": {"type": "integer", "default": 5},
                    },
                    "required": ["error_message"],
                },
            },
            {
                "name": "get_task_summary",
                "description": "Get a summary of all attempts made for a task.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                    },
                    "required": ["task_id"],
                },
            },
            {
                "name": "classify_error",
                "description": "Classify an error message to determine type and routing.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "error_message": {"type": "string"},
                    },
                    "required": ["error_message"],
                },
            },
            {
                "name": "reflect_on_failure",
                "description": "Get a decision from the Reflexion Engine on how to proceed after a failure.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "error_message": {"type": "string"},
                        "proposed_approach": {"type": "string"},
                    },
                    "required": ["task_id", "error_message"],
                },
            },
            {
                "name": "get_memory_stats",
                "description": "Get statistics about the memory system.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
        ]
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an incoming MCP request.
        
        Args:
            request: JSON-RPC style request with method and params
            
        Returns:
            Response with result or error
        """
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")
        
        logger.info(f"Handling request: {method}")
        
        # Handle MCP protocol methods
        if method == "initialize":
            return {
                "id": request_id,
                "result": {
                    "protocolVersion": "0.1.0",
                    "serverInfo": {
                        "name": "agi-reflexion-memory",
                        "version": "0.1.0",
                    },
                    "capabilities": {
                        "tools": {},
                    },
                },
            }
        
        elif method == "tools/list":
            return {
                "id": request_id,
                "result": {
                    "tools": self.get_tool_definitions(),
                },
            }
        
        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})
            
            if tool_name not in self.tools:
                return {
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}",
                    },
                }
            
            try:
                result = await self.tools[tool_name](**tool_args)
                return {
                    "id": request_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result, indent=2),
                            }
                        ],
                    },
                }
            except Exception as e:
                logger.error(f"Tool error: {e}")
                return {
                    "id": request_id,
                    "error": {
                        "code": -32000,
                        "message": str(e),
                    },
                }
        
        else:
            return {
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Unknown method: {method}",
                },
            }


# =============================================================================
# Stdio Transport (for MCP)
# =============================================================================

async def run_stdio_server():
    """Run the MCP server using stdio transport."""
    server = MemoryMCPServer()
    
    logger.info("Starting MCP server on stdio...")
    
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)
    
    writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
        asyncio.streams.FlowControlMixin, sys.stdout
    )
    writer = asyncio.StreamWriter(writer_transport, writer_protocol, reader, asyncio.get_event_loop())
    
    while True:
        try:
            line = await reader.readline()
            if not line:
                break
            
            request = json.loads(line.decode())
            response = await server.handle_request(request)
            
            response_line = json.dumps(response) + "\n"
            writer.write(response_line.encode())
            await writer.drain()
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
        except Exception as e:
            logger.error(f"Server error: {e}")


# =============================================================================
# HTTP Transport (alternative)
# =============================================================================

async def run_http_server(host: str = "127.0.0.1", port: int = 8765):
    """Run the MCP server using HTTP transport."""
    try:
        from aiohttp import web
    except ImportError:
        logger.error("aiohttp not installed. Run: pip install aiohttp")
        return
    
    server = MemoryMCPServer()
    
    async def handle_post(request):
        try:
            data = await request.json()
            response = await server.handle_request(data)
            return web.json_response(response)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_health(request):
        return web.json_response({"status": "ok", "tools": len(server.tools)})
    
    app = web.Application()
    app.router.add_post("/mcp", handle_post)
    app.router.add_get("/health", handle_health)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    
    logger.info(f"Starting HTTP MCP server on {host}:{port}")
    await site.start()
    
    # Keep running
    while True:
        await asyncio.sleep(3600)


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AGI Reflexion Memory MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="http",
        help="Transport protocol (default: http)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="HTTP host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MCP_SERVER_PORT", "8765")),
        help="HTTP port (default: 8765)",
    )
    
    args = parser.parse_args()
    
    if args.transport == "stdio":
        asyncio.run(run_stdio_server())
    else:
        asyncio.run(run_http_server(args.host, args.port))


if __name__ == "__main__":
    main()

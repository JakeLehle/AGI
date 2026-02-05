"""
MCP Server Module for AGI Pipeline.

This module provides Model Context Protocol (MCP) servers that expose
AGI pipeline functionality to external tools and agents.

Available Servers:
    - MemoryMCPServer: Exposes Reflexion Memory for querying past failures
                       and solutions during task execution.

Usage:
    # Start HTTP server
    python -m mcp_server.memory_server --transport http --port 8765
    
    # Start stdio server (for MCP clients)
    python -m mcp_server.memory_server --transport stdio

    # Use the client helper
    from mcp_server import MemoryClient
    
    client = MemoryClient()
    result = await client.check_if_tried("task_001", "Install pandas via pip")
"""

from .memory_server import MemoryMCPServer

__all__ = [
    "MemoryMCPServer",
]

__version__ = "0.1.0"

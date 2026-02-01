"""
AGI Memory Module - Mem0-backed reflexion memory for loop prevention.

This module provides persistent semantic memory for the AGI pipeline's
Reflexion Engine, enabling detection and prevention of infinite retry loops.

Usage:
    from agi.memory import ReflexionMemory, FailureType
    
    memory = ReflexionMemory()
    
    # Store a failure
    memory.store_failure(
        task_id="task_001",
        error_type=FailureType.MISSING_PACKAGE,
        error_message="ModuleNotFoundError: No module named 'pandas'",
        approach_tried="Added import without installing package"
    )
    
    # Check if approach was tried (KEY for loop prevention)
    result = memory.check_if_tried(
        task_id="task_001",
        proposed_approach="Add import pandas statement"
    )
    # Returns: {"tried": True, "similarity": 0.91, ...}
"""

from .reflexion_memory import (
    ReflexionMemory,
    FailureType,
    FailureRecord,
    SolutionRecord,
)

from .config import get_mem0_config

__all__ = [
    "ReflexionMemory",
    "FailureType", 
    "FailureRecord",
    "SolutionRecord",
    "get_mem0_config",
]

__version__ = "0.1.0"

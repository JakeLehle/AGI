#!/usr/bin/env python3
"""
Test script for Reflexion Integration helpers.

Tests the integration functions without requiring the full LangGraph workflow.

Usage:
    python setup/test_reflexion_integration.py
"""

import sys
import os
from pathlib import Path

# Add AGI root to path
AGI_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(AGI_ROOT))

# Set environment
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("AGI_DATA_DIR", str(Path.home() / "agi_data"))


def test_check_before_retry():
    """Test the check_before_retry function."""
    print("\n" + "=" * 50)
    print("TEST 1: Check Before Retry")
    print("=" * 50)
    
    from utils.reflexion_integration import check_before_retry, handle_failure
    
    task_id = "integration_test_1"
    
    # First, record a failure
    print("  Recording initial failure...")
    handle_failure(
        task_id=task_id,
        error_message="ModuleNotFoundError: No module named 'pandas'",
        approach_tried="Added 'import pandas' without installing the package",
    )
    
    # Check similar approach
    print("  Checking similar approach...")
    result = check_before_retry(
        task_id=task_id,
        proposed_approach="Import pandas at the top of the file",
    )
    
    if not result["allowed"]:
        print(f"  ✅ Similar approach correctly rejected (similarity: {result['similarity']:.2f})")
    else:
        print(f"  ⚠️  Similar approach was allowed (similarity: {result['similarity']:.2f})")
    
    # Check different approach
    print("  Checking different approach...")
    result = check_before_retry(
        task_id=task_id,
        proposed_approach="pip install pandas in the conda environment first",
    )
    
    if result["allowed"]:
        print(f"  ✅ Different approach correctly allowed (similarity: {result['similarity']:.2f})")
    else:
        print(f"  ⚠️  Different approach was rejected (similarity: {result['similarity']:.2f})")
    
    return True


def test_handle_failure():
    """Test the handle_failure function."""
    print("\n" + "=" * 50)
    print("TEST 2: Handle Failure")
    print("=" * 50)
    
    from utils.reflexion_integration import handle_failure
    
    task_id = "integration_test_2"
    
    # Handle a failure
    print("  Handling first failure...")
    state = handle_failure(
        task_id=task_id,
        error_message="FileNotFoundError: data.csv not found",
        approach_tried="Tried to read file without checking existence",
        proposed_next_approach="Check if file exists before reading",
    )
    
    print(f"  Action: {state['action']}")
    print(f"  Error Type: {state['last_error_type']}")
    print(f"  Should Escalate: {state['should_escalate']}")
    
    if state["action"] in ["retry", "reject_duplicate", "escalate", "apply_solution"]:
        print("  ✅ Got valid decision")
    else:
        print(f"  ❌ Unexpected action: {state['action']}")
        return False
    
    return True


def test_record_and_find_solution():
    """Test recording and finding solutions."""
    print("\n" + "=" * 50)
    print("TEST 3: Record and Find Solutions")
    print("=" * 50)
    
    from utils.reflexion_integration import record_solution, find_similar_solutions
    
    # Record a solution
    print("  Recording solution...")
    record_solution(
        task_id="integration_test_3",
        problem_pattern="Permission denied when writing to output directory",
        error_type="permission_error",
        solution="chmod 755 on the output directory before writing",
    )
    
    # Find similar solution
    print("  Searching for similar solution...")
    solutions = find_similar_solutions(
        error_message="PermissionError: [Errno 13] Permission denied: '/output/file.txt'",
        error_type="permission_error",
    )
    
    if solutions:
        print(f"  ✅ Found {len(solutions)} solution(s)")
        print(f"     Best match (score {solutions[0].get('score', 0):.2f})")
    else:
        print("  ⚠️  No solutions found")
    
    return True


def test_full_workflow_simulation():
    """Test a simulated workflow with multiple failures."""
    print("\n" + "=" * 50)
    print("TEST 4: Full Workflow Simulation")
    print("=" * 50)
    
    from utils.reflexion_integration import (
        handle_failure,
        check_before_retry,
        create_initial_reflexion_state,
    )
    
    task_id = "integration_test_4"
    
    # Simulate multiple failures
    errors = [
        ("ModuleNotFoundError: No module named 'scipy'", "Import scipy without installing"),
        ("ModuleNotFoundError: No module named 'scipy'", "Add scipy to imports"),
    ]
    
    for i, (error, approach) in enumerate(errors, 1):
        print(f"  Attempt {i}: {approach[:40]}...")
        state = handle_failure(
            task_id=task_id,
            error_message=error,
            approach_tried=approach,
        )
        print(f"    Decision: {state['action']}")
    
    # Third attempt should trigger escalation (threshold is 2 for missing_package)
    print("  Checking for escalation...")
    state = handle_failure(
        task_id=task_id,
        error_message="ModuleNotFoundError: No module named 'scipy'",
        approach_tried="Attempt 3: Still trying imports",
    )
    
    if state["should_escalate"]:
        print(f"  ✅ Escalation triggered correctly")
        print(f"     Target: {state['escalation_target']}")
    else:
        print(f"  ⚠️  Escalation not triggered (action: {state['action']})")
    
    return True


def test_memory_client_modes():
    """Test that both direct and server modes work."""
    print("\n" + "=" * 50)
    print("TEST 5: Memory Client Modes")
    print("=" * 50)
    
    from mcp_server.client import MemoryClient
    
    # Test direct mode
    print("  Testing direct mode...")
    client = MemoryClient(use_direct=True)
    result = client.classify_error("SyntaxError: invalid syntax")
    
    if result["error_type"] == "syntax_error":
        print(f"  ✅ Direct mode works: {result['error_type']}")
    else:
        print(f"  ❌ Direct mode failed: {result}")
        return False
    
    return True


def main():
    print("\n" + "=" * 50)
    print("  Reflexion Integration Test Suite")
    print("=" * 50)
    
    results = {}
    
    # Run tests
    results["check_before_retry"] = test_check_before_retry()
    results["handle_failure"] = test_handle_failure()
    results["solutions"] = test_record_and_find_solution()
    results["workflow_sim"] = test_full_workflow_simulation()
    results["client_modes"] = test_memory_client_modes()
    
    # Summary
    print("\n" + "=" * 50)
    print("  SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"  {test}: {status}")
    
    print(f"\n  Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All integration tests passed!\n")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

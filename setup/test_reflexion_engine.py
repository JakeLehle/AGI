#!/usr/bin/env python3
"""
Test script for the Reflexion Engine.

This tests the core loop-prevention functionality:
1. Error classification
2. Failure recording
3. Duplicate detection
4. Escalation decisions
5. Solution lookup

Usage:
    python setup/test_reflexion_engine.py
    
    OR from the AGI root:
    python -m setup.test_reflexion_engine
"""

import sys
import os
from pathlib import Path

# Add AGI root to path so imports work
AGI_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(AGI_ROOT))

# Set environment
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("AGI_DATA_DIR", str(Path.home() / "agi_data"))


def test_error_classification():
    """Test that errors are classified correctly."""
    print("\n" + "=" * 50)
    print("TEST 1: Error Classification")
    print("=" * 50)
    
    from engines import ReflexionEngine
    from memory import FailureType
    
    engine = ReflexionEngine()
    
    test_cases = [
        ("ModuleNotFoundError: No module named 'pandas'", FailureType.MISSING_PACKAGE),
        ("FileNotFoundError: [Errno 2] No such file or directory: 'data.csv'", FailureType.MISSING_FILE),
        ("SyntaxError: invalid syntax", FailureType.SYNTAX_ERROR),
        ("CUDA out of memory. Tried to allocate 2.00 GiB", FailureType.OUT_OF_MEMORY),
        ("RuntimeError: CUDA error: device-side assert triggered", FailureType.GPU_ERROR),
        ("TimeoutError: Connection timed out", FailureType.TIMEOUT),
        ("PermissionError: [Errno 13] Permission denied", FailureType.PERMISSION_ERROR),
        ("slurmstepd: error: CANCELLED DUE TO TIME LIMIT", FailureType.SLURM_ERROR),
        ("Some random error message", FailureType.UNKNOWN),
    ]
    
    passed = 0
    for error_msg, expected_type in test_cases:
        result = engine.classify_error(error_msg)
        status = "✅" if result == expected_type else "❌"
        print(f"  {status} '{error_msg[:40]}...' -> {result.value}")
        if result == expected_type:
            passed += 1
    
    print(f"\n  Passed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_duplicate_detection():
    """Test that similar approaches are detected as duplicates."""
    print("\n" + "=" * 50)
    print("TEST 2: Duplicate Detection")
    print("=" * 50)
    
    from engines import ReflexionEngine, ReflexionAction
    from memory import FailureType
    
    engine = ReflexionEngine()
    
    task_id = "test_dup_detection"
    
    # Clean up from previous runs
    engine.memory.clear_task_memory(task_id)
    
    # Record first attempt
    print("  Recording first attempt...")
    engine.record_attempt(
        task_id=task_id,
        error_message="ModuleNotFoundError: No module named 'pandas'",
        approach_tried="Added 'import pandas' at the top of the script without installing the package",
    )
    
    # Try similar approach (should be rejected)
    print("  Checking similar approach...")
    decision = engine.reflect_on_failure(
        task_id=task_id,
        error_message="ModuleNotFoundError: No module named 'pandas'",
        proposed_approach="Add the import statement for pandas at the beginning of the file",
    )
    
    if decision.action == ReflexionAction.REJECT_DUPLICATE:
        print(f"  ✅ Similar approach rejected (similarity: {decision.similarity_score:.2f})")
        similar_rejected = True
    else:
        print(f"  ❌ Similar approach NOT rejected (action: {decision.action.value})")
        similar_rejected = False
    
    # Try different approach (should be allowed)
    print("  Checking different approach...")
    decision = engine.reflect_on_failure(
        task_id=task_id,
        error_message="ModuleNotFoundError: No module named 'pandas'",
        proposed_approach="Install pandas using pip in the conda environment before running",
    )
    
    if decision.action == ReflexionAction.RETRY:
        print(f"  ✅ Different approach allowed (similarity: {decision.similarity_score:.2f})")
        different_allowed = True
    else:
        print(f"  ⚠️  Different approach status: {decision.action.value}")
        different_allowed = decision.action != ReflexionAction.REJECT_DUPLICATE
    
    # Cleanup
    engine.memory.clear_task_memory(task_id)
    
    return similar_rejected and different_allowed


def test_escalation():
    """Test that escalation triggers after threshold."""
    print("\n" + "=" * 50)
    print("TEST 3: Escalation After Threshold")
    print("=" * 50)
    
    from engines import ReflexionEngine, ReflexionAction
    
    engine = ReflexionEngine()
    
    task_id = "test_escalation"
    
    # Clean up
    engine.memory.clear_task_memory(task_id)
    
    # Record multiple attempts for same error type
    print("  Recording multiple failed attempts...")
    for i in range(3):
        engine.record_attempt(
            task_id=task_id,
            error_message="ModuleNotFoundError: No module named 'pandas'",
            approach_tried=f"Attempt {i+1}: different approach number {i+1}",
        )
    
    # Check if escalation triggers
    print("  Checking if escalation triggers...")
    decision = engine.reflect_on_failure(
        task_id=task_id,
        error_message="ModuleNotFoundError: No module named 'pandas'",
        proposed_approach="Yet another approach",
    )
    
    if decision.action == ReflexionAction.ESCALATE:
        print(f"  ✅ Escalation triggered after {decision.attempt_count} attempts")
        print(f"     Target: {decision.escalation_target}")
        result = True
    else:
        print(f"  ❌ Escalation NOT triggered (action: {decision.action.value})")
        result = False
    
    # Cleanup
    engine.memory.clear_task_memory(task_id)
    
    return result


def test_solution_lookup():
    """Test that known solutions are found and applied."""
    print("\n" + "=" * 50)
    print("TEST 4: Solution Lookup")
    print("=" * 50)
    
    from engines import ReflexionEngine, ReflexionAction
    from memory import FailureType
    
    engine = ReflexionEngine()
    
    # Store a solution
    print("  Storing a known solution...")
    engine.memory.store_solution(
        task_id="previous_task",
        problem_pattern="ModuleNotFoundError for numpy package",
        error_type=FailureType.MISSING_PACKAGE,
        solution="pip install numpy --break-system-packages",
        context={"env": "conda"},
    )
    
    # Search for similar solution
    print("  Searching for solution to similar problem...")
    solutions = engine.memory.get_working_solutions(
        "ModuleNotFoundError: No module named 'numpy'",
        FailureType.MISSING_PACKAGE,
    )
    
    if solutions:
        print(f"  ✅ Found {len(solutions)} solution(s)")
        print(f"     Best match (score {solutions[0]['score']:.2f}): {solutions[0]['solution'][:50]}...")
        return True
    else:
        print("  ❌ No solutions found")
        return False


def test_full_workflow():
    """Test the complete workflow: fail -> record -> check -> succeed."""
    print("\n" + "=" * 50)
    print("TEST 5: Full Workflow")
    print("=" * 50)
    
    from engines import ReflexionEngine, ReflexionAction
    
    engine = ReflexionEngine()
    
    task_id = "test_full_workflow"
    
    # Clean up
    engine.memory.clear_task_memory(task_id)
    
    # Simulate workflow
    print("  Step 1: First failure occurs...")
    decision = engine.reflect_on_failure(
        task_id=task_id,
        error_message="FileNotFoundError: data.csv not found",
        proposed_approach="Check if the file path is correct",
    )
    print(f"    Decision: {decision.action.value}")
    
    # Record the attempt
    print("  Step 2: Recording the attempt...")
    engine.record_attempt(
        task_id=task_id,
        error_message="FileNotFoundError: data.csv not found",
        approach_tried="Checked file path, found typo",
    )
    
    # Second attempt with similar approach (should be rejected)
    print("  Step 3: Trying similar approach...")
    decision = engine.reflect_on_failure(
        task_id=task_id,
        error_message="FileNotFoundError: data.csv not found",
        proposed_approach="Verify the file path is correct",
    )
    print(f"    Decision: {decision.action.value}")
    
    # Third attempt with different approach
    print("  Step 4: Trying different approach...")
    decision = engine.reflect_on_failure(
        task_id=task_id,
        error_message="FileNotFoundError: data.csv not found",
        proposed_approach="Create the missing data.csv file from template",
    )
    print(f"    Decision: {decision.action.value}")
    
    # Record success
    print("  Step 5: Task succeeded, recording solution...")
    engine.record_success(
        task_id=task_id,
        problem_description="Missing data.csv file",
        solution="Created file from template",
    )
    
    # Check stats
    summary = engine.get_task_summary(task_id)
    print(f"    Total attempts: {summary['total_attempts']}")
    
    # Cleanup
    engine.memory.clear_task_memory(task_id)
    
    print("  ✅ Full workflow completed")
    return True


def main():
    print("\n" + "=" * 50)
    print("  Reflexion Engine Test Suite")
    print("=" * 50)
    
    results = {}
    
    # Test 1: Error classification
    results["classification"] = test_error_classification()
    
    # Test 2: Duplicate detection
    results["duplicates"] = test_duplicate_detection()
    
    # Test 3: Escalation
    results["escalation"] = test_escalation()
    
    # Test 4: Solution lookup
    results["solutions"] = test_solution_lookup()
    
    # Test 5: Full workflow
    results["workflow"] = test_full_workflow()
    
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
        print("\n✅ All tests passed! Reflexion Engine is ready.\n")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

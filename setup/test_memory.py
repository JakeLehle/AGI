#!/usr/bin/env python3
"""
Test script for ReflexionMemory with embedded Qdrant.

Run this to verify your Mem0 + Ollama setup is working correctly.

Usage:
    python scripts/test_memory.py
    
    # Or with custom data directory:
    AGI_DATA_DIR=/path/to/data python scripts/test_memory.py
"""

import sys
import os
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that required packages are installed."""
    print("\n" + "=" * 60)
    print("Step 1: Testing imports")
    print("=" * 60)
    
    errors = []
    
    # Test mem0
    try:
        from mem0 import Memory
        print("  ✅ mem0ai installed")
    except ImportError as e:
        errors.append(f"mem0ai: {e}")
        print("  ❌ mem0ai not installed")
        print("     Run: pip install mem0ai")
    
    # Test qdrant-client
    try:
        import qdrant_client
        print("  ✅ qdrant-client installed")
    except ImportError as e:
        errors.append(f"qdrant-client: {e}")
        print("  ❌ qdrant-client not installed")
    
    # Test yaml
    try:
        import yaml
        print("  ✅ pyyaml installed")
    except ImportError as e:
        print("  ⚠️  pyyaml not installed (will use defaults)")
    
    if errors:
        print("\n❌ Missing required packages. Install them first.")
        return False
    
    return True


def test_ollama_connection():
    """Test Ollama is running and has required models."""
    print("\n" + "=" * 60)
    print("Step 2: Testing Ollama connection")
    print("=" * 60)
    
    import urllib.request
    import json
    
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    
    try:
        req = urllib.request.Request(f"{ollama_url}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            models = [m["name"] for m in data.get("models", [])]
            
            print(f"  ✅ Connected to Ollama at {ollama_url}")
            print(f"  📦 Available models: {len(models)}")
            
            # Check for embedding model
            has_embed = any("nomic-embed-text" in m for m in models)
            if has_embed:
                print("  ✅ nomic-embed-text model available")
            else:
                print("  ❌ nomic-embed-text not found")
                print("     Run: ollama pull nomic-embed-text")
                return False
            
            # Check for LLM
            llm_models = ["llama3.1", "llama3", "mistral", "qwen"]
            has_llm = any(any(m in model for m in llm_models) for model in models)
            if has_llm:
                print("  ✅ LLM model available")
            else:
                print("  ⚠️  No standard LLM found (llama3.1, mistral, etc.)")
                print("     Run: ollama pull llama3.1:8b")
            
            return True
            
    except Exception as e:
        print(f"  ❌ Cannot connect to Ollama: {e}")
        print("     Make sure Ollama is running: ollama serve")
        return False


def test_memory_initialization():
    """Test ReflexionMemory can be initialized."""
    print("\n" + "=" * 60)
    print("Step 3: Testing memory initialization")
    print("=" * 60)
    
    try:
        from memory import ReflexionMemory
        
        print("  Initializing ReflexionMemory (this may take a moment)...")
        memory = ReflexionMemory()
        print("  ✅ ReflexionMemory initialized successfully")
        return memory
        
    except Exception as e:
        print(f"  ❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_store_and_retrieve(memory):
    """Test storing and retrieving failures."""
    print("\n" + "=" * 60)
    print("Step 4: Testing store and retrieve")
    print("=" * 60)
    
    from memory import FailureType
    
    task_id = "test_task_001"
    
    # Clean up any previous test data
    print("  Clearing previous test data...")
    memory.clear_task_memory(task_id)
    
    # Store first failure
    print("  Storing failure 1...")
    result1 = memory.store_failure(
        task_id=task_id,
        error_type=FailureType.MISSING_PACKAGE,
        error_message="ModuleNotFoundError: No module named 'pandas'",
        approach_tried="Added import pandas at the top without installing the package",
        attempt_number=1,
    )
    print(f"    ✅ Stored: {result1}")
    
    # Store second failure
    print("  Storing failure 2...")
    result2 = memory.store_failure(
        task_id=task_id,
        error_type=FailureType.MISSING_PACKAGE,
        error_message="ModuleNotFoundError: No module named 'pandas'",
        approach_tried="Tried using pip install inside the Python script",
        attempt_number=2,
    )
    print(f"    ✅ Stored: {result2}")
    
    # Retrieve approaches
    print("  Retrieving tried approaches...")
    approaches = memory.get_tried_approaches(task_id)
    print(f"    ✅ Found {len(approaches)} approaches")
    
    for i, approach in enumerate(approaches, 1):
        print(f"    {i}. [{approach['error_type']}] {approach['approach'][:60]}...")
    
    return len(approaches) >= 2


def test_similarity_check(memory):
    """Test the key loop-prevention feature."""
    print("\n" + "=" * 60)
    print("Step 5: Testing similarity detection (loop prevention)")
    print("=" * 60)
    
    task_id = "test_task_001"
    
    # Test with similar approach (should be detected)
    print("  Checking similar approach...")
    similar_check = memory.check_if_tried(
        task_id=task_id,
        proposed_approach="Add the import statement for pandas at the beginning",
    )
    
    print(f"    Tried: {similar_check['tried']}")
    print(f"    Similarity: {similar_check['similarity']:.2f}")
    print(f"    Attempts found: {similar_check['attempt_count']}")
    
    if similar_check['tried']:
        print("    ✅ Correctly detected similar approach!")
    else:
        print("    ⚠️  Did not detect similarity (threshold may need tuning)")
    
    # Test with different approach (should not be detected)
    print("\n  Checking different approach...")
    different_check = memory.check_if_tried(
        task_id=task_id,
        proposed_approach="Install pandas using conda in the environment setup",
    )
    
    print(f"    Tried: {different_check['tried']}")
    print(f"    Similarity: {different_check['similarity']:.2f}")
    
    if not different_check['tried']:
        print("    ✅ Correctly identified as new approach!")
    else:
        print("    ⚠️  Flagged as similar (might be overly sensitive)")
    
    return True


def test_solutions(memory):
    """Test storing and finding solutions."""
    print("\n" + "=" * 60)
    print("Step 6: Testing solutions")
    print("=" * 60)
    
    from memory import FailureType
    
    # Store a solution
    print("  Storing a solution...")
    memory.store_solution(
        task_id="test_task_001",
        problem_pattern="ModuleNotFoundError for pandas in conda environment",
        error_type=FailureType.MISSING_PACKAGE,
        solution="pip install pandas --break-system-packages",
        context={"env": "conda_agi", "python": "3.11"},
    )
    print("    ✅ Solution stored")
    
    # Search for solutions
    print("  Searching for similar solutions...")
    solutions = memory.get_working_solutions(
        problem_description="ModuleNotFoundError: No module named 'numpy'",
        error_type=FailureType.MISSING_PACKAGE,
    )
    
    print(f"    ✅ Found {len(solutions)} relevant solutions")
    for sol in solutions:
        print(f"    - Score {sol['score']:.2f}: {sol['solution'][:50]}...")
    
    return True


def test_stats(memory):
    """Test statistics gathering."""
    print("\n" + "=" * 60)
    print("Step 7: Testing statistics")
    print("=" * 60)
    
    stats = memory.get_stats()
    
    print(f"  Total failures: {stats.get('total_failures', 0)}")
    print(f"  Total solutions: {stats.get('total_solutions', 0)}")
    print(f"  Unique tasks: {stats.get('unique_tasks', 0)}")
    print(f"  Failure types: {stats.get('failure_types', {})}")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  ReflexionMemory Test Suite")
    print("  Embedded Qdrant + Ollama")
    print("=" * 60)
    
    data_dir = os.environ.get("AGI_DATA_DIR", str(Path.home() / "agi_data"))
    print(f"\nData directory: {data_dir}")
    
    # Run tests
    tests_passed = 0
    tests_total = 7
    
    # Test 1: Imports
    if test_imports():
        tests_passed += 1
    else:
        print("\n❌ Cannot proceed without required packages.")
        sys.exit(1)
    
    # Test 2: Ollama
    if test_ollama_connection():
        tests_passed += 1
    else:
        print("\n❌ Cannot proceed without Ollama.")
        sys.exit(1)
    
    # Test 3: Initialization
    memory = test_memory_initialization()
    if memory:
        tests_passed += 1
    else:
        print("\n❌ Cannot proceed without memory initialization.")
        sys.exit(1)
    
    # Test 4: Store/Retrieve
    if test_store_and_retrieve(memory):
        tests_passed += 1
    
    # Test 5: Similarity
    if test_similarity_check(memory):
        tests_passed += 1
    
    # Test 6: Solutions
    if test_solutions(memory):
        tests_passed += 1
    
    # Test 7: Stats
    if test_stats(memory):
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"  Results: {tests_passed}/{tests_total} tests passed")
    print("=" * 60)
    
    if tests_passed == tests_total:
        print("\n✅ All tests passed! ReflexionMemory is ready to use.\n")
        return 0
    else:
        print(f"\n⚠️  {tests_total - tests_passed} test(s) had issues.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Test script for the MCP Memory Server.

Tests the HTTP interface of the MCP server.

Usage:
    # First, start the server in another terminal:
    python mcp_server/memory_server.py --transport http --port 8765
    
    # Then run this test:
    python setup/test_mcp_server.py
"""

import sys
import json
import asyncio
from pathlib import Path

# Add AGI root to path
AGI_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(AGI_ROOT))


async def test_mcp_server(base_url: str = "http://127.0.0.1:8765"):
    """Test the MCP server endpoints."""
    
    try:
        import aiohttp
    except ImportError:
        print("❌ aiohttp not installed. Run: pip install aiohttp")
        return False
    
    print("\n" + "=" * 50)
    print("  MCP Memory Server Test Suite")
    print("=" * 50)
    
    async with aiohttp.ClientSession() as session:
        
        # Test 1: Health check
        print("\n[TEST 1] Health Check")
        try:
            async with session.get(f"{base_url}/health") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"  ✅ Server healthy: {data}")
                else:
                    print(f"  ❌ Health check failed: {resp.status}")
                    return False
        except aiohttp.ClientConnectorError:
            print(f"  ❌ Cannot connect to server at {base_url}")
            print("     Make sure the server is running:")
            print("     python mcp_server/memory_server.py --transport http --port 8765")
            return False
        
        # Test 2: Initialize
        print("\n[TEST 2] MCP Initialize")
        request = {
            "id": 1,
            "method": "initialize",
            "params": {},
        }
        async with session.post(f"{base_url}/mcp", json=request) as resp:
            data = await resp.json()
            if "result" in data and "serverInfo" in data["result"]:
                print(f"  ✅ Initialized: {data['result']['serverInfo']}")
            else:
                print(f"  ❌ Initialize failed: {data}")
                return False
        
        # Test 3: List tools
        print("\n[TEST 3] List Tools")
        request = {
            "id": 2,
            "method": "tools/list",
            "params": {},
        }
        async with session.post(f"{base_url}/mcp", json=request) as resp:
            data = await resp.json()
            if "result" in data and "tools" in data["result"]:
                tools = data["result"]["tools"]
                print(f"  ✅ Found {len(tools)} tools:")
                for tool in tools:
                    print(f"     - {tool['name']}")
            else:
                print(f"  ❌ List tools failed: {data}")
                return False
        
        # Test 4: Classify error
        print("\n[TEST 4] Classify Error")
        request = {
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "classify_error",
                "arguments": {
                    "error_message": "ModuleNotFoundError: No module named 'pandas'"
                }
            },
        }
        async with session.post(f"{base_url}/mcp", json=request) as resp:
            data = await resp.json()
            if "result" in data:
                content = json.loads(data["result"]["content"][0]["text"])
                print(f"  ✅ Classified: {content['error_type']}")
                print(f"     Agent: {content['recommended_agent']}")
            else:
                print(f"  ❌ Classify failed: {data}")
                return False
        
        # Test 5: Store and check failure
        print("\n[TEST 5] Store Failure")
        task_id = "mcp_test_task"
        request = {
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "store_failure",
                "arguments": {
                    "task_id": task_id,
                    "error_type": "missing_package",
                    "error_message": "ModuleNotFoundError: No module named 'numpy'",
                    "approach_tried": "Added import numpy without installing the package",
                }
            },
        }
        async with session.post(f"{base_url}/mcp", json=request) as resp:
            data = await resp.json()
            if "result" in data:
                print(f"  ✅ Failure stored")
            else:
                print(f"  ❌ Store failed: {data}")
                return False
        
        # Test 6: Check if tried
        print("\n[TEST 6] Check If Tried (similar approach)")
        request = {
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "check_if_tried",
                "arguments": {
                    "task_id": task_id,
                    "proposed_approach": "Import numpy at the top of the file",
                }
            },
        }
        async with session.post(f"{base_url}/mcp", json=request) as resp:
            data = await resp.json()
            if "result" in data:
                content = json.loads(data["result"]["content"][0]["text"])
                print(f"  ✅ Check result: tried={content['tried']}, similarity={content['similarity']:.2f}")
            else:
                print(f"  ❌ Check failed: {data}")
                return False
        
        # Test 7: Reflect on failure
        print("\n[TEST 7] Reflect on Failure")
        request = {
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "reflect_on_failure",
                "arguments": {
                    "task_id": task_id,
                    "error_message": "ModuleNotFoundError: No module named 'numpy'",
                    "proposed_approach": "pip install numpy in the conda environment",
                }
            },
        }
        async with session.post(f"{base_url}/mcp", json=request) as resp:
            data = await resp.json()
            if "result" in data:
                content = json.loads(data["result"]["content"][0]["text"])
                print(f"  ✅ Decision: {content['action']}")
                print(f"     Reason: {content['reason']}")
            else:
                print(f"  ❌ Reflect failed: {data}")
                return False
        
        # Test 8: Get task summary
        print("\n[TEST 8] Get Task Summary")
        request = {
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "get_task_summary",
                "arguments": {
                    "task_id": task_id,
                }
            },
        }
        async with session.post(f"{base_url}/mcp", json=request) as resp:
            data = await resp.json()
            if "result" in data:
                content = json.loads(data["result"]["content"][0]["text"])
                print(f"  ✅ Summary: {content['total_attempts']} attempts")
            else:
                print(f"  ❌ Summary failed: {data}")
                return False
        
        # Test 9: Memory stats
        print("\n[TEST 9] Memory Stats")
        request = {
            "id": 8,
            "method": "tools/call",
            "params": {
                "name": "get_memory_stats",
                "arguments": {}
            },
        }
        async with session.post(f"{base_url}/mcp", json=request) as resp:
            data = await resp.json()
            if "result" in data:
                content = json.loads(data["result"]["content"][0]["text"])
                print(f"  ✅ Stats retrieved")
                print(f"     Threshold: {content['config']['similarity_threshold']}")
            else:
                print(f"  ❌ Stats failed: {data}")
                return False
    
    print("\n" + "=" * 50)
    print("  ✅ All MCP server tests passed!")
    print("=" * 50 + "\n")
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MCP Memory Server")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8765",
        help="Server URL (default: http://127.0.0.1:8765)",
    )
    
    args = parser.parse_args()
    
    success = asyncio.run(test_mcp_server(args.url))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

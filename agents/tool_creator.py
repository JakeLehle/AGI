"""
Dynamic tool creation capability.
Allows agents to generate new tools when existing ones are insufficient.
"""

from pathlib import Path
from typing import Dict, Any
from langchain_community.llms import Ollama
import importlib.util
import sys

class ToolCreator:
    """Creates new tools dynamically based on agent needs"""
    
    def __init__(self, ollama_model: str = "llama3.1:70b"):
        self.llm = Ollama(model=ollama_model)
        self.dynamic_tools_dir = Path("tools/dynamic_tools")
        self.dynamic_tools_dir.mkdir(parents=True, exist_ok=True)
        self.created_tools = {}
    
    def should_create_tool(self, task_description: str, available_tools: list) -> Dict[str, Any]:
        """Determine if a new tool is needed"""
        
        prompt = f"""
You have these available tools:
{', '.join(available_tools)}

For this task: "{task_description}"

Do you need a new tool that doesn't exist yet? 

Answer with JSON:
{{
    "needs_new_tool": true/false,
    "tool_name": "name_of_needed_tool",
    "reason": "why this tool is needed",
    "functionality": "what the tool should do"
}}

Only answer true if the existing tools are truly insufficient.
"""
        
        response = self.llm.invoke(prompt)
        
        # Parse response (simplified - you'd want better parsing)
        try:
            # Extract JSON from response
            import json
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"needs_new_tool": False}
        except:
            return {"needs_new_tool": False}
    
    def create_tool(self, tool_name: str, functionality: str, context: str) -> Dict[str, Any]:
        """Generate a new tool function"""
        
        prompt = f"""
Create a Python function for this tool:

Tool Name: {tool_name}
Functionality: {functionality}
Context: {context}

Requirements:
1. Function must be called '{tool_name}'
2. Include type hints
3. Include docstring
4. Include error handling with try/except
5. Return a dictionary with 'success' key and relevant data
6. Import only standard library or: pathlib, json, requests, pandas, beautifulsoup4

Example format:
```python
from pathlib import Path
from typing import Dict, Any
import json

def my_tool(param1: str, param2: int) -> Dict[str, Any]:
    \"\"\"
    Tool description here.
    
    Args:
        param1: Description
        param2: Description
    
    Returns:
        Dictionary with results
    \"\"\"
    try:
        # Implementation here
        result = do_something(param1, param2)
        
        return {{
            "success": True,
            "result": result
        }}
    except Exception as e:
        return {{
            "success": False,
            "error": str(e)
        }}
```

Generate ONLY the Python code, no explanation.
"""
        
        code = self.llm.invoke(prompt)
        
        # Clean up code (remove markdown formatting if present)
        code = code.replace("```python", "").replace("```", "").strip()
        
        # Save tool to file
        tool_file = self.dynamic_tools_dir / f"{tool_name}.py"
        tool_file.write_text(code)
        
        # Load the tool
        try:
            spec = importlib.util.spec_from_file_location(tool_name, tool_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[tool_name] = module
            spec.loader.exec_module(module)
            
            # Get the function
            tool_func = getattr(module, tool_name)
            self.created_tools[tool_name] = tool_func
            
            return {
                "success": True,
                "tool_name": tool_name,
                "filepath": str(tool_file),
                "code": code
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to load tool: {str(e)}",
                "code": code
            }
    
    def get_tool(self, tool_name: str):
        """Retrieve a dynamically created tool"""
        return self.created_tools.get(tool_name)
    
    def list_created_tools(self) -> list:
        """List all dynamically created tools"""
        return list(self.created_tools.keys())

tool_creator = ToolCreator()

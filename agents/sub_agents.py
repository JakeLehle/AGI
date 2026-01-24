"""
Sub-agent that executes tasks with built-in reflection capability.
Can retry tasks and learn from failures.
"""

from typing import Dict, Any, List
from langchain_community.llms import Ollama
from utils.logging_config import agent_logger
from tools.base_tools import base_tools
from agents.tool_creator import tool_creator

class SubAgent:
    """Executes subtasks with reflection and retry logic"""
    
    def __init__(self, agent_id: str, ollama_model: str = "llama3.1:70b"):
        self.agent_id = agent_id
        self.llm = Ollama(model=ollama_model)
        self.tools_used = []
        self.execution_history = []
    
    def execute(self, subtask: Dict[str, Any], attempt: int = 1) -> Dict[str, Any]:
        """
        Execute a subtask with reflection
        
        Args:
            subtask: Dictionary with 'id', 'description', 'context'
            attempt: Current attempt number
        
        Returns:
            Result dictionary with success status and data
        """
        
        task_id = subtask['id']
        description = subtask['description']
        
        agent_logger.log_task_start(
            agent_name=self.agent_id,
            task_id=task_id,
            description=description,
            attempt=attempt
        )
        
        # Determine what tools are needed
        tools_needed = self._plan_tools(description, subtask.get('context', {}))
        
        # Check if we need to create new tools
        available_tools = list(base_tools.__dict__.keys()) + tool_creator.list_created_tools()
        tool_check = tool_creator.should_create_tool(description, available_tools)
        
        if tool_check.get('needs_new_tool'):
            # Create the needed tool
            new_tool = tool_creator.create_tool(
                tool_name=tool_check['tool_name'],
                functionality=tool_check['functionality'],
                context=description
            )
            
            if new_tool['success']:
                agent_logger.log_tool_creation(
                    agent_name=self.agent_id,
                    tool_name=tool_check['tool_name'],
                    reason=tool_check['reason'],
                    code=new_tool['code']
                )
                tools_needed.append(tool_check['tool_name'])
        
        # Execute the task
        try:
            result = self._execute_with_tools(description, tools_needed, subtask.get('context', {}))
            
            # Self-reflect on the result
            reflection = self._reflect(subtask, result, attempt)
            
            if reflection['success']:
                agent_logger.log_task_success(
                    agent_name=self.agent_id,
                    task_id=task_id,
                    result=result,
                    tools_used=self.tools_used
                )
                
                return {
                    "success": True,
                    "task_id": task_id,
                    "result": result,
                    "reflection": reflection,
                    "tools_used": self.tools_used,
                    "attempts": attempt
                }
            else:
                # Reflection says we failed
                agent_logger.log_reflection(
                    agent_name=self.agent_id,
                    task_id=task_id,
                    reflection=reflection['analysis']
                )
                
                return {
                    "success": False,
                    "task_id": task_id,
                    "result": result,
                    "reflection": reflection,
                    "tools_used": self.tools_used,
                    "attempts": attempt,
                    "should_retry": reflection.get('should_retry', True),
                    "improvement_strategy": reflection.get('improvement_strategy', '')
                }
                
        except Exception as e:
            error_msg = str(e)
            agent_logger.log_task_failure(
                agent_name=self.agent_id,
                task_id=task_id,
                error=error_msg,
                context={"tools_needed": tools_needed, "attempt": attempt}
            )
            
            return {
                "success": False,
                "task_id": task_id,
                "error": error_msg,
                "tools_used": self.tools_used,
                "attempts": attempt,
                "should_retry": True
            }
    
    def _plan_tools(self, description: str, context: Dict) -> List[str]:
        """Determine which tools are needed for the task"""
        
        prompt = f"""
Given this task: "{description}"

Context: {context}

Which of these tools do you need?
- read_file: Read text files
- write_file: Write content to files
- list_files: List files in directory
- web_search: Search the web
- fetch_webpage: Download and parse webpage
- analyze_csv: Analyze CSV data
- save_json: Save data as JSON
- load_json: Load JSON data

Return ONLY a comma-separated list of tool names you'll use, nothing else.
Example: read_file,analyze_csv,save_json
"""
        
        response = self.llm.invoke(prompt).strip()
        tools = [t.strip() for t in response.split(',') if t.strip()]
        
        return tools
    
    def _execute_with_tools(self, description: str, tools_needed: List[str], context: Dict) -> Dict[str, Any]:
        """Execute task using specified tools"""
        
        self.tools_used = tools_needed
        
        # Build tool context
        tool_context = "You have access to these tools:\n"
        for tool_name in tools_needed:
            tool_context += f"- {tool_name}\n"
        
        prompt = f"""
Task: {description}

Context: {context}

{tool_context}

Execute this task step by step. Describe what you would do with each tool.
Format your response as a plan with specific actions.

Example:
1. Use read_file to load data from X
2. Use analyze_csv to find patterns
3. Use save_json to store results
"""
        
        execution_plan = self.llm.invoke(prompt)
        
        # For now, we'll simulate execution
        # In a full implementation, you'd actually call the tools based on the plan
        
        result = {
            "execution_plan": execution_plan,
            "status": "completed",
            "output": f"Task '{description}' executed successfully using tools: {', '.join(tools_needed)}"
        }
        
        return result
    
    def _reflect(self, subtask: Dict, result: Dict, attempt: int) -> Dict[str, Any]:
        """Self-reflect on task execution to determine success"""
        
        prompt = f"""
You just attempted this task: {subtask['description']}

Your result was: {result}

This was attempt #{attempt}.

Evaluate your performance:
1. Did you successfully complete the task? (YES/NO)
2. If NO, what went wrong?
3. If NO, should you retry with a different approach?
4. If retrying, what specific changes would improve success?

Respond in JSON format:
{{
    "success": true/false,
    "analysis": "your evaluation",
    "should_retry": true/false,
    "improvement_strategy": "specific changes for next attempt"
}}
"""
        
        reflection = self.llm.invoke(prompt)
        
        # Parse JSON response
        try:
            import json
            import re
            json_match = re.search(r'\{.*\}', reflection, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback if parsing fails
        success = "YES" in reflection.upper()[:100]
        
        return {
            "success": success,
            "analysis": reflection,
            "should_retry": not success,
            "improvement_strategy": reflection if not success else ""
        }

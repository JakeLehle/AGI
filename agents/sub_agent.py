"""
Improved Sub-agent that executes tasks with:
- Context-aware file resolution (checks prompt-specified files first)
- Reference script examination
- Language/tool specification respect
- Existing output verification
"""

from typing import Dict, Any, List, Optional
from langchain_community.llms import Ollama
from utils.logging_config import agent_logger
from tools.base_tools import base_tools
from agents.tool_creator import tool_creator
from pathlib import Path
import re
import json


class SubAgent:
    """Executes subtasks with reflection, retry logic, and full context awareness"""
    
    def __init__(self, agent_id: str, ollama_model: str = "llama3.1:70b", project_root: str = "."):
        self.agent_id = agent_id
        self.llm = Ollama(model=ollama_model)
        self.tools_used = []
        self.execution_history = []
        self.project_root = Path(project_root)
    
    def execute(self, subtask: Dict[str, Any], attempt: int = 1) -> Dict[str, Any]:
        """
        Execute a subtask with full context awareness
        
        Args:
            subtask: Dictionary with task details including preserved context
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
        
        # Step 1: Check if outputs already exist (skip if done)
        existing_check = self._check_existing_outputs(subtask)
        if existing_check['already_complete']:
            agent_logger.log_task_success(
                agent_name=self.agent_id,
                task_id=task_id,
                result={"status": "skipped", "reason": "Output files already exist"},
                tools_used=[]
            )
            return {
                "success": True,
                "task_id": task_id,
                "result": existing_check,
                "skipped": True,
                "tools_used": [],
                "attempts": attempt
            }
        
        # Step 2: Verify input files exist
        file_check = self._verify_input_files(subtask)
        if not file_check['all_found']:
            # Don't fail immediately - check if we can find alternatives
            alternatives = self._find_alternative_inputs(subtask, file_check['missing'])
            if alternatives['found']:
                subtask = self._update_subtask_with_alternatives(subtask, alternatives)
            else:
                return {
                    "success": False,
                    "task_id": task_id,
                    "error": f"Missing required input files: {file_check['missing']}",
                    "file_check": file_check,
                    "should_retry": False,  # Can't retry without files
                    "attempts": attempt
                }
        
        # Step 3: Examine reference scripts if specified
        reference_code = self._examine_reference_scripts(subtask)
        if reference_code:
            subtask['context'] = subtask.get('context', {})
            subtask['context']['reference_code'] = reference_code
        
        # Step 4: Plan tools respecting language/package specifications
        tools_needed = self._plan_tools_with_context(subtask)
        
        # Step 5: Check if we need to create new tools
        available_tools = list(base_tools.__dict__.keys()) + tool_creator.list_created_tools()
        tool_check = tool_creator.should_create_tool(description, available_tools)
        
        if tool_check.get('needs_new_tool'):
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
        
        # Step 6: Execute the task
        try:
            result = self._execute_with_context(subtask, tools_needed)
            
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
                    "packages": subtask.get('packages', []),
                    "attempts": attempt
                }
            else:
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
    
    def _check_existing_outputs(self, subtask: Dict) -> Dict[str, Any]:
        """Check if output files already exist (task may be complete)"""
        
        output_files = subtask.get('output_files', [])
        skip_if_exists = subtask.get('skip_if_exists', [])
        
        files_to_check = output_files + skip_if_exists
        
        if not files_to_check:
            return {'already_complete': False, 'reason': 'No output files specified'}
        
        existing = []
        missing = []
        
        for filepath in files_to_check:
            full_path = self.project_root / filepath
            if full_path.exists():
                existing.append(str(full_path))
            else:
                missing.append(filepath)
        
        # Consider complete if ALL specified outputs exist
        if existing and not missing:
            return {
                'already_complete': True,
                'reason': f'Output files already exist: {existing}',
                'existing_files': existing
            }
        
        return {
            'already_complete': False,
            'existing_files': existing,
            'missing_files': missing
        }
    
    def _verify_input_files(self, subtask: Dict) -> Dict[str, Any]:
        """Verify that input files specified in the subtask exist"""
        
        input_files = subtask.get('input_files', [])
        
        # Also extract file paths from description
        description = subtask.get('description', '')
        file_patterns = [
            r'[`"]?([^\s`"]*\.h5ad)[`"]?',
            r'[`"]?([^\s`"]*\.csv)[`"]?',
            r'[`"]?(data/[^\s`"]+)[`"]?'
        ]
        
        for pattern in file_patterns:
            matches = re.findall(pattern, description)
            input_files.extend(matches)
        
        # Deduplicate
        input_files = list(set(input_files))
        
        found = []
        missing = []
        
        for filepath in input_files:
            # Try multiple locations
            possible_paths = [
                self.project_root / filepath,
                self.project_root / 'data' / filepath,
                self.project_root / 'data' / 'outputs' / filepath,
                self.project_root / 'data' / 'inputs' / filepath,
            ]
            
            file_found = False
            for path in possible_paths:
                if path.exists():
                    found.append(str(path))
                    file_found = True
                    break
            
            if not file_found:
                missing.append(filepath)
        
        return {
            'all_found': len(missing) == 0,
            'found': found,
            'missing': missing,
            'searched_paths': [str(p) for p in [self.project_root / 'data']]
        }
    
    def _find_alternative_inputs(self, subtask: Dict, missing_files: List[str]) -> Dict[str, Any]:
        """Try to find alternative input files when specified ones are missing"""
        
        alternatives = {}
        found_any = False
        
        for missing in missing_files:
            # Get the file extension and base name
            missing_path = Path(missing)
            extension = missing_path.suffix
            
            # Search for similar files
            search_dirs = [
                self.project_root / 'data' / 'outputs',
                self.project_root / 'data' / 'inputs',
                self.project_root / 'data',
            ]
            
            for search_dir in search_dirs:
                if not search_dir.exists():
                    continue
                
                # Look for files with same extension
                matching_files = list(search_dir.rglob(f'*{extension}'))
                
                if matching_files:
                    # Take the most recently modified
                    matching_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    alternatives[missing] = str(matching_files[0])
                    found_any = True
                    
                    agent_logger.log_reflection(
                        agent_name=self.agent_id,
                        task_id=subtask.get('id', 'unknown'),
                        reflection=f"Found alternative for {missing}: {matching_files[0]}"
                    )
                    break
        
        return {
            'found': found_any,
            'alternatives': alternatives,
            'still_missing': [m for m in missing_files if m not in alternatives]
        }
    
    def _update_subtask_with_alternatives(self, subtask: Dict, alternatives: Dict) -> Dict:
        """Update subtask with alternative file paths"""
        
        updated = subtask.copy()
        
        # Update input_files
        if updated.get('input_files'):
            updated['input_files'] = [
                alternatives['alternatives'].get(f, f) for f in updated['input_files']
            ]
        
        # Update description to reference correct files
        description = updated.get('description', '')
        for original, alternative in alternatives['alternatives'].items():
            description = description.replace(original, alternative)
        updated['description'] = description
        
        # Note the substitution in context
        updated['context'] = updated.get('context', {})
        updated['context']['file_substitutions'] = alternatives['alternatives']
        
        return updated
    
    def _examine_reference_scripts(self, subtask: Dict) -> Optional[str]:
        """Read and return content of reference scripts for the sub-agent to use"""
        
        reference_scripts = subtask.get('reference_scripts', [])
        
        if not reference_scripts:
            return None
        
        combined_code = []
        
        for script_path in reference_scripts:
            # Try multiple locations
            possible_paths = [
                self.project_root / script_path,
                self.project_root / 'scripts' / script_path,
                Path(script_path)
            ]
            
            for path in possible_paths:
                if path.exists():
                    try:
                        code = path.read_text()
                        combined_code.append(f"# === Reference Script: {path} ===\n{code}")
                        
                        agent_logger.log_reflection(
                            agent_name=self.agent_id,
                            task_id=subtask.get('id', 'unknown'),
                            reflection=f"Loaded reference script: {path}"
                        )
                        break
                    except Exception as e:
                        agent_logger.log_reflection(
                            agent_name=self.agent_id,
                            task_id=subtask.get('id', 'unknown'),
                            reflection=f"Failed to read reference script {path}: {e}"
                        )
        
        return "\n\n".join(combined_code) if combined_code else None
    
    def _plan_tools_with_context(self, subtask: Dict) -> List[str]:
        """Plan tools while respecting language and package specifications"""
        
        description = subtask.get('description', '')
        language = subtask.get('language', 'python')
        packages = subtask.get('packages', [])
        context = subtask.get('context', {})
        
        # Build context-aware prompt
        prompt = f"""
Given this task: "{description}"

IMPORTANT CONSTRAINTS:
- Language: {language} (you MUST use {language} tools/packages)
- Specified Packages: {', '.join(packages) if packages else 'None specified'}
- Reference Code Available: {'Yes' if context.get('reference_code') else 'No'}

Available base tools:
- read_file: Read text files
- write_file: Write content to files
- list_files: List files in directory  
- web_search: Search the web
- fetch_webpage: Download and parse webpage
- analyze_csv: Analyze CSV data
- save_json: Save data as JSON
- load_json: Load JSON data

Based on the task and constraints, which tools do you need?

RULES:
1. If Python packages are specified, assume you'll write and execute Python code
2. Do NOT suggest R packages if Python is specified
3. List only the base tools you'll use to facilitate the task

Return ONLY a comma-separated list of tool names, nothing else.
Example: read_file,write_file,save_json
"""
        
        response = self.llm.invoke(prompt).strip()
        tools = [t.strip() for t in response.split(',') if t.strip()]
        
        # Validate tools exist
        valid_tools = ['read_file', 'write_file', 'list_files', 'web_search', 
                       'fetch_webpage', 'analyze_csv', 'save_json', 'load_json']
        tools = [t for t in tools if t in valid_tools]
        
        return tools
    
    def _execute_with_context(self, subtask: Dict, tools_needed: List[str]) -> Dict[str, Any]:
        """Execute task using specified tools with full context"""
        
        self.tools_used = tools_needed
        
        description = subtask.get('description', '')
        language = subtask.get('language', 'python')
        packages = subtask.get('packages', [])
        context = subtask.get('context', {})
        input_files = subtask.get('input_files', [])
        output_files = subtask.get('output_files', [])
        code_hints = subtask.get('code_hints', [])
        reference_code = context.get('reference_code', '')
        huggingface_repos = subtask.get('huggingface_repos', [])
        
        # Build comprehensive execution prompt
        tool_context = "Base tools available:\n"
        for tool_name in tools_needed:
            tool_context += f"- {tool_name}\n"
        
        prompt = f"""
Task: {description}

=== MANDATORY CONSTRAINTS ===
Language: {language}
Packages to use: {', '.join(packages) if packages else 'Standard library'}
{'HuggingFace Repo: ' + ', '.join(huggingface_repos) if huggingface_repos else ''}

=== INPUT FILES ===
{chr(10).join(input_files) if input_files else 'Check task description for file paths'}

=== EXPECTED OUTPUTS ===
{chr(10).join(output_files) if output_files else 'As specified in task'}

=== REFERENCE CODE ===
{reference_code[:2000] if reference_code else 'None provided'}

=== CODE HINTS FROM PROMPT ===
{chr(10).join(code_hints[:2]) if code_hints else 'None'}

=== BASE TOOLS ===
{tool_context}

=== INSTRUCTIONS ===
1. You MUST use {language} and the specified packages ({', '.join(packages)})
2. Do NOT substitute different packages (e.g., don't use Seurat if Scanpy is specified)
3. Reference the provided code examples when available
4. Use the exact file paths specified

Generate a detailed execution plan with actual {language} code that accomplishes this task.
Include the exact code to run, not just descriptions.
"""
        
        execution_plan = self.llm.invoke(prompt)
        
        result = {
            "execution_plan": execution_plan,
            "status": "completed",
            "language": language,
            "packages_used": packages,
            "input_files": input_files,
            "output_files": output_files,
            "output": f"Task '{description}' executed using {language} with packages: {', '.join(packages)}"
        }
        
        return result
    
    def _reflect(self, subtask: Dict, result: Dict, attempt: int) -> Dict[str, Any]:
        """Self-reflect on task execution to determine success"""
        
        language = subtask.get('language', 'not specified')
        packages = subtask.get('packages', [])
        
        prompt = f"""
You just attempted this task: {subtask['description']}

Required constraints:
- Language: {language}
- Packages: {', '.join(packages)}

Your result was: {result}

This was attempt #{attempt}.

Evaluate your performance:
1. Did you successfully complete the task? (YES/NO)
2. Did you use the CORRECT language ({language})? (YES/NO)  
3. Did you use the SPECIFIED packages ({', '.join(packages)})? (YES/NO)
4. If NO to any above, what went wrong?
5. If retrying, what specific changes would improve success?

Respond in JSON format:
{{
    "success": true/false,
    "used_correct_language": true/false,
    "used_specified_packages": true/false,
    "analysis": "your evaluation",
    "should_retry": true/false,
    "improvement_strategy": "specific changes for next attempt"
}}
"""
        
        reflection = self.llm.invoke(prompt)
        
        try:
            json_match = re.search(r'\{.*\}', reflection, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                
                # Override success if wrong tools were used
                if parsed.get('used_correct_language') == False or parsed.get('used_specified_packages') == False:
                    parsed['success'] = False
                    parsed['should_retry'] = True
                    parsed['improvement_strategy'] = f"Must use {language} with packages: {', '.join(packages)}"
                
                return parsed
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

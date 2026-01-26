"""
Sub-agent that executes tasks with built-in reflection capability.
Supports both interactive execution and SLURM job submission.
Includes file exploration and reasoning about missing resources.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import json
import time
import re

# Use updated import to avoid deprecation warning
try:
    from langchain_ollama import OllamaLLM
except ImportError:
    from langchain_community.llms import Ollama as OllamaLLM

from utils.logging_config import agent_logger
from tools.sandbox import Sandbox
from tools.execution_tools import ExecutionTools
from tools.conda_tools import CondaTools
from tools.web_search_tools import WebSearchTools
from tools.slurm_tools import SlurmTools, SlurmConfig


class SubAgent:
    """
    Executes subtasks with reflection, retry logic, and real code execution.
    Supports SLURM job submission for cluster-based execution.
    Limited to max_iterations attempts before generating final report.
    Includes file exploration capability to find missing resources.
    """
    
    def __init__(
        self,
        agent_id: str,
        sandbox: Sandbox,
        conda_tools: CondaTools,
        slurm_tools: SlurmTools = None,
        ollama_model: str = "llama3.1:70b",
        ollama_base_url: str = "http://127.0.0.1:11434",
        max_iterations: int = 12,
        use_slurm: bool = False,
        slurm_config: Dict[str, Any] = None
    ):
        """
        Initialize sub-agent.
        
        Args:
            agent_id: Unique identifier for this agent
            sandbox: Sandbox instance for file operations
            conda_tools: CondaTools instance for environment management
            slurm_tools: SlurmTools instance for job submission (optional)
            ollama_model: Ollama model to use
            ollama_base_url: Ollama server URL
            max_iterations: Maximum attempts before final report (default 12)
            use_slurm: Whether to use SLURM for script execution
            slurm_config: SLURM job configuration overrides
        """
        self.agent_id = agent_id
        self.sandbox = sandbox
        self.conda_tools = conda_tools
        self.slurm_tools = slurm_tools
        self.max_iterations = max_iterations
        self.use_slurm = use_slurm and slurm_tools is not None
        self.slurm_config = slurm_config or {}
        
        # Initialize LLM with new import
        self.llm = OllamaLLM(
            model=ollama_model,
            base_url=ollama_base_url
        )
        
        # Initialize execution tools (for interactive mode)
        self.execution = ExecutionTools(sandbox, conda_tools)
        self.web_search = WebSearchTools(cache_dir=sandbox.get_temp_dir() / "search_cache")
        
        # Tracking
        self.iteration_count = 0
        self.execution_history = []
        self.files_created = []
        self.files_modified = []
        self.tools_used = []
        self.errors_encountered = []
        self.slurm_jobs = []  # Track submitted SLURM jobs
        self.file_search_cache = {}  # Cache file search results
    
    def execute(
        self,
        subtask: Dict[str, Any],
        env_name: str = None,
        prior_attempts: int = 0  # Track attempts across workflow invocations
    ) -> Dict[str, Any]:
        """
        Execute a subtask with iteration limit.
        
        Args:
            subtask: Dictionary with 'id', 'description', 'context', 'success_criteria'
            env_name: Conda environment to use
            prior_attempts: Number of attempts from previous workflow invocations
            
        Returns:
            Result dictionary with success status, report, and execution details
        """
        task_id = subtask['id']
        description = subtask['description']
        context = subtask.get('context', {})
        success_criteria = subtask.get('success_criteria', '')
        expected_outputs = context.get('expected_outputs', [])
        
        # Calculate effective max iterations accounting for prior attempts
        effective_max = max(1, self.max_iterations - prior_attempts)
        
        agent_logger.log_task_start(
            agent_name=self.agent_id,
            task_id=task_id,
            description=f"{description} (prior_attempts={prior_attempts}, max={effective_max})",
            attempt=prior_attempts + 1
        )
        
        # Reset tracking for this task
        self.iteration_count = 0
        self.execution_history = []
        self.files_created = []
        self.files_modified = []
        self.tools_used = []
        self.errors_encountered = []
        self.slurm_jobs = []
        
        # If we've already exceeded max attempts across invocations, fail immediately
        if prior_attempts >= self.max_iterations:
            agent_logger.log_task_failure(
                agent_name=self.agent_id,
                task_id=task_id,
                error=f"Maximum total attempts ({self.max_iterations}) already exceeded",
                context={"prior_attempts": prior_attempts}
            )
            return self._generate_max_attempts_result(subtask, prior_attempts)
        
        # FIRST: Explore project structure and identify required files
        file_exploration = self._explore_and_identify_files(description, context)
        if file_exploration.get('missing_critical'):
            # Try to resolve missing files before starting main execution
            resolution = self._resolve_missing_files(file_exploration['missing_files'], context)
            if not resolution['resolved'] and resolution.get('blocking'):
                # Can't proceed without these files
                self.errors_encountered.append({
                    "type": "missing_files",
                    "files": file_exploration['missing_files'],
                    "resolution_attempted": resolution
                })
                return {
                    "success": False,
                    "task_id": task_id,
                    "iterations": 0,
                    "max_iterations": self.max_iterations,
                    "total_attempts": prior_attempts,
                    "result": None,
                    "report": self._generate_final_report(subtask, False, None),
                    "files_created": [],
                    "files_modified": [],
                    "tools_used": ["file_exploration"],
                    "errors": self.errors_encountered,
                    "execution_history": [],
                    "slurm_jobs": [],
                    "execution_mode": "blocked_by_missing_files",
                    "should_reformulate": True,
                    "missing_files_info": file_exploration
                }
        
        # Main execution loop
        success = False
        final_result = None
        
        while self.iteration_count < effective_max and not success:
            self.iteration_count += 1
            total_attempts = prior_attempts + self.iteration_count
            
            agent_logger.log_task_start(
                agent_name=self.agent_id,
                task_id=task_id,
                description=f"Iteration {self.iteration_count}/{effective_max} (total: {total_attempts}/{self.max_iterations})",
                attempt=total_attempts
            )
            
            # Get execution plan from LLM (include file exploration results)
            plan = self._get_execution_plan(
                description,
                context,
                success_criteria,
                self.execution_history,
                file_exploration
            )
            
            # Execute the plan (using SLURM or interactive)
            if self.use_slurm:
                iteration_result = self._execute_plan_slurm(plan, env_name)
            else:
                iteration_result = self._execute_plan_interactive(plan, env_name)
            
            # Record this iteration
            self.execution_history.append({
                "iteration": self.iteration_count,
                "total_attempt": total_attempts,
                "plan": plan,
                "result": iteration_result,
                "execution_mode": "slurm" if self.use_slurm else "interactive",
                "timestamp": datetime.now().isoformat()
            })
            
            # Check if we've achieved success
            if expected_outputs:
                output_check = self.execution.check_output_files(expected_outputs)
                if output_check["all_exist"]:
                    success = True
                    final_result = iteration_result
            elif iteration_result.get("success"):
                # If no specific outputs expected, trust the execution result
                success = self._reflect_on_result(
                    subtask,
                    iteration_result,
                    total_attempts
                )
                if success:
                    final_result = iteration_result
        
        # Generate final report
        final_report = self._generate_final_report(
            subtask,
            success,
            final_result
        )
        
        total_attempts_used = prior_attempts + self.iteration_count
        
        if success:
            agent_logger.log_task_success(
                agent_name=self.agent_id,
                task_id=task_id,
                result=final_result,
                tools_used=self.tools_used
            )
        else:
            agent_logger.log_task_failure(
                agent_name=self.agent_id,
                task_id=task_id,
                error=f"Failed after {self.iteration_count} iterations (total: {total_attempts_used})",
                context={"errors": self.errors_encountered}
            )
        
        return {
            "success": success,
            "task_id": task_id,
            "iterations": self.iteration_count,
            "max_iterations": self.max_iterations,
            "total_attempts": total_attempts_used,
            "result": final_result,
            "report": final_report,
            "files_created": self.files_created,
            "files_modified": self.files_modified,
            "tools_used": list(set(self.tools_used)),
            "errors": self.errors_encountered,
            "execution_history": self.execution_history,
            "slurm_jobs": self.slurm_jobs,
            "execution_mode": "slurm" if self.use_slurm else "interactive",
            "file_exploration": file_exploration
        }
    
    def _generate_max_attempts_result(self, subtask: Dict, prior_attempts: int) -> Dict[str, Any]:
        """Generate result when max attempts already exceeded"""
        return {
            "success": False,
            "task_id": subtask['id'],
            "iterations": 0,
            "max_iterations": self.max_iterations,
            "total_attempts": prior_attempts,
            "result": None,
            "report": {
                "task_id": subtask["id"],
                "task_description": subtask["description"],
                "success": False,
                "summary": f"Task abandoned: exceeded maximum {self.max_iterations} total attempts across workflow invocations",
                "recommendation": "SKIP or ESCALATE - do not retry"
            },
            "files_created": [],
            "files_modified": [],
            "tools_used": [],
            "errors": [{"error": "max_attempts_exceeded", "total": prior_attempts}],
            "execution_history": [],
            "slurm_jobs": [],
            "execution_mode": "skipped",
            "should_skip": True  # Signal to workflow to skip, not reformulate
        }
    
    def _explore_and_identify_files(self, description: str, context: Dict) -> Dict[str, Any]:
        """
        Explore project structure and identify files needed for the task.
        This runs BEFORE the main execution loop to catch missing file issues early.
        """
        self.tools_used.append("file_exploration")
        
        # Get current directory structure
        file_tree = self.sandbox.get_directory_tree(max_depth=3)
        
        # Get list of all files
        all_files = self._get_all_files()
        
        prompt = f"""Analyze this task and identify what files are needed:

Task: {description}

Context: {json.dumps(context, indent=2)}

Current project structure:
```
{file_tree}
```

Available files: {json.dumps(all_files[:50], indent=2)}  # First 50 files

Questions to answer:
1. What input files does this task need?
2. Are those files present in the project structure?
3. If files are mentioned but not found, where might they be?
4. Can this task proceed without any missing files?

Respond in JSON:
{{
    "required_files": [
        {{"name": "filename", "purpose": "why needed", "found": true/false, "path": "path if found"}}
    ],
    "missing_files": ["list of files not found"],
    "missing_critical": true/false,
    "suggestions": ["where to look or how to proceed"],
    "can_proceed": true/false,
    "alternative_approach": "if files missing, what else can we do"
}}
"""
        
        try:
            response = self.llm.invoke(prompt)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                result['file_tree'] = file_tree
                result['all_files'] = all_files[:50]
                return result
        except Exception as e:
            agent_logger.log_reflection(
                agent_name=self.agent_id,
                task_id="file_exploration",
                reflection=f"File exploration failed: {e}"
            )
        
        return {
            "required_files": [],
            "missing_files": [],
            "missing_critical": False,
            "can_proceed": True,
            "file_tree": file_tree,
            "all_files": all_files[:50]
        }
    
    def _get_all_files(self) -> List[str]:
        """Get list of all files in project directory"""
        all_files = []
        try:
            for path in self.sandbox.project_dir.rglob("*"):
                if path.is_file():
                    # Skip hidden files and common non-essential directories
                    rel_path = path.relative_to(self.sandbox.project_dir)
                    parts = rel_path.parts
                    if any(p.startswith('.') for p in parts):
                        continue
                    if any(p in ['__pycache__', 'node_modules', '.git', 'logs'] for p in parts):
                        continue
                    all_files.append(str(rel_path))
        except Exception as e:
            pass
        return sorted(all_files)
    
    def _resolve_missing_files(self, missing_files: List[str], context: Dict) -> Dict[str, Any]:
        """
        Try to resolve missing files by searching or suggesting alternatives.
        """
        self.tools_used.append("file_resolution")
        
        resolved = []
        still_missing = []
        
        for filename in missing_files:
            # Search for file by name (case-insensitive)
            found_path = self._search_for_file(filename)
            if found_path:
                resolved.append({"name": filename, "found_at": found_path})
            else:
                still_missing.append(filename)
        
        # If we found some files, update context
        if resolved:
            agent_logger.log_reflection(
                agent_name=self.agent_id,
                task_id="file_resolution",
                reflection=f"Found {len(resolved)} of {len(missing_files)} missing files: {resolved}"
            )
        
        # Ask LLM what to do about still-missing files
        if still_missing:
            file_tree = self.sandbox.get_directory_tree(max_depth=2)
            
            prompt = f"""These files are needed but not found anywhere in the project:

Missing files: {still_missing}

Project structure:
```
{file_tree}
```

Context about the task: {json.dumps(context, indent=2)}

Are these files:
1. Files that should have been created by a previous step?
2. Files that need to be downloaded/fetched?
3. Files with wrong names (typos)?
4. Not actually required (can proceed without)?

Respond in JSON:
{{
    "resolved": true/false,
    "blocking": true/false,
    "reason": "explanation",
    "action": "what to do",
    "alternative_files": ["files that could substitute"],
    "recommendation": "PROCEED/WAIT/FAIL"
}}
"""
            
            try:
                response = self.llm.invoke(prompt)
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    result['resolved_files'] = resolved
                    result['still_missing'] = still_missing
                    return result
            except:
                pass
        
        return {
            "resolved": len(still_missing) == 0,
            "blocking": len(still_missing) > 0,
            "resolved_files": resolved,
            "still_missing": still_missing
        }
    
    def _search_for_file(self, filename: str) -> Optional[str]:
        """Search for a file in the project directory"""
        
        # Check cache first
        if filename in self.file_search_cache:
            return self.file_search_cache[filename]
        
        # Extract just the filename without path
        base_name = Path(filename).name
        
        # Search strategies
        search_patterns = [
            filename,  # Exact match
            f"*/{filename}",  # One level deep
            f"**/{filename}",  # Any depth
            f"**/{base_name}",  # Just filename at any depth
            f"**/*{base_name}*",  # Partial match
        ]
        
        for pattern in search_patterns:
            try:
                matches = list(self.sandbox.project_dir.glob(pattern))
                if matches:
                    result = str(matches[0].relative_to(self.sandbox.project_dir))
                    self.file_search_cache[filename] = result
                    return result
            except Exception:
                continue
        
        self.file_search_cache[filename] = None
        return None
    
    def _get_execution_plan(
        self,
        description: str,
        context: Dict,
        success_criteria: str,
        history: List[Dict],
        file_exploration: Dict = None
    ) -> Dict[str, Any]:
        """Get execution plan from LLM"""
        
        # Build context about previous attempts
        history_summary = ""
        if history:
            history_summary = "\n\nPrevious attempts:\n"
            for h in history[-3:]:  # Last 3 attempts
                history_summary += f"- Iteration {h['iteration']} (total: {h.get('total_attempt', h['iteration'])}): "
                if h['result'].get('success'):
                    history_summary += "Partially successful\n"
                else:
                    errors = h['result'].get('error', h['result'].get('step_results', []))
                    history_summary += f"Failed - {str(errors)[:200]}\n"
        
        # Current file structure
        file_tree = self.sandbox.get_directory_tree(max_depth=2)
        
        # File exploration context
        file_context = ""
        if file_exploration:
            if file_exploration.get('required_files'):
                file_context = "\n\nFile status:\n"
                for f in file_exploration.get('required_files', []):
                    status = "✓ Found" if f.get('found') else "✗ Missing"
                    file_context += f"- {f.get('name')}: {status}"
                    if f.get('path'):
                        file_context += f" at {f['path']}"
                    file_context += "\n"
        
        # Execution mode context
        execution_mode = "SLURM cluster (jobs submitted via sbatch)" if self.use_slurm else "Interactive (direct execution)"
        
        prompt = f"""You are executing this task: {description}

Success criteria: {success_criteria if success_criteria else 'Complete the task as described'}

Execution mode: {execution_mode}

Current project structure:
```
{file_tree}
```

Context: {json.dumps(context, indent=2)}
{file_context}
{history_summary}

Available capabilities:
1. Write and execute Python scripts
2. Write and execute R scripts  
3. Write and execute bash scripts
4. Write and execute Perl scripts
5. Search the web for information
6. Read and write files
7. Install packages via conda (no sudo)
8. List and search for files in the project

{"NOTE: Scripts will be submitted to SLURM. For long-running tasks, this is more efficient." if self.use_slurm else ""}

IMPORTANT: If previous attempts failed, try a DIFFERENT approach. Don't repeat the same steps.

Create a specific execution plan. Respond in JSON format:
{{
    "steps": [
        {{
            "action": "write_script|execute_script|execute_command|search_web|read_file|write_file|install_package|list_files",
            "language": "python|r|bash|perl" (for scripts),
            "filename": "script name" (for write_script),
            "code": "code content" (for write_script),
            "command": "command" (for execute_command),
            "query": "search query" (for search_web),
            "filepath": "path" (for read/write file),
            "content": "content" (for write_file),
            "directory": "path" (for list_files),
            "packages": ["pkg1", "pkg2"] (for install_package),
            "cpus": 4 (optional, for SLURM),
            "memory": "16G" (optional, for SLURM),
            "time": "01:00:00" (optional, for SLURM),
            "reason": "why this step"
        }}
    ],
    "expected_outcomes": ["what files/outputs this should produce"],
    "different_from_previous": "how this plan differs from failed attempts"
}}

Keep the plan focused and achievable. Prioritize getting working code over perfect code.
"""
        
        response = self.llm.invoke(prompt)
        
        # Parse JSON response
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
        
        # Fallback plan
        return {
            "steps": [
                {
                    "action": "list_files",
                    "directory": ".",
                    "reason": "Explore project structure to understand what's available"
                },
                {
                    "action": "execute_command",
                    "command": "echo 'Could not parse plan, listing files' && ls -la",
                    "reason": "Fallback"
                }
            ],
            "expected_outcomes": []
        }
    
    def _execute_plan_interactive(self, plan: Dict, env_name: str) -> Dict[str, Any]:
        """Execute the steps in the plan interactively (original behavior)"""
        
        results = []
        overall_success = True
        
        for step in plan.get("steps", []):
            action = step.get("action", "")
            step_result = {"action": action, "success": False}
            
            try:
                if action == "write_script":
                    self.tools_used.append("write_script")
                    result = self.execution.write_script(
                        code=step.get("code", ""),
                        filename=step.get("filename", "temp_script"),
                        language=step.get("language", "python")
                    )
                    step_result = result
                    if result["success"]:
                        self.files_created.append(result["script_path"])
                
                elif action == "execute_script":
                    self.tools_used.append("execute_script")
                    exec_result = self.execution.execute_script(
                        script_path=step.get("filepath", step.get("filename", "")),
                        args=step.get("args", []),
                        env_name=env_name,
                        timeout=step.get("timeout", 300)
                    )
                    step_result = exec_result.to_dict()
                
                elif action == "execute_command":
                    self.tools_used.append("execute_command")
                    exec_result = self.execution.execute_command(
                        command=step.get("command", ""),
                        env_name=env_name,
                        timeout=step.get("timeout", 300)
                    )
                    step_result = exec_result.to_dict()
                
                elif action == "search_web":
                    self.tools_used.append("web_search")
                    step_result = self.web_search.search(
                        query=step.get("query", ""),
                        max_results=step.get("max_results", 5)
                    )
                
                elif action == "read_file":
                    self.tools_used.append("read_file")
                    step_result = self.execution.read_output_file(
                        filepath=step.get("filepath", "")
                    )
                
                elif action == "write_file":
                    self.tools_used.append("write_file")
                    filepath = self.sandbox.safe_path(step.get("filepath", "temp.txt"))
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    filepath.write_text(step.get("content", ""))
                    step_result = {"success": True, "filepath": str(filepath)}
                    self.files_created.append(str(filepath))
                
                elif action == "list_files":
                    self.tools_used.append("list_files")
                    directory = step.get("directory", ".")
                    try:
                        dir_path = self.sandbox.safe_path(directory)
                        files = list(dir_path.rglob("*"))[:100]  # Limit results
                        step_result = {
                            "success": True,
                            "files": [str(f.relative_to(self.sandbox.project_dir)) for f in files if f.is_file()],
                            "count": len(files)
                        }
                    except Exception as e:
                        step_result = {"success": False, "error": str(e)}
                
                elif action == "install_package":
                    self.tools_used.append("conda_install")
                    packages = step.get("packages", [])
                    if packages and env_name:
                        step_result = self.conda_tools.install_packages(
                            env_name=env_name,
                            packages=packages,
                            use_pip=step.get("use_pip", False)
                        )
                    else:
                        step_result = {"success": False, "error": "No packages or env specified"}
                
                else:
                    step_result = {"success": False, "error": f"Unknown action: {action}"}
                
            except Exception as e:
                step_result = {"success": False, "error": str(e)}
                self.errors_encountered.append({
                    "action": action,
                    "error": str(e),
                    "iteration": self.iteration_count
                })
            
            results.append(step_result)
            
            if not step_result.get("success"):
                overall_success = False
        
        return {
            "success": overall_success,
            "steps_executed": len(results),
            "step_results": results,
            "files_created": self.files_created[-len(results):] if self.files_created else [],
            "execution_mode": "interactive"
        }
    
    def _execute_plan_slurm(self, plan: Dict, env_name: str) -> Dict[str, Any]:
        """Execute the steps in the plan via SLURM job submission"""
        
        results = []
        overall_success = True
        submitted_jobs = []
        
        # Identify parallel steps
        parallel_indices = set(plan.get("parallel_steps", []))
        
        for i, step in enumerate(plan.get("steps", [])):
            action = step.get("action", "")
            step_result = {"action": action, "success": False}
            
            try:
                # Actions that don't need SLURM
                if action in ["search_web", "read_file", "write_file", "install_package", "list_files"]:
                    # Execute these interactively
                    if action == "search_web":
                        self.tools_used.append("web_search")
                        step_result = self.web_search.search(
                            query=step.get("query", ""),
                            max_results=step.get("max_results", 5)
                        )
                    elif action == "read_file":
                        self.tools_used.append("read_file")
                        step_result = self.execution.read_output_file(
                            filepath=step.get("filepath", "")
                        )
                    elif action == "write_file":
                        self.tools_used.append("write_file")
                        filepath = self.sandbox.safe_path(step.get("filepath", "temp.txt"))
                        filepath.parent.mkdir(parents=True, exist_ok=True)
                        filepath.write_text(step.get("content", ""))
                        step_result = {"success": True, "filepath": str(filepath)}
                        self.files_created.append(str(filepath))
                    elif action == "list_files":
                        self.tools_used.append("list_files")
                        directory = step.get("directory", ".")
                        try:
                            dir_path = self.sandbox.safe_path(directory)
                            files = list(dir_path.rglob("*"))[:100]
                            step_result = {
                                "success": True,
                                "files": [str(f.relative_to(self.sandbox.project_dir)) for f in files if f.is_file()],
                                "count": len(files)
                            }
                        except Exception as e:
                            step_result = {"success": False, "error": str(e)}
                    elif action == "install_package":
                        self.tools_used.append("conda_install")
                        packages = step.get("packages", [])
                        if packages and env_name:
                            step_result = self.conda_tools.install_packages(
                                env_name=env_name,
                                packages=packages,
                                use_pip=step.get("use_pip", False)
                            )
                        else:
                            step_result = {"success": False, "error": "No packages or env specified"}
                
                # Script writing (prepare for SLURM)
                elif action == "write_script":
                    self.tools_used.append("write_script")
                    result = self.execution.write_script(
                        code=step.get("code", ""),
                        filename=step.get("filename", "temp_script"),
                        language=step.get("language", "python")
                    )
                    step_result = result
                    if result["success"]:
                        self.files_created.append(result["script_path"])
                
                # Script/command execution via SLURM
                elif action in ["execute_script", "execute_command"]:
                    self.tools_used.append("slurm_submit")
                    
                    if action == "execute_script":
                        script_path = step.get("filepath", step.get("filename", ""))
                        # Read the script content
                        full_path = self.sandbox.safe_path(script_path)
                        if full_path.exists():
                            script_content = full_path.read_text()
                            language = step.get("language", "bash")
                            # Detect language from extension if not specified
                            if not step.get("language"):
                                ext = full_path.suffix.lower()
                                lang_map = {".py": "python", ".r": "r", ".pl": "perl", ".sh": "bash"}
                                language = lang_map.get(ext, "bash")
                        else:
                            step_result = {"success": False, "error": f"Script not found: {script_path}"}
                            results.append(step_result)
                            overall_success = False
                            continue
                    else:
                        # execute_command
                        script_content = step.get("command", "")
                        language = "bash"
                    
                    # Submit to SLURM
                    job_name = f"{self.agent_id}_step{i}"
                    
                    # Determine dependencies
                    dependencies = []
                    if i not in parallel_indices and submitted_jobs:
                        # If not parallel, depend on previous job
                        dependencies = [submitted_jobs[-1]["job_id"]]
                    
                    submit_result = self.slurm_tools.submit_script(
                        script_content=script_content,
                        job_name=job_name,
                        language=language,
                        conda_env=env_name,
                        cpus=step.get("cpus", self.slurm_config.get("cpus", 4)),
                        memory=step.get("memory", self.slurm_config.get("memory", "16G")),
                        time_limit=step.get("time", self.slurm_config.get("time", "04:00:00")),
                        partition=step.get("partition", self.slurm_config.get("partition")),
                        gpus=step.get("gpus", self.slurm_config.get("gpus", 0)),
                        gpu_type=step.get("gpu_type", self.slurm_config.get("gpu_type")),
                        nodelist=step.get("nodelist", self.slurm_config.get("nodelist")),
                        exclude_nodes=step.get("exclude_nodes", self.slurm_config.get("exclude_nodes")),
                        dependencies=dependencies if dependencies else None
                    )
                    
                    if submit_result["success"]:
                        submitted_jobs.append({
                            "job_id": submit_result["job_id"],
                            "step_index": i,
                            "action": action
                        })
                        self.slurm_jobs.append(submit_result)
                        step_result = {
                            "success": True,
                            "job_id": submit_result["job_id"],
                            "message": f"Job submitted: {submit_result['job_id']}"
                        }
                    else:
                        step_result = submit_result
                        overall_success = False
                
                else:
                    step_result = {"success": False, "error": f"Unknown action: {action}"}
                
            except Exception as e:
                step_result = {"success": False, "error": str(e)}
                self.errors_encountered.append({
                    "action": action,
                    "error": str(e),
                    "iteration": self.iteration_count
                })
                overall_success = False
            
            results.append(step_result)
        
        # Wait for all submitted SLURM jobs to complete
        if submitted_jobs:
            job_ids = [j["job_id"] for j in submitted_jobs]
            wait_result = self.slurm_tools.wait_for_jobs(
                job_ids,
                poll_interval=self.slurm_config.get("poll_interval", 10),
                max_attempts=self.slurm_config.get("max_poll_attempts", 360)
            )
            
            # Check job results
            if wait_result["success"]:
                for job_id, status in wait_result["jobs"].items():
                    if status.get("exit_code") and status["exit_code"] != "0":
                        overall_success = False
                        # Get job output for error info
                        output = self.slurm_tools.get_job_output(job_id)
                        self.errors_encountered.append({
                            "job_id": job_id,
                            "error": f"Job failed with exit code {status['exit_code']}",
                            "stderr": output.get("stderr", "")[:500],
                            "iteration": self.iteration_count
                        })
            else:
                overall_success = False
                self.errors_encountered.append({
                    "error": "Some SLURM jobs did not complete",
                    "timed_out": wait_result.get("timed_out", []),
                    "iteration": self.iteration_count
                })
        
        return {
            "success": overall_success,
            "steps_executed": len(results),
            "step_results": results,
            "jobs_submitted": len(submitted_jobs),
            "slurm_jobs": submitted_jobs,
            "files_created": self.files_created[-len(results):] if self.files_created else [],
            "execution_mode": "slurm"
        }
    
    def _reflect_on_result(
        self,
        subtask: Dict,
        result: Dict,
        total_attempts: int
    ) -> bool:
        """Have LLM reflect on whether the task is complete"""
        
        file_tree = self.sandbox.get_directory_tree(max_depth=2)
        
        # If SLURM was used, get job outputs
        job_outputs = ""
        if result.get("execution_mode") == "slurm" and self.slurm_jobs:
            job_outputs = "\n\nSLURM Job Outputs:\n"
            for job in self.slurm_jobs[-3:]:  # Last 3 jobs
                if job.get("job_id"):
                    output = self.slurm_tools.get_job_output(job["job_id"])
                    job_outputs += f"\nJob {job['job_id']}:\n"
                    job_outputs += f"stdout (last 500 chars): {output.get('stdout', '')[-500:]}\n"
                    if output.get("stderr"):
                        job_outputs += f"stderr: {output.get('stderr', '')[:200]}\n"
        
        prompt = f"""Evaluate if this task is complete:

Task: {subtask['description']}
Success Criteria: {subtask.get('success_criteria', 'Task completed as described')}

Attempt {total_attempts} of {self.max_iterations} maximum result:
{json.dumps(result, indent=2, default=str)[:2000]}
{job_outputs}

Current project files:
```
{file_tree}
```

Is the task COMPLETE? Be strict - only say complete if the success criteria are actually met.

Answer with JSON:
{{
    "complete": true/false,
    "reasoning": "brief explanation",
    "missing": ["what's still needed if not complete"],
    "confidence": 0.0-1.0
}}
"""
        
        response = self.llm.invoke(prompt)
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                reflection = json.loads(json_match.group())
                return reflection.get("complete", False)
        except:
            pass
        
        # Default to checking if "complete" or "success" appears
        return "complete" in response.lower() or "yes" in response.lower()[:50]
    
    def _generate_final_report(
        self,
        subtask: Dict,
        success: bool,
        final_result: Optional[Dict]
    ) -> Dict[str, Any]:
        """Generate final report for the master agent"""
        
        file_tree = self.sandbox.get_directory_tree(max_depth=3)
        
        # Summarize execution history
        history_summary = []
        for h in self.execution_history:
            history_summary.append({
                "iteration": h["iteration"],
                "total_attempt": h.get("total_attempt", h["iteration"]),
                "success": h["result"].get("success", False),
                "steps_executed": h["result"].get("steps_executed", 0),
                "mode": h.get("execution_mode", "interactive")
            })
        
        # SLURM job summary
        slurm_summary = ""
        if self.slurm_jobs:
            slurm_summary = f"\nSLURM Jobs Submitted: {len(self.slurm_jobs)}"
            for job in self.slurm_jobs:
                slurm_summary += f"\n  - Job {job.get('job_id', 'unknown')}"
        
        # Get LLM to generate human-readable summary
        prompt = f"""Generate a summary report for this completed subtask:

Task: {subtask['description']}
Success: {success}
Iterations used: {self.iteration_count}/{self.max_iterations}
Execution mode: {"SLURM cluster" if self.use_slurm else "Interactive"}
{slurm_summary}

Files created: {self.files_created}
Tools used: {list(set(self.tools_used))}
Errors encountered: {len(self.errors_encountered)}

Error details (if any): {json.dumps(self.errors_encountered[:3], default=str)[:500]}

Final project structure:
```
{file_tree}
```

Write a concise summary (3-5 sentences) of:
1. What was accomplished (or what failed)
2. Key files created
3. Any issues encountered and their likely causes
4. Recommendations for next steps (especially if failed)
"""
        
        summary = self.llm.invoke(prompt)
        
        report = {
            "task_id": subtask["id"],
            "task_description": subtask["description"],
            "success": success,
            "iterations_used": self.iteration_count,
            "max_iterations": self.max_iterations,
            "execution_mode": "slurm" if self.use_slurm else "interactive",
            "summary": summary,
            "files_created": self.files_created,
            "files_modified": self.files_modified,
            "tools_used": list(set(self.tools_used)),
            "errors": self.errors_encountered,
            "slurm_jobs": self.slurm_jobs,
            "execution_history": history_summary,
            "file_structure": file_tree,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save report to file
        report_path = self.sandbox.get_reports_dir() / f"subtask_{subtask['id']}_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report

"""
Script-First Sub-Agent Architecture

This SubAgent follows a script-generation paradigm:
1. Receive subtask with context (packages, files, reference scripts)
2. Generate actual Python/R script file
3. Generate task-specific conda environment YAML
4. Create and submit SLURM sbatch job
5. Monitor job completion
6. On success: return script path, output files
7. On failure: parse logs, optionally run diagnostics, update script

Token-based context management replaces iteration limits.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import re
import json
import time
from datetime import datetime

# Use non-deprecated import
try:
    from langchain_ollama import OllamaLLM
except ImportError:
    from langchain_community.llms import Ollama as OllamaLLM

from utils.logging_config import agent_logger
from utils.context_manager import ContextManager, context_manager


class ScriptFirstSubAgent:
    """
    Executes subtasks by generating scripts and submitting SLURM jobs.
    
    Workflow:
    1. Analyze subtask requirements
    2. Generate script file (Python/R/bash)
    3. Generate conda_env.yml for task
    4. Submit via SLURM sbatch
    5. Monitor completion
    6. Return results or diagnose failures
    
    Uses token-based context limits (70K) instead of iteration counts.
    """
    
    # Token limits
    MAX_CONTEXT_TOKENS = 70_000
    MAX_TOOL_OUTPUT_TOKENS = 25_000
    MIN_TOKENS_FOR_RETRY = 10_000  # Minimum tokens needed to attempt another iteration
    
    def __init__(
        self,
        agent_id: str,
        sandbox=None,
        conda_tools=None,
        slurm_tools=None,
        ollama_model: str = "llama3.1:70b",
        ollama_base_url: str = "http://127.0.0.1:11434",
        use_slurm: bool = True,
        slurm_config: Dict[str, Any] = None,
        project_root: str = ".",
        **kwargs
    ):
        """
        Initialize ScriptFirstSubAgent.
        
        Args:
            agent_id: Unique identifier for this agent
            sandbox: Sandbox instance for file operations
            conda_tools: CondaTools instance for environment management
            slurm_tools: SlurmTools instance for job submission
            ollama_model: Ollama model for LLM calls
            ollama_base_url: Ollama server URL
            use_slurm: Whether to use SLURM (if False, runs locally)
            slurm_config: SLURM configuration options
            project_root: Fallback project root
        """
        self.agent_id = agent_id
        self.llm = OllamaLLM(model=ollama_model, base_url=ollama_base_url)
        
        # Infrastructure
        self.sandbox = sandbox
        self.conda_tools = conda_tools
        self.slurm_tools = slurm_tools
        self.use_slurm = use_slurm
        self.slurm_config = slurm_config or {}
        
        # Set project root
        if sandbox:
            self.project_root = Path(sandbox.project_dir)
        else:
            self.project_root = Path(project_root)
        
        # Context management
        self.context_mgr = ContextManager(
            max_context_tokens=self.MAX_CONTEXT_TOKENS,
            max_tool_output_tokens=self.MAX_TOOL_OUTPUT_TOKENS,
            llm_for_summarization=self.llm
        )
        self.context_window = self.context_mgr.create_context_window(agent_id)
        
        # Execution tracking
        self.scripts_generated: List[Path] = []
        self.jobs_submitted: List[str] = []
        self.iteration_count = 0
        
        # Store any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def execute(
        self,
        subtask: Dict[str, Any],
        env_name: str = None,
        prior_attempts: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute subtask using script-first paradigm.
        
        Args:
            subtask: Subtask dictionary with description, packages, files, etc.
            env_name: Base conda environment name (task-specific env will be created)
            prior_attempts: Previous attempts (for logging, not limits)
            
        Returns:
            Result dictionary with script_path, output_files, success status
        """
        task_id = subtask.get('id', 'unknown')
        description = subtask.get('description', subtask.get('title', 'Unknown task'))
        
        agent_logger.log_task_start(
            agent_name=self.agent_id,
            task_id=task_id,
            description=f"Script-first execution: {description[:100]}",
            attempt=prior_attempts + 1
        )
        
        # Initialize context with task information
        self._initialize_context(subtask)
        
        # Check if outputs already exist
        existing_check = self._check_existing_outputs(subtask)
        if existing_check['already_complete']:
            return self._success_result(
                task_id=task_id,
                message="Outputs already exist",
                skipped=True,
                existing_files=existing_check.get('existing_files', [])
            )
        
        # Main execution loop - token-based, not iteration-based
        while self._can_continue():
            self.iteration_count += 1
            
            try:
                # Step 1: Generate script
                script_result = self._generate_script(subtask)
                if not script_result['success']:
                    self._add_to_context('assistant', f"Script generation failed: {script_result.get('error')}")
                    continue
                
                script_path = script_result['script_path']
                self.scripts_generated.append(Path(script_path))
                
                # Step 2: Generate/update conda environment
                env_result = self._setup_environment(subtask, task_id)
                if not env_result['success']:
                    self._add_to_context('assistant', f"Environment setup failed: {env_result.get('error')}")
                    # Continue anyway - might work with base env
                
                task_env_name = env_result.get('env_name', env_name)
                
                # Step 3: Submit SLURM job or run locally
                if self.use_slurm and self.slurm_tools:
                    run_result = self._submit_slurm_job(
                        script_path=script_path,
                        task_id=task_id,
                        env_name=task_env_name,
                        subtask=subtask
                    )
                else:
                    run_result = self._run_locally(
                        script_path=script_path,
                        env_name=task_env_name
                    )
                
                # Step 4: Check results
                if run_result['success']:
                    # Verify output files exist
                    outputs_check = self._verify_outputs(subtask, run_result)
                    
                    if outputs_check['all_exist']:
                        return self._success_result(
                            task_id=task_id,
                            script_path=str(script_path),
                            output_files=outputs_check['found_files'],
                            job_id=run_result.get('job_id'),
                            env_yaml=env_result.get('yaml_path'),
                            iterations=self.iteration_count
                        )
                    else:
                        # Script ran but outputs missing
                        self._add_to_context(
                            'tool',
                            f"Script completed but outputs missing: {outputs_check['missing_files']}"
                        )
                        continue
                else:
                    # Job failed - parse logs and decide next action
                    diagnosis = self._diagnose_failure(run_result, script_path)
                    self._add_to_context('tool', f"Failure diagnosis:\n{diagnosis['summary']}")
                    
                    if diagnosis['recoverable']:
                        # Update subtask context with diagnosis for next iteration
                        subtask = self._update_subtask_with_diagnosis(subtask, diagnosis)
                        continue
                    else:
                        # Unrecoverable failure
                        return self._failure_result(
                            task_id=task_id,
                            error=diagnosis['summary'],
                            script_path=str(script_path),
                            logs=diagnosis.get('logs', {}),
                            iterations=self.iteration_count
                        )
                        
            except Exception as e:
                self._add_to_context('tool', f"Exception during execution: {str(e)}")
                agent_logger.log_task_failure(
                    agent_name=self.agent_id,
                    task_id=task_id,
                    error=str(e),
                    context={"iteration": self.iteration_count}
                )
        
        # Context exhausted
        return self._failure_result(
            task_id=task_id,
            error=f"Context window exhausted after {self.iteration_count} iterations",
            script_path=str(self.scripts_generated[-1]) if self.scripts_generated else None,
            context_status=self.context_mgr.get_context_status(self.agent_id),
            iterations=self.iteration_count
        )
    
    def _can_continue(self) -> bool:
        """Check if we have enough context budget to continue"""
        can_continue, reason = self.context_mgr.should_continue(
            self.agent_id, 
            min_tokens_needed=self.MIN_TOKENS_FOR_RETRY
        )
        
        if not can_continue:
            agent_logger.log_reflection(
                agent_name=self.agent_id,
                task_id="context_check",
                reflection=f"Stopping: {reason}"
            )
        
        return can_continue
    
    def _add_to_context(self, role: str, content: str, metadata: Dict = None):
        """Add content to context window with truncation if needed"""
        success, warning = self.context_mgr.add_to_context(
            self.agent_id, role, content, metadata
        )
        if warning:
            agent_logger.log_reflection(
                agent_name=self.agent_id,
                task_id="context_management",
                reflection=warning
            )
    
    def _initialize_context(self, subtask: Dict):
        """Initialize context window with task information"""
        # System message with task context
        system_content = f"""You are a computational biology script generator.

Task: {subtask.get('description', 'Unknown')}

Language: {subtask.get('language', 'python')}
Required packages: {', '.join(subtask.get('packages', []))}
Input files: {', '.join(subtask.get('input_files', []))}
Expected outputs: {', '.join(subtask.get('output_files', []))}

Your role:
1. Generate complete, runnable scripts
2. Handle errors gracefully
3. Verify outputs exist after execution
"""
        self._add_to_context('system', system_content)
        
        # Add reference code if available
        reference_code = self._load_reference_scripts(subtask)
        if reference_code:
            truncated_ref = self.context_mgr.format_for_context(
                reference_code, 'code', max_tokens=10000
            )
            self._add_to_context('system', f"Reference code:\n{truncated_ref}")
    
    def _load_reference_scripts(self, subtask: Dict) -> Optional[str]:
        """Load reference scripts specified in subtask"""
        reference_scripts = subtask.get('reference_scripts', [])
        if not reference_scripts:
            return None
        
        combined = []
        for script_path in reference_scripts:
            possible_paths = [
                self.project_root / script_path,
                self.project_root / 'scripts' / script_path,
                self.project_root / 'scripts' / 'example_reference_scripts' / Path(script_path).name,
            ]
            
            for path in possible_paths:
                if path.exists():
                    try:
                        code = path.read_text()
                        combined.append(f"# === {path.name} ===\n{code}")
                        break
                    except Exception:
                        pass
        
        return "\n\n".join(combined) if combined else None
    
    def _generate_script(self, subtask: Dict) -> Dict[str, Any]:
        """
        Generate a complete, runnable script for the subtask.
        
        Returns:
            Dict with script_path, script_content, success status
        """
        task_id = subtask.get('id', 'task')
        language = subtask.get('language', 'python')
        description = subtask.get('description', '')
        packages = subtask.get('packages', [])
        input_files = subtask.get('input_files', [])
        output_files = subtask.get('output_files', [])
        code_hints = subtask.get('code_hints', [])
        huggingface_repos = subtask.get('huggingface_repos', [])
        
        # Get context from previous failures if any
        previous_attempts = subtask.get('context', {}).get('previous_attempts', [])
        
        prompt = f"""Generate a complete, runnable {language} script for this task.

TASK: {description}

REQUIREMENTS:
- Language: {language}
- Packages: {', '.join(packages)}
- Input files: {', '.join(input_files)}
- Expected output files: {', '.join(output_files)}
{f'- HuggingFace repos: {", ".join(huggingface_repos)}' if huggingface_repos else ''}

SCRIPT REQUIREMENTS:
1. Include all necessary imports at the top
2. Use absolute paths based on working directory
3. Add error handling with informative messages
4. Print progress messages to stdout
5. Verify input files exist before processing
6. Save outputs to the exact paths specified
7. Print "SUCCESS: Task completed" at the end if successful

{f'Previous attempt errors to avoid: {previous_attempts[-1] if previous_attempts else "None"}' if previous_attempts else ''}

{f'Code hints from prompt: {code_hints[0][:1000] if code_hints else "None"}'}

Generate ONLY the script code, no explanations. Start with imports."""

        self._add_to_context('user', f"Generate script for: {description[:200]}")
        
        try:
            script_content = self.llm.invoke(prompt)
            
            # Clean up the response
            script_content = self._clean_script_content(script_content, language)
            
            # Add standard header
            script_content = self._add_script_header(script_content, subtask, language)
            
            # Save script file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_task_id = re.sub(r'[^\w\-]', '_', task_id)[:30]
            
            if language == 'python':
                script_path = self.project_root / 'scripts' / f"{safe_task_id}_{timestamp}.py"
            elif language == 'r':
                script_path = self.project_root / 'scripts' / f"{safe_task_id}_{timestamp}.R"
            else:
                script_path = self.project_root / 'scripts' / f"{safe_task_id}_{timestamp}.sh"
            
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text(script_content)
            
            self._add_to_context('assistant', f"Generated script: {script_path.name}")
            
            agent_logger.log_reflection(
                agent_name=self.agent_id,
                task_id=task_id,
                reflection=f"Generated script: {script_path}"
            )
            
            return {
                'success': True,
                'script_path': str(script_path),
                'script_content': script_content,
                'language': language
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _clean_script_content(self, content: str, language: str) -> str:
        """Clean LLM-generated script content"""
        # Remove markdown code blocks
        content = re.sub(r'^```(?:python|r|bash|sh)?\n?', '', content, flags=re.MULTILINE)
        content = re.sub(r'\n?```$', '', content, flags=re.MULTILINE)
        content = content.strip()
        
        # Ensure proper shebang for bash
        if language == 'bash' and not content.startswith('#!'):
            content = '#!/bin/bash\nset -e\n\n' + content
        
        return content
    
    def _add_script_header(self, content: str, subtask: Dict, language: str) -> str:
        """Add informative header to script"""
        task_id = subtask.get('id', 'unknown')
        description = subtask.get('description', '')[:100]
        
        if language == 'python':
            header = f'''#!/usr/bin/env python3
"""
Auto-generated script for: {task_id}
Task: {description}
Generated: {datetime.now().isoformat()}

Packages: {', '.join(subtask.get('packages', []))}
"""

import os
import sys
from pathlib import Path

# Set working directory
os.chdir("{self.project_root}")

'''
        elif language == 'r':
            header = f'''#!/usr/bin/env Rscript
# Auto-generated script for: {task_id}
# Task: {description}
# Generated: {datetime.now().isoformat()}

setwd("{self.project_root}")

'''
        else:
            header = f'''#!/bin/bash
# Auto-generated script for: {task_id}
# Task: {description}
# Generated: {datetime.now().isoformat()}

set -e
cd "{self.project_root}"

'''
        
        # Don't duplicate shebang
        if content.startswith('#!'):
            lines = content.split('\n', 1)
            content = lines[1] if len(lines) > 1 else ''
        
        return header + content
    
    def _setup_environment(self, subtask: Dict, task_id: str) -> Dict[str, Any]:
        """Create task-specific conda environment"""
        if not self.conda_tools:
            return {'success': False, 'error': 'No conda_tools available'}
        
        packages = subtask.get('packages', [])
        language = subtask.get('language', 'python')
        
        # Create environment YAML
        env_name = f"task_{task_id}_{datetime.now().strftime('%Y%m%d')}"
        
        # Map common package names to conda packages
        conda_packages = ['pip']
        pip_packages = []
        
        package_mapping = {
            'scanpy': ('conda-forge', 'scanpy'),
            'squidpy': ('conda-forge', 'squidpy'),
            'anndata': ('conda-forge', 'anndata'),
            'pandas': ('conda-forge', 'pandas'),
            'numpy': ('conda-forge', 'numpy'),
            'scipy': ('conda-forge', 'scipy'),
            'matplotlib': ('conda-forge', 'matplotlib'),
            'seaborn': ('conda-forge', 'seaborn'),
            'scikit-learn': ('conda-forge', 'scikit-learn'),
        }
        
        pip_only = ['popv', 'popV', 'scvi-tools', 'cellxgene', 'celltypist']
        
        channels = set(['defaults', 'conda-forge'])
        
        for pkg in packages:
            pkg_lower = pkg.lower()
            if pkg_lower in package_mapping:
                channel, pkg_name = package_mapping[pkg_lower]
                channels.add(channel)
                conda_packages.append(pkg_name)
            elif pkg_lower in [p.lower() for p in pip_only]:
                pip_packages.append(pkg)
            else:
                # Default to pip for unknown packages
                pip_packages.append(pkg)
        
        # Generate environment YAML
        env_yaml = {
            'name': env_name,
            'channels': list(channels),
            'dependencies': conda_packages
        }
        
        if pip_packages:
            env_yaml['dependencies'].append({'pip': pip_packages})
        
        # Save YAML
        yaml_path = self.project_root / 'envs' / f"{env_name}.yml"
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        import yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(env_yaml, f, default_flow_style=False)
        
        # Try to create the environment
        try:
            result = self.conda_tools.create_from_yaml(str(yaml_path), env_name)
            if result['success']:
                return {
                    'success': True,
                    'env_name': env_name,
                    'yaml_path': str(yaml_path)
                }
            else:
                # Environment creation failed but we have the YAML
                return {
                    'success': True,  # YAML created successfully
                    'env_name': None,
                    'yaml_path': str(yaml_path),
                    'warning': f"Could not create env: {result.get('error')}"
                }
        except Exception as e:
            return {
                'success': True,  # YAML created
                'env_name': None,
                'yaml_path': str(yaml_path),
                'warning': str(e)
            }
    
    def _submit_slurm_job(
        self,
        script_path: str,
        task_id: str,
        env_name: str,
        subtask: Dict
    ) -> Dict[str, Any]:
        """Submit script as SLURM job and wait for completion"""
        
        language = subtask.get('language', 'python')
        
        # Read script content
        script_content = Path(script_path).read_text()
        
        # Determine resources from subtask or defaults
        cpus = subtask.get('cpus', self.slurm_config.get('cpus', 4))
        memory = subtask.get('memory', self.slurm_config.get('memory', '32G'))
        time_limit = subtask.get('time', self.slurm_config.get('time', '04:00:00'))
        partition = subtask.get('partition', self.slurm_config.get('partition'))
        gpus = subtask.get('gpus', self.slurm_config.get('gpus', 0))
        
        # Submit via slurm_tools
        submit_result = self.slurm_tools.submit_script(
            script_content=script_content,
            job_name=f"agi_{task_id}",
            language=language,
            conda_env=env_name,
            cpus=cpus,
            memory=memory,
            time_limit=time_limit,
            partition=partition,
            gpus=gpus
        )
        
        if not submit_result['success']:
            return submit_result
        
        job_id = submit_result['job_id']
        self.jobs_submitted.append(job_id)
        
        agent_logger.log_slurm_job(
            agent_name=self.agent_id,
            job_id=job_id,
            status="SUBMITTED",
            details={"script": script_path, "partition": partition}
        )
        
        self._add_to_context('tool', f"SLURM job {job_id} submitted")
        
        # Wait for completion
        wait_result = self.slurm_tools.wait_for_job(
            job_id=job_id,
            poll_interval=self.slurm_config.get('poll_interval', 10),
            max_attempts=self.slurm_config.get('max_poll_attempts', 720)
        )
        
        # Get job output
        output_result = self.slurm_tools.get_job_output(job_id)
        
        # Check success
        job_succeeded = (
            wait_result.get('success') and 
            wait_result.get('status') == 'COMPLETED' and
            wait_result.get('exit_code', '1') == '0'
        )
        
        # Log output (truncated for context)
        stdout = output_result.get('stdout', '')
        stderr = output_result.get('stderr', '')
        
        truncated_stdout = self.context_mgr.format_for_context(stdout, 'log', 5000)
        truncated_stderr = self.context_mgr.format_for_context(stderr, 'error', 5000)
        
        self._add_to_context('tool', f"Job output:\n{truncated_stdout}")
        if stderr:
            self._add_to_context('tool', f"Job errors:\n{truncated_stderr}")
        
        agent_logger.log_slurm_job(
            agent_name=self.agent_id,
            job_id=job_id,
            status=wait_result.get('status', 'UNKNOWN'),
            details={
                "exit_code": wait_result.get('exit_code'),
                "success": job_succeeded
            }
        )
        
        return {
            'success': job_succeeded,
            'job_id': job_id,
            'status': wait_result.get('status'),
            'exit_code': wait_result.get('exit_code'),
            'stdout': stdout,
            'stderr': stderr,
            'output_file': output_result.get('output_file'),
            'error_file': output_result.get('error_file')
        }
    
    def _run_locally(self, script_path: str, env_name: str) -> Dict[str, Any]:
        """Run script locally (when SLURM not available)"""
        import subprocess
        
        script_path = Path(script_path)
        language = 'python' if script_path.suffix == '.py' else 'bash'
        
        if env_name and self.conda_tools:
            # Run in conda environment
            if language == 'python':
                cmd = f"python {script_path}"
            else:
                cmd = f"bash {script_path}"
            
            result = self.conda_tools.run_in_environment(
                env_name=env_name,
                command=cmd,
                timeout=3600  # 1 hour timeout
            )
        else:
            # Run directly
            try:
                if language == 'python':
                    cmd = ['python', str(script_path)]
                else:
                    cmd = ['bash', str(script_path)]
                
                proc_result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600,
                    cwd=str(self.project_root)
                )
                
                result = {
                    'success': proc_result.returncode == 0,
                    'stdout': proc_result.stdout,
                    'stderr': proc_result.stderr,
                    'return_code': proc_result.returncode
                }
            except subprocess.TimeoutExpired:
                result = {'success': False, 'error': 'Timeout after 1 hour'}
            except Exception as e:
                result = {'success': False, 'error': str(e)}
        
        return result
    
    def _verify_outputs(self, subtask: Dict, run_result: Dict) -> Dict[str, Any]:
        """Verify expected output files exist"""
        output_files = subtask.get('output_files', [])
        
        if not output_files:
            # No specific outputs expected - check for SUCCESS in stdout
            stdout = run_result.get('stdout', '')
            if 'SUCCESS' in stdout:
                return {'all_exist': True, 'found_files': [], 'missing_files': []}
            return {'all_exist': True, 'found_files': [], 'missing_files': []}
        
        found = []
        missing = []
        
        for filepath in output_files:
            possible_paths = [
                self.project_root / filepath,
                self.project_root / 'data' / 'outputs' / filepath,
                self.project_root / 'data' / 'outputs' / Path(filepath).name,
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
            'all_exist': len(missing) == 0,
            'found_files': found,
            'missing_files': missing
        }
    
    def _check_existing_outputs(self, subtask: Dict) -> Dict[str, Any]:
        """Check if output files already exist"""
        output_files = subtask.get('output_files', [])
        skip_if_exists = subtask.get('skip_if_exists', [])
        
        files_to_check = output_files + skip_if_exists
        if not files_to_check:
            return {'already_complete': False}
        
        existing = []
        for filepath in files_to_check:
            possible_paths = [
                self.project_root / filepath,
                self.project_root / 'data' / 'outputs' / filepath,
            ]
            for path in possible_paths:
                if path.exists():
                    existing.append(str(path))
                    break
        
        if len(existing) == len(files_to_check):
            return {
                'already_complete': True,
                'existing_files': existing
            }
        
        return {'already_complete': False, 'existing_files': existing}
    
    def _diagnose_failure(self, run_result: Dict, script_path: str) -> Dict[str, Any]:
        """Diagnose why a job failed and determine if recoverable"""
        
        stdout = run_result.get('stdout', '')
        stderr = run_result.get('stderr', '')
        exit_code = run_result.get('exit_code', 'unknown')
        
        # Common error patterns and their solutions
        error_patterns = [
            (r'ModuleNotFoundError: No module named [\'"](\w+)[\'"]', 
             'missing_package', True),
            (r'FileNotFoundError:.*[\'"]([^"\']+)[\'"]',
             'missing_file', True),
            (r'MemoryError|OutOfMemoryError|OOM',
             'out_of_memory', True),
            (r'CUDA out of memory|CUDA error',
             'gpu_memory', True),
            (r'Permission denied',
             'permission_error', False),
            (r'Timeout|exceeded time limit',
             'timeout', True),
            (r'SyntaxError|IndentationError',
             'syntax_error', True),
        ]
        
        diagnosis = {
            'recoverable': True,
            'error_type': 'unknown',
            'summary': '',
            'suggested_fix': '',
            'logs': {
                'stdout': stdout[-5000:] if stdout else '',
                'stderr': stderr[-5000:] if stderr else ''
            }
        }
        
        combined_output = stdout + '\n' + stderr
        
        for pattern, error_type, recoverable in error_patterns:
            match = re.search(pattern, combined_output, re.IGNORECASE)
            if match:
                diagnosis['error_type'] = error_type
                diagnosis['recoverable'] = recoverable
                diagnosis['match'] = match.group(0)
                
                if error_type == 'missing_package':
                    pkg = match.group(1)
                    diagnosis['summary'] = f"Missing package: {pkg}"
                    diagnosis['suggested_fix'] = f"Add '{pkg}' to packages list"
                    diagnosis['missing_package'] = pkg
                    
                elif error_type == 'missing_file':
                    filepath = match.group(1)
                    diagnosis['summary'] = f"Missing file: {filepath}"
                    diagnosis['suggested_fix'] = "Verify input file paths"
                    diagnosis['missing_file'] = filepath
                    
                elif error_type == 'out_of_memory':
                    diagnosis['summary'] = "Out of memory error"
                    diagnosis['suggested_fix'] = "Request more memory or process in chunks"
                    
                elif error_type == 'gpu_memory':
                    diagnosis['summary'] = "GPU memory exhausted"
                    diagnosis['suggested_fix'] = "Reduce batch size or use CPU"
                    
                elif error_type == 'syntax_error':
                    diagnosis['summary'] = "Script has syntax errors"
                    diagnosis['suggested_fix'] = "Regenerate script with fixes"
                
                break
        
        if diagnosis['error_type'] == 'unknown':
            # Extract last error line
            error_lines = [l for l in stderr.split('\n') if 'error' in l.lower()]
            if error_lines:
                diagnosis['summary'] = error_lines[-1][:200]
            else:
                diagnosis['summary'] = f"Job failed with exit code {exit_code}"
        
        return diagnosis
    
    def _update_subtask_with_diagnosis(self, subtask: Dict, diagnosis: Dict) -> Dict:
        """Update subtask context with failure diagnosis for next attempt"""
        updated = subtask.copy()
        updated['context'] = updated.get('context', {})
        
        # Track previous attempts
        if 'previous_attempts' not in updated['context']:
            updated['context']['previous_attempts'] = []
        
        updated['context']['previous_attempts'].append({
            'error_type': diagnosis['error_type'],
            'summary': diagnosis['summary'],
            'suggested_fix': diagnosis['suggested_fix']
        })
        
        # Add missing package if identified
        if diagnosis.get('missing_package'):
            packages = updated.get('packages', [])
            if diagnosis['missing_package'] not in packages:
                packages.append(diagnosis['missing_package'])
                updated['packages'] = packages
        
        return updated
    
    def _success_result(self, task_id: str, **kwargs) -> Dict[str, Any]:
        """Build success result dictionary"""
        result = {
            'success': True,
            'task_id': task_id,
            'context_status': self.context_mgr.get_context_status(self.agent_id),
            **kwargs
        }
        
        agent_logger.log_task_success(
            agent_name=self.agent_id,
            task_id=task_id,
            result=result,
            tools_used=['script_generator', 'slurm' if self.use_slurm else 'local']
        )
        
        return result
    
    def _failure_result(self, task_id: str, error: str, **kwargs) -> Dict[str, Any]:
        """Build failure result dictionary"""
        result = {
            'success': False,
            'task_id': task_id,
            'error': error,
            'context_status': self.context_mgr.get_context_status(self.agent_id),
            **kwargs
        }
        
        agent_logger.log_task_failure(
            agent_name=self.agent_id,
            task_id=task_id,
            error=error,
            context=kwargs
        )
        
        return result


# Alias for backward compatibility
SubAgent = ScriptFirstSubAgent

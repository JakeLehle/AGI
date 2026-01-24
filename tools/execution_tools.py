"""
Script execution tools for agents.
Supports Python, R, bash, Java, and Perl execution within sandbox.
"""

import subprocess
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json
import shlex

from tools.sandbox import Sandbox, SandboxViolation, sandboxed


class ExecutionResult:
    """Container for execution results"""
    
    def __init__(
        self,
        success: bool,
        stdout: str = "",
        stderr: str = "",
        return_code: int = 0,
        execution_time: float = 0,
        error: str = None
    ):
        self.success = success
        self.stdout = stdout
        self.stderr = stderr
        self.return_code = return_code
        self.execution_time = execution_time
        self.error = error
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_code": self.return_code,
            "execution_time": self.execution_time,
            "error": self.error
        }


class ExecutionTools:
    """
    Executes scripts and commands within a sandboxed environment.
    Supports multiple languages and conda environments.
    """
    
    # Supported languages and their interpreters
    LANGUAGE_CONFIG = {
        "python": {
            "extension": ".py",
            "interpreter": "python",
            "shebang": "#!/usr/bin/env python3"
        },
        "r": {
            "extension": ".R",
            "interpreter": "Rscript",
            "shebang": "#!/usr/bin/env Rscript"
        },
        "bash": {
            "extension": ".sh",
            "interpreter": "bash",
            "shebang": "#!/bin/bash"
        },
        "perl": {
            "extension": ".pl",
            "interpreter": "perl",
            "shebang": "#!/usr/bin/env perl"
        },
        "java": {
            "extension": ".java",
            "interpreter": "java",
            "compile_cmd": "javac",
            "shebang": None
        }
    }
    
    def __init__(self, sandbox: Sandbox, conda_tools=None):
        """
        Initialize execution tools.
        
        Args:
            sandbox: Sandbox instance for path validation
            conda_tools: CondaTools instance for environment management
        """
        self.sandbox = sandbox
        self.conda_tools = conda_tools
        self.execution_log = []
    
    def write_script(
        self,
        code: str,
        filename: str,
        language: str = "python"
    ) -> Dict[str, Any]:
        """
        Write code to a script file in the scripts directory.
        
        Args:
            code: The script code
            filename: Name for the script file
            language: Programming language
            
        Returns:
            Result with script path
        """
        if language not in self.LANGUAGE_CONFIG:
            return {
                "success": False,
                "error": f"Unsupported language: {language}. Supported: {list(self.LANGUAGE_CONFIG.keys())}"
            }
        
        config = self.LANGUAGE_CONFIG[language]
        
        # Ensure proper extension
        if not filename.endswith(config["extension"]):
            filename = filename + config["extension"]
        
        # Create script path
        script_path = self.sandbox.get_scripts_dir() / filename
        
        # Add shebang if not present and language has one
        if config["shebang"] and not code.startswith("#!"):
            code = config["shebang"] + "\n" + code
        
        try:
            script_path.write_text(code)
            
            # Make executable for bash/perl
            if language in ["bash", "perl"]:
                os.chmod(script_path, 0o755)
            
            return {
                "success": True,
                "script_path": str(script_path),
                "language": language,
                "size_bytes": len(code)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def execute_script(
        self,
        script_path: str,
        args: List[str] = None,
        env_name: str = None,
        timeout: int = 300,
        working_dir: str = None
    ) -> ExecutionResult:
        """
        Execute a script file.
        
        Args:
            script_path: Path to script file
            args: Command line arguments
            env_name: Conda environment to use
            timeout: Timeout in seconds
            working_dir: Working directory (defaults to project dir)
            
        Returns:
            ExecutionResult with output
        """
        start_time = datetime.now()
        
        # Validate path is in sandbox
        try:
            validated_path = self.sandbox.validate_path(script_path, must_exist=True)
        except SandboxViolation as e:
            return ExecutionResult(
                success=False,
                error=str(e)
            )
        except FileNotFoundError as e:
            return ExecutionResult(
                success=False,
                error=str(e)
            )
        
        # Determine language from extension
        extension = validated_path.suffix.lower()
        language = None
        for lang, config in self.LANGUAGE_CONFIG.items():
            if config["extension"].lower() == extension:
                language = lang
                break
        
        if not language:
            return ExecutionResult(
                success=False,
                error=f"Unknown script extension: {extension}"
            )
        
        # Build command
        config = self.LANGUAGE_CONFIG[language]
        
        if language == "java":
            # Java requires compilation first
            return self._execute_java(validated_path, args, env_name, timeout, working_dir)
        
        cmd = [config["interpreter"], str(validated_path)]
        if args:
            cmd.extend(args)
        
        # Set working directory
        if working_dir:
            try:
                work_dir = self.sandbox.validate_path(working_dir)
            except SandboxViolation:
                work_dir = self.sandbox.get_working_dir()
        else:
            work_dir = self.sandbox.get_working_dir()
        
        # Execute
        return self._run_command(cmd, env_name, timeout, work_dir, start_time)
    
    def execute_code(
        self,
        code: str,
        language: str = "python",
        env_name: str = None,
        timeout: int = 300,
        script_name: str = None
    ) -> ExecutionResult:
        """
        Write and execute code in one step.
        
        Args:
            code: Code to execute
            language: Programming language
            env_name: Conda environment
            timeout: Timeout in seconds
            script_name: Optional name for script file
            
        Returns:
            ExecutionResult
        """
        if not script_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            script_name = f"temp_{timestamp}"
        
        # Write script
        write_result = self.write_script(code, script_name, language)
        if not write_result["success"]:
            return ExecutionResult(
                success=False,
                error=write_result["error"]
            )
        
        # Execute
        return self.execute_script(
            write_result["script_path"],
            env_name=env_name,
            timeout=timeout
        )
    
    def execute_command(
        self,
        command: str,
        env_name: str = None,
        timeout: int = 300,
        working_dir: str = None
    ) -> ExecutionResult:
        """
        Execute a shell command.
        
        Args:
            command: Shell command to execute
            env_name: Conda environment
            timeout: Timeout in seconds
            working_dir: Working directory
            
        Returns:
            ExecutionResult
        """
        start_time = datetime.now()
        
        # Validate command safety
        try:
            self.sandbox.validate_command(command)
        except SandboxViolation as e:
            return ExecutionResult(
                success=False,
                error=str(e)
            )
        
        # Set working directory
        if working_dir:
            try:
                work_dir = self.sandbox.validate_path(working_dir)
            except SandboxViolation:
                work_dir = self.sandbox.get_working_dir()
        else:
            work_dir = self.sandbox.get_working_dir()
        
        cmd = ["bash", "-c", command]
        
        return self._run_command(cmd, env_name, timeout, work_dir, start_time)
    
    def _run_command(
        self,
        cmd: List[str],
        env_name: str,
        timeout: int,
        working_dir: Path,
        start_time: datetime
    ) -> ExecutionResult:
        """Internal method to run a command with optional conda environment"""
        
        # If using conda environment, wrap command
        if env_name and self.conda_tools:
            full_env_name = env_name if env_name.startswith("agi_") else f"agi_{env_name}"
            
            # Use conda run
            original_cmd = " ".join(shlex.quote(c) for c in cmd)
            conda_result = self.conda_tools.run_in_environment(
                full_env_name,
                original_cmd,
                timeout=timeout
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = ExecutionResult(
                success=conda_result["success"],
                stdout=conda_result.get("stdout", ""),
                stderr=conda_result.get("stderr", ""),
                return_code=conda_result.get("return_code", 1 if not conda_result["success"] else 0),
                execution_time=execution_time,
                error=conda_result.get("error")
            )
        else:
            # Run directly
            try:
                proc_result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=str(working_dir),
                    env={**os.environ, "PROJECT_DIR": str(self.sandbox.project_dir)}
                )
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                result = ExecutionResult(
                    success=proc_result.returncode == 0,
                    stdout=proc_result.stdout,
                    stderr=proc_result.stderr,
                    return_code=proc_result.returncode,
                    execution_time=execution_time
                )
            except subprocess.TimeoutExpired:
                execution_time = (datetime.now() - start_time).total_seconds()
                result = ExecutionResult(
                    success=False,
                    error=f"Command timed out after {timeout} seconds",
                    execution_time=execution_time
                )
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                result = ExecutionResult(
                    success=False,
                    error=str(e),
                    execution_time=execution_time
                )
        
        # Log execution
        self._log_execution(cmd, result, env_name)
        
        return result
    
    def _execute_java(
        self,
        java_file: Path,
        args: List[str],
        env_name: str,
        timeout: int,
        working_dir: str
    ) -> ExecutionResult:
        """Handle Java compilation and execution"""
        start_time = datetime.now()
        
        # Compile
        compile_cmd = ["javac", str(java_file)]
        
        try:
            compile_result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=timeout // 2,
                cwd=str(java_file.parent)
            )
            
            if compile_result.returncode != 0:
                return ExecutionResult(
                    success=False,
                    stderr=compile_result.stderr,
                    error="Java compilation failed"
                )
            
            # Run
            class_name = java_file.stem
            run_cmd = ["java", class_name]
            if args:
                run_cmd.extend(args)
            
            run_result = subprocess.run(
                run_cmd,
                capture_output=True,
                text=True,
                timeout=timeout // 2,
                cwd=str(java_file.parent)
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ExecutionResult(
                success=run_result.returncode == 0,
                stdout=run_result.stdout,
                stderr=run_result.stderr,
                return_code=run_result.returncode,
                execution_time=execution_time
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                error=f"Java execution timed out"
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e)
            )
    
    def _log_execution(self, cmd: List[str], result: ExecutionResult, env_name: str):
        """Log execution for tracking"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "command": cmd,
            "env_name": env_name,
            "success": result.success,
            "return_code": result.return_code,
            "execution_time": result.execution_time,
            "stdout_preview": result.stdout[:500] if result.stdout else "",
            "stderr_preview": result.stderr[:500] if result.stderr else "",
            "error": result.error
        }
        self.execution_log.append(log_entry)
        
        # Also write to execution log file
        log_file = self.sandbox.get_logs_dir() / "execution_log.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def read_output_file(self, filepath: str) -> Dict[str, Any]:
        """
        Read an output file generated by a script.
        
        Args:
            filepath: Path to output file
            
        Returns:
            Dict with file contents
        """
        try:
            validated_path = self.sandbox.validate_path(filepath, must_exist=True)
            
            # Determine how to read based on extension
            extension = validated_path.suffix.lower()
            
            if extension == ".json":
                with open(validated_path, 'r') as f:
                    content = json.load(f)
                return {
                    "success": True,
                    "content": content,
                    "type": "json"
                }
            elif extension in [".csv", ".tsv"]:
                content = validated_path.read_text()
                return {
                    "success": True,
                    "content": content,
                    "type": "csv",
                    "lines": len(content.split("\n"))
                }
            else:
                content = validated_path.read_text()
                return {
                    "success": True,
                    "content": content,
                    "type": "text",
                    "size_bytes": len(content)
                }
        except SandboxViolation as e:
            return {"success": False, "error": str(e)}
        except FileNotFoundError:
            return {"success": False, "error": f"File not found: {filepath}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def check_output_files(self, expected_files: List[str]) -> Dict[str, Any]:
        """
        Check if expected output files were created.
        
        Args:
            expected_files: List of expected file paths
            
        Returns:
            Dict with status of each file
        """
        results = {}
        all_exist = True
        
        for filepath in expected_files:
            try:
                # Handle relative paths
                if not Path(filepath).is_absolute():
                    full_path = self.sandbox.project_dir / filepath
                else:
                    full_path = self.sandbox.validate_path(filepath)
                
                exists = full_path.exists()
                size = full_path.stat().st_size if exists else 0
                
                results[filepath] = {
                    "exists": exists,
                    "size_bytes": size,
                    "path": str(full_path)
                }
                
                if not exists:
                    all_exist = False
                    
            except (SandboxViolation, Exception) as e:
                results[filepath] = {
                    "exists": False,
                    "error": str(e)
                }
                all_exist = False
        
        return {
            "success": all_exist,
            "all_exist": all_exist,
            "files": results
        }
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all executions in this session"""
        total = len(self.execution_log)
        successful = sum(1 for e in self.execution_log if e["success"])
        
        return {
            "total_executions": total,
            "successful": successful,
            "failed": total - successful,
            "total_execution_time": sum(e.get("execution_time", 0) for e in self.execution_log),
            "recent_executions": self.execution_log[-5:]  # Last 5
        }

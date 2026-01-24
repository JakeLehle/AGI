"""
Sandbox utilities for restricting agent file operations.
Ensures all file operations stay within the project directory.
"""

from pathlib import Path
from typing import Union, Optional
import os
import functools

class SandboxViolation(Exception):
    """Raised when an operation attempts to escape the sandbox"""
    pass

class Sandbox:
    """
    Enforces directory restrictions for agent operations.
    All file paths are validated to be within the project directory.
    """
    
    def __init__(self, project_dir: Union[str, Path]):
        """
        Initialize sandbox with project directory.
        
        Args:
            project_dir: Root directory for all agent operations
        """
        self.project_dir = Path(project_dir).resolve()
        
        # Ensure project directory exists
        self.project_dir.mkdir(parents=True, exist_ok=True)
        
        # Create standard subdirectories
        self.dirs = {
            "data": self.project_dir / "data",
            "inputs": self.project_dir / "data" / "inputs",
            "outputs": self.project_dir / "data" / "outputs",
            "scripts": self.project_dir / "scripts",
            "logs": self.project_dir / "logs",
            "envs": self.project_dir / "envs",  # For environment.yaml files
            "reports": self.project_dir / "reports",
            "temp": self.project_dir / "temp",
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def validate_path(self, path: Union[str, Path], must_exist: bool = False) -> Path:
        """
        Validate that a path is within the sandbox.
        
        Args:
            path: Path to validate
            must_exist: If True, raise error if path doesn't exist
            
        Returns:
            Resolved absolute path
            
        Raises:
            SandboxViolation: If path escapes sandbox
            FileNotFoundError: If must_exist=True and path doesn't exist
        """
        # Resolve to absolute path
        resolved = Path(path).resolve()
        
        # Check if it's within project directory
        try:
            resolved.relative_to(self.project_dir)
        except ValueError:
            raise SandboxViolation(
                f"Path '{path}' resolves to '{resolved}' which is outside "
                f"the project directory '{self.project_dir}'. "
                f"Agents cannot access files outside the project."
            )
        
        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"Path does not exist: {resolved}")
        
        return resolved
    
    def safe_path(self, relative_path: Union[str, Path]) -> Path:
        """
        Convert a relative path to a safe absolute path within sandbox.
        
        Args:
            relative_path: Path relative to project directory
            
        Returns:
            Absolute path within sandbox
        """
        # Remove any leading slashes or parent references
        clean_path = str(relative_path).lstrip('/').lstrip('\\')
        
        # Replace any remaining parent directory references
        while '..' in clean_path:
            clean_path = clean_path.replace('..', '')
        
        full_path = self.project_dir / clean_path
        return self.validate_path(full_path)
    
    def validate_command(self, command: str) -> str:
        """
        Validate a shell command for safety.
        
        Args:
            command: Shell command to validate
            
        Returns:
            The command if safe
            
        Raises:
            SandboxViolation: If command is unsafe
        """
        # List of dangerous commands/patterns
        dangerous_patterns = [
            'sudo ',
            'su ',
            'rm -rf /',
            'rm -rf ~',
            'chmod 777 /',
            '> /dev/',
            'mkfs',
            'dd if=',
            ':(){',  # Fork bomb
            'wget -O- | sh',
            'curl | sh',
            'eval ',
        ]
        
        command_lower = command.lower()
        
        for pattern in dangerous_patterns:
            if pattern.lower() in command_lower:
                raise SandboxViolation(
                    f"Command contains dangerous pattern '{pattern}'. "
                    f"This operation is not allowed."
                )
        
        # Check for sudo explicitly
        if command.strip().startswith('sudo'):
            raise SandboxViolation(
                "sudo commands are not allowed. "
                "Use conda for package installation instead."
            )
        
        return command
    
    def get_working_dir(self) -> Path:
        """Get the main working directory for scripts"""
        return self.project_dir
    
    def get_data_dir(self, subdir: str = None) -> Path:
        """Get data directory, optionally with subdirectory"""
        base = self.dirs["data"]
        if subdir:
            path = base / subdir
            path.mkdir(parents=True, exist_ok=True)
            return path
        return base
    
    def get_scripts_dir(self) -> Path:
        """Get scripts directory"""
        return self.dirs["scripts"]
    
    def get_logs_dir(self) -> Path:
        """Get logs directory"""
        return self.dirs["logs"]
    
    def get_outputs_dir(self) -> Path:
        """Get outputs directory"""
        return self.dirs["outputs"]
    
    def get_inputs_dir(self) -> Path:
        """Get inputs directory"""
        return self.dirs["inputs"]
    
    def get_envs_dir(self) -> Path:
        """Get environments directory (for yaml files)"""
        return self.dirs["envs"]
    
    def get_reports_dir(self) -> Path:
        """Get reports directory"""
        return self.dirs["reports"]
    
    def get_temp_dir(self) -> Path:
        """Get temporary files directory"""
        return self.dirs["temp"]
    
    def list_files(self, subdir: str = None, pattern: str = "*") -> list:
        """
        List files in project directory or subdirectory.
        
        Args:
            subdir: Optional subdirectory to list
            pattern: Glob pattern for filtering
            
        Returns:
            List of file paths (relative to project dir)
        """
        if subdir:
            base = self.safe_path(subdir)
        else:
            base = self.project_dir
        
        files = []
        for path in base.rglob(pattern):
            if path.is_file():
                try:
                    rel_path = path.relative_to(self.project_dir)
                    files.append(str(rel_path))
                except ValueError:
                    pass  # Skip files outside sandbox
        
        return sorted(files)
    
    def get_directory_tree(self, max_depth: int = 3) -> str:
        """
        Generate a tree view of the project directory.
        
        Args:
            max_depth: Maximum depth to traverse
            
        Returns:
            ASCII tree representation
        """
        def _tree(dir_path: Path, prefix: str = "", depth: int = 0) -> list:
            if depth > max_depth:
                return []
            
            lines = []
            try:
                items = sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
                
                # Filter out hidden files and common ignore patterns
                items = [i for i in items if not i.name.startswith('.') 
                        and i.name not in ['__pycache__', 'node_modules', '.git']]
                
                for i, item in enumerate(items):
                    is_last = i == len(items) - 1
                    current_prefix = "└── " if is_last else "├── "
                    lines.append(f"{prefix}{current_prefix}{item.name}")
                    
                    if item.is_dir():
                        extension = "    " if is_last else "│   "
                        lines.extend(_tree(item, prefix + extension, depth + 1))
            except PermissionError:
                pass
            
            return lines
        
        tree_lines = [str(self.project_dir.name) + "/"]
        tree_lines.extend(_tree(self.project_dir))
        return "\n".join(tree_lines)
    
    def cleanup_temp(self):
        """Remove all files in temp directory"""
        import shutil
        temp_dir = self.dirs["temp"]
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)


def sandboxed(method):
    """
    Decorator to ensure a method operates within sandbox.
    The decorated method's class must have a 'sandbox' attribute.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, 'sandbox') or self.sandbox is None:
            raise SandboxViolation(
                "No sandbox configured. Initialize sandbox before file operations."
            )
        return method(self, *args, **kwargs)
    return wrapper

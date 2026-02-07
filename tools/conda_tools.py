"""
Conda environment management tools for agents.
Handles environment creation, package installation, and cleanup.
All without requiring sudo access.
"""

import subprocess
import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import shutil

class CondaTools:
    """
    Manages conda environments for agent tasks.
    Each project can have multiple environments that are documented and reproducible.
    """
    
    def __init__(self, project_dir: Path, envs_dir: Path = None):
        """
        Initialize conda tools.
        
        Args:
            project_dir: Project root directory
            envs_dir: Directory to store environment.yaml files
        """
        self.project_dir = Path(project_dir)
        self.envs_dir = envs_dir or (self.project_dir / "envs")
        self.envs_dir.mkdir(parents=True, exist_ok=True)
        
        # Track environments created for this project
        self.env_registry_path = self.envs_dir / "env_registry.json"
        self.env_registry = self._load_registry()
        
        # Detect conda installation
        self.conda_path = self._find_conda()
    
    def _find_conda(self) -> str:
        """Find conda executable path"""
        # Try common locations
        possible_paths = [
            "conda",  # In PATH
            os.path.expanduser("~/anaconda3/bin/conda"),
            os.path.expanduser("~/miniconda3/bin/conda"),
            "/opt/conda/bin/conda",
        ]
        
        # Also check CONDA_EXE environment variable
        conda_exe = os.environ.get("CONDA_EXE")
        if conda_exe:
            possible_paths.insert(0, conda_exe)
        
        for path in possible_paths:
            try:
                result = subprocess.run(
                    [path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    return path
            except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
                continue
        
        raise RuntimeError(
            "Could not find conda installation. "
            "Please ensure conda is installed and in your PATH."
        )
    
    def _load_registry(self) -> Dict:
        """Load environment registry from file"""
        if self.env_registry_path.exists():
            try:
                with open(self.env_registry_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {"environments": {}, "created_at": datetime.now().isoformat()}
    
    def _save_registry(self):
        """Save environment registry to file"""
        self.env_registry["updated_at"] = datetime.now().isoformat()
        with open(self.env_registry_path, 'w') as f:
            json.dump(self.env_registry, f, indent=2)
    
    def _run_conda(self, args: List[str], timeout: int = 300) -> Dict[str, Any]:
        """
        Run a conda command.
        
        Args:
            args: Command arguments (without 'conda')
            timeout: Command timeout in seconds
            
        Returns:
            Dict with success status, stdout, stderr
        """
        cmd = [self.conda_path] + args
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.project_dir)
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "command": " ".join(cmd)
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after {timeout} seconds",
                "command": " ".join(cmd)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "command": " ".join(cmd)
            }
    
    def create_environment(
        self,
        env_name: str,
        python_version: str = "3.10",
        packages: List[str] = None,
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Create a new conda environment for the project.
        
        Args:
            env_name: Name of the environment
            python_version: Python version to use
            packages: Initial packages to install
            description: Description of environment purpose
            
        Returns:
            Result dictionary with success status
        """
        # Prefix environment name with project identifier
        full_env_name = f"agi_{env_name}"
        
        # Check if environment already exists
        if self.environment_exists(full_env_name):
            return {
                "success": True,
                "message": f"Environment '{full_env_name}' already exists",
                "env_name": full_env_name,
                "already_existed": True
            }

        # Build create command
        create_args = [
            "create", "-n", full_env_name,
            f"python={python_version}",
            "-y"  # Don't ask for confirmation
        ]
        
        if packages:
            create_args.extend(packages)
        
        result = self._run_conda(create_args, timeout=600)
        
        if result["success"]:
            # Register the environment
            self.env_registry["environments"][full_env_name] = {
                "created_at": datetime.now().isoformat(),
                "python_version": python_version,
                "description": description,
                "packages_installed": packages or [],
                "status": "active"
            }
            self._save_registry()
            
            # Export initial environment.yaml
            self.export_environment(full_env_name)
            
            return {
                "success": True,
                "env_name": full_env_name,
                "message": f"Environment '{full_env_name}' created successfully",
                "python_version": python_version
            }
        else:
            return {
                "success": False,
                "env_name": full_env_name,
                "error": result.get("stderr", result.get("error", "Unknown error")),
                "command": result.get("command", "")
            }
    
    def install_packages(
        self,
        env_name: str,
        packages: List[str],
        use_pip: bool = False,
        channel: str = None
    ) -> Dict[str, Any]:
        """
        Install packages into an environment.
        
        Args:
            env_name: Environment name
            packages: List of packages to install
            use_pip: Use pip instead of conda
            channel: Conda channel to use (e.g., conda-forge, bioconda)
            
        Returns:
            Result dictionary
        """
        full_env_name = env_name if env_name.startswith("agi_") else f"agi_{env_name}"
        
        if not self.environment_exists(full_env_name):
            return {
                "success": False,
                "error": f"Environment '{full_env_name}' does not exist"
            }
        
        if use_pip:
            # Use pip within the conda environment
            pip_cmd = [
                "run", "-n", full_env_name,
                "pip", "install"
            ] + packages
            result = self._run_conda(pip_cmd, timeout=600)
        else:
            # Use conda install
            install_args = ["install", "-n", full_env_name, "-y"]
            if channel:
                install_args.extend(["-c", channel])
            install_args.extend(packages)
            result = self._run_conda(install_args, timeout=600)
        
        if result["success"]:
            # Update registry
            if full_env_name in self.env_registry["environments"]:
                self.env_registry["environments"][full_env_name]["packages_installed"].extend(packages)
                self.env_registry["environments"][full_env_name]["last_updated"] = datetime.now().isoformat()
                self._save_registry()
            
            # Update environment.yaml
            self.export_environment(full_env_name)
            
            return {
                "success": True,
                "message": f"Installed {len(packages)} package(s) in '{full_env_name}'",
                "packages": packages
            }
        else:
            return {
                "success": False,
                "error": result.get("stderr", result.get("error", "Unknown error")),
                "packages": packages
            }
    
    def environment_exists(self, env_name: str) -> bool:
        """Check if a conda environment exists"""
        full_env_name = env_name if env_name.startswith("agi_") else f"agi_{env_name}"
        result = self._run_conda(["env", "list"])
        
        if result["success"]:
            return full_env_name in result["stdout"]
        return False
    
    def export_environment(self, env_name: str) -> Dict[str, Any]:
        """
        Export environment to YAML file.
        
        Args:
            env_name: Environment name
            
        Returns:
            Result dictionary with yaml_path
        """
        full_env_name = env_name if env_name.startswith("agi_") else f"agi_{env_name}"
        
        result = self._run_conda(["env", "export", "-n", full_env_name])
        
        if result["success"]:
            yaml_path = self.envs_dir / f"{full_env_name}.yaml"
            
            # Parse and clean up the YAML (remove prefix path)
            try:
                env_dict = yaml.safe_load(result["stdout"])
                # Remove the prefix line as it's machine-specific
                if "prefix" in env_dict:
                    del env_dict["prefix"]
                
                with open(yaml_path, 'w') as f:
                    yaml.dump(env_dict, f, default_flow_style=False)
                
                return {
                    "success": True,
                    "yaml_path": str(yaml_path),
                    "message": f"Environment exported to {yaml_path}"
                }
            except yaml.YAMLError as e:
                # Fall back to raw output
                yaml_path.write_text(result["stdout"])
                return {
                    "success": True,
                    "yaml_path": str(yaml_path),
                    "message": f"Environment exported to {yaml_path} (raw format)"
                }
        else:
            return {
                "success": False,
                "error": result.get("stderr", result.get("error", "Unknown error"))
            }
    
    def create_from_yaml(self, yaml_path: str, env_name: str = None) -> Dict[str, Any]:
        """
        Create environment from YAML file.
        
        Args:
            yaml_path: Path to environment.yaml
            env_name: Override environment name (optional)
            
        Returns:
            Result dictionary
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            return {
                "success": False,
                "error": f"YAML file not found: {yaml_path}"
            }
        
        create_args = ["env", "create", "-f", str(yaml_path)]
        if env_name:
            full_env_name = f"agi_{env_name}" if not env_name.startswith("agi_") else env_name
            create_args.extend(["-n", full_env_name])
        
        result = self._run_conda(create_args, timeout=900)
        
        if result["success"]:
            return {
                "success": True,
                "message": f"Environment created from {yaml_path}"
            }
        else:
            return {
                "success": False,
                "error": result.get("stderr", result.get("error", "Unknown error"))
            }
    
    def remove_environment(self, env_name: str) -> Dict[str, Any]:
        """
        Remove a conda environment.
        
        Args:
            env_name: Environment name
            
        Returns:
            Result dictionary
        """
        full_env_name = env_name if env_name.startswith("agi_") else f"agi_{env_name}"
        
        # First export to preserve the environment definition
        self.export_environment(full_env_name)
        
        result = self._run_conda(["env", "remove", "-n", full_env_name, "-y"])
        
        if result["success"]:
            # Update registry
            if full_env_name in self.env_registry["environments"]:
                self.env_registry["environments"][full_env_name]["status"] = "removed"
                self.env_registry["environments"][full_env_name]["removed_at"] = datetime.now().isoformat()
                self._save_registry()
            
            return {
                "success": True,
                "message": f"Environment '{full_env_name}' removed",
                "yaml_preserved": str(self.envs_dir / f"{full_env_name}.yaml")
            }
        else:
            return {
                "success": False,
                "error": result.get("stderr", result.get("error", "Unknown error"))
            }
    
    def list_environments(self) -> Dict[str, Any]:
        """List all project environments"""
        result = self._run_conda(["env", "list"])
        
        if result["success"]:
            # Filter to only agi_ prefixed environments
            lines = result["stdout"].strip().split("\n")
            project_envs = []
            
            for line in lines:
                if "agi_" in line:
                    parts = line.split()
                    if parts:
                        project_envs.append({
                            "name": parts[0],
                            "path": parts[1] if len(parts) > 1 else "",
                            "registered": parts[0] in self.env_registry["environments"]
                        })
            
            return {
                "success": True,
                "environments": project_envs,
                "registry": self.env_registry["environments"]
            }
        else:
            return {
                "success": False,
                "error": result.get("stderr", result.get("error", "Unknown error"))
            }
    
    def run_in_environment(
        self,
        env_name: str,
        command: str,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """
        Run a command within a conda environment.
        
        Args:
            env_name: Environment name
            command: Command to run
            timeout: Timeout in seconds
            
        Returns:
            Result dictionary with stdout/stderr
        """
        full_env_name = env_name if env_name.startswith("agi_") else f"agi_{env_name}"
        
        # Use conda run to execute in environment
        run_args = ["run", "-n", full_env_name, "bash", "-c", command]
        
        return self._run_conda(run_args, timeout=timeout)
    
    def get_environment_info(self, env_name: str) -> Dict[str, Any]:
        """Get detailed information about an environment"""
        full_env_name = env_name if env_name.startswith("agi_") else f"agi_{env_name}"
        
        result = self._run_conda(["list", "-n", full_env_name])
        
        if result["success"]:
            # Parse package list
            packages = []
            for line in result["stdout"].strip().split("\n"):
                if line and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 2:
                        packages.append({
                            "name": parts[0],
                            "version": parts[1],
                            "channel": parts[3] if len(parts) > 3 else "unknown"
                        })
            
            registry_info = self.env_registry["environments"].get(full_env_name, {})
            
            return {
                "success": True,
                "env_name": full_env_name,
                "packages": packages,
                "package_count": len(packages),
                "registry_info": registry_info
            }
        else:
            return {
                "success": False,
                "error": result.get("stderr", result.get("error", "Unknown error"))
            }
    
    def cleanup_all_project_environments(self) -> Dict[str, Any]:
        """
        Remove all environments created for this project.
        Preserves YAML files for recreation.
        
        Returns:
            Result dictionary with cleanup summary
        """
        removed = []
        failed = []
        preserved_yamls = []
        
        for env_name in list(self.env_registry["environments"].keys()):
            if self.env_registry["environments"][env_name].get("status") == "active":
                # Export before removing
                export_result = self.export_environment(env_name)
                if export_result["success"]:
                    preserved_yamls.append(export_result["yaml_path"])
                
                # Remove environment
                result = self.remove_environment(env_name)
                if result["success"]:
                    removed.append(env_name)
                else:
                    failed.append({"name": env_name, "error": result.get("error")})
        
        return {
            "success": len(failed) == 0,
            "removed": removed,
            "failed": failed,
            "preserved_yamls": preserved_yamls,
            "message": f"Removed {len(removed)} environments, {len(failed)} failed"
        }

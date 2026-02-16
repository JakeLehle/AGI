"""
Conda environment management tools for agents.
Handles environment creation, package installation, and cleanup.
All without requiring sudo access.

v1.2.0 Updates:
- rebuild_env_from_yaml(): YAML-as-source-of-truth rebuild (remove + recreate)
- install_package(): Single package install with conda→pip fallback + YAML sync
- update_yaml_with_package(): Surgical YAML editor for adding/removing packages
- remove_package_from_yaml(): Remove a package from YAML conda or pip section
- validate_env_has_package(): Verify installation via import/library/which check
- DiskManager integration: ensure_space_for_build() called before env creation
- All env mutations keep the YAML in sync as the single source of truth
"""

import subprocess
import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import shutil
import re
import logging

logger = logging.getLogger(__name__)


class CondaTools:
    """
    Manages conda environments for agent tasks.
    Each project can have multiple environments that are documented and reproducible.

    v1.2.0: The YAML file is the single source of truth for every environment.
    All install/remove/rebuild operations update the YAML first, then mutate the
    live environment. This means any env can be fully reconstructed from its YAML.
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

        # v1.2.0: Optional DiskManager integration
        # Lazily initialized on first use to avoid circular imports
        self._disk_manager = None
        self._disk_manager_initialized = False
    
    # =========================================================================
    # DISK MANAGER INTEGRATION (v1.2.0)
    # =========================================================================

    def _get_disk_manager(self):
        """Lazily initialize DiskManager on first use.

        Returns None if DiskManager is unavailable (graceful degradation).
        """
        if not self._disk_manager_initialized:
            self._disk_manager_initialized = True
            try:
                from utils.disk_manager import DiskManager
                self._disk_manager = DiskManager()
                logger.debug("DiskManager initialized for conda_tools")
            except ImportError:
                logger.debug("DiskManager not available — disk checks disabled")
                self._disk_manager = None
            except Exception as e:
                logger.warning(f"DiskManager init failed: {e}")
                self._disk_manager = None
        return self._disk_manager

    def _ensure_disk_space(self, estimated_gb: float = 5.0) -> Dict[str, Any]:
        """Check and ensure sufficient disk space before environment operations.

        Calls DiskManager.ensure_space_for_build() if available. Returns a dict
        with 'ok' status and optional 'cleaned_gb' if cleanup was performed.

        If DiskManager is unavailable, returns ok=True (optimistic fallback).
        """
        dm = self._get_disk_manager()
        if dm is None:
            return {"ok": True, "reason": "disk_manager_unavailable"}

        try:
            result = dm.ensure_space_for_build(estimated_size_gb=estimated_gb)
            if result:
                return {"ok": True, "space_report": dm.get_space_report()}
            else:
                # Last resort: emergency cleanup
                logger.warning("Disk space insufficient even after proactive cleanup, attempting emergency cleanup")
                dm.emergency_cleanup(keep_envs=[])
                # Check again
                report = dm.get_space_report()
                if report.get("status") != "critical":
                    return {"ok": True, "space_report": report, "emergency_cleanup": True}
                return {
                    "ok": False,
                    "reason": "insufficient_disk_space",
                    "space_report": report,
                }
        except Exception as e:
            logger.warning(f"Disk space check failed: {e}")
            return {"ok": True, "reason": f"disk_check_error: {e}"}

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

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

    def _normalize_env_name(self, env_name: str) -> str:
        """Ensure environment name has agi_ prefix."""
        return env_name if env_name.startswith("agi_") else f"agi_{env_name}"

    # =========================================================================
    # ENVIRONMENT LIFECYCLE
    # =========================================================================
    
    def create_environment(
        self,
        env_name: str,
        python_version: str = "3.10",
        packages: List[str] = None,
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Create a new conda environment for the project.
        
        v1.2.0: Calls ensure_space_for_build() before creating.

        Args:
            env_name: Name of the environment
            python_version: Python version to use
            packages: Initial packages to install
            description: Description of environment purpose
            
        Returns:
            Result dictionary with success status
        """
        full_env_name = self._normalize_env_name(env_name)
        
        # Check if environment already exists
        if self.environment_exists(full_env_name):
            return {
                "success": True,
                "message": f"Environment '{full_env_name}' already exists",
                "env_name": full_env_name,
                "already_existed": True
            }

        # v1.2.0: Check disk space before creating
        disk_check = self._ensure_disk_space()
        if not disk_check.get("ok"):
            return {
                "success": False,
                "env_name": full_env_name,
                "error": f"Insufficient disk space: {disk_check.get('reason', 'unknown')}",
                "space_report": disk_check.get("space_report"),
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

    def create_from_yaml(self, yaml_path: str, env_name: str = None) -> Dict[str, Any]:
        """
        Create environment from YAML file.

        v1.2.0: Calls ensure_space_for_build() before creating.
        
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

        # v1.2.0: Check disk space before creating
        disk_check = self._ensure_disk_space()
        if not disk_check.get("ok"):
            return {
                "success": False,
                "error": f"Insufficient disk space: {disk_check.get('reason', 'unknown')}",
                "space_report": disk_check.get("space_report"),
            }

        create_args = ["env", "create", "-f", str(yaml_path)]
        if env_name:
            full_env_name = self._normalize_env_name(env_name)
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
        full_env_name = self._normalize_env_name(env_name)
        
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

    # =========================================================================
    # REBUILD FROM YAML (v1.2.0)
    # =========================================================================

    def rebuild_env_from_yaml(
        self,
        yaml_path: str,
        env_name: str,
        force: bool = True,
    ) -> Dict[str, Any]:
        """
        Remove existing environment and rebuild from YAML (source of truth).

        This is the core YAML-as-source-of-truth rebuild operation. The
        diagnostic agent calls this after updating the YAML with new packages
        or version pins. It ensures the live environment exactly matches the
        YAML specification.

        v1.2.0: New method.

        Args:
            yaml_path: Path to the authoritative environment YAML
            env_name: Environment name (agi_ prefix added if needed)
            force: If True, remove existing env before rebuild

        Returns:
            Result dictionary with success status, pip_fallback if applicable
        """
        full_env_name = self._normalize_env_name(env_name)
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            return {"success": False, "error": f"YAML not found: {yaml_path}"}

        # Step 1: Check disk space
        disk_check = self._ensure_disk_space()
        if not disk_check.get("ok"):
            return {
                "success": False,
                "error": f"Insufficient disk space for rebuild: {disk_check.get('reason')}",
                "space_report": disk_check.get("space_report"),
            }

        # Step 2: Remove existing environment if it exists
        if force and self.environment_exists(full_env_name):
            logger.info(f"Removing existing env '{full_env_name}' for rebuild")
            remove_result = self._run_conda(
                ["env", "remove", "-n", full_env_name, "-y"],
                timeout=120,
            )
            if not remove_result["success"]:
                logger.warning(
                    f"Failed to remove env '{full_env_name}': "
                    f"{remove_result.get('stderr', '')[:200]}"
                )
                # Continue anyway — conda env create with -f may overwrite

        # Step 3: Create from YAML
        result = self._run_conda(
            ["env", "create", "-f", str(yaml_path), "-n", full_env_name],
            timeout=900,
        )

        if result["success"] or "already exists" in result.get("stderr", ""):
            # Update registry
            self.env_registry["environments"][full_env_name] = {
                "created_at": datetime.now().isoformat(),
                "yaml_source": str(yaml_path),
                "status": "active",
                "rebuild": True,
            }
            self._save_registry()

            return {
                "success": True,
                "env_name": full_env_name,
                "message": f"Environment '{full_env_name}' rebuilt from {yaml_path}",
            }

        # Step 4: Conda failed — attempt pip fallback for problematic packages
        stderr = result.get("stderr", "")
        logger.warning(f"Conda env create failed for rebuild, attempting pip fallback")

        failed_pkgs = self._detect_failed_packages(stderr, yaml_path)
        if not failed_pkgs:
            return {
                "success": False,
                "error": f"Rebuild failed: {stderr[:500]}",
                "env_name": full_env_name,
            }

        # Create stripped YAML without failing packages
        stripped_yaml_path = yaml_path.with_suffix(".stripped.yml")
        stripped_ok = self._write_stripped_yaml(yaml_path, stripped_yaml_path, failed_pkgs)
        if not stripped_ok:
            return {
                "success": False,
                "error": f"Could not strip failed packages from YAML: {failed_pkgs}",
            }

        # Create env from stripped YAML
        result2 = self._run_conda(
            ["env", "create", "-f", str(stripped_yaml_path), "-n", full_env_name],
            timeout=900,
        )
        if not result2["success"] and "already exists" not in result2.get("stderr", ""):
            return {
                "success": False,
                "error": f"Stripped env also failed: {result2.get('stderr', '')[:300]}",
            }

        # Pip install the failed packages
        pip_installed, pip_failed = self._pip_install_packages(full_env_name, failed_pkgs)

        # Clean up stripped YAML
        try:
            stripped_yaml_path.unlink(missing_ok=True)
        except Exception:
            pass

        if pip_failed:
            return {
                "success": False,
                "error": f"Pip fallback failed for: {', '.join(pip_failed)}",
                "pip_installed": pip_installed,
                "pip_failed": pip_failed,
            }

        return {
            "success": True,
            "env_name": full_env_name,
            "message": f"Rebuilt with pip fallback for: {', '.join(pip_installed)}",
            "pip_fallback": pip_installed,
        }

    def _detect_failed_packages(self, stderr: str, yaml_path: Path) -> List[str]:
        """Parse conda error output to identify packages that caused failure.

        Checks against known pip-only packages as a fallback if stderr parsing
        yields nothing.
        """
        failed = []

        # Pattern 1: PackagesNotFoundError / ResolvePackageNotFound
        not_found = re.findall(
            r'[-–]\s*([a-zA-Z0-9_\-]+(?:[><=!]+\S*)?)',
            stderr
        )
        noise_words = {
            'the', 'following', 'packages', 'not', 'available', 'from',
            'current', 'channels', 'with', 'specs', 'and', 'your',
        }
        for pkg in not_found:
            name = re.split(r'[><=!~]', pkg)[0].strip()
            if name and len(name) > 1 and not name.startswith('-'):
                if name.lower() not in noise_words:
                    failed.append(name)

        # Pattern 2: "nothing provides <pkg>"
        nothing_provides = re.findall(
            r'nothing provides\s+([a-zA-Z0-9_\-]+)', stderr, re.IGNORECASE
        )
        failed.extend(nothing_provides)

        # Pattern 3: Cross-reference YAML with known pip-only packages
        if not failed and yaml_path.exists():
            try:
                from utils.dependency_parser import KNOWN_PIP_ONLY
                content = yaml_path.read_text()
                parsed = yaml.safe_load(content)
                if parsed and 'dependencies' in parsed:
                    for dep in parsed['dependencies']:
                        if isinstance(dep, str):
                            name = re.split(r'[><=!~]', dep)[0].strip().lower()
                            if name in KNOWN_PIP_ONLY:
                                failed.append(dep)
            except ImportError:
                # dependency_parser not available — use inline fallback set
                _FALLBACK_PIP_ONLY = {
                    'popv', 'celltypist', 'scvi-tools', 'decoupler',
                    'episcanpy', 'cell2location', 'moscot', 'pertpy',
                }
                try:
                    content = yaml_path.read_text()
                    parsed = yaml.safe_load(content)
                    if parsed and 'dependencies' in parsed:
                        for dep in parsed['dependencies']:
                            if isinstance(dep, str):
                                name = re.split(r'[><=!~]', dep)[0].strip().lower()
                                if name in _FALLBACK_PIP_ONLY:
                                    failed.append(dep)
                except Exception:
                    pass
            except Exception:
                pass

        # Deduplicate preserving order
        seen = set()
        unique = []
        for pkg in failed:
            name = re.split(r'[><=!~]', pkg)[0].strip().lower()
            if name not in seen:
                seen.add(name)
                unique.append(pkg)

        return unique

    def _write_stripped_yaml(
        self,
        original: Path,
        output: Path,
        remove_packages: List[str],
    ) -> bool:
        """Write a copy of the env YAML with specified packages removed."""
        try:
            content = original.read_text()
            parsed = yaml.safe_load(content)
            if not parsed or 'dependencies' not in parsed:
                return False

            remove_names = {
                re.split(r'[><=!~]', p)[0].strip().lower()
                for p in remove_packages
            }

            new_deps = []
            for dep in parsed['dependencies']:
                if isinstance(dep, str):
                    name = re.split(r'[><=!~]', dep)[0].strip().lower()
                    if name not in remove_names:
                        new_deps.append(dep)
                elif isinstance(dep, dict) and 'pip' in dep:
                    new_pip = [
                        p for p in dep['pip']
                        if re.split(r'[><=!~]', str(p))[0].strip().lower()
                        not in remove_names
                    ]
                    if new_pip:
                        new_deps.append({'pip': new_pip})
                else:
                    new_deps.append(dep)

            parsed['dependencies'] = new_deps

            with open(output, 'w') as f:
                yaml.dump(parsed, f, default_flow_style=False)
            return True

        except Exception as e:
            logger.warning(f"Failed to write stripped YAML: {e}")
            return False

    def _pip_install_packages(
        self,
        env_name: str,
        packages: List[str],
    ) -> tuple:
        """Install packages via pip into a conda environment.

        Returns:
            (pip_installed, pip_failed) — two lists of package names
        """
        installed = []
        failed = []

        for pkg in packages:
            try:
                result = subprocess.run(
                    [self.conda_path, 'run', '-n', env_name,
                     'pip', 'install', pkg],
                    capture_output=True, text=True, timeout=300,
                )
                if result.returncode == 0:
                    installed.append(pkg)
                    logger.info(f"pip installed '{pkg}' into '{env_name}'")
                else:
                    failed.append(pkg)
                    logger.warning(f"pip failed for '{pkg}': {result.stderr[:200]}")
            except Exception as e:
                failed.append(pkg)
                logger.warning(f"pip install exception for '{pkg}': {e}")

        return installed, failed

    # =========================================================================
    # SINGLE PACKAGE INSTALL (v1.2.0)
    # =========================================================================

    def install_package(
        self,
        env_name: str,
        package: str,
        method: str = "conda",
        channel: str = "conda-forge",
        yaml_path: str = None,
    ) -> Dict[str, Any]:
        """
        Install a single package into an existing environment.

        Tries the requested method first. If conda fails, falls back to pip.
        Optionally updates the environment YAML to keep it in sync.

        v1.2.0: New method. The diagnostic agent uses this for targeted
        single-package installs discovered during error investigation.

        Args:
            env_name: Environment name (agi_ prefix added if needed)
            package: Package name (with optional version spec, e.g. "celltypist>=0.6")
            method: "conda" or "pip" — which installer to try first
            channel: Conda channel for conda installs (default: conda-forge)
            yaml_path: If provided, update this YAML file after successful install

        Returns:
            Dict with success, method_used, and optional yaml_updated fields
        """
        full_env_name = self._normalize_env_name(env_name)

        if not self.environment_exists(full_env_name):
            return {
                "success": False,
                "error": f"Environment '{full_env_name}' does not exist",
            }

        pkg_name = re.split(r'[><=!~]', package)[0].strip()
        method_used = None

        if method == "conda":
            # Try conda first
            result = self._run_conda(
                ["install", "-n", full_env_name, "-c", channel, "-y", package],
                timeout=300,
            )
            if result["success"]:
                method_used = "conda"
            else:
                # Fall back to pip
                logger.info(f"Conda install failed for '{package}', trying pip")
                pip_result = subprocess.run(
                    [self.conda_path, 'run', '-n', full_env_name,
                     'pip', 'install', package],
                    capture_output=True, text=True, timeout=300,
                )
                if pip_result.returncode == 0:
                    method_used = "pip"
                else:
                    return {
                        "success": False,
                        "error": (
                            f"Both conda and pip failed for '{package}'. "
                            f"conda: {result.get('stderr', '')[:200]}. "
                            f"pip: {pip_result.stderr[:200]}"
                        ),
                    }
        else:
            # pip directly
            pip_result = subprocess.run(
                [self.conda_path, 'run', '-n', full_env_name,
                 'pip', 'install', package],
                capture_output=True, text=True, timeout=300,
            )
            if pip_result.returncode == 0:
                method_used = "pip"
            else:
                return {
                    "success": False,
                    "error": f"pip install failed for '{package}': {pip_result.stderr[:300]}",
                }

        # Update YAML if requested
        yaml_updated = False
        if yaml_path:
            section = "pip" if method_used == "pip" else "conda"
            yaml_updated = self.update_yaml_with_package(
                yaml_path, package, section=section
            )

        # Update registry
        if full_env_name in self.env_registry["environments"]:
            pkg_list = self.env_registry["environments"][full_env_name].get(
                "packages_installed", []
            )
            pkg_list.append(f"{package} (via {method_used})")
            self.env_registry["environments"][full_env_name]["packages_installed"] = pkg_list
            self.env_registry["environments"][full_env_name]["last_updated"] = (
                datetime.now().isoformat()
            )
            self._save_registry()

        return {
            "success": True,
            "package": package,
            "method_used": method_used,
            "env_name": full_env_name,
            "yaml_updated": yaml_updated,
            "message": f"Installed '{package}' via {method_used} into '{full_env_name}'",
        }

    # =========================================================================
    # YAML MANIPULATION (v1.2.0)
    # =========================================================================

    def update_yaml_with_package(
        self,
        yaml_path: str,
        package: str,
        section: str = "pip",
    ) -> bool:
        """
        Add a package to the environment YAML file.

        Adds to the conda dependencies list or the pip sub-list depending
        on the section parameter. Avoids duplicates (checks base package name).

        v1.2.0: New method. Called by install_package() and the diagnostic
        agent to keep the YAML in sync after every mutation.

        Args:
            yaml_path: Path to environment YAML
            package: Package spec (e.g., "celltypist>=0.6")
            section: "conda" or "pip"

        Returns:
            True if YAML was updated, False on error
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            logger.warning(f"Cannot update YAML — file not found: {yaml_path}")
            return False

        try:
            content = yaml_path.read_text()
            parsed = yaml.safe_load(content)
            if not parsed or 'dependencies' not in parsed:
                logger.warning(f"Cannot update YAML — no dependencies key: {yaml_path}")
                return False

            pkg_name = re.split(r'[><=!~]', package)[0].strip().lower()

            if section == "pip":
                # Find or create pip sub-dict
                pip_dict = None
                for dep in parsed['dependencies']:
                    if isinstance(dep, dict) and 'pip' in dep:
                        pip_dict = dep
                        break

                if pip_dict is None:
                    # Ensure 'pip' package is in conda deps for pip to work
                    conda_names = [
                        re.split(r'[><=!~]', d)[0].strip().lower()
                        for d in parsed['dependencies']
                        if isinstance(d, str)
                    ]
                    if 'pip' not in conda_names:
                        parsed['dependencies'].append('pip')
                    pip_dict = {'pip': []}
                    parsed['dependencies'].append(pip_dict)

                # Check for duplicates
                existing_pip_names = [
                    re.split(r'[><=!~]', str(p))[0].strip().lower()
                    for p in pip_dict['pip']
                ]
                if pkg_name not in existing_pip_names:
                    pip_dict['pip'].append(package)
                else:
                    logger.debug(f"Package '{pkg_name}' already in pip section")
                    return True  # Already present

            else:
                # Add to conda dependencies
                existing_conda_names = [
                    re.split(r'[><=!~]', d)[0].strip().lower()
                    for d in parsed['dependencies']
                    if isinstance(d, str)
                ]
                if pkg_name not in existing_conda_names:
                    # Insert before any pip dict
                    insert_idx = len(parsed['dependencies'])
                    for i, dep in enumerate(parsed['dependencies']):
                        if isinstance(dep, dict):
                            insert_idx = i
                            break
                    parsed['dependencies'].insert(insert_idx, package)
                else:
                    logger.debug(f"Package '{pkg_name}' already in conda section")
                    return True  # Already present

            with open(yaml_path, 'w') as f:
                yaml.dump(parsed, f, default_flow_style=False)

            logger.info(f"Updated YAML '{yaml_path}': added '{package}' to {section}")
            return True

        except Exception as e:
            logger.warning(f"Failed to update YAML with package '{package}': {e}")
            return False

    def remove_package_from_yaml(
        self,
        yaml_path: str,
        package: str,
    ) -> bool:
        """
        Remove a package from the environment YAML file.

        Removes from both conda and pip sections. Matches by base package name
        (ignoring version specifiers).

        v1.2.0: New method.

        Args:
            yaml_path: Path to environment YAML
            package: Package name to remove (version spec ignored for matching)

        Returns:
            True if package was found and removed, False otherwise
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            return False

        try:
            content = yaml_path.read_text()
            parsed = yaml.safe_load(content)
            if not parsed or 'dependencies' not in parsed:
                return False

            pkg_name = re.split(r'[><=!~]', package)[0].strip().lower()
            removed = False

            # Remove from conda deps
            new_deps = []
            for dep in parsed['dependencies']:
                if isinstance(dep, str):
                    name = re.split(r'[><=!~]', dep)[0].strip().lower()
                    if name == pkg_name:
                        removed = True
                        continue
                    new_deps.append(dep)
                elif isinstance(dep, dict) and 'pip' in dep:
                    new_pip = []
                    for p in dep['pip']:
                        name = re.split(r'[><=!~]', str(p))[0].strip().lower()
                        if name == pkg_name:
                            removed = True
                            continue
                        new_pip.append(p)
                    if new_pip:
                        new_deps.append({'pip': new_pip})
                    # If pip list is now empty, drop the pip dict entirely
                else:
                    new_deps.append(dep)

            if removed:
                parsed['dependencies'] = new_deps
                with open(yaml_path, 'w') as f:
                    yaml.dump(parsed, f, default_flow_style=False)
                logger.info(f"Removed '{pkg_name}' from YAML '{yaml_path}'")

            return removed

        except Exception as e:
            logger.warning(f"Failed to remove package '{package}' from YAML: {e}")
            return False

    # =========================================================================
    # PACKAGE VALIDATION (v1.2.0)
    # =========================================================================

    def validate_env_has_package(
        self,
        env_name: str,
        package: str,
        language: str = "python",
    ) -> Dict[str, Any]:
        """
        Verify that a package is actually importable/available in the environment.

        Runs a language-appropriate check command inside the conda environment:
          - python: python -c "import {package}"
          - r: Rscript -e "library({package})"
          - bash/system: which {package}
          - perl: perl -e "use {package}"

        v1.2.0: New method. Used by the diagnostic agent to verify that an
        install_package() actually made the package usable.

        Args:
            env_name: Environment name
            package: Package/module name to check (import name, not PyPI name)
            language: "python", "r", "bash", "perl"

        Returns:
            Dict with success (bool), output, and error fields
        """
        full_env_name = self._normalize_env_name(env_name)

        if not self.environment_exists(full_env_name):
            return {
                "success": False,
                "error": f"Environment '{full_env_name}' does not exist",
            }

        # Build the check command based on language
        if language == "python":
            check_cmd = f'python -c "import {package}"'
        elif language == "r":
            check_cmd = f'Rscript -e "library({package})"'
        elif language in ("bash", "system"):
            check_cmd = f"which {package}"
        elif language == "perl":
            check_cmd = f'perl -e "use {package}"'
        else:
            check_cmd = f'python -c "import {package}"'

        try:
            result = subprocess.run(
                [self.conda_path, 'run', '-n', full_env_name,
                 'bash', '-c', check_cmd],
                capture_output=True, text=True, timeout=60,
            )
            return {
                "success": result.returncode == 0,
                "package": package,
                "language": language,
                "env_name": full_env_name,
                "output": result.stdout.strip(),
                "error": result.stderr.strip() if result.returncode != 0 else None,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Validation check timed out for '{package}'",
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Validation check failed: {e}",
            }

    # =========================================================================
    # EXISTING METHODS (unchanged from v3.2)
    # =========================================================================

    def install_packages(
        self,
        env_name: str,
        packages: List[str],
        use_pip: bool = False,
        channel: str = None
    ) -> Dict[str, Any]:
        """
        Install packages into an environment (batch install).
        
        Args:
            env_name: Environment name
            packages: List of packages to install
            use_pip: Use pip instead of conda
            channel: Conda channel to use (e.g., conda-forge, bioconda)
            
        Returns:
            Result dictionary
        """
        full_env_name = self._normalize_env_name(env_name)
        
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
        full_env_name = self._normalize_env_name(env_name)
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
        full_env_name = self._normalize_env_name(env_name)
        
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
            Result dictionary with stdout, stderr
        """
        full_env_name = self._normalize_env_name(env_name)
        
        if not self.environment_exists(full_env_name):
            return {
                "success": False,
                "error": f"Environment '{full_env_name}' does not exist"
            }
        
        # Use conda run
        result = self._run_conda(
            ["run", "-n", full_env_name, "bash", "-c", command],
            timeout=timeout
        )
        
        return {
            "success": result["success"],
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "return_code": result.get("return_code"),
            "env_name": full_env_name
        }
    
    def cleanup_all_environments(self) -> Dict[str, Any]:
        """
        Remove all project environments.
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

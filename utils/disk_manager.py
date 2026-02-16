"""
Disk Manager — ~/.conda quota monitoring and cleanup for AGI Pipeline v1.2.0

Manages the limited home directory space (~100GB) on HPC clusters where
conda environments can quickly consume available quota. Provides:

  - Proactive space checks before environment builds
  - Automated conda cache cleanup
  - Stale AGI environment removal
  - Emergency cleanup when disk is critically low
  - Space reporting for diagnostics

Environment variables (set in RUN_AGI_PIPELINE_*.sh):
  AGI_HOME_QUOTA_GB              Total home quota (default: 100)
  AGI_DISK_CLEANUP_THRESHOLD_GB  Trigger proactive cleanup (default: 15)
  AGI_DISK_EMERGENCY_THRESHOLD_GB  Trigger emergency cleanup (default: 5)

Usage:
    from utils.disk_manager import DiskManager

    dm = DiskManager()

    # Before building an environment
    if dm.ensure_space_for_build(estimated_size_gb=5):
        # Safe to proceed with conda env create
        ...
    else:
        # Could not free enough space — escalate
        ...

    # Get a full space report (useful for diagnostic agent)
    report = dm.get_space_report()

    # Reactive cleanup after a disk quota error
    recovered = dm.emergency_cleanup(keep_envs=["agi_step_03"])
"""

import os
import re
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class DiskManager:
    """
    Manages ~/.conda disk usage and home directory quota on HPC clusters.

    Designed for environments where:
    - Home directory has a hard quota (default 100GB)
    - Conda environments are the primary space consumer in ~/
    - Multiple AGI pipeline runs create/destroy environments frequently
    - The conda package cache (~/.conda/pkgs) grows unbounded without cleanup
    """

    # Defaults — overridden by environment variables or constructor args
    DEFAULT_HOME_QUOTA_GB = 100
    DEFAULT_CLEANUP_THRESHOLD_GB = 15
    DEFAULT_EMERGENCY_THRESHOLD_GB = 5
    DEFAULT_ESTIMATED_ENV_SIZE_GB = 5
    DEFAULT_STALE_ENV_PREFIX = "agi_"

    def __init__(
        self,
        home_quota_gb: int = None,
        cleanup_threshold_gb: int = None,
        emergency_threshold_gb: int = None,
        stale_env_prefix: str = None,
    ):
        self.home_quota_gb = (
            home_quota_gb
            or _env_int("AGI_HOME_QUOTA_GB", self.DEFAULT_HOME_QUOTA_GB)
        )
        self.cleanup_threshold_gb = (
            cleanup_threshold_gb
            or _env_int(
                "AGI_DISK_CLEANUP_THRESHOLD_GB",
                self.DEFAULT_CLEANUP_THRESHOLD_GB,
            )
        )
        self.emergency_threshold_gb = (
            emergency_threshold_gb
            or _env_int(
                "AGI_DISK_EMERGENCY_THRESHOLD_GB",
                self.DEFAULT_EMERGENCY_THRESHOLD_GB,
            )
        )
        self.stale_env_prefix = (
            stale_env_prefix or self.DEFAULT_STALE_ENV_PREFIX
        )
        self.home_dir = Path.home()

    # =========================================================================
    # SPACE QUERIES
    # =========================================================================

    def check_available_space(self, path: str = None) -> float:
        """
        Return available disk space in GB for the filesystem containing *path*.

        Tries three strategies in order:
          1. ``quota -s`` (HPC systems with enforced quotas)
          2. ``df`` (general POSIX)
          3. Python ``os.statvfs`` (fallback)

        Args:
            path: Directory to check. Defaults to $HOME.

        Returns:
            Available space in GB (float). Returns -1.0 if detection fails.
        """
        target = path or str(self.home_dir)

        # Strategy 1: quota command (most accurate on HPC)
        free_gb = self._try_quota(target)
        if free_gb is not None:
            return free_gb

        # Strategy 2: df
        free_gb = self._try_df(target)
        if free_gb is not None:
            return free_gb

        # Strategy 3: os.statvfs
        free_gb = self._try_statvfs(target)
        if free_gb is not None:
            return free_gb

        logger.warning("Could not determine available disk space")
        return -1.0

    def get_conda_cache_size(self) -> float:
        """
        Return the size of ~/.conda/pkgs in GB.

        Returns:
            Size in GB, or 0.0 if the directory does not exist.
        """
        pkgs_dir = self.home_dir / ".conda" / "pkgs"
        if not pkgs_dir.exists():
            return 0.0
        return self._dir_size_gb(pkgs_dir)

    def get_conda_envs_size(self) -> float:
        """
        Return the total size of ~/.conda/envs in GB.
        """
        envs_dir = self.home_dir / ".conda" / "envs"
        if not envs_dir.exists():
            return 0.0
        return self._dir_size_gb(envs_dir)

    def list_agi_environments(self) -> List[Dict[str, Any]]:
        """
        List all conda environments whose names start with the stale prefix.

        Returns:
            List of dicts with keys: name, path, size_gb (size may be None
            if the environment is not in the default envs directory).
        """
        envs = []
        try:
            result = subprocess.run(
                ["conda", "env", "list", "--json"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                return envs

            import json
            data = json.loads(result.stdout)
            for env_path_str in data.get("envs", []):
                env_path = Path(env_path_str)
                name = env_path.name
                if name.startswith(self.stale_env_prefix):
                    size = None
                    if env_path.exists():
                        size = self._dir_size_gb(env_path)
                    envs.append({
                        "name": name,
                        "path": str(env_path),
                        "size_gb": size,
                    })
        except Exception as e:
            logger.warning(f"Failed to list conda environments: {e}")

        return envs

    # =========================================================================
    # SPACE REPORT
    # =========================================================================

    def get_space_report(self) -> Dict[str, Any]:
        """
        Comprehensive disk usage report for diagnostics and logging.

        Returns:
            Dict with home_free_gb, conda_cache_gb, conda_envs_gb,
            agi_env_count, agi_envs_total_gb, home_quota_gb, and status
            ("ok", "warning", "critical").
        """
        free_gb = self.check_available_space()
        cache_gb = self.get_conda_cache_size()
        envs_gb = self.get_conda_envs_size()
        agi_envs = self.list_agi_environments()
        agi_total = sum(e["size_gb"] for e in agi_envs if e["size_gb"])

        if free_gb < 0:
            status = "unknown"
        elif free_gb < self.emergency_threshold_gb:
            status = "critical"
        elif free_gb < self.cleanup_threshold_gb:
            status = "warning"
        else:
            status = "ok"

        return {
            "home_free_gb": round(free_gb, 2),
            "home_quota_gb": self.home_quota_gb,
            "conda_cache_gb": round(cache_gb, 2),
            "conda_envs_gb": round(envs_gb, 2),
            "agi_env_count": len(agi_envs),
            "agi_envs_total_gb": round(agi_total, 2),
            "agi_envs": agi_envs,
            "status": status,
        }

    # =========================================================================
    # CLEANUP ACTIONS
    # =========================================================================

    def clean_conda_cache(self) -> Dict[str, Any]:
        """
        Run ``conda clean --all --yes`` to remove cached packages, tarballs,
        and index caches.

        Returns:
            Dict with success, space_before_gb, space_after_gb, freed_gb.
        """
        before = self.check_available_space()
        logger.info("Running conda clean --all --yes ...")

        try:
            result = subprocess.run(
                ["conda", "clean", "--all", "--yes"],
                capture_output=True,
                text=True,
                timeout=300,
            )
            success = result.returncode == 0
            if not success:
                logger.warning(f"conda clean stderr: {result.stderr[:500]}")
        except subprocess.TimeoutExpired:
            logger.warning("conda clean timed out after 300s")
            success = False
        except Exception as e:
            logger.warning(f"conda clean failed: {e}")
            success = False

        after = self.check_available_space()
        freed = max(0, after - before) if (after >= 0 and before >= 0) else 0

        logger.info(
            f"conda clean: freed ~{freed:.1f}GB "
            f"({before:.1f}GB -> {after:.1f}GB free)"
        )
        return {
            "success": success,
            "space_before_gb": round(before, 2),
            "space_after_gb": round(after, 2),
            "freed_gb": round(freed, 2),
        }

    def remove_stale_environments(
        self,
        keep_list: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Remove conda environments matching the stale prefix, except those
        in *keep_list*.

        Args:
            keep_list: Environment names to preserve (e.g. the currently
                       active task's env).

        Returns:
            Dict with removed (list of names), failed (list), freed_gb.
        """
        keep = set(keep_list or [])
        agi_envs = self.list_agi_environments()
        before = self.check_available_space()

        removed = []
        failed = []

        for env in agi_envs:
            name = env["name"]
            if name in keep:
                logger.info(f"Keeping environment: {name}")
                continue

            logger.info(f"Removing stale environment: {name}")
            try:
                result = subprocess.run(
                    ["conda", "env", "remove", "-n", name, "-y"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode == 0:
                    removed.append(name)
                else:
                    failed.append(name)
                    logger.warning(
                        f"Failed to remove {name}: {result.stderr[:200]}"
                    )
            except Exception as e:
                failed.append(name)
                logger.warning(f"Exception removing {name}: {e}")

        after = self.check_available_space()
        freed = max(0, after - before) if (after >= 0 and before >= 0) else 0

        logger.info(
            f"Stale env cleanup: removed {len(removed)}, "
            f"failed {len(failed)}, freed ~{freed:.1f}GB"
        )
        return {
            "removed": removed,
            "failed": failed,
            "freed_gb": round(freed, 2),
        }

    def proactive_cleanup(
        self,
        threshold_gb: float = None,
        keep_envs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        If free space is below *threshold_gb*, run cache cleanup.
        If still below, remove stale AGI environments.

        Args:
            threshold_gb: Override cleanup threshold.
            keep_envs: Env names to preserve.

        Returns:
            Dict with action_taken (bool), steps performed, final free space.
        """
        threshold = threshold_gb or self.cleanup_threshold_gb
        free = self.check_available_space()
        steps = []

        if free >= threshold:
            return {
                "action_taken": False,
                "free_gb": round(free, 2),
                "steps": [],
                "message": f"Space OK ({free:.1f}GB free >= {threshold}GB threshold)",
            }

        logger.info(
            f"Proactive cleanup triggered: {free:.1f}GB free "
            f"< {threshold}GB threshold"
        )

        # Step 1: conda cache
        cache_result = self.clean_conda_cache()
        steps.append({"action": "conda_clean", **cache_result})
        free = self.check_available_space()

        if free >= threshold:
            return {
                "action_taken": True,
                "free_gb": round(free, 2),
                "steps": steps,
                "message": f"Cache cleanup sufficient ({free:.1f}GB free)",
            }

        # Step 2: stale environments
        env_result = self.remove_stale_environments(keep_list=keep_envs)
        steps.append({"action": "remove_stale_envs", **env_result})
        free = self.check_available_space()

        return {
            "action_taken": True,
            "free_gb": round(free, 2),
            "steps": steps,
            "message": (
                f"Cleanup complete ({free:.1f}GB free). "
                f"{'OK' if free >= threshold else 'Still below threshold!'}"
            ),
        }

    def emergency_cleanup(
        self,
        keep_envs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Aggressive cleanup for disk-full situations. Runs all cleanup
        strategies and reports total space recovered.

        This is called reactively when a ``No space left on device`` or
        ``Disk quota exceeded`` error is caught.

        Args:
            keep_envs: Env names to preserve.

        Returns:
            Dict with total_freed_gb, final_free_gb, steps.
        """
        logger.warning("EMERGENCY disk cleanup triggered!")
        before = self.check_available_space()
        steps = []

        # 1. Conda cache (tarballs + packages)
        cache_result = self.clean_conda_cache()
        steps.append({"action": "conda_clean_all", **cache_result})

        # 2. Extra: clean just packages if anything remains
        try:
            subprocess.run(
                ["conda", "clean", "--packages", "--yes"],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except Exception:
            pass

        # 3. Remove stale agi_ environments
        env_result = self.remove_stale_environments(keep_list=keep_envs)
        steps.append({"action": "remove_stale_envs", **env_result})

        # 4. Clean pip cache if present
        pip_result = self._clean_pip_cache()
        if pip_result:
            steps.append({"action": "pip_cache_clean", **pip_result})

        after = self.check_available_space()
        total_freed = max(0, after - before) if (after >= 0 and before >= 0) else 0

        logger.info(
            f"Emergency cleanup complete: freed ~{total_freed:.1f}GB, "
            f"now {after:.1f}GB free"
        )

        return {
            "total_freed_gb": round(total_freed, 2),
            "final_free_gb": round(after, 2),
            "steps": steps,
        }

    # =========================================================================
    # PRE-BUILD CHECK
    # =========================================================================

    def ensure_space_for_build(
        self,
        estimated_size_gb: float = None,
        keep_envs: Optional[List[str]] = None,
    ) -> bool:
        """
        Ensure there is enough free space to build a new conda environment.

        Runs proactive cleanup if needed. Returns True if the build can
        proceed, False if space is still insufficient after all cleanup.

        Args:
            estimated_size_gb: Expected size of the new environment.
            keep_envs: Env names to preserve during cleanup.

        Returns:
            True if there is enough space to proceed.
        """
        est = estimated_size_gb or self.DEFAULT_ESTIMATED_ENV_SIZE_GB
        needed = est + self.emergency_threshold_gb  # buffer

        free = self.check_available_space()

        if free >= needed:
            logger.info(
                f"Disk check passed: {free:.1f}GB free >= "
                f"{needed:.1f}GB needed"
            )
            return True

        logger.info(
            f"Disk check: {free:.1f}GB free < {needed:.1f}GB needed, "
            f"running proactive cleanup..."
        )
        self.proactive_cleanup(
            threshold_gb=needed, keep_envs=keep_envs
        )

        free = self.check_available_space()

        if free >= needed:
            logger.info(f"Disk check passed after cleanup: {free:.1f}GB free")
            return True

        logger.warning(
            f"Disk check FAILED: {free:.1f}GB free < {needed:.1f}GB needed "
            f"even after cleanup"
        )
        return False

    # =========================================================================
    # HELPERS — SPACE DETECTION
    # =========================================================================

    def _try_quota(self, path: str) -> Optional[float]:
        """Try the ``quota`` command to get remaining space in GB."""
        try:
            result = subprocess.run(
                ["quota", "-s"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode != 0:
                return None

            # Parse quota output — format varies by system, but typically:
            #   Filesystem  blocks  quota  limit  grace  files  ...
            # Look for lines with numbers and units (K, M, G, T)
            for line in result.stdout.splitlines():
                if "/home" in line or "home" in line.lower():
                    parts = line.split()
                    if len(parts) >= 4:
                        used = _parse_size_to_gb(parts[1])
                        limit = _parse_size_to_gb(parts[3])
                        if used is not None and limit is not None and limit > 0:
                            return max(0, limit - used)

            return None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        except Exception:
            return None

    def _try_df(self, path: str) -> Optional[float]:
        """Try ``df -BG`` to get available space in GB."""
        try:
            result = subprocess.run(
                ["df", "-BG", path],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode != 0:
                return None

            lines = result.stdout.strip().splitlines()
            if len(lines) < 2:
                return None

            # The available column is typically the 4th field
            # Handle wrapped lines (filesystem name too long)
            data_line = lines[-1]
            parts = data_line.split()
            for part in parts:
                # Look for the "Available" column — a number ending in G
                if part.endswith("G") and part[:-1].isdigit():
                    # Could be Used, Available, or Size — we want Available
                    pass

            # More reliable: parse the 4th column
            if len(parts) >= 4:
                avail_str = parts[3].rstrip("G")
                if avail_str.isdigit():
                    return float(avail_str)

            return None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        except Exception:
            return None

    def _try_statvfs(self, path: str) -> Optional[float]:
        """Use os.statvfs as a last resort."""
        try:
            stat = os.statvfs(path)
            free_bytes = stat.f_bavail * stat.f_frsize
            return free_bytes / (1024 ** 3)
        except Exception:
            return None

    # =========================================================================
    # HELPERS — SIZE CALCULATION
    # =========================================================================

    def _dir_size_gb(self, path: Path) -> float:
        """
        Get directory size in GB using ``du -s``. Falls back to Python
        walk if du is unavailable.
        """
        try:
            result = subprocess.run(
                ["du", "-s", "--block-size=1G", str(path)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                size_str = result.stdout.split()[0]
                return float(size_str)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        except Exception:
            pass

        # Fallback: Python walk (slower but portable)
        try:
            total = 0
            for dirpath, _dirnames, filenames in os.walk(path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    try:
                        total += os.path.getsize(fp)
                    except OSError:
                        pass
            return total / (1024 ** 3)
        except Exception:
            return 0.0

    def _clean_pip_cache(self) -> Optional[Dict[str, Any]]:
        """Clean pip cache as an additional space recovery measure."""
        try:
            before = self.check_available_space()
            result = subprocess.run(
                ["pip", "cache", "purge"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            after = self.check_available_space()
            freed = max(0, after - before) if (after >= 0 and before >= 0) else 0
            return {
                "success": result.returncode == 0,
                "freed_gb": round(freed, 2),
            }
        except Exception:
            return None


# =============================================================================
# MODULE-LEVEL HELPERS
# =============================================================================

def _env_int(var_name: str, default: int) -> int:
    """Read an integer from an environment variable with a default."""
    val = os.environ.get(var_name)
    if val is not None:
        try:
            return int(val)
        except ValueError:
            pass
    return default


def _parse_size_to_gb(size_str: str) -> Optional[float]:
    """
    Parse a human-readable size string (e.g. '45G', '120M', '1.2T', '500K',
    '10485760') into GB.

    Handles suffixes: K, M, G, T (case-insensitive) and also bare numbers
    which are assumed to be in KB (common quota output format).
    """
    if not size_str:
        return None

    size_str = size_str.strip().rstrip("*")  # quota sometimes adds *

    match = re.match(
        r"^(\d+(?:\.\d+)?)\s*([KMGT]?)(?:i?B)?$",
        size_str,
        re.IGNORECASE,
    )
    if not match:
        return None

    value = float(match.group(1))
    unit = match.group(2).upper() if match.group(2) else ""

    multipliers = {"": 1 / (1024 * 1024), "K": 1 / (1024 * 1024), "M": 1 / 1024, "G": 1.0, "T": 1024.0}

    if unit == "" and value > 1000:
        # Bare large number — likely KB (common in quota output)
        return value / (1024 * 1024)
    elif unit == "":
        # Small bare number — ambiguous, assume GB
        return value

    return value * multipliers.get(unit, 1.0)


# =============================================================================
# CONVENIENCE SINGLETON
# =============================================================================

_default_instance: Optional[DiskManager] = None


def get_disk_manager() -> DiskManager:
    """Return a module-level DiskManager singleton."""
    global _default_instance
    if _default_instance is None:
        _default_instance = DiskManager()
    return _default_instance

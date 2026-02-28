"""
ManifestManager — Output Manifest for AGI Pipeline v1.2.8

Writes and reads the canonical output_manifest.json file that lives at
reports/output_manifest.json inside every project directory.

The manifest is the single source of truth for:
  - Which output files each step produced and where they live
  - Which paths are protected from cleanup scripts
  - The state of each step's conda environment
  - Validation status of each output (populated by v1.2.9 Validation Agent)

Cleanup scripts (PARTIAL_CLEAN_PROJECT.sh, FULL_CLEAN_PROJECT.sh) read
get_all_protected_paths() before removing anything to ensure they never
touch scientific outputs or protected artifacts.

The env_state field drives v1.2.10 human-in-the-loop installation:
  - agent_created  : default; sub-agent may remove and rebuild
  - human_modified : sub-agent repair loop skips removal entirely
  - pinned         : reserved for future use; never removed without DB wipe

All path values stored in the manifest are relative to project_root so the
manifest remains valid if the project directory is moved or accessed from a
different absolute path.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Manifest schema version — bump when the schema changes in a breaking way
MANIFEST_SCHEMA_VERSION = "1.2.8"

# Valid env_state values
ENV_STATE_AGENT_CREATED = "agent_created"
ENV_STATE_HUMAN_MODIFIED = "human_modified"
ENV_STATE_PINNED = "pinned"

VALID_ENV_STATES = {
    ENV_STATE_AGENT_CREATED,
    ENV_STATE_HUMAN_MODIFIED,
    ENV_STATE_PINNED,
}

# Valid validation_status values (extended in v1.2.9)
VALIDATION_STATUS_NOT_VALIDATED = "not_validated"
VALIDATION_STATUS_PASSED = "passed"
VALIDATION_STATUS_FAILED = "failed"
VALIDATION_STATUS_SKIPPED = "skipped"
VALIDATION_STATUS_ESCALATED = "escalated"

# Directories that are ALWAYS protected — cleanup scripts must never touch these
PROTECTED_DIRECTORIES = {
    "data",
    "outputs",
    "envs",
    "scripts",
    "slurm/scripts",
    "reports",
}


class ManifestManager:
    """
    Read/write interface for reports/output_manifest.json.

    Instantiate once per sub-agent execution. All writes are atomic — the file
    is loaded, modified in memory, then written in full. This is safe because
    only one sub-agent writes to the manifest per step at any given time
    (parallel sub-agents write to different step_ids).

    Args:
        project_root: Absolute path to the project directory. All paths stored
                      in the manifest are relative to this root.
    """

    MANIFEST_FILENAME = "output_manifest.json"

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()
        self.manifest_path = (
            self.project_root / "reports" / self.MANIFEST_FILENAME
        )
        self._ensure_reports_dir()

    # =========================================================================
    # PUBLIC API — WRITE
    # =========================================================================

    def write_step_entry(
        self,
        step_id: str,
        outputs: List[Dict[str, Any]],
        env_yaml: Optional[str] = None,
        script: Optional[str] = None,
        sbatch: Optional[str] = None,
        env_state: str = ENV_STATE_AGENT_CREATED,
        env_name: Optional[str] = None,
        status: str = "completed",
    ) -> None:
        """
        Write or overwrite the manifest entry for a completed step.

        Called by the sub-agent at the end of Phase 4 (successful completion).
        All path arguments should be relative to project_root; absolute paths
        are automatically converted.

        Args:
            step_id:   Step identifier, e.g. "step_07" or "step_07_clustering".
            outputs:   List of output dicts. Each dict should contain at minimum
                       {"path": "outputs/step_07/clustered.h5ad", "type": "h5ad"}.
                       See _build_output_entry() for the full schema.
            env_yaml:  Path to the conda YAML spec, e.g. "envs/step_07_env.yaml".
            script:    Path to the generated analysis script.
            sbatch:    Path to the generated sbatch file.
            env_state: One of agent_created | human_modified | pinned.
            env_name:  Conda environment name, e.g. "agi_step_07_clustering".
            status:    Step status string, default "completed".
        """
        if env_state not in VALID_ENV_STATES:
            logger.warning(
                f"[{step_id}] Unknown env_state '{env_state}'; "
                f"defaulting to '{ENV_STATE_AGENT_CREATED}'"
            )
            env_state = ENV_STATE_AGENT_CREATED

        manifest = self._load()

        manifest["steps"][step_id] = {
            "status": status,
            "completed_at": datetime.now().isoformat(),
            "outputs": [
                self._build_output_entry(o) for o in (outputs or [])
            ],
            "env_yaml": self._to_relative(env_yaml),
            "script": self._to_relative(script),
            "sbatch": self._to_relative(sbatch),
            "env_name": env_name,
            "env_state": env_state,
        }

        self._save(manifest)
        logger.info(
            f"[{step_id}] Manifest entry written "
            f"({len(outputs or [])} output(s), env_state={env_state})"
        )

    def update_step_status(self, step_id: str, status: str) -> None:
        """
        Update the status field of an existing step entry.

        Used to transition a step from "completed" to "validation_escalated"
        or other terminal states without rewriting the full entry.
        """
        manifest = self._load()
        if step_id not in manifest["steps"]:
            logger.warning(
                f"[{step_id}] update_step_status: step not in manifest, skipping"
            )
            return
        manifest["steps"][step_id]["status"] = status
        manifest["steps"][step_id]["updated_at"] = datetime.now().isoformat()
        self._save(manifest)

    def set_env_state(
        self,
        step_id: str,
        state: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Set the env_state for a step and optionally attach human install metadata.

        Called by INJECT_HINTS.sh handler when HUMAN_INSTALL_STEP is set,
        and by FORCE_RESET_STEP to revert a step back to agent_created.

        Args:
            step_id:  Step identifier.
            state:    One of agent_created | human_modified | pinned.
            metadata: Optional dict stored under "human_env_metadata". Used by
                      v1.2.10 to record binary paths, install notes, etc.
        """
        if state not in VALID_ENV_STATES:
            raise ValueError(
                f"Invalid env_state '{state}'. "
                f"Must be one of: {sorted(VALID_ENV_STATES)}"
            )

        manifest = self._load()

        if step_id not in manifest["steps"]:
            # Create a minimal stub entry so env_state is recorded even if
            # the step hasn't run yet (e.g. HUMAN_INSTALL_STEP before first run)
            manifest["steps"][step_id] = {
                "status": "pending",
                "completed_at": None,
                "outputs": [],
                "env_yaml": None,
                "script": None,
                "sbatch": None,
                "env_name": None,
                "env_state": state,
                "human_env_metadata": metadata or {},
            }
        else:
            manifest["steps"][step_id]["env_state"] = state
            if metadata:
                manifest["steps"][step_id]["human_env_metadata"] = metadata
            manifest["steps"][step_id]["updated_at"] = datetime.now().isoformat()

        self._save(manifest)
        logger.info(f"[{step_id}] env_state set to '{state}'")

    def update_validation_status(
        self,
        step_id: str,
        output_path: str,
        status: str,
        sample_repr: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update the validation_status of a specific output file within a step.

        Called by the Validation Agent (v1.2.9) after structural or semantic
        checks. Also stores the sample_repr from successful validation for
        use by the Librarian Agent (v1.3.0).

        Args:
            step_id:     Step identifier.
            output_path: The output path to update (relative or absolute).
            status:      One of: not_validated | passed | failed | skipped | escalated.
            sample_repr: Optional dict with shape, columns, head, key_metadata.
        """
        rel_path = self._to_relative(output_path)
        manifest = self._load()

        if step_id not in manifest["steps"]:
            logger.warning(
                f"[{step_id}] update_validation_status: step not in manifest"
            )
            return

        outputs = manifest["steps"][step_id].get("outputs", [])
        matched = False
        for output in outputs:
            if output.get("path") == rel_path:
                output["validation_status"] = status
                output["validated_at"] = datetime.now().isoformat()
                if sample_repr is not None:
                    output["sample_repr"] = sample_repr
                matched = True
                break

        if not matched:
            logger.warning(
                f"[{step_id}] update_validation_status: "
                f"path '{rel_path}' not found in manifest outputs"
            )
            return

        self._save(manifest)
        logger.info(
            f"[{step_id}] Validation status for '{rel_path}' → '{status}'"
        )

    # =========================================================================
    # PUBLIC API — READ
    # =========================================================================

    def read_step_entry(self, step_id: str) -> Dict[str, Any]:
        """
        Return the full manifest entry for a step.

        Returns an empty dict if the step has no entry yet.
        """
        manifest = self._load()
        return manifest["steps"].get(step_id, {})

    def get_output_paths(self, step_id: str) -> List[str]:
        """
        Return the list of output paths registered for a step.

        Paths are relative to project_root. Returns an empty list if the step
        has no manifest entry or produced no registered outputs.
        """
        entry = self.read_step_entry(step_id)
        return [o["path"] for o in entry.get("outputs", []) if o.get("path")]

    def get_all_protected_paths(self) -> List[str]:
        """
        Return the full list of paths that must never be deleted by any
        cleanup script.

        Includes:
          - All output paths from all completed steps
          - All env_yaml paths
          - All script paths
          - All sbatch paths (slurm/scripts/ is protected)
          - The manifest file itself

        Paths are relative to project_root. Cleanup scripts should resolve
        these against the project root before comparing to deletion candidates.
        """
        manifest = self._load()
        protected: List[str] = []

        # Always protect the manifest itself
        protected.append(str(self.manifest_path.relative_to(self.project_root)))

        for step_id, entry in manifest["steps"].items():
            # Output files
            for output in entry.get("outputs", []):
                path = output.get("path")
                if path:
                    protected.append(path)

            # Artifact paths
            for key in ("env_yaml", "script", "sbatch"):
                path = entry.get(key)
                if path:
                    protected.append(path)

        return sorted(set(protected))

    def get_env_state(self, step_id: str) -> str:
        """
        Return the env_state for a step.

        Returns ENV_STATE_AGENT_CREATED if the step has no manifest entry,
        since that is the safe default (sub-agent may rebuild freely).
        """
        entry = self.read_step_entry(step_id)
        return entry.get("env_state", ENV_STATE_AGENT_CREATED)

    def get_human_env_metadata(self, step_id: str) -> Dict[str, Any]:
        """
        Return the human install metadata for a step.

        Returns an empty dict if no metadata has been recorded.
        Used by sub-agent Phase 2 to retrieve HUMAN_ENV_BIN_PATH and
        related fields set by INJECT_HINTS.sh.
        """
        entry = self.read_step_entry(step_id)
        return entry.get("human_env_metadata", {})

    def get_next_step_requirements(self, step_id: str) -> Dict[str, Any]:
        """
        Return the input requirements that the *next* step has registered
        for the output of step_id.

        Stub for v1.2.9 Validation Agent integration. Returns an empty dict
        until the Validation Agent populates next_step_requirements fields.
        The Validation Agent will call write_step_entry with an additional
        "next_step_requirements" key after decomposition.
        """
        entry = self.read_step_entry(step_id)
        return entry.get("next_step_requirements", {})

    def get_all_steps(self) -> Dict[str, Dict[str, Any]]:
        """Return all step entries from the manifest."""
        manifest = self._load()
        return manifest.get("steps", {})

    def get_project_id(self) -> str:
        """Return the project_id stored in the manifest header."""
        manifest = self._load()
        return manifest.get("project_id", "unknown")

    def is_step_complete(self, step_id: str) -> bool:
        """Return True if the step has a 'completed' or 'completed_validated' status."""
        entry = self.read_step_entry(step_id)
        return entry.get("status", "") in {
            "completed",
            "completed_validated",
            "completed_unvalidated",
        }

    # =========================================================================
    # PROTECTED PATH HELPERS (for cleanup scripts)
    # =========================================================================

    def is_path_protected(self, path: str | Path) -> bool:
        """
        Return True if the given path (absolute or relative to project_root)
        is protected and must not be deleted.

        Checks both the explicit path list from get_all_protected_paths() and
        the PROTECTED_DIRECTORIES whitelist (any path inside a protected
        directory is protected).
        """
        try:
            rel = Path(path)
            if rel.is_absolute():
                rel = rel.relative_to(self.project_root)
            rel_str = str(rel)
        except ValueError:
            # Path is outside project_root — not our concern
            return False

        # Check directory-level protection
        for protected_dir in PROTECTED_DIRECTORIES:
            if rel_str.startswith(protected_dir + "/") or rel_str == protected_dir:
                return True

        # Check explicit protected paths
        protected = self.get_all_protected_paths()
        return rel_str in protected

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _ensure_reports_dir(self) -> None:
        """Create reports/ if it doesn't exist."""
        reports_dir = self.project_root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

    def _load(self) -> Dict[str, Any]:
        """
        Load the manifest from disk.

        Returns a fresh skeleton if the file does not exist yet. This means
        the first call to any write method creates the file automatically.
        """
        if not self.manifest_path.exists():
            return self._skeleton()

        try:
            with open(self.manifest_path, "r") as f:
                data = json.load(f)
            # Ensure required top-level keys are present (forward compat)
            if "steps" not in data:
                data["steps"] = {}
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.error(
                f"Failed to load manifest at {self.manifest_path}: {e}. "
                f"Returning fresh skeleton."
            )
            return self._skeleton()

    def _save(self, manifest: Dict[str, Any]) -> None:
        """
        Write the manifest to disk atomically.

        Writes to a .tmp file then renames to avoid partial writes on
        crash or SLURM cancellation mid-write.
        """
        manifest["last_updated"] = datetime.now().isoformat()
        tmp_path = self.manifest_path.with_suffix(".json.tmp")

        try:
            with open(tmp_path, "w") as f:
                json.dump(manifest, f, indent=2)
            tmp_path.replace(self.manifest_path)
        except OSError as e:
            logger.error(f"Failed to write manifest: {e}")
            raise

    def _skeleton(self) -> Dict[str, Any]:
        """Return a fresh manifest skeleton with metadata populated from the project."""
        project_id = self._read_project_id()
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "project_id": project_id,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "steps": {},
        }

    def _read_project_id(self) -> str:
        """
        Read the project name from project.yaml.

        Returns "unknown" if project.yaml is absent or malformed.
        """
        project_yaml = self.project_root / "project.yaml"
        if not project_yaml.exists():
            return "unknown"
        try:
            import yaml  # PyYAML — already a pipeline dependency
            with open(project_yaml, "r") as f:
                data = yaml.safe_load(f)
            return data.get("project", {}).get("name", "unknown")
        except Exception:
            return "unknown"

    def _to_relative(self, path: Optional[str | Path]) -> Optional[str]:
        """
        Convert an absolute path to a path relative to project_root.

        Returns the original string unchanged if it is already relative,
        None, or cannot be made relative (path outside project_root).
        """
        if path is None:
            return None
        p = Path(path)
        if not p.is_absolute():
            return str(p)
        try:
            return str(p.relative_to(self.project_root))
        except ValueError:
            logger.warning(
                f"Path '{path}' is outside project_root '{self.project_root}' "
                f"— storing as-is"
            )
            return str(path)

    def _build_output_entry(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize and fill defaults for a single output entry.

        Input dict requires at minimum {"path": "..."}.
        Optional keys: type, description, protected, size_bytes.
        """
        path = self._to_relative(output.get("path", ""))

        # Auto-detect file type from extension if not provided
        file_type = output.get("type")
        if not file_type and path:
            suffix = Path(path).suffix.lstrip(".")
            # Handle compound extensions like .fastq.gz
            if path.endswith(".fastq.gz"):
                suffix = "fastq.gz"
            elif path.endswith(".vcf.gz"):
                suffix = "vcf.gz"
            file_type = suffix or "unknown"

        # Resolve file size from disk if not provided
        size_bytes = output.get("size_bytes")
        if size_bytes is None:
            abs_path = self.project_root / path if path else None
            if abs_path and abs_path.exists():
                try:
                    size_bytes = abs_path.stat().st_size
                except OSError:
                    size_bytes = None

        return {
            "path": path,
            "type": file_type,
            "description": output.get("description", ""),
            "protected": True,  # All registered outputs are always protected
            "size_bytes": size_bytes,
            "validation_status": output.get(
                "validation_status", VALIDATION_STATUS_NOT_VALIDATED
            ),
        }

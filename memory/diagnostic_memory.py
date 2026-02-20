"""
DiagnosticMemory — Global solution knowledge base for AGI Pipeline v1.2.0

A persistent, semantic memory layer for the Diagnostic Agent that maps
error patterns to validated solutions.  Unlike ReflexionMemory (which is
per-task and prevents retry loops), DiagnosticMemory is **global** — a
fix discovered while debugging step_01 is instantly available to step_07,
and even to future pipeline runs on different projects.

Over time this accumulates institutional knowledge about the HPC
environment's quirks:
  - "celltypist is pip-only, not on conda-forge"
  - "scvi-tools requires jax<0.5 on V100 nodes"
  - "conda-forge scanpy 1.10 conflicts with numpy 2.0"
  - "samtools must be installed via bioconda, not pip"

Architecture:
  - Backend: Mem0 + embedded Qdrant (same infrastructure as ReflexionMemory)
  - Collection: ``agi_diagnostic_solutions`` (separate from reflexion memory)
  - Embedder: nomic-embed-text via Ollama (768 dimensions)
  - Namespaces: solutions are stored under user_id = "agi_diagnostics"
  - Persistence: survives pipeline restarts and works across projects

Usage:
    from memory.diagnostic_memory import DiagnosticMemory

    dm = DiagnosticMemory()

    # Before investigating an error, check if we already know the fix
    known = dm.find_solution(
        error_pattern="ModuleNotFoundError: No module named 'celltypist'",
        error_type="missing_package",
    )
    if known and known[0]["confidence"] > 0.85:
        # Apply known fix directly — skip investigation
        fix = known[0]
        ...

    # After a successful fix, store it for future use
    dm.store_solution(
        error_pattern="ModuleNotFoundError: No module named 'celltypist'",
        solution="pip install celltypist (not available on conda-forge/bioconda)",
        error_type="missing_package",
        fix_target="env",
        env_context={"cluster": "arc_compute1", "python": "3.10"},
    )
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SolutionEntry:
    """
    A validated solution record for storage in diagnostic memory.

    Attributes:
        error_pattern: The error message or pattern this solution addresses.
        solution: Human-readable description of the fix.
        error_type: Classified error type (matches memory.FailureType values).
        fix_target: Which artifact the fix modifies — "env", "script",
                    "sbatch", "config", or "system".
        fix_actions: Machine-readable list of actions taken, e.g.
                     ``[{"action": "pip_install", "package": "celltypist"}]``.
        env_context: Environment details when the fix was validated
                     (cluster, python version, OS, etc.).
        success_count: How many times this solution has been successfully reused.
        last_used: ISO timestamp of the most recent successful application.
        task_ids: List of task IDs where this solution was applied.
        created_at: ISO timestamp of initial creation.
        confidence: Computed confidence score (0.0–1.0), based on success_count
                    and recency.
    """
    error_pattern: str
    solution: str
    error_type: str = "unknown"
    fix_target: str = "unknown"
    fix_actions: List[Dict[str, Any]] = field(default_factory=list)
    env_context: Dict[str, Any] = field(default_factory=dict)
    success_count: int = 1
    last_used: str = field(default_factory=lambda: datetime.now().isoformat())
    task_ids: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    confidence: float = 0.0

    def to_memory_text(self) -> str:
        """Serialize to a text block for Mem0 storage."""
        parts = [
            f"Error Type: {self.error_type}",
            f"Error Pattern: {self.error_pattern}",
            f"Solution: {self.solution}",
            f"Fix Target: {self.fix_target}",
            f"Success Count: {self.success_count}",
        ]
        if self.fix_actions:
            parts.append(f"Fix Actions: {json.dumps(self.fix_actions)}")
        if self.env_context:
            ctx_parts = [f"{k}={v}" for k, v in self.env_context.items()]
            parts.append(f"Environment: {', '.join(ctx_parts)}")
        return "\n".join(parts)

    def to_metadata(self) -> Dict[str, Any]:
        """Produce the metadata dict stored alongside the Mem0 memory."""
        return {
            "error_type": self.error_type,
            "fix_target": self.fix_target,
            "fix_actions": json.dumps(self.fix_actions),
            "env_context": json.dumps(self.env_context),
            "success_count": self.success_count,
            "last_used": self.last_used,
            "task_ids": json.dumps(self.task_ids),
            "created_at": self.created_at,
            "source": "diagnostic_agent",
        }


# =============================================================================
# DIAGNOSTIC MEMORY
# =============================================================================

# Separate Qdrant collection name — keeps diagnostic solutions isolated from
# the reflexion retry-tracking data.
DIAGNOSTIC_COLLECTION = "agi_diagnostic_solutions"

# Mem0 user namespace — all diagnostic solutions stored under this user.
DIAGNOSTIC_USER = "agi_diagnostics"

# Default similarity threshold above which a known solution is considered a
# confident match.  Overridden by AGI_SOLUTION_MEMORY_THRESHOLD env var or
# constructor argument.
DEFAULT_SIMILARITY_THRESHOLD = 0.85


class DiagnosticMemory:
    """
    Global solution knowledge base for the Diagnostic Agent.

    Backed by Mem0 + embedded Qdrant with a **separate collection** from
    ReflexionMemory.  Solutions are not scoped to a single task — they are
    shared across all tasks, pipeline runs, and projects so that knowledge
    accumulates over time.

    Key operations:
      - ``find_solution()``  — semantic search for known fixes
      - ``store_solution()`` — record a validated fix (or increment its
        success count if a near-duplicate already exists)
      - ``get_stats()``      — summary of stored knowledge

    Thread Safety:
        Mem0/Qdrant handle concurrent reads.  Writes are append-only so
        concurrent writers are unlikely to corrupt data, but there is no
        built-in locking.  In the AGI pipeline, only one diagnostic agent
        instance writes at a time per subtask, so this is fine.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict] = None,
        similarity_threshold: float = None,
    ):
        """
        Initialize DiagnosticMemory.

        Uses the same Mem0 config loading path as ReflexionMemory but
        overrides the collection name to ``agi_diagnostic_solutions``.

        Args:
            config_path: Path to mem0_config.yaml (searches standard
                         locations if None).
            config: Direct Mem0 config dict (overrides config_path).
            similarity_threshold: Minimum similarity score to consider
                                  a stored solution a confident match.
        """
        self.similarity_threshold = (
            similarity_threshold
            or _env_float(
                "AGI_SOLUTION_MEMORY_THRESHOLD",
                DEFAULT_SIMILARITY_THRESHOLD,
            )
        )

        # --- Load Mem0 config, then override collection name ----------------
        try:
            from mem0 import Memory
        except ImportError:
            raise ImportError(
                "mem0ai package not installed. "
                "Install with: pip install mem0ai"
            )

        if config is not None:
            mem0_config = _deep_copy_dict(config)
        else:
            from .config import get_mem0_config
            config_data = get_mem0_config(config_path)
            mem0_config = _deep_copy_dict(config_data["config"])

        # Point at the diagnostic-specific collection
        vs_cfg = mem0_config.get("vector_store", {}).get("config", {})
        vs_cfg["collection_name"] = DIAGNOSTIC_COLLECTION
        if "path" in vs_cfg:
            vs_cfg["path"] = str(Path(vs_cfg["path"]).parent / "qdrant_diagnostic")
        mem0_config.setdefault("vector_store", {})["config"] = vs_cfg

        logger.info(
            f"Initializing DiagnosticMemory "
            f"(collection={DIAGNOSTIC_COLLECTION}, "
            f"threshold={self.similarity_threshold})"
        )
        self.memory = Memory.from_config(mem0_config)
        logger.info("DiagnosticMemory initialized successfully")

    # =========================================================================
    # SOLUTION LOOKUP
    # =========================================================================

    def find_solution(
        self,
        error_pattern: str,
        error_type: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search for known solutions matching an error pattern.

        This is the **first thing** the diagnostic agent calls when it
        receives an error.  If a high-confidence match is found, the
        agent can skip its investigation protocol entirely and apply the
        known fix.

        Args:
            error_pattern: The error message or description to match.
            error_type: Optional filter (e.g. "missing_package").
            limit: Maximum number of results to return.

        Returns:
            List of solution dicts sorted by relevance, each containing:
              - solution: The fix description
              - error_pattern: The original error it was stored against
              - error_type: Classified error type
              - fix_target: Which file/artifact the fix targets
              - fix_actions: Machine-readable action list
              - confidence: Computed confidence (0.0–1.0)
              - success_count: Number of successful applications
              - similarity: Semantic similarity score from Qdrant
              - memory_id: Mem0 memory ID (for updates)
        """
        query = f"Error: {error_pattern}"
        if error_type:
            query += f"\nError Type: {error_type}"

        try:
            results = self.memory.search(
                query,
                user_id=DIAGNOSTIC_USER,
                limit=limit,
            )
        except Exception as e:
            logger.warning(f"DiagnosticMemory search failed: {e}")
            return []

        solutions = []
        for result in results.get("results", []):
            meta = result.get("metadata", {})
            similarity = result.get("score", 0.0)
            success_count = _safe_int(meta.get("success_count", 1))

            # Compute confidence from similarity and historical success
            confidence = _compute_confidence(similarity, success_count)

            # Optionally filter by error_type
            if error_type and meta.get("error_type"):
                if meta["error_type"] != error_type:
                    # Reduce confidence for type mismatch but don't exclude —
                    # the solution may still be relevant.
                    confidence *= 0.7

            # Parse fix_actions back from JSON
            fix_actions = _safe_json_loads(meta.get("fix_actions", "[]"), [])
            task_ids = _safe_json_loads(meta.get("task_ids", "[]"), [])
            env_context = _safe_json_loads(meta.get("env_context", "{}"), {})

            solutions.append({
                "solution": _extract_field(result.get("memory", ""), "Solution"),
                "error_pattern": _extract_field(
                    result.get("memory", ""), "Error Pattern"
                ),
                "error_type": meta.get("error_type", "unknown"),
                "fix_target": meta.get("fix_target", "unknown"),
                "fix_actions": fix_actions,
                "env_context": env_context,
                "confidence": round(confidence, 3),
                "success_count": success_count,
                "similarity": round(similarity, 3),
                "task_ids": task_ids,
                "memory_id": result.get("id"),
                "last_used": meta.get("last_used", ""),
                "created_at": meta.get("created_at", ""),
            })

        # Sort by confidence descending
        solutions.sort(key=lambda s: s["confidence"], reverse=True)

        if solutions:
            best = solutions[0]
            logger.info(
                f"DiagnosticMemory: best match confidence={best['confidence']}, "
                f"success_count={best['success_count']}, "
                f"error_type={best['error_type']}"
            )

        return solutions

    def has_confident_solution(
        self,
        error_pattern: str,
        error_type: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Convenience method: return the best solution if it exceeds the
        confidence threshold, otherwise None.

        Args:
            error_pattern: Error to look up.
            error_type: Optional error type filter.

        Returns:
            Best solution dict if confidence >= threshold, else None.
        """
        solutions = self.find_solution(
            error_pattern, error_type=error_type, limit=1
        )
        if solutions and solutions[0]["confidence"] >= self.similarity_threshold:
            return solutions[0]
        return None

    # =========================================================================
    # SOLUTION STORAGE
    # =========================================================================

    def store_solution(
        self,
        error_pattern: str,
        solution: str,
        error_type: str = "unknown",
        fix_target: str = "unknown",
        fix_actions: Optional[List[Dict[str, Any]]] = None,
        env_context: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Store a validated solution, or increment the success count if a
        near-duplicate already exists.

        The deduplication logic:
          1. Search for existing solutions matching the error pattern.
          2. If a match with similarity >= threshold exists, update its
             ``success_count``, ``last_used``, and ``task_ids`` metadata.
          3. Otherwise, create a new solution entry.

        This means the knowledge base naturally converges: the first time
        a fix is discovered it gets stored with success_count=1.  Each
        subsequent time the same fix works, the count increases and the
        solution becomes higher-confidence for future lookups.

        Args:
            error_pattern: The error message this solution addresses.
            solution: Human-readable description of the fix.
            error_type: Classified error type string.
            fix_target: "env", "script", "sbatch", "config", or "system".
            fix_actions: Machine-readable action list.
            env_context: Environment details (cluster, python version, etc.).
            task_id: Task where this fix was validated.

        Returns:
            Dict with ``stored`` (bool), ``memory_id``, ``is_update`` (bool),
            and ``success_count``.
        """
        entry = SolutionEntry(
            error_pattern=error_pattern,
            solution=solution,
            error_type=error_type,
            fix_target=fix_target,
            fix_actions=fix_actions or [],
            env_context=env_context or {},
            task_ids=[task_id] if task_id else [],
        )

        # --- Check for existing near-duplicate --------------------------------
        existing = self.find_solution(
            error_pattern, error_type=error_type, limit=1
        )

        if existing and existing[0]["similarity"] >= self.similarity_threshold:
            return self._update_existing_solution(existing[0], entry)

        # --- Store new solution -----------------------------------------------
        return self._store_new_solution(entry)

    def _store_new_solution(
        self, entry: SolutionEntry
    ) -> Dict[str, Any]:
        """Store a brand-new solution entry in Mem0."""
        try:
            result = self.memory.add(
                entry.to_memory_text(),
                user_id=DIAGNOSTIC_USER,
                metadata=entry.to_metadata(),
            )

            memory_id = None
            if isinstance(result, dict):
                results_list = result.get("results", [])
                if results_list:
                    memory_id = results_list[0].get("id")

            logger.info(
                f"Stored new diagnostic solution: "
                f"error_type={entry.error_type}, "
                f"fix_target={entry.fix_target}, "
                f"memory_id={memory_id}"
            )

            return {
                "stored": True,
                "memory_id": memory_id,
                "is_update": False,
                "success_count": 1,
            }
        except Exception as e:
            logger.error(f"Failed to store diagnostic solution: {e}")
            return {
                "stored": False,
                "error": str(e),
                "is_update": False,
                "success_count": 0,
            }

    def _update_existing_solution(
        self, existing: Dict[str, Any], new_entry: SolutionEntry
    ) -> Dict[str, Any]:
        """
        Increment success_count and update metadata on an existing solution.

        Mem0 does not natively support metadata-only updates, so we delete
        the old entry and re-add with updated counts.  This is safe because
        the semantic content is essentially identical.
        """
        memory_id = existing.get("memory_id")
        old_count = existing.get("success_count", 1)
        new_count = old_count + 1

        # Merge task IDs
        old_task_ids = existing.get("task_ids", [])
        new_task_ids = list(set(old_task_ids + new_entry.task_ids))

        # Merge env_context (keep all seen contexts)
        merged_context = existing.get("env_context", {})
        merged_context.update(new_entry.env_context)

        # Build updated entry
        updated = SolutionEntry(
            error_pattern=new_entry.error_pattern,
            solution=existing.get("solution") or new_entry.solution,
            error_type=new_entry.error_type,
            fix_target=existing.get("fix_target") or new_entry.fix_target,
            fix_actions=new_entry.fix_actions or existing.get("fix_actions", []),
            env_context=merged_context,
            success_count=new_count,
            last_used=datetime.now().isoformat(),
            task_ids=new_task_ids,
            created_at=existing.get("created_at", datetime.now().isoformat()),
        )

        try:
            # Delete old entry
            if memory_id:
                self.memory.delete(memory_id)

            # Re-add with updated metadata
            result = self.memory.add(
                updated.to_memory_text(),
                user_id=DIAGNOSTIC_USER,
                metadata=updated.to_metadata(),
            )

            new_memory_id = None
            if isinstance(result, dict):
                results_list = result.get("results", [])
                if results_list:
                    new_memory_id = results_list[0].get("id")

            logger.info(
                f"Updated diagnostic solution: "
                f"success_count {old_count} → {new_count}, "
                f"memory_id={new_memory_id}"
            )

            return {
                "stored": True,
                "memory_id": new_memory_id,
                "is_update": True,
                "success_count": new_count,
            }
        except Exception as e:
            logger.error(f"Failed to update diagnostic solution: {e}")
            return {
                "stored": False,
                "error": str(e),
                "is_update": True,
                "success_count": old_count,
            }

    # =========================================================================
    # BULK OPERATIONS
    # =========================================================================

    def store_known_solutions(
        self, solutions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Seed the memory with a batch of known solutions.

        Useful for bootstrapping the knowledge base with solutions you
        already know from experience (e.g. "popv is pip-only").

        Args:
            solutions: List of dicts, each with at minimum ``error_pattern``
                       and ``solution`` keys.

        Returns:
            Dict with ``stored_count`` and ``failed_count``.
        """
        stored = 0
        failed = 0

        for sol in solutions:
            result = self.store_solution(
                error_pattern=sol.get("error_pattern", ""),
                solution=sol.get("solution", ""),
                error_type=sol.get("error_type", "unknown"),
                fix_target=sol.get("fix_target", "unknown"),
                fix_actions=sol.get("fix_actions"),
                env_context=sol.get("env_context"),
                task_id=sol.get("task_id"),
            )
            if result.get("stored"):
                stored += 1
            else:
                failed += 1

        logger.info(
            f"Bulk store: {stored} stored, {failed} failed "
            f"out of {len(solutions)}"
        )
        return {"stored_count": stored, "failed_count": failed}

    # =========================================================================
    # STATISTICS & MAINTENANCE
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Summary statistics of the diagnostic knowledge base.

        Returns:
            Dict with total_solutions, solutions_by_type,
            solutions_by_target, avg_success_count, and
            most_reused (top 5 by success_count).
        """
        try:
            all_memories = self.memory.get_all(user_id=DIAGNOSTIC_USER)
        except Exception as e:
            logger.warning(f"Failed to get diagnostic stats: {e}")
            return {"error": str(e)}

        results = all_memories.get("results", [])

        by_type: Dict[str, int] = {}
        by_target: Dict[str, int] = {}
        success_counts: List[int] = []
        top_solutions: List[Dict[str, Any]] = []

        for r in results:
            meta = r.get("metadata", {})
            etype = meta.get("error_type", "unknown")
            ftarget = meta.get("fix_target", "unknown")
            count = _safe_int(meta.get("success_count", 1))

            by_type[etype] = by_type.get(etype, 0) + 1
            by_target[ftarget] = by_target.get(ftarget, 0) + 1
            success_counts.append(count)

            top_solutions.append({
                "solution": _extract_field(r.get("memory", ""), "Solution"),
                "error_type": etype,
                "success_count": count,
                "last_used": meta.get("last_used", ""),
            })

        # Sort top solutions by success_count
        top_solutions.sort(key=lambda s: s["success_count"], reverse=True)

        return {
            "total_solutions": len(results),
            "solutions_by_type": by_type,
            "solutions_by_target": by_target,
            "avg_success_count": (
                round(sum(success_counts) / len(success_counts), 1)
                if success_counts else 0
            ),
            "most_reused": top_solutions[:5],
        }

    def prune_stale_solutions(
        self, max_age_days: int = 90
    ) -> Dict[str, Any]:
        """
        Remove solutions that haven't been used recently.

        Solutions with high success_count are kept even if stale (they
        represent well-established knowledge).

        Args:
            max_age_days: Remove solutions not used within this many days,
                          unless success_count >= 3.

        Returns:
            Dict with pruned_count and kept_count.
        """
        try:
            all_memories = self.memory.get_all(user_id=DIAGNOSTIC_USER)
        except Exception as e:
            logger.warning(f"Failed to get memories for pruning: {e}")
            return {"error": str(e)}

        results = all_memories.get("results", [])
        now = datetime.now()
        pruned = 0
        kept = 0

        for r in results:
            meta = r.get("metadata", {})
            memory_id = r.get("id")
            success_count = _safe_int(meta.get("success_count", 1))

            # Keep high-success solutions regardless of age
            if success_count >= 3:
                kept += 1
                continue

            # Check age
            last_used_str = meta.get("last_used") or meta.get("created_at", "")
            if last_used_str:
                try:
                    last_used = datetime.fromisoformat(last_used_str)
                    age_days = (now - last_used).days
                    if age_days <= max_age_days:
                        kept += 1
                        continue
                except (ValueError, TypeError):
                    pass

            # Prune this entry
            if memory_id:
                try:
                    self.memory.delete(memory_id)
                    pruned += 1
                except Exception as e:
                    logger.warning(f"Failed to prune {memory_id}: {e}")
                    kept += 1
            else:
                kept += 1

        logger.info(f"Pruned {pruned} stale solutions, kept {kept}")
        return {"pruned_count": pruned, "kept_count": kept}

    def clear_all(self, confirm: bool = False) -> bool:
        """
        Delete ALL diagnostic solutions.  DESTRUCTIVE.

        Args:
            confirm: Must be True to proceed.

        Returns:
            True if reset succeeded.
        """
        if not confirm:
            logger.warning("Clear not confirmed. Pass confirm=True.")
            return False

        try:
            self.memory.reset()
            logger.warning("ALL DIAGNOSTIC MEMORY CLEARED")
            return True
        except Exception as e:
            logger.error(f"Clear failed: {e}")
            return False


# =============================================================================
# BOOTSTRAP DATA
# =============================================================================

# Pre-seed solutions for common bioinformatics environment issues.
# Call ``dm.store_known_solutions(BOOTSTRAP_SOLUTIONS)`` on first run
# to give the diagnostic agent a head-start.

BOOTSTRAP_SOLUTIONS: List[Dict[str, Any]] = [
    {
        "error_pattern": "ModuleNotFoundError: No module named 'celltypist'",
        "solution": "pip install celltypist — not available on conda-forge or bioconda",
        "error_type": "missing_package",
        "fix_target": "env",
        "fix_actions": [{"action": "pip_install", "package": "celltypist"}],
    },
    {
        "error_pattern": "ModuleNotFoundError: No module named 'popv'",
        "solution": "pip install popv — not available on conda-forge or bioconda",
        "error_type": "missing_package",
        "fix_target": "env",
        "fix_actions": [{"action": "pip_install", "package": "popv"}],
    },
    {
        "error_pattern": "ModuleNotFoundError: No module named 'scvi'",
        "solution": "pip install scvi-tools — the import name is scvi but the package name is scvi-tools, pip only",
        "error_type": "missing_package",
        "fix_target": "env",
        "fix_actions": [{"action": "pip_install", "package": "scvi-tools"}],
    },
    {
        "error_pattern": "ModuleNotFoundError: No module named 'decoupler'",
        "solution": "pip install decoupler — not available on conda-forge or bioconda",
        "error_type": "missing_package",
        "fix_target": "env",
        "fix_actions": [{"action": "pip_install", "package": "decoupler"}],
    },
    {
        "error_pattern": "sbatch: error: --mem specification not allowed on GPU partition",
        "solution": "Remove --mem from sbatch directives when targeting GPU partitions on ARC. GPU nodes do not accept memory requests.",
        "error_type": "sbatch_config_error",
        "fix_target": "sbatch",
        "fix_actions": [{"action": "remove_sbatch_directive", "directive": "--mem"}],
    },
    {
        "error_pattern": "CUDA out of memory",
        "solution": "Reduce batch size or add torch.cuda.empty_cache() between major operations. If persistent, route to a node with more VRAM (a100 vs v100).",
        "error_type": "gpu_error",
        "fix_target": "script",
    },
    {
        "error_pattern": "Disk quota exceeded",
        "solution": "Run 'conda clean --all --yes' then remove stale agi_* environments. Check ~/.conda/pkgs and pip cache.",
        "error_type": "disk_quota_error",
        "fix_target": "system",
        "fix_actions": [
            {"action": "conda_clean"},
            {"action": "remove_stale_envs"},
        ],
    },
    {
        "error_pattern": "samtools: command not found",
        "solution": "Install samtools via bioconda: add samtools to conda dependencies in env YAML.",
        "error_type": "binary_not_found",
        "fix_target": "env",
        "fix_actions": [{"action": "conda_install", "package": "samtools", "channel": "bioconda"}],
    },
    {
        "error_pattern": "bedtools: command not found",
        "solution": "Install bedtools via bioconda: add bedtools to conda dependencies in env YAML.",
        "error_type": "binary_not_found",
        "fix_target": "env",
        "fix_actions": [{"action": "conda_install", "package": "bedtools", "channel": "bioconda"}],
    },
    {
        "error_pattern": "numpy.core.multiarray failed to import",
        "solution": "Version conflict between numpy 2.x and packages compiled against numpy 1.x. Pin numpy<2.0 in env YAML.",
        "error_type": "missing_package",
        "fix_target": "env",
        "fix_actions": [{"action": "version_pin", "package": "numpy", "pin": "<2.0"}],
    },
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _compute_confidence(similarity: float, success_count: int) -> float:
    """
    Compute a confidence score from semantic similarity and success count.

    Formula:
      confidence = similarity * (1.0 + log2(success_count) * 0.05)

    This means:
      - success_count=1 → confidence ≈ similarity
      - success_count=4 → confidence ≈ similarity * 1.10
      - success_count=16 → confidence ≈ similarity * 1.20

    The boost is modest — similarity is always the dominant factor — but
    well-proven solutions get a small edge over untested ones.

    Capped at 1.0.
    """
    import math

    if success_count <= 0:
        return similarity

    boost = 1.0 + math.log2(max(1, success_count)) * 0.05
    return min(1.0, similarity * boost)


def _extract_field(memory_text: str, field_name: str) -> str:
    """
    Extract a named field from Mem0's stored text.

    The stored text format is ``Field Name: value`` on separate lines.
    """
    for line in memory_text.splitlines():
        if line.startswith(f"{field_name}:"):
            return line[len(field_name) + 1:].strip()
    return memory_text  # Fallback: return entire text


def _safe_int(value: Any) -> int:
    """Safely convert a value to int."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def _safe_json_loads(value: str, default: Any) -> Any:
    """Safely parse a JSON string."""
    if not value or not isinstance(value, str):
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _env_float(var_name: str, default: float) -> float:
    """Read a float from an environment variable."""
    val = os.environ.get(var_name)
    if val is not None:
        try:
            return float(val)
        except ValueError:
            pass
    return default


def _deep_copy_dict(d: Dict) -> Dict:
    """
    Deep copy a dict without importing copy (keeps dependencies minimal).
    Uses JSON round-trip which handles the types present in Mem0 configs.
    """
    return json.loads(json.dumps(d))


# =============================================================================
# CONVENIENCE SINGLETON
# =============================================================================

_default_instance: Optional[DiagnosticMemory] = None


def get_diagnostic_memory() -> DiagnosticMemory:
    """Return a module-level DiagnosticMemory singleton."""
    global _default_instance
    if _default_instance is None:
        _default_instance = DiagnosticMemory()
    return _default_instance

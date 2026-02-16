"""
DiagnosticAgent — Specialist failure investigator for AGI Pipeline v1.2.0

A semi-autonomous agent that activates on job failure, investigates root
cause, and returns a structured FixPrescription telling the sub-agent
exactly what to change and where.

Design principles:
  - Gets its OWN 25K token budget (independent of sub-agent)
  - Checks DiagnosticMemory FIRST for known solutions before investigating
  - Can execute limited actions directly (package installs, diagnostic
    commands, conda cleanup) — low-risk, high-frequency operations
  - Returns code/config changes as prescriptions for the sub-agent to
    apply — high-risk operations stay under sub-agent control
  - Stores validated fixes in DiagnosticMemory for future reuse

Investigation protocol by error type:
  missing_package  → parse module name, try install, update YAML
  code_error       → read traceback + code, form hypothesis, run diagnostic
  data_structure   → inspect data object, dump structure, compare to script
  memory_error     → check data sizes, recommend mem increase or chunking
  disk_quota       → delegate to DiskManager for cleanup
  binary_not_found → search conda-forge for binary, install if found
  gpu_error        → check nvidia-smi, CUDA version, package GPU support
  sbatch_config    → parse SLURM error, suggest config fix
  syntax_error     → parse traceback, identify file + line
  permission_error → check directory permissions, suggest fix
  runtime_logic    → read traceback, isolate failing function
  network_error    → check connectivity, suggest retry or mirror

All LLM calls use invoke_resilient() with exponential backoff retry.
No artificial timeouts — the 3-day SLURM wall time is the only limit.
"""

import os
import re
import json
import subprocess
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FixPrescription:
    """
    Structured fix returned by the Diagnostic Agent to the sub-agent.

    The sub-agent reads this to decide what action to take:
      - env_updated   → diagnostic agent already handled it, just retry
      - edit_code     → apply the specified changes to the target file
      - change_config → modify the sbatch script
      - rebuild_env   → reset env_created, go back to Phase 2
      - add_package   → install package (already done), retry
      - escalate      → give up on this error type

    Attributes:
        target_file: Which artifact to modify — "script", "env", "sbatch",
                     or a specific path.
        fix_type: Category of fix applied or recommended.
        changes: List of specific changes (dicts with action-specific keys).
        confidence: 0.0–1.0, how confident the agent is in this fix.
        diagnostic_actions_taken: What the agent already executed directly.
        packages_installed: Packages the agent directly installed into the env.
        env_yaml_updated: Whether the YAML was modified by the agent.
        explanation: Human-readable summary of the diagnosis and fix.
        from_memory: True if this fix came from DiagnosticMemory (no investigation).
        error_type: The classified error type that was investigated.
    """
    target_file: str = "unknown"
    fix_type: str = "escalate"
    changes: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    diagnostic_actions_taken: List[str] = field(default_factory=list)
    packages_installed: List[str] = field(default_factory=list)
    env_yaml_updated: bool = False
    explanation: str = ""
    from_memory: bool = False
    error_type: str = "unknown"


# =============================================================================
# DIAGNOSTIC AGENT
# =============================================================================

class DiagnosticAgent:
    """
    Specialist agent that investigates job failures and produces targeted
    fix prescriptions.

    Operates with its own independent token budget. The sub-agent
    instantiates a new DiagnosticAgent per error investigation, so each
    invocation gets a fresh context window.

    Option C architecture: can execute limited actions directly (package
    installs, diagnostic commands) but returns code/config changes as
    prescriptions for the sub-agent to apply.
    """

    # Maximum bash commands per investigation (prevent runaway diagnostics)
    MAX_DIAGNOSTIC_COMMANDS = 10

    def __init__(
        self,
        agent_id: str,
        step_id: str,
        project_root: str,
        conda_tools=None,
        slurm_tools=None,
        ollama_model: str = None,
        ollama_base_url: str = None,
        solution_memory=None,
        env_name: str = None,
        env_yaml_path: str = None,
    ):
        """
        Initialize the Diagnostic Agent.

        Args:
            agent_id: Unique identifier for this invocation.
            step_id: The step prefix for finding files (e.g. "step_01").
            project_root: Project directory path.
            conda_tools: CondaTools instance for package management.
            slurm_tools: SlurmTools instance for job context.
            ollama_model: LLM model name (resolved via model_config if None).
            ollama_base_url: Ollama API URL.
            solution_memory: DiagnosticMemory instance for known solutions.
            env_name: Conda environment name for this step.
            env_yaml_path: Path to the step's environment YAML.
        """
        self.agent_id = agent_id
        self.step_id = step_id
        self.project_root = Path(project_root)
        self.conda_tools = conda_tools
        self.slurm_tools = slurm_tools
        self.solution_memory = solution_memory
        self.env_name = env_name
        self.env_yaml_path = env_yaml_path

        # Track actions taken during this investigation
        self._actions_taken: List[str] = []
        self._packages_installed: List[str] = []
        self._commands_run: int = 0

        # Initialize LLM
        try:
            from langchain_ollama import OllamaLLM
            from utils.model_config import resolve_model, resolve_base_url

            resolved_model = resolve_model(ollama_model)
            self.ollama_base_url = resolve_base_url(ollama_base_url)
            self.llm = OllamaLLM(
                model=resolved_model, base_url=self.ollama_base_url
            )
        except ImportError:
            logger.warning(
                "LangChain/Ollama not available — LLM diagnostics disabled"
            )
            self.llm = None
            self.ollama_base_url = ollama_base_url or "http://127.0.0.1:11434"

        # Discover step files
        self._step_files = self._read_step_files()

        logger.info(
            f"DiagnosticAgent initialized: agent_id={agent_id}, "
            f"step_id={step_id}, env={env_name}, "
            f"files_found={list(self._step_files.keys())}"
        )

    # =========================================================================
    # CORE INVESTIGATION
    # =========================================================================

    def investigate(
        self,
        error_classification: Dict[str, Any],
        logs: Dict[str, str],
        subtask_description: str = "",
    ) -> FixPrescription:
        """
        Main entry point. Investigate a job failure and produce a fix.

        Flow:
          1. Check DiagnosticMemory for a known solution
          2. If no confident match, run error-type-specific investigation
          3. Store the fix in memory if it succeeds

        Args:
            error_classification: Dict with at minimum "error_type" and
                "error_message" keys from _analyze_job_result().
            logs: Dict with "stdout" and "stderr" from the failed job.
            subtask_description: The subtask text for context.

        Returns:
            FixPrescription with the diagnosis and recommended/applied fix.
        """
        error_type = error_classification.get("error_type", "unknown")
        error_message = error_classification.get("error_message", "")
        stderr = logs.get("stderr", "")
        stdout = logs.get("stdout", "")

        logger.info(
            f"[{self.agent_id}] Investigating error_type={error_type}: "
            f"{error_message[:120]}"
        )

        # Step 1: Check solution memory
        memory_fix = self._check_solution_memory(error_message, error_type)
        if memory_fix is not None:
            return memory_fix

        # Step 2: Dispatch to error-type-specific investigation
        dispatch = {
            "missing_package": self._investigate_missing_package,
            "code_error": self._investigate_code_error,
            "data_structure_error": self._investigate_data_structure,
            "memory_error": self._investigate_memory_error,
            "disk_quota_error": self._investigate_disk_quota,
            "binary_not_found": self._investigate_binary_not_found,
            "gpu_error": self._investigate_gpu_error,
            "sbatch_config_error": self._investigate_sbatch_config,
            "syntax_error": self._investigate_syntax_error,
            "permission_error": self._investigate_permission_error,
            "runtime_logic_error": self._investigate_code_error,
            "network_error": self._investigate_network_error,
        }

        handler = dispatch.get(error_type, self._investigate_generic)

        try:
            prescription = handler(
                error_message=error_message,
                stderr=stderr,
                stdout=stdout,
                subtask_description=subtask_description,
                classification=error_classification,
            )
        except Exception as e:
            logger.error(f"[{self.agent_id}] Investigation failed: {e}")
            prescription = FixPrescription(
                fix_type="escalate",
                explanation=f"Investigation raised exception: {e}",
                error_type=error_type,
                diagnostic_actions_taken=self._actions_taken,
            )

        # Populate common fields
        prescription.error_type = error_type
        prescription.diagnostic_actions_taken = self._actions_taken
        prescription.packages_installed = self._packages_installed

        return prescription

    # =========================================================================
    # SOLUTION MEMORY
    # =========================================================================

    def _check_solution_memory(
        self, error_message: str, error_type: str
    ) -> Optional[FixPrescription]:
        """Check DiagnosticMemory for a known fix before investigating."""
        if self.solution_memory is None:
            return None

        try:
            known = self.solution_memory.has_confident_solution(
                error_pattern=error_message,
                error_type=error_type,
            )
        except Exception as e:
            logger.warning(f"Solution memory lookup failed: {e}")
            return None

        if known is None:
            return None

        logger.info(
            f"[{self.agent_id}] Found known solution "
            f"(confidence={known['confidence']}, "
            f"success_count={known['success_count']}): "
            f"{known['solution'][:100]}"
        )

        # Attempt to apply the known fix directly
        fix_actions = known.get("fix_actions", [])
        applied = self._apply_known_fix_actions(fix_actions, error_type)

        if applied:
            return FixPrescription(
                target_file=known.get("fix_target", "env"),
                fix_type="env_updated" if known.get("fix_target") == "env" else "edit_code",
                changes=fix_actions,
                confidence=known["confidence"],
                explanation=f"Applied known solution: {known['solution']}",
                from_memory=True,
                env_yaml_updated=any(
                    a.get("action") in ("pip_install", "conda_install", "version_pin")
                    for a in fix_actions
                ),
                error_type=error_type,
                diagnostic_actions_taken=self._actions_taken,
                packages_installed=self._packages_installed,
            )

        # Known fix couldn't be applied — fall through to investigation
        logger.info(
            f"[{self.agent_id}] Known fix actions could not be applied, "
            f"proceeding with investigation"
        )
        return None

    def _apply_known_fix_actions(
        self, fix_actions: List[Dict[str, Any]], error_type: str
    ) -> bool:
        """Try to execute machine-readable fix actions from memory.

        Only handles low-risk direct actions (package installs).
        Returns True if at least one action succeeded.
        """
        if not fix_actions or not self.conda_tools or not self.env_name:
            return False

        any_success = False
        for action in fix_actions:
            act = action.get("action", "")
            pkg = action.get("package", "")

            if act == "pip_install" and pkg:
                result = self._install_package_into_env(
                    pkg, method="pip"
                )
                if result:
                    any_success = True

            elif act == "conda_install" and pkg:
                channel = action.get("channel", "conda-forge")
                result = self._install_package_into_env(
                    pkg, method="conda", channel=channel
                )
                if result:
                    any_success = True

            elif act == "version_pin" and pkg:
                pin = action.get("pin", "")
                if pin and self.env_yaml_path:
                    # Remove existing, add pinned version
                    self.conda_tools.remove_package_from_yaml(
                        self.env_yaml_path, pkg
                    )
                    self.conda_tools.update_yaml_with_package(
                        self.env_yaml_path, f"{pkg}{pin}", section="conda"
                    )
                    self._actions_taken.append(
                        f"Pinned {pkg}{pin} in YAML"
                    )
                    # Rebuild env from updated YAML
                    any_success = True

            elif act in ("conda_clean", "remove_stale_envs"):
                self._run_disk_cleanup()
                any_success = True

        return any_success

    def _store_solution(
        self, error_message: str, prescription: FixPrescription
    ):
        """Store a successful fix in DiagnosticMemory for future reuse."""
        if self.solution_memory is None:
            return

        try:
            self.solution_memory.store_solution(
                error_pattern=error_message,
                solution=prescription.explanation,
                error_type=prescription.error_type,
                fix_target=prescription.target_file,
                fix_actions=prescription.changes,
                env_context={
                    "env_name": self.env_name or "",
                    "step_id": self.step_id,
                },
                task_id=self.step_id,
            )
        except Exception as e:
            logger.warning(f"Failed to store solution in memory: {e}")

    # =========================================================================
    # ERROR-TYPE-SPECIFIC INVESTIGATORS
    # =========================================================================

    def _investigate_missing_package(self, **kwargs) -> FixPrescription:
        """Investigate ModuleNotFoundError / ImportError.

        Protocol:
          1. Parse the module/package name from the error
          2. Try installing via conda, then pip
          3. Update the env YAML
          4. Verify the import works
        """
        error_message = kwargs["error_message"]
        stderr = kwargs["stderr"]

        # Parse module name from traceback
        module_name = self._parse_missing_module(error_message, stderr)
        if not module_name:
            return FixPrescription(
                fix_type="escalate",
                confidence=0.2,
                explanation="Could not parse module name from error",
            )

        # Map import name to package name (common mismatches)
        package_name = self._import_to_package_name(module_name)

        self._actions_taken.append(
            f"Parsed missing module: {module_name} → package: {package_name}"
        )

        # Try installing
        installed = self._install_package_into_env(package_name)

        if installed:
            # Verify it actually works now
            validation = self._validate_package(module_name)

            if validation:
                rx = FixPrescription(
                    target_file="env",
                    fix_type="env_updated",
                    changes=[{
                        "action": "pip_install" if "pip" in self._actions_taken[-1] else "conda_install",
                        "package": package_name,
                    }],
                    confidence=0.95,
                    env_yaml_updated=True,
                    explanation=(
                        f"Installed {package_name} and verified "
                        f"'import {module_name}' succeeds"
                    ),
                )
                self._store_solution(error_message, rx)
                return rx
            else:
                return FixPrescription(
                    target_file="env",
                    fix_type="env_updated",
                    changes=[{"action": "install_unverified", "package": package_name}],
                    confidence=0.6,
                    env_yaml_updated=True,
                    explanation=(
                        f"Installed {package_name} but import verification "
                        f"failed — may need different package name or version"
                    ),
                )

        return FixPrescription(
            fix_type="escalate",
            confidence=0.3,
            explanation=(
                f"Could not install package for module '{module_name}'. "
                f"Tried: {package_name}"
            ),
        )

    def _investigate_code_error(self, **kwargs) -> FixPrescription:
        """Investigate runtime/logic/type errors in the script.

        Protocol:
          1. Parse traceback → file, line, error
          2. Read the relevant script section
          3. Ask LLM to diagnose and propose a fix
          4. Optionally run a diagnostic script to inspect data
          5. Return edit_code prescription with specific line changes
        """
        error_message = kwargs["error_message"]
        stderr = kwargs["stderr"]
        subtask_description = kwargs.get("subtask_description", "")

        # Parse traceback
        tb = self._parse_traceback(stderr)
        if not tb["file"]:
            tb = self._parse_traceback(error_message)

        # Read script content
        script_content = self._read_script_content(tb.get("file"))
        if not script_content:
            # Try finding script by step_id
            for ext in ("py", "R", "sh", "pl"):
                key = f"{self.step_id}.{ext}"
                if key in self._step_files:
                    script_content = self._step_files[key].get("content", "")
                    break

        if not script_content:
            return FixPrescription(
                fix_type="escalate",
                confidence=0.2,
                explanation="Could not read script content for diagnosis",
            )

        # Ask LLM for diagnosis
        if self.llm is None:
            return FixPrescription(
                fix_type="escalate",
                confidence=0.3,
                explanation="LLM not available for code diagnosis",
            )

        diagnosis = self._llm_diagnose_code_error(
            script_content=script_content,
            error_message=error_message,
            stderr_tail=stderr[-3000:] if stderr else "",
            traceback_info=tb,
            subtask_description=subtask_description,
        )

        if diagnosis and diagnosis.get("fix"):
            rx = FixPrescription(
                target_file=tb.get("file") or "script",
                fix_type="edit_code",
                changes=[{
                    "action": "replace_script",
                    "fixed_content": diagnosis["fix"],
                    "diagnosis": diagnosis.get("diagnosis", ""),
                }],
                confidence=diagnosis.get("confidence", 0.6),
                explanation=diagnosis.get("diagnosis", "LLM-diagnosed code fix"),
            )
            return rx

        return FixPrescription(
            fix_type="escalate",
            confidence=0.3,
            explanation=f"LLM could not produce a fix. Traceback: {tb}",
        )

    def _investigate_data_structure(self, **kwargs) -> FixPrescription:
        """Investigate KeyError, shape mismatch, missing column errors.

        Runs a small diagnostic script to inspect the data object, then
        asks the LLM to compare against what the script expects.
        """
        error_message = kwargs["error_message"]
        stderr = kwargs["stderr"]
        subtask_description = kwargs.get("subtask_description", "")

        # Try to identify the data file from the error
        data_info = self._run_data_inspection(stderr)

        # Fall through to code error handler with extra data context
        kwargs["subtask_description"] = (
            f"{subtask_description}\n\n"
            f"DATA INSPECTION RESULTS:\n{data_info}"
        )
        return self._investigate_code_error(**kwargs)

    def _investigate_memory_error(self, **kwargs) -> FixPrescription:
        """Investigate MemoryError / OOM.

        Checks data file sizes and recommends either increasing --mem
        in sbatch or adding chunking logic to the script.
        """
        stderr = kwargs["stderr"]

        # Check data file sizes
        size_info = self._run_diagnostic_command(
            "find . -name '*.h5ad' -o -name '*.csv' -o -name '*.parquet' "
            "| head -20 | xargs ls -lh 2>/dev/null"
        )
        self._actions_taken.append(f"Checked data sizes: {size_info[:200]}")

        # Determine if this is a SLURM OOM or Python OOM
        is_slurm_oom = "oom-kill" in stderr.lower() or "OUT_OF_MEMORY" in stderr

        if is_slurm_oom:
            return FixPrescription(
                target_file="sbatch",
                fix_type="change_config",
                changes=[{
                    "action": "increase_memory",
                    "suggestion": "Double --mem value or remove --mem on GPU partitions",
                }],
                confidence=0.7,
                explanation=(
                    f"SLURM OOM kill detected. Data sizes: {size_info[:200]}. "
                    f"Recommend increasing memory allocation."
                ),
            )
        else:
            return FixPrescription(
                target_file="script",
                fix_type="edit_code",
                changes=[{
                    "action": "add_chunking",
                    "suggestion": "Process data in chunks or use backed mode for AnnData",
                }],
                confidence=0.5,
                explanation=(
                    f"Python MemoryError. Data sizes: {size_info[:200]}. "
                    f"Recommend chunked processing or backed mode."
                ),
            )

    def _investigate_disk_quota(self, **kwargs) -> FixPrescription:
        """Investigate disk quota exceeded errors.

        Direct action: run cleanup, then report space recovered.
        """
        space_before = self._run_diagnostic_command("df -h ~ | tail -1")
        self._run_disk_cleanup()
        space_after = self._run_diagnostic_command("df -h ~ | tail -1")

        self._actions_taken.append(
            f"Disk cleanup: before={space_before.strip()}, "
            f"after={space_after.strip()}"
        )

        rx = FixPrescription(
            target_file="system",
            fix_type="env_updated",
            changes=[
                {"action": "conda_clean"},
                {"action": "remove_stale_envs"},
            ],
            confidence=0.8,
            explanation=(
                f"Ran disk cleanup. Space before: {space_before.strip()}, "
                f"after: {space_after.strip()}. Retry should succeed."
            ),
        )
        self._store_solution(kwargs["error_message"], rx)
        return rx

    def _investigate_binary_not_found(self, **kwargs) -> FixPrescription:
        """Investigate command-not-found errors for CLI tools."""
        error_message = kwargs["error_message"]
        stderr = kwargs["stderr"]

        # Parse binary name
        binary = None
        patterns = [
            r"(\S+): command not found",
            r"(\S+): No such file or directory",
            r"FileNotFoundError.*'(\S+)'",
        ]
        for pat in patterns:
            m = re.search(pat, error_message + "\n" + stderr)
            if m:
                binary = m.group(1).strip("'\"")
                break

        if not binary:
            return FixPrescription(
                fix_type="escalate",
                confidence=0.2,
                explanation="Could not parse binary name from error",
            )

        # Try installing via conda (bioconda for bio tools)
        installed = self._install_package_into_env(
            binary, method="conda", channel="bioconda"
        )
        if not installed:
            # Try conda-forge
            installed = self._install_package_into_env(
                binary, method="conda", channel="conda-forge"
            )

        if installed:
            rx = FixPrescription(
                target_file="env",
                fix_type="env_updated",
                changes=[{"action": "conda_install", "package": binary}],
                confidence=0.85,
                env_yaml_updated=True,
                explanation=f"Installed missing binary '{binary}' into environment",
            )
            self._store_solution(error_message, rx)
            return rx

        return FixPrescription(
            fix_type="escalate",
            confidence=0.3,
            explanation=(
                f"Could not find '{binary}' on conda-forge or bioconda. "
                f"May need system-level installation."
            ),
        )

    def _investigate_gpu_error(self, **kwargs) -> FixPrescription:
        """Investigate CUDA / GPU errors."""
        stderr = kwargs["stderr"]

        # Check GPU availability
        gpu_info = self._run_diagnostic_command("nvidia-smi 2>&1 | head -20")
        cuda_info = self._run_diagnostic_command(
            "python -c \"import torch; print(torch.cuda.is_available(), "
            "torch.version.cuda)\" 2>&1"
        )

        self._actions_taken.append(f"GPU check: {gpu_info[:100]}")

        if "CUDA out of memory" in stderr:
            return FixPrescription(
                target_file="script",
                fix_type="edit_code",
                changes=[{
                    "action": "reduce_gpu_memory",
                    "suggestion": (
                        "Add torch.cuda.empty_cache(), reduce batch_size, "
                        "or use torch.cuda.amp for mixed precision"
                    ),
                }],
                confidence=0.6,
                explanation=f"CUDA OOM. GPU info: {gpu_info[:200]}",
            )

        if "no CUDA" in stderr.lower() or "not available" in cuda_info.lower():
            return FixPrescription(
                target_file="sbatch",
                fix_type="change_config",
                changes=[{
                    "action": "ensure_gpu_partition",
                    "suggestion": "Route to GPU partition with --gres=gpu:1",
                }],
                confidence=0.7,
                explanation=(
                    f"GPU not available on current node. "
                    f"CUDA check: {cuda_info.strip()}"
                ),
            )

        return FixPrescription(
            fix_type="escalate",
            confidence=0.4,
            explanation=f"GPU error. nvidia-smi: {gpu_info[:200]}, CUDA: {cuda_info[:200]}",
        )

    def _investigate_sbatch_config(self, **kwargs) -> FixPrescription:
        """Investigate SLURM submission/configuration errors."""
        error_message = kwargs["error_message"]
        stderr = kwargs["stderr"]
        combined = f"{error_message}\n{stderr}"

        # Common sbatch error patterns and fixes
        rules = [
            (r"Invalid partition", "change_config", "Fix partition name to match available partitions"),
            (r"Invalid account", "change_config", "Fix account name"),
            (r"invalid resource specification", "change_config", "Fix --gres or --mem format"),
            (r"--mem.*not allowed", "change_config", "Remove --mem on GPU partitions"),
            (r"Requested node configuration is not available", "change_config", "Reduce resource request"),
        ]

        for pattern, fix_type, suggestion in rules:
            if re.search(pattern, combined, re.IGNORECASE):
                rx = FixPrescription(
                    target_file="sbatch",
                    fix_type=fix_type,
                    changes=[{"action": "fix_sbatch", "suggestion": suggestion}],
                    confidence=0.8,
                    explanation=f"SLURM config error: {suggestion}",
                )
                self._store_solution(error_message, rx)
                return rx

        return FixPrescription(
            target_file="sbatch",
            fix_type="change_config",
            confidence=0.4,
            explanation=f"Unrecognized SLURM error: {combined[:300]}",
        )

    def _investigate_syntax_error(self, **kwargs) -> FixPrescription:
        """Investigate SyntaxError — typically a script generation issue."""
        # Syntax errors are fundamentally code errors, delegate
        return self._investigate_code_error(**kwargs)

    def _investigate_permission_error(self, **kwargs) -> FixPrescription:
        """Investigate PermissionError on file/directory access."""
        error_message = kwargs["error_message"]

        # Try to extract the path
        m = re.search(r"PermissionError.*'([^']+)'", error_message)
        path = m.group(1) if m else None

        if path:
            # Check permissions
            perms = self._run_diagnostic_command(f"ls -la {path} 2>&1")
            parent = self._run_diagnostic_command(
                f"ls -la $(dirname {path}) 2>&1 | head -5"
            )
            self._actions_taken.append(f"Checked permissions: {perms.strip()}")

            return FixPrescription(
                target_file="script",
                fix_type="edit_code",
                changes=[{
                    "action": "fix_path",
                    "original_path": path,
                    "suggestion": "Use PROJECT_ROOT-relative path or /work/$USER/ path",
                }],
                confidence=0.6,
                explanation=(
                    f"Permission denied on '{path}'. "
                    f"Permissions: {perms.strip()}. "
                    f"Parent dir: {parent.strip()}"
                ),
            )

        return FixPrescription(
            fix_type="escalate",
            confidence=0.3,
            explanation=f"Permission error but could not parse path: {error_message[:200]}",
        )

    def _investigate_network_error(self, **kwargs) -> FixPrescription:
        """Investigate download/network failures."""
        error_message = kwargs["error_message"]

        return FixPrescription(
            target_file="script",
            fix_type="edit_code",
            changes=[{
                "action": "add_retry_logic",
                "suggestion": (
                    "Add retry with exponential backoff for network operations, "
                    "or pre-download data files before the main script"
                ),
            }],
            confidence=0.5,
            explanation=f"Network error: {error_message[:200]}. Suggest adding retry logic.",
        )

    def _investigate_generic(self, **kwargs) -> FixPrescription:
        """Fallback investigation using LLM analysis."""
        error_message = kwargs["error_message"]
        stderr = kwargs["stderr"]

        if self.llm is None:
            return FixPrescription(
                fix_type="escalate",
                confidence=0.1,
                explanation=f"Unknown error type, no LLM available: {error_message[:200]}",
            )

        # Ask LLM for general diagnosis
        diagnosis = self._llm_diagnose_code_error(
            script_content=self._get_any_script_content(),
            error_message=error_message,
            stderr_tail=stderr[-3000:] if stderr else "",
            traceback_info=self._parse_traceback(stderr),
            subtask_description=kwargs.get("subtask_description", ""),
        )

        if diagnosis and diagnosis.get("fix"):
            return FixPrescription(
                target_file="script",
                fix_type="edit_code",
                changes=[{
                    "action": "replace_script",
                    "fixed_content": diagnosis["fix"],
                    "diagnosis": diagnosis.get("diagnosis", ""),
                }],
                confidence=diagnosis.get("confidence", 0.4),
                explanation=diagnosis.get("diagnosis", "LLM generic diagnosis"),
            )

        return FixPrescription(
            fix_type="escalate",
            confidence=0.1,
            explanation=f"Could not diagnose: {error_message[:300]}",
        )

    # =========================================================================
    # HELPER: PACKAGE INSTALLATION
    # =========================================================================

    def _install_package_into_env(
        self,
        package: str,
        method: str = "conda",
        channel: str = "conda-forge",
    ) -> bool:
        """Try to install a package into the step's conda environment.

        Tries the requested method first, falls back to the other.
        Updates the env YAML on success.

        Returns True on success.
        """
        if not self.conda_tools or not self.env_name:
            return False

        result = self.conda_tools.install_package(
            env_name=self.env_name,
            package=package,
            method=method,
            channel=channel,
            yaml_path=self.env_yaml_path,
        )

        if result.get("success"):
            method_used = result.get("method_used", method)
            self._packages_installed.append(package)
            self._actions_taken.append(
                f"Installed {package} via {method_used}"
            )
            return True

        self._actions_taken.append(
            f"Failed to install {package}: {result.get('error', '')[:150]}"
        )
        return False

    def _validate_package(self, module_name: str) -> bool:
        """Verify a package is importable in the environment."""
        if not self.conda_tools or not self.env_name:
            return False

        # Detect language from step files
        language = "python"
        for key in self._step_files:
            if key.endswith(".R"):
                language = "r"
            elif key.endswith(".pl"):
                language = "perl"

        result = self.conda_tools.validate_env_has_package(
            env_name=self.env_name,
            package=module_name,
            language=language,
        )
        return result.get("success", False)

    # =========================================================================
    # HELPER: PARSING
    # =========================================================================

    def _parse_missing_module(self, error_message: str, stderr: str) -> Optional[str]:
        """Extract the module name from a ModuleNotFoundError."""
        combined = f"{error_message}\n{stderr}"
        patterns = [
            r"ModuleNotFoundError: No module named '([^']+)'",
            r"ImportError: No module named '([^']+)'",
            r"ImportError: cannot import name '(\w+)'",
            r"ModuleNotFoundError: No module named \"([^\"]+)\"",
        ]
        for pat in patterns:
            m = re.search(pat, combined)
            if m:
                # Return top-level module (e.g. "sklearn" from "sklearn.utils")
                return m.group(1).split(".")[0]
        return None

    def _import_to_package_name(self, module_name: str) -> str:
        """Map Python import names to PyPI/conda package names.

        Many packages have different import and install names.
        """
        mapping = {
            "sklearn": "scikit-learn",
            "skimage": "scikit-image",
            "cv2": "opencv-python",
            "PIL": "pillow",
            "yaml": "pyyaml",
            "Bio": "biopython",
            "scvi": "scvi-tools",
            "leidenalg": "leidenalg",
            "umap": "umap-learn",
            "mpl_toolkits": "matplotlib",
            "tables": "pytables",
            "hdf5plugin": "hdf5plugin",
            "magic": "magic-impute",
            "louvain": "louvain",
        }
        return mapping.get(module_name, module_name)

    def _parse_traceback(self, text: str) -> Dict[str, Any]:
        """Extract file, line number, and error from a Python traceback."""
        result = {"file": None, "line": None, "error": None, "context": ""}

        if not text:
            return result

        # Python traceback: File "path", line N
        tb_lines = re.findall(
            r'File "([^"]+)", line (\d+)', text
        )
        if tb_lines:
            # Last entry is the actual error location
            result["file"] = tb_lines[-1][0]
            result["line"] = int(tb_lines[-1][1])

        # R traceback: Error in <func>(<file>:<line>)
        if not result["file"]:
            r_match = re.search(
                r'Error in.*?:\s*(\S+\.R):(\d+)', text, re.IGNORECASE
            )
            if r_match:
                result["file"] = r_match.group(1)
                result["line"] = int(r_match.group(2))

        # Extract the actual error message (last line after traceback)
        lines = text.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith("File ") and not line.startswith("Traceback"):
                result["error"] = line
                break

        # Extract a few lines of context around the error
        if result["line"] and result["file"]:
            result["context"] = self._get_code_context(
                result["file"], result["line"]
            )

        return result

    def _get_code_context(self, filepath: str, line_num: int, window: int = 5) -> str:
        """Read lines around a specific line in a file."""
        try:
            path = Path(filepath)
            if not path.is_absolute():
                path = self.project_root / path
            if not path.exists():
                return ""

            lines = path.read_text().split("\n")
            start = max(0, line_num - window - 1)
            end = min(len(lines), line_num + window)

            context_lines = []
            for i in range(start, end):
                marker = ">>>" if i == line_num - 1 else "   "
                context_lines.append(f"{marker} {i + 1:4d} | {lines[i]}")
            return "\n".join(context_lines)
        except Exception:
            return ""

    # =========================================================================
    # HELPER: FILE DISCOVERY
    # =========================================================================

    def _read_step_files(self) -> Dict[str, Dict[str, Any]]:
        """Discover all files matching {step_id}.* prefix.

        Returns a dict mapping filename → {path, content, size} for each
        discovered file. Content is loaded lazily (only for small files).
        """
        files = {}
        search_dirs = [
            self.project_root / "scripts",
            self.project_root / "envs",
            self.project_root / "slurm" / "scripts",
        ]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for path in search_dir.iterdir():
                if path.name.startswith(self.step_id):
                    entry = {
                        "path": str(path),
                        "size": path.stat().st_size if path.exists() else 0,
                    }
                    # Load content for small files (<100KB)
                    if entry["size"] < 100_000:
                        try:
                            entry["content"] = path.read_text()
                        except Exception:
                            entry["content"] = ""
                    files[path.name] = entry

        return files

    def _read_script_content(self, filepath: Optional[str]) -> str:
        """Read script content from a path, with fallback resolution."""
        if not filepath:
            return ""

        path = Path(filepath)
        if not path.is_absolute():
            path = self.project_root / path

        if path.exists():
            try:
                return path.read_text()
            except Exception:
                return ""

        # Try resolving relative to scripts/
        alt = self.project_root / "scripts" / path.name
        if alt.exists():
            try:
                return alt.read_text()
            except Exception:
                pass

        return ""

    def _get_any_script_content(self) -> str:
        """Get the content of any script file for this step."""
        for ext in ("py", "R", "sh", "pl", "java"):
            key = f"{self.step_id}.{ext}"
            if key in self._step_files and self._step_files[key].get("content"):
                return self._step_files[key]["content"]

        # Fallback: return any file content we have
        for info in self._step_files.values():
            if info.get("content"):
                return info["content"]
        return ""

    # =========================================================================
    # HELPER: DIAGNOSTIC COMMANDS
    # =========================================================================

    def _run_diagnostic_command(self, command: str, timeout: int = 30) -> str:
        """Execute a bash command and return its output.

        Budgeted: stops after MAX_DIAGNOSTIC_COMMANDS to prevent runaway.
        """
        if self._commands_run >= self.MAX_DIAGNOSTIC_COMMANDS:
            return "[diagnostic command limit reached]"

        self._commands_run += 1

        try:
            result = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.project_root),
            )
            output = result.stdout + result.stderr
            return output[:2000]  # Truncate to save tokens
        except subprocess.TimeoutExpired:
            return f"[command timed out after {timeout}s]"
        except Exception as e:
            return f"[command failed: {e}]"

    def _run_data_inspection(self, stderr: str) -> str:
        """Run a diagnostic script to inspect data objects.

        Attempts to identify the data file from the error, load it,
        and dump its structure.
        """
        # Find h5ad/csv files referenced in the error
        data_files = re.findall(
            r'["\']?([^\s"\']+\.(?:h5ad|csv|tsv|parquet))["\']?', stderr
        )

        if not data_files:
            # Search project directory
            find_result = self._run_diagnostic_command(
                "find . -name '*.h5ad' -o -name '*.csv' | head -5"
            )
            return f"No data files in error. Available: {find_result}"

        results = []
        for df in data_files[:2]:  # Inspect up to 2 files
            if df.endswith(".h5ad"):
                script = (
                    f"import scanpy as sc; "
                    f"adata = sc.read_h5ad('{df}'); "
                    f"print('Shape:', adata.shape); "
                    f"print('obs columns:', list(adata.obs.columns[:20])); "
                    f"print('var columns:', list(adata.var.columns[:20])); "
                    f"print('layers:', list(adata.layers.keys()) if adata.layers else 'none'); "
                    f"print('obsm:', list(adata.obsm.keys()) if adata.obsm else 'none')"
                )
                if self.env_name and self.conda_tools:
                    out = self.conda_tools.run_in_environment(
                        self.env_name,
                        f'python -c "{script}"',
                        timeout=60,
                    )
                    results.append(
                        f"{df}: {out.get('stdout', out.get('stderr', 'no output'))}"
                    )
                else:
                    results.append(f"{df}: [cannot inspect — no env available]")
            else:
                head = self._run_diagnostic_command(f"head -5 '{df}'")
                wc = self._run_diagnostic_command(f"wc -l '{df}'")
                results.append(f"{df}: lines={wc.strip()}, head:\n{head}")

        return "\n".join(results)

    def _run_disk_cleanup(self):
        """Run disk cleanup operations."""
        # Try DiskManager first
        try:
            from utils.disk_manager import DiskManager
            dm = DiskManager()
            dm.emergency_cleanup()
            self._actions_taken.append("DiskManager emergency cleanup executed")
            return
        except Exception:
            pass

        # Fallback: direct conda clean
        self._run_diagnostic_command("conda clean --all --yes")
        self._actions_taken.append("Ran 'conda clean --all --yes'")

    # =========================================================================
    # HELPER: LLM DIAGNOSIS
    # =========================================================================

    def _llm_diagnose_code_error(
        self,
        script_content: str,
        error_message: str,
        stderr_tail: str,
        traceback_info: Dict[str, Any],
        subtask_description: str,
    ) -> Optional[Dict[str, Any]]:
        """Ask the LLM to diagnose a code error and propose a fix.

        Returns dict with 'diagnosis', 'fix' (complete script), and
        'confidence' keys, or None on failure.
        """
        if self.llm is None:
            return None

        # Build focused prompt
        prompt = f"""You are a diagnostic agent analyzing a failed script execution.

TASK DESCRIPTION:
{subtask_description[:2000]}

SCRIPT (may contain the bug):
```
{script_content[:6000]}
```

ERROR MESSAGE:
{error_message[:500]}

STDERR (last 3000 chars):
```
{stderr_tail}
```

TRACEBACK INFO:
- File: {traceback_info.get('file', 'unknown')}
- Line: {traceback_info.get('line', 'unknown')}
- Error: {traceback_info.get('error', 'unknown')}
- Context:
{traceback_info.get('context', 'unavailable')}

INSTRUCTIONS:
1. Diagnose the root cause of the error
2. Provide the COMPLETE fixed script (not just the changed lines)
3. Explain what you changed and why

Respond in this exact format:

### DIAGNOSIS ###
[Your diagnosis here]
### CONFIDENCE ###
[A number between 0.0 and 1.0]
### FIXED ###
```
[Complete fixed script here]
```
### END ###"""

        try:
            from utils.llm_invoke import invoke_resilient

            response = invoke_resilient(
                self.llm,
                prompt,
                ollama_base_url=self.ollama_base_url,
                max_retries=10,
                initial_backoff=15.0,
            )

            self._actions_taken.append("LLM diagnosis completed")

            # Parse response
            diagnosis = ""
            confidence = 0.5
            fix = None

            diag_match = re.search(
                r'### DIAGNOSIS ###\s*(.*?)\s*### CONFIDENCE ###',
                response, re.DOTALL
            )
            if diag_match:
                diagnosis = diag_match.group(1).strip()

            conf_match = re.search(
                r'### CONFIDENCE ###\s*([\d.]+)', response
            )
            if conf_match:
                try:
                    confidence = float(conf_match.group(1))
                    confidence = min(1.0, max(0.0, confidence))
                except ValueError:
                    pass

            fix_match = re.search(
                r'### FIXED ###\s*```(?:\w*)\s*(.*?)\s*```\s*### END ###',
                response, re.DOTALL
            )
            if not fix_match:
                # Fallback: try extracting any code block
                fix_match = re.search(
                    r'### FIXED ###\s*```(?:\w*)\s*(.*?)\s*```',
                    response, re.DOTALL
                )
            if not fix_match:
                fix_match = re.search(
                    r'```(?:python|r|bash)\s*(.*?)\s*```',
                    response, re.DOTALL
                )

            if fix_match:
                fix = fix_match.group(1).strip()
                # Sanity check: fix should be at least 50 chars
                if len(fix) < 50:
                    fix = None

            return {
                "diagnosis": diagnosis,
                "confidence": confidence,
                "fix": fix,
            }

        except Exception as e:
            logger.warning(f"LLM diagnosis failed: {e}")
            self._actions_taken.append(f"LLM diagnosis failed: {e}")
            return None

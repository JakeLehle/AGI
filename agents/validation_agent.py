"""
ValidationAgent v1.2.9 — Phase 5 Output Validation for AGI Pipeline

A specialist agent that generates validation scripts, submits them as
lightweight SLURM jobs (--dependency=afterok:<phase4_job_id>), parses
structured validation reports, and synthesizes correction guidance when
outputs fail checks.

Design principles:
  - Operates AFTER Phase 4 success — validates that analysis outputs are
    scientifically sound, not just that the script ran without errors
  - Validation runs as a SEPARATE SLURM job with --dependency=afterok
    so it only executes after the analysis job completes successfully
  - Lightweight resources: 2 CPUs, 16 GB RAM, 30-min cap, no GPU
  - Phase 5 failures do NOT invoke DiagnosticAgent — they are handled
    by synthesize_correction() which produces targeted guidance for
    the next Phase 1 script regeneration attempt
  - Uses the existing MemoryClient/MCP server infrastructure with
    "val_" task_id prefix to prevent retry loops (same pattern as
    Phase 4 but distinguishable in similarity searches)
  - Graceful degradation: if ValidationAgent is unavailable, Phase 5
    is skipped entirely and the step completes as it would in v1.2.8

Validation report contract (stdout JSON):
  {
    "status": "passed" | "failed",
    "step_id": "step_03",
    "outputs": [
      {
        "path": "outputs/step_03/clustered.h5ad",
        "checks": { "exists": true, "size_bytes": 843201024, ... },
        "status": "passed" | "failed",
        "issues": []
      }
    ],
    "issues": [],
    "validation_version": "1.2.9"
  }

All LLM calls use invoke_resilient() with exponential backoff retry.
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
class ValidationReport:
    """Parsed validation report from a Phase 5 SLURM job.

    Attributes:
        status: Top-level "passed" or "failed".
        step_id: Which pipeline step was validated.
        outputs: Per-output check results.
        issues: Top-level issues list.
        raw_json: The original parsed JSON dict.
        validation_version: Report schema version.
    """
    status: str = "unknown"
    step_id: str = ""
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    raw_json: Dict[str, Any] = field(default_factory=dict)
    validation_version: str = "1.2.9"

    @property
    def passed(self) -> bool:
        """Step passes only when top-level AND all outputs pass."""
        if self.status != "passed":
            return False
        return all(o.get("status") == "passed" for o in self.outputs)

    @property
    def summary(self) -> str:
        """One-line human-readable summary for logs and memory."""
        n_outputs = len(self.outputs)
        n_passed = sum(1 for o in self.outputs if o.get("status") == "passed")
        if self.passed:
            return f"PASSED: {n_passed}/{n_outputs} outputs validated"
        failed_paths = [
            o.get("path", "?") for o in self.outputs
            if o.get("status") != "passed"
        ]
        issue_str = "; ".join(self.issues[:3]) if self.issues else "see per-output issues"
        return (
            f"FAILED: {n_passed}/{n_outputs} passed — "
            f"failed: {', '.join(failed_paths[:3])} — {issue_str}"
        )

    @property
    def sample_repr(self) -> Dict[str, Any]:
        """Extract a compact sample representation for manifest storage.

        Pulls key numeric/structural fields from per-output checks so
        the Librarian (v1.3.0) can build a searchable knowledge base
        without re-reading the actual output files.
        """
        samples = {}
        for output in self.outputs:
            path = output.get("path", "unknown")
            checks = output.get("checks", {})
            # Keep only the informative check values, skip booleans
            informative = {
                k: v for k, v in checks.items()
                if not isinstance(v, bool) or k in ("loadable",)
            }
            if informative:
                samples[path] = informative
        return samples


# =============================================================================
# FILE TYPE DISPATCH — Validation check hints for the LLM
# =============================================================================

FILE_TYPE_DISPATCH = {
    ".h5ad": {
        "library": "anndata",
        "load_cmd": "adata = anndata.read_h5ad(path)",
        "checks": [
            "adata.n_obs > 0",
            "adata.n_vars > 0",
            "not adata.obs.isnull().all().any()",
        ],
        "fields": ["n_obs", "n_vars", "obs_columns", "var_columns"],
    },
    ".csv": {
        "library": "pandas",
        "load_cmd": "df = pandas.read_csv(path)",
        "checks": [
            "len(df) > 0",
            "len(df.columns) > 0",
            "not df.isnull().all().any()",
        ],
        "fields": ["n_rows", "n_cols", "columns"],
    },
    ".tsv": {
        "library": "pandas",
        "load_cmd": "df = pandas.read_csv(path, sep='\\t')",
        "checks": [
            "len(df) > 0",
            "len(df.columns) > 0",
        ],
        "fields": ["n_rows", "n_cols", "columns"],
    },
    ".h5": {
        "library": "h5py",
        "load_cmd": "f = h5py.File(path, 'r')",
        "checks": [
            "len(f.keys()) > 0",
        ],
        "fields": ["top_level_keys"],
    },
    ".bam": {
        "library": "pysam",
        "load_cmd": "bam = pysam.AlignmentFile(path, 'rb')",
        "checks": [
            "bam.mapped > 0 or bam.unmapped > 0",
        ],
        "fields": ["mapped", "unmapped"],
    },
    ".bed": {
        "library": None,
        "load_cmd": "lines = open(path).readlines()",
        "checks": [
            "len(lines) > 0",
            "all(len(l.split('\\t')) >= 3 for l in lines[:10])",
        ],
        "fields": ["n_lines"],
    },
    ".png": {
        "library": None,
        "load_cmd": None,
        "checks": [],
        "fields": ["size_bytes"],
    },
    ".pdf": {
        "library": None,
        "load_cmd": None,
        "checks": [],
        "fields": ["size_bytes"],
    },
    ".rds": {
        "library": None,
        "load_cmd": None,
        "checks": [],
        "fields": ["size_bytes"],
        "note": "R binary format — check exists + non-zero size only",
    },
}

# Generic fallback for unrecognized extensions
GENERIC_DISPATCH = {
    "library": None,
    "load_cmd": None,
    "checks": [],
    "fields": ["size_bytes"],
}


def _get_dispatch_for_path(path: str) -> Dict[str, Any]:
    """Return the file type dispatch entry for a given output path."""
    p = Path(path)
    # Handle compound extensions: .fastq.gz, .vcf.gz
    suffixes = p.suffixes
    if len(suffixes) >= 2:
        compound = "".join(suffixes[-2:])
        if compound in FILE_TYPE_DISPATCH:
            return FILE_TYPE_DISPATCH[compound]
    ext = p.suffix.lower()
    return FILE_TYPE_DISPATCH.get(ext, GENERIC_DISPATCH)


# =============================================================================
# VALIDATION SCRIPT TEMPLATE
# =============================================================================

# The LLM generates checking logic ABOVE this fixed footer.
# The footer is appended verbatim — never LLM-generated — so the report
# contract is always satisfied even if the LLM produces buggy checks.
VALIDATION_SCRIPT_FOOTER = '''
# ═══════════════════════════════════════════════════════════════════════
# FIXED FOOTER — DO NOT MODIFY (ValidationAgent v1.2.9)
# This block assembles and prints the final JSON validation report.
# ═══════════════════════════════════════════════════════════════════════

# Determine top-level status from per-output results
_all_passed = all(o["status"] == "passed" for o in _output_results)
_top_issues = []
for _o in _output_results:
    for _iss in _o.get("issues", []):
        _top_issues.append(f"{_o['path']}: {_iss}")

_report = {
    "status": "passed" if _all_passed else "failed",
    "step_id": _STEP_ID,
    "outputs": _output_results,
    "issues": _top_issues,
    "validation_version": "1.2.9",
}

import json as _json
print(_json.dumps(_report))
'''


# =============================================================================
# VALIDATION AGENT
# =============================================================================

class ValidationAgent:
    """
    Specialist agent for Phase 5 output validation.

    Generates a validation script via LLM, wraps it in a lightweight
    SLURM job with --dependency=afterok, parses the structured JSON
    report from stdout, and synthesizes correction guidance on failure.

    Thread Safety:
        Each sub-agent instance gets its own ValidationAgent (or shares
        one passed from the workflow). The agent is stateless between
        calls — all state lives in the checkpoint and subtask dict.
    """

    # Maximum length of analysis script content sent to LLM for context
    MAX_SCRIPT_CONTEXT_CHARS = 8000

    def __init__(
        self,
        agent_id: str = "validation_agent",
        project_root: str = None,
        ollama_model: str = None,
        ollama_base_url: str = None,
        memory_client=None,
    ):
        """
        Initialize the Validation Agent.

        Args:
            agent_id: Unique identifier (for logging).
            project_root: Project directory path.
            ollama_model: LLM model name (resolved via model_config if None).
            ollama_base_url: Ollama API URL.
            memory_client: Shared MemoryClient instance for loop prevention.
                           Uses the existing MCP server infrastructure.
        """
        self.agent_id = agent_id
        self.project_root = Path(project_root) if project_root else Path(".")
        self.memory_client = memory_client

        # Initialize LLM (same pattern as DiagnosticAgent)
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
                "LangChain/Ollama not available — "
                "ValidationAgent LLM features disabled"
            )
            self.llm = None
            self.ollama_base_url = ollama_base_url or "http://127.0.0.1:11434"

        logger.info(
            f"ValidationAgent initialized: agent_id={agent_id}, "
            f"project_root={self.project_root}, "
            f"memory={'enabled' if memory_client else 'disabled'}"
        )

    # =========================================================================
    # PUBLIC METHOD 1: GENERATE VALIDATION SCRIPT (Phase 5a)
    # =========================================================================

    def generate_validation_script(
        self,
        subtask: Dict[str, Any],
        analysis_script_path: str,
        checkpoint: Any,
    ) -> Dict[str, Any]:
        """Generate a Python validation script for the step's outputs.

        The LLM receives the step description, expected outputs, the full
        analysis script contents, and file-type-specific check hints.
        It generates checking logic that populates ``_output_results``,
        and the fixed footer (never LLM-generated) assembles and prints
        the final JSON report.

        Args:
            subtask: The subtask dict with id, description, output_files, etc.
            analysis_script_path: Path to the Phase 1 analysis script.
            checkpoint: TaskCheckpoint (used for step metadata).

        Returns:
            {"success": True, "script_path": str} on success.
            {"success": False, "error": str} on failure.
        """
        step_id = subtask.get("id", "task")
        safe_id = re.sub(r"[^\w\-]", "_", step_id)[:30]

        if self.llm is None:
            return {"success": False, "error": "LLM not available"}

        # Read analysis script for context
        analysis_content = ""
        try:
            ap = Path(analysis_script_path)
            if ap.exists():
                analysis_content = ap.read_text()[:self.MAX_SCRIPT_CONTEXT_CHARS]
        except Exception as e:
            logger.warning(f"Could not read analysis script: {e}")

        # Collect output file info and dispatch hints
        output_files = subtask.get("output_files", [])
        if not output_files:
            return {
                "success": False,
                "error": "No output_files defined — nothing to validate",
            }

        dispatch_hints = []
        for out_path in output_files:
            dispatch = _get_dispatch_for_path(out_path)
            dispatch_hints.append({
                "path": out_path,
                "dispatch": dispatch,
            })

        # Build the LLM prompt
        prompt = self._build_validation_script_prompt(
            step_id=step_id,
            description=subtask.get("description", ""),
            output_files=output_files,
            dispatch_hints=dispatch_hints,
            analysis_script=analysis_content,
            success_criteria=subtask.get("success_criteria", ""),
        )

        # Generate via LLM
        try:
            from utils.llm_invoke import invoke_resilient, LLMInvocationError

            response = invoke_resilient(
                self.llm,
                prompt,
                ollama_base_url=self.ollama_base_url,
                max_retries=10,
                initial_backoff=15.0,
            )
        except Exception as e:
            return {"success": False, "error": f"LLM invocation failed: {e}"}

        # Extract code from response
        script_body = self._extract_code_from_response(response)
        if not script_body or len(script_body) < 30:
            return {
                "success": False,
                "error": "LLM returned insufficient validation code",
            }

        # Assemble full validation script: header + LLM body + fixed footer
        full_script = self._assemble_validation_script(
            step_id=step_id,
            llm_body=script_body,
            output_files=output_files,
        )

        # Write to scripts/ directory (protected by manifest)
        script_path = self.project_root / "scripts" / f"{safe_id}_validate.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(full_script)

        logger.info(
            f"[{step_id}] Validation script generated: {script_path} "
            f"({len(full_script)} chars, {len(output_files)} outputs)"
        )
        return {"success": True, "script_path": str(script_path)}

    # =========================================================================
    # PUBLIC METHOD 2: GENERATE VALIDATION SBATCH (Phase 5b)
    # =========================================================================

    def generate_validation_sbatch(
        self,
        subtask: Dict[str, Any],
        checkpoint: Any,
        phase4_job_id: str,
        slurm_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate and submit a lightweight validation SLURM job.

        The validation job uses --dependency=afterok:<phase4_job_id> so
        SLURM will only start it after the analysis job completes
        successfully.  Resources are minimal: 2 CPUs, 16 GB RAM,
        30-minute cap, no GPU.

        Args:
            subtask: The subtask dict.
            checkpoint: TaskCheckpoint with validation_script_path set.
            phase4_job_id: SLURM job ID from the Phase 4 production run.
            slurm_config: Routed SLURM settings (partition, account, etc.).

        Returns:
            {"success": True, "sbatch_path": str, "job_id": str} on success.
            {"success": False, "error": str} on failure.
        """
        step_id = subtask.get("id", "task")
        safe_id = re.sub(r"[^\w\-]", "_", step_id)[:30]

        val_script = getattr(checkpoint, "validation_script_path", None)
        if not val_script or not Path(val_script).exists():
            return {
                "success": False,
                "error": "Validation script not found on disk",
            }

        env_name = getattr(checkpoint, "env_name", None)
        if not env_name:
            return {"success": False, "error": "No env_name in checkpoint"}

        # Build sbatch content
        sbatch_content = self._build_validation_sbatch(
            step_id=step_id,
            safe_id=safe_id,
            val_script_path=val_script,
            env_name=env_name,
            phase4_job_id=phase4_job_id,
            slurm_config=slurm_config,
        )

        # Write to slurm/scripts/ (protected directory)
        sbatch_path = (
            self.project_root / "slurm" / "scripts" / f"{safe_id}_validate.sbatch"
        )
        sbatch_path.parent.mkdir(parents=True, exist_ok=True)
        sbatch_path.write_text(sbatch_content)

        logger.info(f"[{step_id}] Validation sbatch written: {sbatch_path}")

        # Submit via sbatch
        try:
            result = subprocess.run(
                ["sbatch", str(sbatch_path)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.project_root),
            )

            if result.returncode != 0:
                return {
                    "success": False,
                    "error": f"sbatch submission failed: {result.stderr}",
                    "sbatch_path": str(sbatch_path),
                }

            match = re.search(r"Submitted batch job (\d+)", result.stdout)
            if not match:
                return {
                    "success": False,
                    "error": f"Cannot parse job ID: {result.stdout}",
                    "sbatch_path": str(sbatch_path),
                }

            job_id = match.group(1)
            logger.info(
                f"[{step_id}] Validation job submitted: {job_id} "
                f"(depends on {phase4_job_id})"
            )
            return {
                "success": True,
                "sbatch_path": str(sbatch_path),
                "job_id": job_id,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "sbatch command timed out",
                "sbatch_path": str(sbatch_path),
            }
        except FileNotFoundError:
            return {
                "success": False,
                "error": "sbatch command not found — SLURM not available",
                "sbatch_path": str(sbatch_path),
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"sbatch submission error: {e}",
                "sbatch_path": str(sbatch_path),
            }

    # =========================================================================
    # PUBLIC METHOD 3: PARSE VALIDATION OUTPUT (Phase 5c — after job completes)
    # =========================================================================

    def parse_validation_output(
        self,
        slurm_log_path: str,
    ) -> Dict[str, Any]:
        """Parse the structured JSON report from validation job stdout.

        Reads the SLURM stdout log, finds the last line that parses as
        valid JSON matching the report schema, and returns a
        ValidationReport.

        Args:
            slurm_log_path: Path to the validation job stdout log
                            (slurm/logs/{step_id}_validate_{job_id}.out).

        Returns:
            {"parsed": True, "report": ValidationReport} on success.
            {"parsed": False, "raw_tail": str, "error": str} on failure.
        """
        log_path = Path(slurm_log_path)
        if not log_path.exists():
            return {
                "parsed": False,
                "raw_tail": "",
                "error": f"Log file not found: {slurm_log_path}",
            }

        try:
            content = log_path.read_text()
        except Exception as e:
            return {
                "parsed": False,
                "raw_tail": "",
                "error": f"Cannot read log file: {e}",
            }

        # Search from the bottom for the last valid JSON line
        lines = content.strip().splitlines()
        for line in reversed(lines):
            line = line.strip()
            if not line or not line.startswith("{"):
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Validate minimum schema: must have "status" and "outputs"
            if "status" not in data or "outputs" not in data:
                continue

            report = ValidationReport(
                status=data.get("status", "unknown"),
                step_id=data.get("step_id", ""),
                outputs=data.get("outputs", []),
                issues=data.get("issues", []),
                raw_json=data,
                validation_version=data.get("validation_version", "unknown"),
            )

            logger.info(
                f"Validation report parsed: {report.summary}"
            )
            return {"parsed": True, "report": report}

        # No valid JSON found — return tail for LLM diagnosis
        tail = "\n".join(lines[-30:]) if lines else ""
        return {
            "parsed": False,
            "raw_tail": tail,
            "error": "No valid JSON validation report found in stdout",
        }

    # =========================================================================
    # PUBLIC METHOD 4: SYNTHESIZE CORRECTION (Phase 5c — on failure)
    # =========================================================================

    def synthesize_correction(
        self,
        subtask: Dict[str, Any],
        validation_report: Optional[ValidationReport] = None,
        raw_log_tail: str = "",
    ) -> Dict[str, Any]:
        """Produce correction guidance after a validation failure.

        The LLM reads the validation report (or raw log tail if the
        report was unparseable) and produces 3–5 concrete, actionable
        correction hints.  These are dual-injected into the subtask:
          - subtask['code_hints'] += hints
          - subtask['expanded_plan'] += [VALIDATION FAILURE] block

        Before generating corrections, checks MemoryClient to prevent
        suggesting the same fix that was already tried.

        Args:
            subtask: The subtask dict (will be modified in place by caller).
            validation_report: Parsed ValidationReport (preferred).
            raw_log_tail: Raw stdout tail if report was unparseable.

        Returns:
            {
                "hints": List[str],
                "guidance_block": str,
                "summary": str,
                "from_memory": bool,
                "duplicate_detected": bool,
            }
        """
        step_id = subtask.get("id", "task")

        # Build failure context for the LLM
        if validation_report:
            failure_context = self._format_report_for_llm(validation_report)
            summary = validation_report.summary
        else:
            failure_context = f"Raw validation log tail:\n{raw_log_tail[-3000:]}"
            summary = "Validation script failed to produce parseable report"

        # ── Memory check: has this correction been tried before? ─────────
        proposed_summary = f"Correction for: {summary}"
        duplicate_detected = False

        if self.memory_client:
            try:
                from utils.reflexion_integration import check_before_retry
                check = check_before_retry(
                    task_id=f"val_{step_id}",
                    proposed_approach=proposed_summary,
                )
                if check.get("tried"):
                    duplicate_detected = True
                    logger.warning(
                        f"[{step_id}] Validation correction already tried "
                        f"(similarity={check.get('similarity', 0):.2f}). "
                        f"Will request a DIFFERENT approach from LLM."
                    )
            except Exception as e:
                logger.warning(f"Memory check failed: {e}")

        # ── Generate correction via LLM ──────────────────────────────────
        hints = []
        if self.llm is not None:
            hints = self._llm_generate_corrections(
                step_id=step_id,
                description=subtask.get("description", ""),
                failure_context=failure_context,
                output_files=subtask.get("output_files", []),
                duplicate_detected=duplicate_detected,
            )

        if not hints:
            hints = [
                f"Previous validation failed: {summary}",
                "Review the output file generation logic carefully",
                "Ensure all expected output files are written with correct format",
            ]

        # ── Build guidance block for expanded_plan injection ─────────────
        guidance_block = self._build_guidance_block(hints, summary)

        # ── Store this correction attempt in memory ──────────────────────
        correction_summary = "; ".join(hints[:3])
        if self.memory_client:
            try:
                self.memory_client.store_failure(
                    task_id=f"val_{step_id}",
                    error_type="validation_failure",
                    error_message=summary,
                    approach_tried=correction_summary,
                )
            except Exception as e:
                logger.warning(f"Memory store_failure failed: {e}")

        logger.info(
            f"[{step_id}] Synthesized {len(hints)} correction hints "
            f"(duplicate_detected={duplicate_detected})"
        )

        return {
            "hints": hints,
            "guidance_block": guidance_block,
            "summary": summary,
            "from_memory": False,
            "duplicate_detected": duplicate_detected,
        }

    # =========================================================================
    # MEMORY: Store validation success (called by sub_agent on Phase 5 pass)
    # =========================================================================

    def store_validation_success(
        self,
        step_id: str,
        report: ValidationReport,
    ):
        """Record a successful validation in memory for future reference.

        Called by the sub_agent after Phase 5c passes. Stores the summary
        and sample_repr so the Librarian (v1.3.0) can reference validated
        output characteristics.

        Args:
            step_id: The pipeline step ID.
            report: The passing ValidationReport.
        """
        if not self.memory_client:
            return

        try:
            self.memory_client.store_solution(
                task_id=f"val_{step_id}",
                problem_pattern=f"Validation of {step_id} outputs",
                error_type="validation_success",
                solution=report.summary,
            )
            logger.info(f"[{step_id}] Stored validation success in memory")
        except Exception as e:
            logger.warning(f"Memory store_solution failed: {e}")

    # =========================================================================
    # PRIVATE: Build validation script prompt
    # =========================================================================

    def _build_validation_script_prompt(
        self,
        step_id: str,
        description: str,
        output_files: List[str],
        dispatch_hints: List[Dict],
        analysis_script: str,
        success_criteria: str,
    ) -> str:
        """Build the LLM prompt for validation script generation."""

        # Format dispatch hints for the prompt
        hints_text = ""
        for dh in dispatch_hints:
            path = dh["path"]
            disp = dh["dispatch"]
            lib = disp.get("library", "None")
            load = disp.get("load_cmd", "# exists + size check only")
            checks = disp.get("checks", [])
            fields = disp.get("fields", [])

            hints_text += f"\n  File: {path}\n"
            hints_text += f"    Library: {lib}\n"
            hints_text += f"    Load: {load}\n"
            if checks:
                hints_text += f"    Suggested checks: {', '.join(checks)}\n"
            if fields:
                hints_text += f"    Fields to report: {', '.join(fields)}\n"

        return f"""Generate validation checking logic for a bioinformatics pipeline step.

STEP ID: {step_id}
TASK DESCRIPTION:
{description[:3000]}

{f"SUCCESS CRITERIA: {success_criteria}" if success_criteria else ""}

EXPECTED OUTPUT FILES:
{chr(10).join(f"  - {f}" for f in output_files)}

FILE TYPE HINTS (use these libraries and checks):
{hints_text}

ANALYSIS SCRIPT (for context — understand what outputs should contain):
```python
{analysis_script[:6000]}
```

YOUR TASK:
Generate Python code that validates each output file. The code will be inserted
into a script that already has these variables defined:

  _STEP_ID = "{step_id}"           # string
  _PROJECT_ROOT = Path(...)         # resolved project root
  _output_results = []              # list you must append to

For EACH output file, your code must:
1. Resolve the full path: path = _PROJECT_ROOT / "<relative_path>"
2. Check if the file exists and has non-zero size
3. If the file type has a specific library (see hints above), load it and run
   the suggested checks
4. Build a result dict and append to _output_results:
   _output_results.append({{
       "path": "<relative_path>",
       "checks": {{
           "exists": True/False,
           "size_bytes": <int>,
           "loadable": True/False,   # if applicable
           # ... any additional check fields
       }},
       "status": "passed" or "failed",
       "issues": ["list of specific issues found"]
   }})

CRITICAL RULES:
- Generate ONLY the checking logic — no imports of json, no final print statement.
  The script header handles imports and the fixed footer handles the final print.
- Every code path must append exactly one dict per output file to _output_results.
- Use try/except around each file check so one broken file doesn't skip the rest.
- Set status="failed" if exists is False OR if any domain check fails.
- Include specific, actionable issue descriptions (not generic messages).
- Wrap each output check in its own try/except block.

Generate ONLY the Python checking code — no markdown fences, no explanation."""

    # =========================================================================
    # PRIVATE: Assemble the full validation script
    # =========================================================================

    def _assemble_validation_script(
        self,
        step_id: str,
        llm_body: str,
        output_files: List[str],
    ) -> str:
        """Combine header + LLM body + fixed footer into a complete script."""
        header = f'''#!/usr/bin/env python3
"""
Validation script for {step_id} — auto-generated by ValidationAgent v1.2.9

Checks that the analysis outputs exist, are loadable, and meet basic
quality criteria. Prints a single JSON report on the last line of stdout.

DO NOT EDIT the footer section — it is the contract between this script
and the ValidationAgent parser.
"""

import os
import sys
from pathlib import Path

# ── Project root (same as analysis script) ──────────────────────────────
_PROJECT_ROOT = Path(os.environ.get("PROJECT_DIR", ".")).resolve()
os.chdir(_PROJECT_ROOT)

_STEP_ID = "{step_id}"
_output_results = []

print(f"[Validation] step={{_STEP_ID}}, project={{_PROJECT_ROOT}}")
print(f"[Validation] checking {len(output_files)} output(s)...")

# ═══════════════════════════════════════════════════════════════════════
# LLM-GENERATED CHECKING LOGIC (ValidationAgent v1.2.9)
# ═══════════════════════════════════════════════════════════════════════
'''
        return header + "\n" + llm_body + "\n" + VALIDATION_SCRIPT_FOOTER

    # =========================================================================
    # PRIVATE: Build validation sbatch
    # =========================================================================

    def _build_validation_sbatch(
        self,
        step_id: str,
        safe_id: str,
        val_script_path: str,
        env_name: str,
        phase4_job_id: str,
        slurm_config: Dict[str, Any],
    ) -> str:
        """Generate the sbatch script for the validation SLURM job."""

        partition = slurm_config.get("partition", "normal")
        account = slurm_config.get("account", "")

        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name=val_{safe_id}",
            f"#SBATCH --output=slurm/logs/{safe_id}_validate_%j.out",
            f"#SBATCH --error=slurm/logs/{safe_id}_validate_%j.err",
            f"#SBATCH --dependency=afterok:{phase4_job_id}",
            f"#SBATCH --partition={partition}",
        ]

        if account:
            lines.append(f"#SBATCH --account={account}")

        lines.extend([
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=2",
            "#SBATCH --mem=16G",
            "#SBATCH --time=00:30:00",
            "",
            "#" + "=" * 70,
            f"# Validation job for {step_id}",
            f"# Depends on analysis job {phase4_job_id}",
            f"# Resources: 2 CPUs, 16G RAM, 30 min, no GPU",
            "#" + "=" * 70,
            "",
            "set -e",
            "",
            "echo '=============================================='",
            "echo 'Validation Job ID: '$SLURM_JOB_ID",
            "echo 'Node: '$(hostname)",
            "echo 'Start: '$(date)",
            f"echo 'Step: {step_id}'",
            f"echo 'Depends on: {phase4_job_id}'",
            "echo '=============================================='",
            "",
            "# Conda setup",
            f'CONDA_ENV="{env_name}"',
            "",
            'echo ">>> Loading conda..."',
            'if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then',
            '    source "$HOME/anaconda3/etc/profile.d/conda.sh"',
            'elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then',
            '    source "$HOME/miniconda3/etc/profile.d/conda.sh"',
            'elif command -v conda &> /dev/null; then',
            '    eval "$(conda shell.bash hook)"',
            'else',
            '    echo "ERROR: conda not found!"; exit 1',
            'fi',
            "",
            'echo ">>> Activating ${CONDA_ENV}..."',
            'conda activate "${CONDA_ENV}"',
            "",
            f'export PROJECT_DIR="{self.project_root}"',
            "",
            f'echo ">>> Running validation script..."',
            f"python {val_script_path}",
            "",
            'echo ""',
            'echo "=============================================="',
            'echo "Validation complete: $(date)"',
            'echo "=============================================="',
        ])

        return "\n".join(lines) + "\n"

    # =========================================================================
    # PRIVATE: LLM correction generation
    # =========================================================================

    def _llm_generate_corrections(
        self,
        step_id: str,
        description: str,
        failure_context: str,
        output_files: List[str],
        duplicate_detected: bool,
    ) -> List[str]:
        """Ask the LLM to generate correction hints from a validation failure."""

        duplicate_instruction = ""
        if duplicate_detected:
            duplicate_instruction = """
IMPORTANT: A very similar correction was already attempted and FAILED.
You MUST suggest a FUNDAMENTALLY DIFFERENT approach — not a rewording
of the same fix. Consider:
  - The output generation logic may need a completely different algorithm
  - The issue may be upstream (wrong input data, missing preprocessing step)
  - The data format assumptions may be wrong
  - A different library or method may be needed
"""

        prompt = f"""You are a validation agent analyzing why a bioinformatics pipeline
step produced invalid outputs.

STEP: {step_id}
TASK DESCRIPTION:
{description[:2000]}

EXPECTED OUTPUTS:
{chr(10).join(f"  - {f}" for f in output_files)}

VALIDATION FAILURE DETAILS:
{failure_context[:4000]}
{duplicate_instruction}

Generate 3-5 CONCRETE, ACTIONABLE correction hints. Each hint should be
a specific instruction that a code-generation LLM can follow to fix the
analysis script. Be precise about:
  - Which output file failed and why
  - What the script should do differently
  - Specific function calls, parameters, or logic changes needed

Format: Return ONLY a JSON array of strings, nothing else.
Example: ["Fix 1: ...", "Fix 2: ...", "Fix 3: ..."]"""

        try:
            from utils.llm_invoke import invoke_resilient

            response = invoke_resilient(
                self.llm,
                prompt,
                ollama_base_url=self.ollama_base_url,
                max_retries=10,
                initial_backoff=15.0,
            )

            # Parse JSON array from response
            hints = self._parse_hints_from_response(response)
            if hints:
                return hints

        except Exception as e:
            logger.warning(f"LLM correction generation failed: {e}")

        return []

    # =========================================================================
    # PRIVATE: Formatting helpers
    # =========================================================================

    def _format_report_for_llm(self, report: ValidationReport) -> str:
        """Format a ValidationReport into a readable string for LLM context."""
        parts = [f"Validation status: {report.status}"]

        for output in report.outputs:
            parts.append(f"\n  Output: {output.get('path', '?')}")
            parts.append(f"    Status: {output.get('status', '?')}")
            checks = output.get("checks", {})
            for k, v in checks.items():
                parts.append(f"    {k}: {v}")
            issues = output.get("issues", [])
            for iss in issues:
                parts.append(f"    ISSUE: {iss}")

        if report.issues:
            parts.append("\n  Top-level issues:")
            for iss in report.issues:
                parts.append(f"    - {iss}")

        return "\n".join(parts)

    def _build_guidance_block(
        self, hints: List[str], summary: str
    ) -> str:
        """Build the guidance block for expanded_plan injection.

        Mirrors the INJECT_HINTS.sh dual-injection pattern:
        subtask['code_hints'] += hints (caller does this)
        subtask['expanded_plan'] += guidance_block (caller does this)
        """
        hint_lines = "\n".join(f"- {h}" for h in hints)
        return f"""

[VALIDATION FAILURE — FOLLOW EXACTLY]
The previous attempt produced outputs that failed validation.
Validation report summary:
  {summary}

Required corrections:
{hint_lines}

You MUST address ALL of the above issues in the regenerated script.
Do not repeat the same approach that produced invalid outputs.
"""

    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response, stripping markdown fences."""
        # Try python-tagged code blocks first
        for tag in ["python", "py", ""]:
            pattern = rf"```{tag}\s*\n(.*?)\n```"
            m = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if m:
                return m.group(1).strip()

        # No code block — strip any remaining fences and return as-is
        content = re.sub(r"^```(?:\w+)?\n?", "", response, flags=re.MULTILINE)
        content = re.sub(r"\n?```\s*$", "", content, flags=re.MULTILINE)
        return content.strip()

    def _parse_hints_from_response(self, response: str) -> List[str]:
        """Parse a JSON array of hint strings from LLM response."""
        # Try to find a JSON array in the response
        # First try the whole response
        text = response.strip()

        # Strip markdown fences if present
        text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
        text = text.strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(h) for h in parsed if h]
        except json.JSONDecodeError:
            pass

        # Fallback: find the first [...] in the response
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    return [str(h) for h in parsed if h]
            except json.JSONDecodeError:
                pass

        # Last resort: extract numbered/bulleted lines
        hints = []
        for line in text.splitlines():
            line = line.strip()
            # Match "1. Fix ..." or "- Fix ..." or "* Fix ..."
            m = re.match(r"^(?:\d+[\.\)]\s*|[-*]\s+)(.+)", line)
            if m and len(m.group(1)) > 10:
                hints.append(m.group(1))

        return hints[:5] if hints else []

    # =========================================================================
    # PRIVATE: Find validation SLURM log
    # =========================================================================

    def find_validation_log(
        self,
        step_id: str,
        job_id: str,
    ) -> Optional[str]:
        """Locate the stdout log file for a validation SLURM job.

        Searches slurm/logs/ for the expected filename pattern:
        {safe_id}_validate_{job_id}.out

        Args:
            step_id: Pipeline step ID.
            job_id: SLURM job ID.

        Returns:
            Path string if found, None otherwise.
        """
        safe_id = re.sub(r"[^\w\-]", "_", step_id)[:30]
        log_dir = self.project_root / "slurm" / "logs"

        # Exact match first
        exact = log_dir / f"{safe_id}_validate_{job_id}.out"
        if exact.exists():
            return str(exact)

        # Glob fallback (SLURM may format job ID differently)
        pattern = f"{safe_id}_validate_*.out"
        matches = sorted(log_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
        if matches:
            # Return most recent
            return str(matches[-1])

        return None

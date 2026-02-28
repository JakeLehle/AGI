#!/bin/bash
# =============================================================================
# AGI Pipeline - PARTIAL Clean Script (v1.2.8)
# =============================================================================
# Selective cleanup for mid-pipeline restarts. Removes log files and
# safe intermediates while preserving everything needed to resume from
# the current pipeline state without re-running completed steps.
#
# REMOVES (safe to delete — regenerated or irrelevant on next run):
#   - slurm/logs/        All SLURM job stdout/stderr from previous run
#                        (diagnostic agent reads only the CURRENT job's logs
#                        by job_id — old logs are never consulted)
#   - slurm_logs/        Master pipeline job stdout/stderr
#   - logs/              Agent JSONL logs, Ollama logs, error logs
#                        Exception: logs/steps/ is PRESERVED (see below)
#   - prompts/prompt_*   Auto-generated decomposition JSONs
#   - reports/pipeline_status.md  (regenerated each run)
#
# PRESERVES (required for partial restart or cleanup protection):
#   - data/raw/                          Source input files. Never modified
#                                        by the pipeline; never safe to remove.
#   - outputs/                           Scientific step outputs. Protected
#                                        by output_manifest.json.
#   - scripts/                           Generated + user scripts.
#   - slurm/scripts/                     Generated sbatch scripts. v1.2.8:
#                                        These are OVERWRITTEN on regen, never
#                                        deleted. Preserving them lets a resume
#                                        skip Phase 3 when the sbatch is valid.
#   - envs/                              Conda YAML specs. Needed for env
#                                        recreation or audit.
#   - reports/master_prompt_state.json   Pipeline state: which steps
#                                        completed, failed, or are pending.
#                                        The router reads this to skip
#                                        completed steps and resume failed ones.
#   - reports/output_manifest.json       Canonical output registry (v1.2.8).
#                                        Lists all protected output paths.
#                                        Cleanup scripts consult this before
#                                        removing anything.
#   - temp/checkpoints/step_*_checkpoint.json
#                                        Phase progress per step. Contains
#                                        script_path, env_name, sbatch_path,
#                                        and phase. Sub-agent resumes from
#                                        the correct phase on restart.
#   - logs/steps/                        Per-step phase logs (v1.2.5). These
#                                        survive PARTIAL clean so Phase 1/2/3
#                                        failures remain diagnosable after
#                                        the main logs are cleared.
#   - Conda environments (agi_step_*)    Existing envs are reused — not
#                                        recreating them saves the biggest
#                                        time cost in a partial restart.
#   - prompts/*.md                       Master prompt files.
#   - config/                            Pipeline configuration.
#
# v1.2.8 CHANGES FROM v1.2.5:
#   - slurm/scripts/ moved from REMOVES → PRESERVES (sbatch scripts are now
#     overwritten on regeneration rather than deleted and recreated)
#   - Manifest pre-flight check: reads output_manifest.json before removing
#     anything and aborts if protected paths would be affected
#   - outputs/ added to preserved display and pre-flight check
#   - data/raw/ replaces data/ in preserved display (canonical input dir)
#
# Usage:
#   bash PARTIAL_CLEAN_PROJECT.sh              # interactive
#   bash PARTIAL_CLEAN_PROJECT.sh --dry-run    # preview only, no changes
#   bash PARTIAL_CLEAN_PROJECT.sh --yes        # skip confirmation
#
# Run from your project root directory.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

DRY_RUN=false
AUTO_YES=false

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --yes|-y)  AUTO_YES=true ;;
        --help|-h)
            echo "Usage: bash PARTIAL_CLEAN_PROJECT.sh [--dry-run] [--yes]"
            echo ""
            echo "  --dry-run   Show what would be deleted without removing anything"
            echo "  --yes, -y   Skip confirmation prompt"
            echo ""
            echo "Use this between runs when you want to resume from current pipeline"
            echo "state (preserving completed steps and conda environments)."
            echo ""
            echo "Use FULL_CLEAN_PROJECT.sh to wipe everything for a fresh start."
            exit 0
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Verify we're in a project directory
# ---------------------------------------------------------------------------
if [ ! -f "project.yaml" ] && [ ! -d "temp/checkpoints" ] && [ ! -d "logs" ]; then
    echo -e "${RED}ERROR: This doesn't look like an AGI project directory.${NC}"
    echo "  Expected: project.yaml, temp/checkpoints/, or logs/"
    echo "  Run from your project root (e.g. slide-TCR-seq-working/)"
    exit 1
fi

PROJECT_DIR="$(pwd)"

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  AGI Pipeline — PARTIAL Clean (v1.2.8)${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "  Project: ${CYAN}${PROJECT_DIR}${NC}"
if [ "$DRY_RUN" = true ]; then
    echo -e "  Mode:    ${YELLOW}DRY RUN — nothing will be changed${NC}"
else
    echo -e "  Mode:    ${BLUE}PARTIAL — logs removed, pipeline state preserved${NC}"
fi
echo ""

# ---------------------------------------------------------------------------
# Manifest pre-flight check (v1.2.8)
# ---------------------------------------------------------------------------
# Read output_manifest.json and verify that none of the paths we are about
# to delete are listed as protected outputs. PARTIAL clean should never
# touch protected paths, but this is a belt-and-suspenders safety check.
# ---------------------------------------------------------------------------

MANIFEST_FILE="reports/output_manifest.json"
MANIFEST_PROTECTED_DIRS=("data/raw" "outputs" "scripts" "slurm/scripts" "envs" "reports")

if [ -f "$MANIFEST_FILE" ] && command -v python3 &>/dev/null; then
    python3 - "$MANIFEST_FILE" <<'PYEOF' || true
import json, sys
from pathlib import Path

manifest_path = sys.argv[1]
try:
    with open(manifest_path) as f:
        manifest = json.load(f)
except Exception as e:
    print(f"  ⚠  Could not parse manifest: {e}")
    sys.exit(0)

steps = manifest.get("steps", {})
protected = []
for step_id, entry in steps.items():
    for output in entry.get("outputs", []):
        path = output.get("path", "")
        if output.get("protected", True) and path:
            protected.append((step_id, path))

if protected:
    print(f"  Manifest: {len(steps)} step(s), {len(protected)} protected output(s) registered")
    print(f"  ✓  None of these are in the PARTIAL clean scope")
else:
    print(f"  Manifest: {len(steps)} step(s) registered, no outputs yet")
PYEOF
    echo ""
elif [ -f "$MANIFEST_FILE" ]; then
    echo -e "  ${CYAN}→${NC} Manifest found: ${MANIFEST_FILE} (python3 not available for pre-flight check)"
    echo ""
else
    echo -e "  ${CYAN}→${NC} No manifest yet (first run or manifest not written)"
    echo ""
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOTAL_FILES=0
TOTAL_SIZE=0

count_dir() {
    local label="$1"
    local dir="$2"
    if [ -d "$dir" ]; then
        local count size hs
        count=$(find "$dir" -type f 2>/dev/null | wc -l || echo 0)
        size=$(du -sb "$dir" 2>/dev/null | awk '{print $1}' || echo 0)
        hs=$(numfmt --to=iec --suffix=B "${size:-0}" 2>/dev/null || echo "${size}B")
        if [ "$count" -gt 0 ]; then
            echo -e "  ${YELLOW}●${NC} ${label}: ${count} files (${hs})"
            TOTAL_FILES=$((TOTAL_FILES + count))
            TOTAL_SIZE=$((TOTAL_SIZE + ${size:-0}))
        else
            echo -e "  ${GREEN}○${NC} ${label}: clean"
        fi
    else
        echo -e "  ${GREEN}○${NC} ${label}: not present"
    fi
}

count_glob() {
    local label="$1"
    shift
    local count=0
    local size=0
    for pattern in "$@"; do
        while IFS= read -r -d '' f; do
            count=$((count + 1))
            size=$((size + $(stat -c%s "$f" 2>/dev/null || echo 0)))
        done < <(find . -maxdepth 5 -path "$pattern" -type f -print0 2>/dev/null)
    done
    if [ "$count" -gt 0 ]; then
        local hs
        hs=$(numfmt --to=iec --suffix=B "$size" 2>/dev/null || echo "${size}B")
        echo -e "  ${YELLOW}●${NC} ${label}: ${count} files (${hs})"
        TOTAL_FILES=$((TOTAL_FILES + count))
        TOTAL_SIZE=$((TOTAL_SIZE + size))
    else
        echo -e "  ${GREEN}○${NC} ${label}: clean"
    fi
}

remove_dir_contents() {
    local label="$1"
    local dir="$2"
    if [ -d "$dir" ]; then
        local count
        count=$(find "$dir" -type f 2>/dev/null | wc -l)
        if [ "$count" -gt 0 ]; then
            if [ "$DRY_RUN" = false ]; then
                find "$dir" -type f -delete
                find "$dir" -mindepth 1 -type d -empty -delete 2>/dev/null || true
            fi
            echo -e "  ${GREEN}✓${NC} Cleared ${count} file(s): ${label}"
        fi
    fi
}

remove_glob() {
    local label="$1"
    shift
    local count=0
    for pattern in "$@"; do
        while IFS= read -r -d '' f; do
            if [ "$DRY_RUN" = false ]; then
                rm -f "$f"
            fi
            count=$((count + 1))
        done < <(find . -maxdepth 5 -path "$pattern" -type f -print0 2>/dev/null)
    done
    [ "$count" -gt 0 ] && echo -e "  ${GREEN}✓${NC} Removed ${count} file(s): ${label}"
}

count_logs_main() {
    # Count files in logs/ excluding the logs/steps/ subdirectory.
    # logs/steps/ contains per-step phase logs (v1.2.5) that survive
    # PARTIAL clean so Phase 1/2/3 failures remain diagnosable.
    local label="$1"
    if [ -d "logs" ]; then
        local count size hs
        count=$(find logs -type f \
                    -not -path "logs/steps/*" \
                    2>/dev/null | wc -l || echo 0)
        size=$(find logs -type f \
                   -not -path "logs/steps/*" \
                   2>/dev/null \
               | xargs du -sc 2>/dev/null | tail -1 | awk '{print $1}' \
               || echo 0)
        hs=$(numfmt --to=iec --suffix=B "${size:-0}" 2>/dev/null \
             || echo "${size:-0}B")
        if [ "${count:-0}" -gt 0 ]; then
            echo -e "  ${YELLOW}●${NC} ${label}: ${count} files (${hs})"
            TOTAL_FILES=$((TOTAL_FILES + count))
            TOTAL_SIZE=$((TOTAL_SIZE + ${size:-0}))
        else
            echo -e "  ${GREEN}○${NC} ${label}: clean"
        fi
    else
        echo -e "  ${GREEN}○${NC} ${label}: not present"
    fi
}

remove_logs_main() {
    # Remove files in logs/ but preserve the logs/steps/ subdirectory
    # and all its contents (per-step phase logs, v1.2.5).
    local label="$1"
    if [ -d "logs" ]; then
        local count
        count=$(find logs -type f \
                    -not -path "logs/steps/*" \
                    2>/dev/null | wc -l || echo 0)
        if [ "${count:-0}" -gt 0 ]; then
            if [ "$DRY_RUN" = false ]; then
                find logs -mindepth 1 -type f \
                    -not -path "logs/steps/*" \
                    -delete 2>/dev/null || true
                # Remove empty subdirs but never remove logs/steps itself
                find logs -mindepth 1 -type d \
                    -not -path "logs/steps" \
                    -not -path "logs/steps/*" \
                    -empty -delete 2>/dev/null || true
            fi
            echo -e "  ${GREEN}✓${NC} Cleared ${count} file(s): ${label}"
        fi
    fi
}


# ---------------------------------------------------------------------------
# Show current pipeline state (always, even in dry run)
# ---------------------------------------------------------------------------

STATE_FILE="reports/master_prompt_state.json"
echo "Current pipeline state:"
if [ -f "$STATE_FILE" ]; then
    if command -v python3 &>/dev/null; then
        python3 - "$STATE_FILE" <<'PYEOF' || true
import json, sys
try:
    with open(sys.argv[1]) as f:
        state = json.load(f)
    steps = state.get('steps', {})
    order = state.get('step_order', list(steps.keys()))
    counts = {'completed': 0, 'failed': 0, 'pending': 0, 'running': 0, 'blocked': 0}
    for sid in order:
        s = steps.get(sid, {})
        st = s.get('status', 'pending')
        counts[st] = counts.get(st, 0) + 1
    icons = {'completed': '✅', 'failed': '❌', 'pending': '⏳', 'running': '🔄', 'blocked': '🚫'}
    print(f"  Steps: {len(order)} total  |  ", end="")
    print("  ".join(f"{icons.get(k,'?')} {k}: {v}" for k, v in counts.items() if v > 0))
    print("")
    print("  Per-step status:")
    for sid in order:
        s = steps.get(sid, {})
        st = s.get('status', 'pending')
        icon = icons.get(st, '?')
        title = s.get('title', sid)[:55]
        err = f"  → {s['error_summary'][:80]}" if s.get('error_summary') and st == 'failed' else ""
        print(f"    {icon} {sid:<12} {title}{err}")
except Exception as e:
    print(f"  (Could not parse state file: {e})")
PYEOF
    else
        echo "  (python3 not available — cannot parse state file)"
    fi
else
    echo -e "  ${YELLOW}⚠${NC} No master_prompt_state.json found"
    echo "     This may be a first run, or state was already cleared."
    echo ""
fi

# ---------------------------------------------------------------------------
# Inventory what will be removed
# ---------------------------------------------------------------------------

echo ""
echo "Scanning artifacts to remove..."
echo ""

count_logs_main  "Agent/Ollama logs        (logs/ excl. steps/)"
count_dir  "SLURM subtask logs       (slurm/logs/)"             "slurm/logs"
count_dir  "SLURM master logs        (slurm_logs/)"             "slurm_logs"
count_glob "Generated prompt JSONs   (prompts/prompt_*.json)"   "./prompts/prompt_*.json"
count_glob "Pipeline status MD       (reports/pipeline_status.md)" "./reports/pipeline_status.md"

# v1.2.8: slurm/scripts/ is no longer scanned or removed.
# Sbatch scripts are OVERWRITTEN on regeneration (not deleted and recreated),
# so they belong in the PROTECTED set alongside scripts/ and envs/.
# Preserving them also lets a resumed sub-agent skip Phase 3 when the
# existing sbatch is still valid (checkpoint.sbatch_path is set and file exists).

echo ""

# Show phase log summary separately — these are PRESERVED
if [ -d "logs/steps" ]; then
    STEP_LOG_COUNT=$(find logs/steps -type f -name "*.log" 2>/dev/null | wc -l || echo 0)
    STEP_LOG_STEPS=$(find logs/steps -type f -name "*.log" 2>/dev/null \
                     | sed 's|logs/steps/||;s|_phases\.log||' | sort | tr '\n' ' ')
    if [ "${STEP_LOG_COUNT:-0}" -gt 0 ]; then
        echo -e "  ${GREEN}◆${NC} Phase logs (PRESERVED) (logs/steps/): ${STEP_LOG_COUNT} file(s)"
        echo -e "    Steps with logs: ${CYAN}${STEP_LOG_STEPS}${NC}"
    fi
fi

echo ""

if [ "$TOTAL_FILES" -eq 0 ]; then
    echo -e "${GREEN}Nothing to remove — logs already clean.${NC}"
    echo ""
    echo "To make step changes before restarting, edit:"
    echo -e "  ${CYAN}reports/master_prompt_state.json${NC}  (step status, hints, input_files)"
    echo -e "  ${CYAN}temp/checkpoints/step_N_checkpoint.json${NC}  (phase progress)"
    echo ""
    echo "To inspect a Phase 1/2/3 failure:"
    echo -e "  ${CYAN}cat logs/steps/step_N_phases.log${NC}"
    exit 0
fi

TOTAL_HUMAN=$(numfmt --to=iec --suffix=B "$TOTAL_SIZE" 2>/dev/null || echo "${TOTAL_SIZE} bytes")
echo -e "  Will remove: ${YELLOW}${TOTAL_FILES} files${NC} (${TOTAL_HUMAN})"
echo ""

# ---------------------------------------------------------------------------
# Show what is explicitly preserved
# ---------------------------------------------------------------------------

echo -e "${GREEN}Preserved (will NOT be touched):${NC}"

# Count checkpoints
CP_COUNT=$(find temp/checkpoints -name "step_*_checkpoint.json" 2>/dev/null | wc -l || echo 0)
echo -e "  ● temp/checkpoints/          ${CP_COUNT} checkpoint(s) — phase progress per step"

# Show checkpoint states
if [ "$CP_COUNT" -gt 0 ] && command -v python3 &>/dev/null; then
    python3 - <<'PYEOF' || true
import json, glob, os
from pathlib import Path

files = sorted(glob.glob("temp/checkpoints/step_*_checkpoint.json"))
phase_names = {0: "init", 1: "script", 2: "env", 3: "sbatch", 4: "executing", 99: "done"}
for fp in files:
    try:
        with open(fp) as f:
            cp = json.load(f)
        task_id = cp.get("task_id", Path(fp).stem)
        phase = cp.get("phase_completed", 0)
        status = cp.get("status", "unknown")
        env = cp.get("env_name", "")
        env_note = f" (env: {env})" if env else ""
        phase_name = phase_names.get(phase, f"phase {phase}")
        print(f"    → {task_id:<20} phase={phase_name:<12} status={status}{env_note}")
    except Exception:
        print(f"    → {fp} (unreadable)")
PYEOF
fi

# Show manifest summary if present
if [ -f "$MANIFEST_FILE" ] && command -v python3 &>/dev/null; then
    python3 - "$MANIFEST_FILE" <<'PYEOF' || true
import json, sys
from pathlib import Path
try:
    with open(sys.argv[1]) as f:
        m = json.load(f)
    steps = m.get("steps", {})
    completed = [s for s, v in steps.items() if v.get("status") == "completed"]
    output_count = sum(len(v.get("outputs", [])) for v in steps.values())
    print(f"  ● reports/output_manifest.json   {len(completed)}/{len(steps)} step(s) completed, {output_count} protected output(s)")
except Exception:
    print(f"  ● reports/output_manifest.json   (present)")
PYEOF
else
    echo -e "  ● reports/output_manifest.json   (not yet created)"
fi

echo -e "  ● reports/master_prompt_state.json  — pipeline step status"
# v1.2.8: slurm/scripts/ is now protected
SBATCH_COUNT=$(find slurm/scripts -type f -name "*.sbatch" 2>/dev/null | wc -l || echo 0)
echo -e "  ● slurm/scripts/             ${SBATCH_COUNT} sbatch file(s) — overwritten on regen, not deleted"
echo -e "  ● envs/                      conda YAML specs"
echo -e "  ● scripts/                   generated + user scripts"
echo -e "  ● data/raw/                  source input files (never modified)"
echo -e "  ● outputs/                   step scientific outputs (manifest-protected)"
echo -e "  ● conda envs (agi_step_*)   existing environments (not removed)"
echo -e "  ● prompts/*.md               master prompt files"
echo -e "  ● config/                    pipeline configuration"
echo ""

# ---------------------------------------------------------------------------
# Confirm
# ---------------------------------------------------------------------------

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}Dry run complete — no files were changed.${NC}"
    exit 0
fi

if [ "$AUTO_YES" = false ]; then
    read -p "Remove ${TOTAL_FILES} log/intermediate files? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
fi

echo ""
echo "Cleaning..."
echo ""

# ---------------------------------------------------------------------------
# Execute removal
# v1.2.8: slurm/scripts/ is intentionally absent from this block.
#         It was removed from v1.2.5's deletion list because sbatch scripts
#         are now overwritten on regeneration rather than deleted and recreated.
# ---------------------------------------------------------------------------

remove_logs_main     "agent + ollama logs (logs/ excl. steps/)"
remove_dir_contents  "SLURM subtask logs"   "slurm/logs"
remove_dir_contents  "SLURM master logs"    "slurm_logs"
remove_glob          "generated prompt JSONs"      "./prompts/prompt_*.json"
remove_glob          "pipeline status report"      "./reports/pipeline_status.md"

# ---------------------------------------------------------------------------
# Done — show next-step guidance
# ---------------------------------------------------------------------------

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  PARTIAL clean complete.${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo -e "  Pipeline state preserved at: ${CYAN}reports/master_prompt_state.json${NC}"
echo ""
echo "  Before resubmitting, you can edit that file to:"
echo "    - Change a step's 'status' from 'failed' to 'pending' to re-run it"
echo "    - Add entries to a step's 'code_hints' list to guide script generation"
echo "    - Correct 'input_files' paths for steps that had argument errors"
echo "    - Set 'attempts' back to 0 to reset retry exhaustion"
echo ""
echo "  To reset a specific step's phase progress (force full regeneration):"
echo -e "    ${CYAN}rm temp/checkpoints/step_N_checkpoint.json${NC}"
echo "    (the existing sbatch in slurm/scripts/ will be overwritten by Phase 3)"
echo ""
echo "  To inject human guidance for failed steps before restarting:"
echo -e "    ${CYAN}bash INJECT_HINTS.sh${NC}"
echo ""
echo "  To inspect Phase 1/2/3 failures before deciding what to inject:"
echo -e "    ${CYAN}cat logs/steps/step_N_phases.log${NC}         (full phase output)"
echo -e "    ${CYAN}tail -50 logs/steps/step_N_phases.log${NC}    (last 50 lines)"
echo -e "    ${CYAN}ls -lh logs/steps/${NC}                       (all available step logs)"
echo ""
echo "  Then resubmit:"
echo -e "    ${CYAN}sbatch RUN_AGI_PIPELINE_GPU.sh${NC}"
echo ""

#!/bin/bash
# =============================================================================
# AGI Pipeline - Human Guidance Injection Tool (v1.2.8)
# =============================================================================
# Form-based tool for injecting guidance into failed pipeline steps before
# a restart. Safely edits master_prompt_state.json via Python — no manual
# JSON editing required.
#
# Supports up to 4 steps per session. For each step you can:
#   1. Add implementation hints  — tell the LLM exactly how to code something
#   2. Fix input file paths      — correct missing or wrong input_files
#   3. Override the approach     — replace the expanded plan entirely
#   4. Skip the step             — mark completed to unblock downstream steps
#   5. Reset and retry           — clear failed status, reset attempt count
#
# Changes are previewed before writing. Optionally clears the step checkpoint
# (forces full restart from Phase 1) or just clears sbatch_path in the
# checkpoint (Phase 3 regenerates and OVERWRITES the existing sbatch file).
#
# NOTE on sbatch handling (v1.2.8):
#   slurm/scripts/ is now a PROTECTED directory — sbatch files are never
#   deleted, only overwritten. The "delete_sbatch" checkpoint action clears
#   the sbatch_path field in the checkpoint JSON only, which causes Phase 3
#   to regenerate and overwrite the sbatch on the next run. The file itself
#   is not removed.
#
# Usage:
#   bash INJECT_HINTS.sh                       # interactive
#   bash INJECT_HINTS.sh --dry-run             # preview changes without writing
#   bash INJECT_HINTS.sh --reset-step=step_03  # non-interactive single step reset
#   bash INJECT_HINTS.sh --reset-all-envs      # non-interactive: force env rebuild
#
# Non-interactive flags:
#   --reset-step=STEP_ID   Reset a specific step to pending and delete its
#                          checkpoint. Skips the interactive form entirely.
#                          Useful in scripts or after a targeted fix.
#   --reset-all-envs       Walk all step checkpoints and set env_created=False.
#                          Forces Phase 2 to rebuild every conda environment on
#                          the next run. Scripts, sbatch files, and the manifest
#                          are untouched. Use when envs are corrupted cluster-wide.
#
# Run from your project root directory.
#
# v1.2.4 fixes (carried forward):
#   - read_single and read_multiline now open /dev/tty directly so they
#     work inside a bash heredoc (stdin is consumed by the script itself)
#   - All display output (menus, step info, prompts) uses display() which
#     writes to stderr — visible on the terminal but not captured by the
#     > "$INSTRUCTIONS_FILE" redirect on the heredoc
#   - Only the final json.dumps() goes to stdout → instructions file
#
# v1.2.8 changes:
#   - Added --reset-step=STEP_ID non-interactive flag
#   - Added --reset-all-envs non-interactive flag
#   - delete_sbatch checkpoint action now clears sbatch_path in checkpoint
#     only (does NOT delete the actual file — slurm/scripts/ is protected)
#   - Version bump throughout
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
BOLD='\033[1m'
NC='\033[0m'

DRY_RUN=false
RESET_STEP=""
RESET_ALL_ENVS=false

for arg in "$@"; do
    case "$arg" in
        --dry-run)          DRY_RUN=true ;;
        --reset-all-envs)   RESET_ALL_ENVS=true ;;
        --reset-step=*)     RESET_STEP="${arg#--reset-step=}" ;;
        --help|-h)
            echo "Usage: bash INJECT_HINTS.sh [OPTIONS]"
            echo ""
            echo "  --dry-run              Preview all changes without writing to disk"
            echo "  --reset-step=STEP_ID   Non-interactively reset a step to pending"
            echo "                         and delete its phase checkpoint."
            echo "  --reset-all-envs       Non-interactively mark every step's conda"
            echo "                         environment for rebuild on next run."
            echo ""
            echo "Run from your project root. Edits reports/master_prompt_state.json"
            echo "safely via Python to inject human guidance before a pipeline restart."
            exit 0
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Verify project directory
# ---------------------------------------------------------------------------
STATE_FILE="reports/master_prompt_state.json"
CHECKPOINT_DIR="temp/checkpoints"

if [ ! -f "$STATE_FILE" ]; then
    echo -e "${RED}ERROR: No master_prompt_state.json found.${NC}"
    echo "  Expected: ${STATE_FILE}"
    echo "  Run from your project root, and ensure the pipeline has run at least once."
    exit 1
fi

if [ ! -d "$CHECKPOINT_DIR" ]; then
    mkdir -p "$CHECKPOINT_DIR"
fi

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  AGI Pipeline — Human Guidance Injection (v1.2.8)${NC}"
echo -e "${BLUE}============================================================${NC}"
if [ "$DRY_RUN" = true ]; then
    echo -e "  Mode: ${YELLOW}DRY RUN — changes will be shown but not written${NC}"
fi
echo ""

# =============================================================================
# NON-INTERACTIVE: --reset-all-envs
# =============================================================================
# Walk every checkpoint in temp/checkpoints/ and set env_created=False.
# This forces Phase 2 to rebuild all conda environments on the next run.
# Scripts, sbatch files, and the output manifest are untouched.
# =============================================================================

if [ "$RESET_ALL_ENVS" = true ]; then
    echo -e "  ${CYAN}→${NC} Mode: --reset-all-envs (non-interactive)"
    echo ""

    python3 - "$CHECKPOINT_DIR" "$DRY_RUN" <<'PYEOF'
import json, sys, glob
from pathlib import Path

cp_dir  = Path(sys.argv[1])
dry_run = sys.argv[2].lower() == "true"

checkpoints = sorted(cp_dir.glob("*.json"))
if not checkpoints:
    print("  No checkpoints found — nothing to reset.")
    sys.exit(0)

count = 0
for cp_path in checkpoints:
    try:
        with open(cp_path) as f:
            cp = json.load(f)
    except Exception as e:
        print(f"  ⚠  Could not read {cp_path.name}: {e}")
        continue

    task_id     = cp.get("task_id", cp_path.stem)
    was_created = cp.get("env_created", False)
    env_name    = cp.get("env_name", "")

    if was_created or env_name:
        cp["env_created"] = False
        marker = "✓" if not dry_run else "(dry-run)"
        print(f"  {marker}  {task_id:<30}  env_created → False  (env: {env_name})")
        if not dry_run:
            tmp = str(cp_path) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(cp, f, indent=2)
            Path(tmp).replace(cp_path)
        count += 1
    else:
        print(f"  ○  {task_id:<30}  already False — skipped")

print(f"\n  {'Would reset' if dry_run else 'Reset'} env_created on {count} checkpoint(s).")
print("  Conda environments themselves are NOT removed.")
print("  Phase 2 will rebuild them on the next pipeline run.")
PYEOF

    echo ""
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}Dry run complete — no files were written.${NC}"
    else
        echo -e "${GREEN}Done. Resubmit to trigger Phase 2 env rebuild for all steps.${NC}"
        echo -e "  ${CYAN}sbatch RUN_AGI_PIPELINE_GPU.sh${NC}"
    fi
    echo ""
    exit 0
fi

# =============================================================================
# NON-INTERACTIVE: --reset-step=STEP_ID
# =============================================================================
# Reset a single named step to pending and delete its checkpoint.
# The step's scripts, sbatch, env YAML, and conda environment are untouched.
# =============================================================================

if [ -n "$RESET_STEP" ]; then
    echo -e "  ${CYAN}→${NC} Mode: --reset-step=${RESET_STEP} (non-interactive)"
    echo ""

    python3 - "$STATE_FILE" "$CHECKPOINT_DIR" "$RESET_STEP" "$DRY_RUN" <<'PYEOF'
import json, sys, re
from pathlib import Path
from datetime import datetime

state_file = sys.argv[1]
cp_dir     = Path(sys.argv[2])
step_id    = sys.argv[3]
dry_run    = sys.argv[4].lower() == "true"

with open(state_file) as f:
    state = json.load(f)

steps = state.get("steps", {})
if step_id not in steps:
    print(f"  ERROR: Step '{step_id}' not found in state file.")
    print(f"  Available: {', '.join(steps.keys())}")
    sys.exit(1)

step = steps[step_id]
old_status = step.get("status", "unknown")

# Reset step status
step["status"]       = "pending"
step["attempts"]     = 0
step["last_updated"] = datetime.now().isoformat()
step.pop("error_summary", None)
steps[step_id] = step

print(f"  Step:    {step_id}")
print(f"  Status:  {old_status} → pending")
print(f"  Attempts reset to 0")

# Delete checkpoint
safe_id  = re.sub(r"[^\w\-]", "_", step_id)[:50]
cp_file  = cp_dir / f"{safe_id}_checkpoint.json"
if not cp_file.exists():
    # Try bare step_id
    cp_file2 = cp_dir / f"{step_id}_checkpoint.json"
    if cp_file2.exists():
        cp_file = cp_file2

if cp_file.exists():
    print(f"  Checkpoint: {cp_file.name} → deleted")
    if not dry_run:
        cp_file.unlink()
else:
    print(f"  Checkpoint: not found (already clean)")

# Write state
if not dry_run:
    state["steps"] = steps
    state["last_updated"] = datetime.now().isoformat()
    tmp = state_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    Path(tmp).replace(state_file)
    print(f"\n  ✅ Written: {state_file}")
else:
    print(f"\n  DRY RUN — no files written.")
PYEOF

    echo ""
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}Dry run complete — no files were written.${NC}"
    else
        echo -e "${GREEN}Done. Step '${RESET_STEP}' will restart from Phase 1 on next run.${NC}"
        echo ""
        echo "  Recommended next steps:"
        echo -e "  1. ${CYAN}bash PARTIAL_CLEAN_PROJECT.sh --yes${NC}   (clear old logs)"
        echo -e "  2. ${CYAN}sbatch RUN_AGI_PIPELINE_GPU.sh${NC}        (resubmit)"
    fi
    echo ""
    exit 0
fi

# =============================================================================
# INTERACTIVE MODE (default)
# =============================================================================

# ---------------------------------------------------------------------------
# Show pipeline state and collect failed step IDs
# ---------------------------------------------------------------------------
echo "Current pipeline state:"
echo ""

FAILED_STEPS=$(python3 - "$STATE_FILE" <<'PYEOF'
import json, sys

try:
    with open(sys.argv[1]) as f:
        state = json.load(f)

    steps = state.get('steps', {})
    failed = []

    STATUS_ICON = {
        'completed': '✅',
        'failed':    '❌',
        'running':   '🔄',
        'pending':   '⏳',
        'blocked':   '🚫',
    }

    for sid, s in steps.items():
        st       = s.get('status', 'pending')
        icon     = STATUS_ICON.get(st, '❓')
        title    = s.get('title', sid)[:60]
        attempts = s.get('attempts', 0)
        err = ""
        if s.get('error_summary') and st in ('failed', 'running'):
            err = f"\n       Error: {s['error_summary'][:100]}"
        print(f"  {icon} {sid:<14} [{st:<10}] attempts={attempts}  {title}{err}")
        if st in ('failed', 'running', 'blocked'):
            failed.append(sid)

    print(f"\nSteps needing attention: {', '.join(failed) if failed else 'none'}")
    print(f"\n__FAILED__:{','.join(failed)}")

except Exception as e:
    print(f"Error reading state: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
)

echo "$FAILED_STEPS" | grep -v "^__FAILED__:"
echo ""

# Extract the failed step IDs
FAILED_ID_LIST=$(echo "$FAILED_STEPS" | grep "^__FAILED__:" | sed 's/__FAILED__://')

if [ -z "$FAILED_ID_LIST" ] || [ "$FAILED_ID_LIST" = "" ]; then
    echo -e "${GREEN}No failed or blocked steps found. Nothing to inject.${NC}"
    echo ""
    echo "  For non-interactive resets, try:"
    echo -e "    ${CYAN}bash INJECT_HINTS.sh --reset-step=step_N${NC}"
    echo -e "    ${CYAN}bash INJECT_HINTS.sh --reset-all-envs${NC}"
    exit 0
fi

# Convert to array
IFS=',' read -ra FAILED_ARRAY <<< "$FAILED_ID_LIST"

# ---------------------------------------------------------------------------
# Let user select up to 4 steps to address
# ---------------------------------------------------------------------------
echo -e "${BOLD}Select steps to inject guidance into (up to 4):${NC}"
echo "  Available: ${FAILED_ID_LIST//,/ }"
echo "  Enter step IDs separated by spaces, or press Enter to use all failed steps"
echo "  (Only the first 4 will be processed)"
echo ""
read -p "  Step IDs: " -r SELECTED_RAW

# If empty, use all failed steps
if [ -z "$SELECTED_RAW" ]; then
    SELECTED_RAW="${FAILED_ID_LIST//,/ }"
fi

# Validate and limit to 4
SELECTED=()
for sid in $SELECTED_RAW; do
    if python3 -c "
import json, sys
with open('$STATE_FILE') as f:
    s = json.load(f)
sys.exit(0 if '$sid' in s.get('steps', {}) else 1)
" 2>/dev/null; then
        SELECTED+=("$sid")
    else
        echo -e "  ${YELLOW}⚠${NC} Unknown step ID: ${sid} — skipping"
    fi
    [ "${#SELECTED[@]}" -ge 4 ] && break
done

if [ "${#SELECTED[@]}" -eq 0 ]; then
    echo -e "${RED}No valid steps selected. Exiting.${NC}"
    exit 1
fi

echo ""
echo -e "Processing ${#SELECTED[@]} step(s): ${SELECTED[*]}"
echo ""

# ---------------------------------------------------------------------------
# Collect guidance for each selected step
#
# KEY DESIGN: the heredoc stdout is redirected to INSTRUCTIONS_FILE, so
# only the final json.dumps() ends up there. All interactive display uses
# display() which writes to stderr — always visible on the terminal.
# read_single / read_multiline open /dev/tty directly, bypassing the
# stdin-consuming heredoc entirely.
# ---------------------------------------------------------------------------
INSTRUCTIONS_FILE=$(mktemp /tmp/agi_inject_XXXXXX.json)
trap 'rm -f "$INSTRUCTIONS_FILE"' EXIT

python3 - "$STATE_FILE" "${SELECTED[@]}" <<'PYEOF' > "$INSTRUCTIONS_FILE"
import json, sys

state_file = sys.argv[1]
selected   = sys.argv[2:]

with open(state_file) as f:
    state = json.load(f)

steps = state.get('steps', {})
instructions = []

# ---------------------------------------------------------------------------
# display() — all terminal output goes to stderr so it is never captured
# by the > "$INSTRUCTIONS_FILE" redirect on this heredoc.
# ---------------------------------------------------------------------------
def display(*args, **kwargs):
    kwargs['file'] = sys.stderr
    print(*args, **kwargs)

# ---------------------------------------------------------------------------
# read_single / read_multiline — open /dev/tty directly so they work
# inside a heredoc where stdin is the script text, not the keyboard.
# ---------------------------------------------------------------------------
def read_single(prompt_text, default=""):
    """Read a single line from the terminal."""
    try:
        tty = open('/dev/tty', 'r')
    except OSError:
        return default
    with tty:
        if default:
            sys.stderr.write(f"  {prompt_text} [{default}]: ")
        else:
            sys.stderr.write(f"  {prompt_text}: ")
        sys.stderr.flush()
        line = tty.readline()
        val  = line.rstrip('\n').strip()
        return val if val else default

def read_multiline(prompt_text):
    """Read multi-line input from the terminal. Empty line = done."""
    display(f"\n  {prompt_text}")
    display("  (Enter each line, then an empty line to finish)")
    lines = []
    try:
        tty = open('/dev/tty', 'r')
    except OSError:
        return lines
    with tty:
        while True:
            sys.stderr.write("  > ")
            sys.stderr.flush()
            line = tty.readline()
            if not line or line.rstrip('\n') == "":
                break
            lines.append(line.rstrip('\n'))
    return lines

# ---------------------------------------------------------------------------
# Menus
# ---------------------------------------------------------------------------
MENU = """
  Injection type:
    1) Add implementation hints   — guide how the LLM writes the script
    2) Fix input file paths       — correct missing/wrong input_files
    3) Override approach          — replace expanded_plan entirely
    4) Skip this step             — mark completed, unblock downstream
    5) Reset and retry            — clear failed status, reset attempts
    6) Skip (no change)
"""

PHASE_MENU = """
  Checkpoint action:
    1) Delete checkpoint          — force full restart from Phase 1 (script generation)
    2) Keep checkpoint            — resume from where it left off
    3) Clear sbatch reference     — keep env + script, Phase 3 regenerates sbatch
                                    (v1.2.8: overwrites existing file, does not delete)
"""

# ---------------------------------------------------------------------------
# Per-step guidance collection loop
# ---------------------------------------------------------------------------
for step_id in selected:
    step = steps.get(step_id, {})

    display(f"\n{'='*60}")
    display(f"  STEP: {step_id}")
    display(f"  Title: {step.get('title', 'unknown')}")
    display(f"  Status: {step.get('status', 'unknown')}  |  Attempts: {step.get('attempts', 0)}")

    # Show full error summary — truncated only if very long
    if step.get('error_summary'):
        err = step['error_summary']
        if len(err) <= 400:
            display(f"  Last error: {err}")
        else:
            display(f"  Last error: {err[:400]}")
            display(f"              ... ({len(err)} chars total — see phase log for full detail)")

    if step.get('input_files'):
        display(f"  Input files: {step['input_files']}")
    current_hints = step.get('code_hints', [])
    if current_hints:
        display(f"  Existing hints ({len(current_hints)}): {current_hints[0][:80]}...")

    # Show phase log path and last 5 lines for Phase 1/2/3 failures.
    # Phase 4 failures have SLURM logs in slurm/logs/ instead.
    import os
    from pathlib import Path as _Path

    project_root = _Path(state_file).parent.parent
    phase_log = project_root / 'logs' / 'steps' / f"{step_id}_phases.log"

    if phase_log.exists():
        log_size = phase_log.stat().st_size
        display(f"")
        display(f"  Phase log: logs/steps/{step_id}_phases.log  "
                f"({log_size:,} bytes)")
        display(f"  Last 5 lines:")
        try:
            lines = phase_log.read_text(errors='replace').splitlines()
            tail = lines[-5:] if len(lines) >= 5 else lines
            for ln in tail:
                display(f"    {ln}")
        except Exception as e:
            display(f"    (could not read log: {e})")
        display(f"  To read full log: cat logs/steps/{step_id}_phases.log")
    else:
        status = step.get('status', '')
        err_text = step.get('error_summary', '')
        is_phase123_failure = (
            status == 'failed' and (
                'Phase 1' in err_text or
                'Phase 2' in err_text or
                'Phase 3' in err_text
            )
        )
        if is_phase123_failure:
            display(f"")
            display(f"  Phase log: not found at logs/steps/{step_id}_phases.log")
            display(f"  (log is created on next run — this step has not run "
                    f"under v1.2.5+ yet)")

    display(f"{'='*60}")

    display(MENU)
    choice = read_single("Choice", "6").strip()
    instr = {"step_id": step_id, "action": None, "checkpoint_action": "keep"}

    # ── Option 1: Add implementation hints ──────────────────────────────────
    if choice == "1":
        instr["action"] = "add_hints"
        existing = step.get('code_hints', [])
        if existing:
            display(f"\n  Existing hints will be KEPT. New hints will be ADDED.")
            display(f"  Current hints:")
            for i, h in enumerate(existing):
                display(f"    {i+1}. {h}")
            keep = read_single("Keep existing hints? [Y/n]", "Y").strip().lower()
            if keep in ("n", "no"):
                instr["clear_existing_hints"] = True

        hints = read_multiline(
            "Enter implementation hints (one per line):\n"
            "  Examples:\n"
            "    Use scanpy.read_h5ad(sys.argv[1]) to load the input\n"
            "    The AnnData object uses obs['cell_type'] not obs['celltype']\n"
            "    Save output to outputs/step_03/ not scripts/"
        )
        instr["hints"] = hints

        display(PHASE_MENU)
        cp = read_single("Checkpoint action", "1").strip()
        instr["checkpoint_action"] = {
            "1": "delete", "2": "keep", "3": "clear_sbatch_ref"
        }.get(cp, "delete")

    # ── Option 2: Fix input file paths ──────────────────────────────────────
    elif choice == "2":
        instr["action"] = "fix_inputs"
        display("\n  Current input_files:", step.get('input_files', []))
        display("  Enter the correct input file paths (absolute or relative to project root):")
        paths = read_multiline("Paths (one per line)")
        instr["input_files"] = paths

        also_hint = read_single("Also add a load hint? [Y/n]", "Y").strip().lower()
        if also_hint not in ("n", "no") and paths:
            lang = step.get('language', 'python')
            if lang == 'r':
                default_hint = "Load input with: adata <- anndata::read_h5ad(commandArgs(TRUE)[1])"
            else:
                default_hint = "Load input with: import sys, scanpy as sc; adata = sc.read_h5ad(sys.argv[1])"
            hint = read_single("Load hint (edit or accept default)", default_hint)
            instr["hints"] = [hint] if hint else []

        display(PHASE_MENU)
        cp = read_single("Checkpoint action", "3").strip()
        instr["checkpoint_action"] = {
            "1": "delete", "2": "keep", "3": "clear_sbatch_ref"
        }.get(cp, "clear_sbatch_ref")

    # ── Option 3: Override approach ──────────────────────────────────────────
    elif choice == "3":
        instr["action"] = "override_approach"
        display("\n  Current expanded_plan (first 400 chars):")
        display(f"  {step.get('expanded_plan', '')[:400]}")
        display("\n  Enter your replacement approach. Be as specific as possible.")
        display("  Include: exact methods, file paths, data structures, expected output format.")
        lines = read_multiline("New approach (one paragraph, empty line to finish)")
        instr["expanded_plan"] = " ".join(lines)

        display(PHASE_MENU)
        cp = read_single("Checkpoint action", "1").strip()
        instr["checkpoint_action"] = {
            "1": "delete", "2": "keep", "3": "clear_sbatch_ref"
        }.get(cp, "delete")

    # ── Option 4: Skip step ──────────────────────────────────────────────────
    elif choice == "4":
        instr["action"] = "skip"
        display(f"\n  Step {step_id} will be marked as COMPLETED.")
        display("  Downstream steps that depend on this will be unblocked.")
        confirm = read_single("Confirm skip? [y/N]", "N").strip().lower()
        if confirm not in ("y", "yes"):
            display("  Cancelled — no change for this step.")
            instr["action"] = None

    # ── Option 5: Reset and retry ────────────────────────────────────────────
    elif choice == "5":
        instr["action"] = "reset"
        display(PHASE_MENU)
        cp = read_single("Checkpoint action", "2").strip()
        instr["checkpoint_action"] = {
            "1": "delete", "2": "keep", "3": "clear_sbatch_ref"
        }.get(cp, "keep")

    # ── Option 6 / default: no change ───────────────────────────────────────
    else:
        instr["action"] = None
        display(f"  Skipping {step_id} — no changes.")

    instructions.append(instr)

# Only stdout → captured by > "$INSTRUCTIONS_FILE"
print(json.dumps(instructions, indent=2))
PYEOF

# ---------------------------------------------------------------------------
# Apply changes via Python
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}Applying changes...${NC}"
echo ""

python3 - "$STATE_FILE" "$INSTRUCTIONS_FILE" "$CHECKPOINT_DIR" "$DRY_RUN" <<'PYEOF'
import json, sys, os, shutil
from pathlib import Path
from datetime import datetime

state_file = sys.argv[1]
instr_file = sys.argv[2]
cp_dir     = Path(sys.argv[3])
dry_run    = sys.argv[4].lower() == "true"

with open(state_file) as f:
    state = json.load(f)

with open(instr_file) as f:
    instructions = json.load(f)

steps = state.get('steps', {})
changes_summary = []

for instr in instructions:
    step_id = instr.get('step_id')
    action  = instr.get('action')

    if not action or step_id not in steps:
        continue

    step = steps[step_id]
    step_changes = []

    # ── add_hints ─────────────────────────────────────────────────────────
    if action == "add_hints":
        new_hints = instr.get('hints', [])
        if not new_hints:
            print(f"  [{step_id}] No hints provided — skipping")
            continue

        if instr.get('clear_existing_hints'):
            step['code_hints'] = new_hints
            step_changes.append(f"Replaced code_hints with {len(new_hints)} new hint(s)")
        else:
            existing = step.get('code_hints', [])
            step['code_hints'] = existing + new_hints
            step_changes.append(
                f"Added {len(new_hints)} hint(s) "
                f"(total: {len(step['code_hints'])})"
            )

        # Dual-injection: also append to expanded_plan so hints reach the
        # sub-agent even if the code_hints patch path has an issue
        guidance_block = (
            "\n\n[HUMAN GUIDANCE — FOLLOW EXACTLY]\n" +
            "\n".join(f"- {h}" for h in new_hints)
        )
        plan = step.get('expanded_plan', step.get('description', ''))
        if '[HUMAN GUIDANCE' in plan:
            plan = plan[:plan.index('[HUMAN GUIDANCE')]
        step['expanded_plan'] = plan.rstrip() + guidance_block
        step_changes.append("Appended guidance block to expanded_plan")

        if step.get('status') in ('failed', 'blocked', 'running'):
            step['status'] = 'pending'
            step['attempts'] = 0
            step_changes.append("Status → pending, attempts → 0")

    # ── fix_inputs ────────────────────────────────────────────────────────
    elif action == "fix_inputs":
        new_paths = instr.get('input_files', [])
        if new_paths:
            step['input_files'] = new_paths
            step_changes.append(f"input_files → {new_paths}")

        extra_hints = instr.get('hints', [])
        if extra_hints:
            existing = step.get('code_hints', [])
            step['code_hints'] = existing + extra_hints
            step_changes.append(f"Added {len(extra_hints)} load hint(s)")
            guidance_block = (
                "\n\n[HUMAN GUIDANCE — FOLLOW EXACTLY]\n" +
                "\n".join(f"- {h}" for h in extra_hints)
            )
            plan = step.get('expanded_plan', step.get('description', ''))
            if '[HUMAN GUIDANCE' in plan:
                plan = plan[:plan.index('[HUMAN GUIDANCE')]
            step['expanded_plan'] = plan.rstrip() + guidance_block

        if step.get('status') in ('failed', 'blocked', 'running'):
            step['status'] = 'pending'
            step['attempts'] = 0
            step_changes.append("Status → pending, attempts → 0")

    # ── override_approach ─────────────────────────────────────────────────
    elif action == "override_approach":
        new_plan = instr.get('expanded_plan', '').strip()
        if new_plan:
            step['expanded_plan'] = new_plan
            step_changes.append(f"expanded_plan replaced ({len(new_plan)} chars)")

        if step.get('status') in ('failed', 'blocked', 'running'):
            step['status'] = 'pending'
            step['attempts'] = 0
            step_changes.append("Status → pending, attempts → 0")

    # ── skip ──────────────────────────────────────────────────────────────
    elif action == "skip":
        step['status'] = 'completed'
        step['skip_reason'] = 'Manually skipped via INJECT_HINTS'
        step_changes.append("Status → completed (manually skipped)")

    # ── reset ─────────────────────────────────────────────────────────────
    elif action == "reset":
        step['status'] = 'pending'
        step['attempts'] = 0
        step.pop('error_summary', None)
        step_changes.append("Status → pending, attempts → 0, error cleared")

    else:
        continue

    # ── checkpoint action ─────────────────────────────────────────────────
    import re
    cp_action = instr.get('checkpoint_action', 'keep')
    safe_id   = re.sub(r'[^\w\-]', '_', step_id)[:50]
    cp_file   = cp_dir / f"{safe_id}_checkpoint.json"

    # Try alternate naming pattern if first not found
    if not cp_file.exists():
        cp_file2 = cp_dir / f"{step_id}_checkpoint.json"
        if cp_file2.exists():
            cp_file = cp_file2

    if cp_action == "delete":
        if cp_file.exists():
            step_changes.append(f"Checkpoint deleted: {cp_file.name}")
            if not dry_run:
                cp_file.unlink()
        else:
            step_changes.append("Checkpoint not found (already clean)")

    elif cp_action == "clear_sbatch_ref":
        # v1.2.8: slurm/scripts/ is a PROTECTED directory — sbatch files are
        # never deleted, only overwritten. This action clears only the
        # sbatch_path field in the checkpoint JSON. Phase 3 will then
        # regenerate and OVERWRITE the existing sbatch file on the next run.
        if cp_file.exists():
            try:
                with open(cp_file) as f:
                    cp = json.load(f)
                sbatch = cp.get('sbatch_path')
                if sbatch:
                    cp['sbatch_path'] = None
                    step_changes.append(
                        f"Checkpoint sbatch_path cleared "
                        f"(Phase 3 will overwrite: {Path(sbatch).name})"
                    )
                    if not dry_run:
                        with open(cp_file, 'w') as f:
                            json.dump(cp, f, indent=2)
                else:
                    step_changes.append("Checkpoint sbatch_path already empty")
            except Exception as e:
                step_changes.append(f"WARNING: Could not update checkpoint: {e}")
        else:
            step_changes.append("Checkpoint not found — nothing to clear")

    elif cp_action == "keep":
        step_changes.append("Checkpoint kept (will resume from last phase)")

    # Timestamp
    step['last_updated'] = datetime.now().isoformat()
    steps[step_id] = step
    changes_summary.append((step_id, step_changes))

# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------
print(f"\n{'─'*60}")
print("  CHANGE PREVIEW")
print(f"{'─'*60}")
for step_id, ch_list in changes_summary:
    step = steps.get(step_id, {})
    print(
        f"\n  [{step_id}]  →  "
        f"status={step.get('status')}  "
        f"attempts={step.get('attempts', 0)}"
    )
    for ch in ch_list:
        print(f"    • {ch}")
    if step.get('code_hints'):
        print(f"    • code_hints ({len(step['code_hints'])}):")
        for h in step['code_hints'][:3]:
            print(f'        "{h[:80]}"')
        if len(step['code_hints']) > 3:
            print(f"        ... and {len(step['code_hints'])-3} more")

print(f"\n{'─'*60}")

if dry_run:
    print("\n  DRY RUN — no files were written.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------
state['steps'] = steps
state['last_updated'] = datetime.now().isoformat()

tmp = state_file + ".tmp"
with open(tmp, 'w') as f:
    json.dump(state, f, indent=2)
os.replace(tmp, state_file)

print(f"\n  ✅ Written: {state_file}")
PYEOF

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}Dry run complete — no files were written.${NC}"
else
    echo -e "${GREEN}${BOLD}============================================================${NC}"
    echo -e "${GREEN}${BOLD}  Injection complete. Ready to resubmit.${NC}"
    echo -e "${GREEN}${BOLD}============================================================${NC}"
    echo ""
    echo "  Recommended next steps:"
    echo -e "  1. ${CYAN}bash PARTIAL_CLEAN_PROJECT.sh --yes${NC}   (clear old logs)"
    echo -e "  2. ${CYAN}sbatch RUN_AGI_PIPELINE_GPU.sh${NC}        (resubmit)"
fi
echo ""

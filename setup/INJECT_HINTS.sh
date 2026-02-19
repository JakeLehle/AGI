#!/bin/bash
# =============================================================================
# AGI Pipeline - Human Guidance Injection Tool (v1.2.3)
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
# Changes are previewed before writing. Optionally deletes step checkpoints
# to control which phases re-run on the next submission.
#
# Usage:
#   bash INJECT_HINTS.sh              # interactive
#   bash INJECT_HINTS.sh --dry-run    # preview changes without writing
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
BOLD='\033[1m'
NC='\033[0m'

DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --help|-h)
            echo "Usage: bash INJECT_HINTS.sh [--dry-run]"
            echo ""
            echo "  --dry-run   Preview all changes without writing to disk"
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

if ! command -v python3 &>/dev/null; then
    echo -e "${RED}ERROR: python3 not found in PATH.${NC}"
    echo "  Activate your conda environment: conda activate AGI"
    exit 1
fi

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
echo ""
echo -e "${BLUE}${BOLD}============================================================${NC}"
echo -e "${BLUE}${BOLD}  AGI Pipeline — Human Guidance Injection (v1.2.3)${NC}"
echo -e "${BLUE}${BOLD}============================================================${NC}"
if [ "$DRY_RUN" = true ]; then
    echo -e "  Mode: ${YELLOW}DRY RUN — changes will be shown but not written${NC}"
fi
echo ""

# ---------------------------------------------------------------------------
# Show current pipeline state and collect failed step IDs
# ---------------------------------------------------------------------------

FAILED_STEPS=$(python3 - "$STATE_FILE" <<'PYEOF'
import json, sys
try:
    with open(sys.argv[1]) as f:
        state = json.load(f)
    steps = state.get('steps', {})
    order = state.get('step_order', list(steps.keys()))

    print("Current pipeline state:\n")
    icons = {'completed': '✅', 'failed': '❌', 'pending': '⏳',
             'running': '🔄', 'blocked': '🚫'}
    failed = []
    for sid in order:
        s = steps.get(sid, {})
        st = s.get('status', 'pending')
        icon = icons.get(st, '?')
        title = s.get('title', sid)[:60]
        attempts = s.get('attempts', 0)
        err = ""
        if s.get('error_summary') and st in ('failed', 'running'):
            err = f"\n       Error: {s['error_summary'][:100]}"
        print(f"  {icon} {sid:<14} [{st:<10}] attempts={attempts}  {title}{err}")
        if st in ('failed', 'running', 'blocked'):
            failed.append(sid)

    print(f"\nSteps needing attention: {', '.join(failed) if failed else 'none'}")
    # Output just the IDs to a separate line for bash to capture
    import sys
    print(f"\n__FAILED__:{ ','.join(failed)}")
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
    # Verify it exists in state
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
# ---------------------------------------------------------------------------
# We'll build a JSON instructions file and pass it to the Python editor
INSTRUCTIONS_FILE=$(mktemp /tmp/agi_inject_XXXXXX.json)
trap 'rm -f "$INSTRUCTIONS_FILE"' EXIT

python3 - "$STATE_FILE" "${SELECTED[@]}" <<'PYEOF' > "$INSTRUCTIONS_FILE"
import json, sys

state_file = sys.argv[1]
selected = sys.argv[2:]

with open(state_file) as f:
    state = json.load(f)

steps = state.get('steps', {})
instructions = []

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
    3) Delete only sbatch         — keep env + script, regenerate sbatch only
"""

def read_multiline(prompt_text):
    """Read multi-line input. Empty line = done."""
    print(f"\n  {prompt_text}")
    print("  (Enter each line, then an empty line to finish)")
    lines = []
    while True:
        try:
            line = input("  > ")
            if line == "":
                break
            lines.append(line)
        except EOFError:
            break
    return lines

def read_single(prompt_text, default=""):
    """Read a single line."""
    if default:
        val = input(f"  {prompt_text} [{default}]: ").strip()
        return val if val else default
    else:
        return input(f"  {prompt_text}: ").strip()

for step_id in selected:
    step = steps.get(step_id, {})
    print(f"\n{'='*60}")
    print(f"  STEP: {step_id}")
    print(f"  Title: {step.get('title', 'unknown')}")
    print(f"  Status: {step.get('status', 'unknown')}  |  Attempts: {step.get('attempts', 0)}")
    if step.get('error_summary'):
        print(f"  Last error: {step['error_summary'][:200]}")
    if step.get('input_files'):
        print(f"  Input files: {step['input_files']}")
    current_hints = step.get('code_hints', [])
    if current_hints:
        print(f"  Existing hints ({len(current_hints)}): {current_hints[0][:80]}...")
    print(f"{'='*60}")

    print(MENU)
    choice = read_single("Choice", "6").strip()

    instr = {"step_id": step_id, "action": None, "checkpoint_action": "keep"}

    if choice == "1":
        instr["action"] = "add_hints"
        existing = step.get('code_hints', [])
        if existing:
            print(f"\n  Existing hints will be KEPT. New hints will be ADDED.")
            print(f"  Current hints:")
            for i, h in enumerate(existing):
                print(f"    {i+1}. {h}")
            keep = read_single("Keep existing hints? [Y/n]", "Y").strip().lower()
            if keep in ("n", "no"):
                instr["clear_existing_hints"] = True

        hints = read_multiline(
            "Enter implementation hints (one per line):\n"
            "  Examples:\n"
            "    Use scanpy.read_h5ad(sys.argv[1]) to load the input\n"
            "    The AnnData object uses obs['cell_type'] not obs['celltype']\n"
            "    Save output to data/outputs/analysis/processed/ not scripts/"
        )
        instr["hints"] = hints

        print(PHASE_MENU)
        cp = read_single("Checkpoint action", "1").strip()
        instr["checkpoint_action"] = {"1": "delete", "2": "keep", "3": "delete_sbatch"}.get(cp, "keep")

    elif choice == "2":
        instr["action"] = "fix_inputs"
        print("\n  Current input_files:", step.get('input_files', []))
        print("  Enter the correct input file paths (absolute or relative to project root):")
        paths = read_multiline("Paths (one per line)")
        instr["input_files"] = paths

        also_hint = read_single("Also add a load hint? [Y/n]", "Y").strip().lower()
        if also_hint not in ("n", "no") and paths:
            first = paths[0]
            lang = step.get('language', 'python')
            if lang == 'r':
                default_hint = f"Load input with: adata <- anndata::read_h5ad(commandArgs(TRUE)[1])"
            else:
                default_hint = f"Load input with: import scanpy as sc; adata = sc.read_h5ad(import sys; sys.argv[1])"
            hint = read_single("Load hint (edit or accept default)", default_hint)
            instr["hints"] = [hint] if hint else []

        print(PHASE_MENU)
        cp = read_single("Checkpoint action", "3").strip()
        instr["checkpoint_action"] = {"1": "delete", "2": "keep", "3": "delete_sbatch"}.get(cp, "delete_sbatch")

    elif choice == "3":
        instr["action"] = "override_approach"
        print("\n  Current expanded_plan (first 400 chars):")
        print(f"  {step.get('expanded_plan', '')[:400]}")
        print("\n  Enter your replacement approach. Be as specific as possible.")
        print("  Include: exact methods, file paths, data structures, expected output format.")
        lines = read_multiline("New approach (one paragraph, Enter=blank line to finish)")
        instr["expanded_plan"] = " ".join(lines)

        print(PHASE_MENU)
        cp = read_single("Checkpoint action", "1").strip()
        instr["checkpoint_action"] = {"1": "delete", "2": "keep", "3": "delete_sbatch"}.get(cp, "delete")

    elif choice == "4":
        instr["action"] = "skip"
        print(f"\n  Step {step_id} will be marked as COMPLETED.")
        print("  Downstream steps that depend on this will be unblocked.")
        confirm = read_single("Confirm skip? [y/N]", "N").strip().lower()
        if confirm not in ("y", "yes"):
            print("  Cancelled — no change for this step.")
            instr["action"] = None

    elif choice == "5":
        instr["action"] = "reset"
        print(PHASE_MENU)
        cp = read_single("Checkpoint action", "2").strip()
        instr["checkpoint_action"] = {"1": "delete", "2": "keep", "3": "delete_sbatch"}.get(cp, "keep")

    else:
        instr["action"] = None
        print(f"  Skipping {step_id} — no changes.")

    instructions.append(instr)

print(json.dumps(instructions, indent=2))
PYEOF

# ---------------------------------------------------------------------------
# Preview and apply changes via Python
# ---------------------------------------------------------------------------

echo ""
echo -e "${BOLD}Applying changes...${NC}"
echo ""

python3 - "$STATE_FILE" "$INSTRUCTIONS_FILE" "$CHECKPOINT_DIR" "$DRY_RUN" <<'PYEOF'
import json, sys, os, shutil
from pathlib import Path
from datetime import datetime

state_file  = sys.argv[1]
instr_file  = sys.argv[2]
cp_dir      = Path(sys.argv[3])
dry_run     = sys.argv[4].lower() == "true"

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

    # -------------------------------------------------------------------------
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
            step_changes.append(f"Added {len(new_hints)} hint(s) (total: {len(step['code_hints'])})")

        # Also append a guidance block to expanded_plan so it reaches
        # the sub-agent even before the sub_agent.py patch is applied
        guidance_block = (
            "\n\n[HUMAN GUIDANCE — FOLLOW EXACTLY]\n" +
            "\n".join(f"- {h}" for h in new_hints)
        )
        plan = step.get('expanded_plan', step.get('description', ''))
        # Remove any previous guidance block before appending fresh
        if '[HUMAN GUIDANCE' in plan:
            plan = plan[:plan.index('[HUMAN GUIDANCE')]
        step['expanded_plan'] = plan.rstrip() + guidance_block
        step_changes.append("Appended guidance block to expanded_plan")

        # Reset status so the router dispatches it
        if step.get('status') in ('failed', 'blocked', 'running'):
            step['status'] = 'pending'
            step['attempts'] = 0
            step_changes.append("Status → pending, attempts → 0")

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    elif action == "override_approach":
        new_plan = instr.get('expanded_plan', '').strip()
        if new_plan:
            step['expanded_plan'] = new_plan
            step['description']   = new_plan
            step_changes.append(f"expanded_plan replaced ({len(new_plan)} chars)")

        if step.get('status') in ('failed', 'blocked', 'running'):
            step['status'] = 'pending'
            step['attempts'] = 0
            step_changes.append("Status → pending, attempts → 0")

    # -------------------------------------------------------------------------
    elif action == "skip":
        step['status'] = 'completed'
        step['error_summary'] = None
        step_changes.append("Status → completed (skipped by human)")

    # -------------------------------------------------------------------------
    elif action == "reset":
        step['status'] = 'pending'
        step['attempts'] = 0
        step['error_summary'] = None
        step_changes.append("Status → pending, attempts → 0")

    # -------------------------------------------------------------------------
    # Checkpoint management
    # -------------------------------------------------------------------------
    cp_action = instr.get('checkpoint_action', 'keep')
    cp_file = cp_dir / f"{step_id}_checkpoint.json"

    if cp_action == "delete" and cp_file.exists():
        step_changes.append(f"Checkpoint DELETED (will restart from Phase 1)")
        if not dry_run:
            cp_file.unlink()

    elif cp_action == "delete_sbatch" and cp_file.exists():
        # Read checkpoint, null out sbatch_path
        try:
            with open(cp_file) as f:
                cp = json.load(f)
            sbatch = cp.get('sbatch_path')
            if sbatch and Path(sbatch).exists():
                step_changes.append(f"Sbatch file deleted: {Path(sbatch).name}")
                if not dry_run:
                    Path(sbatch).unlink()
            cp['sbatch_path'] = None
            step_changes.append("Checkpoint sbatch_path cleared (Phase 3 will regenerate)")
            if not dry_run:
                with open(cp_file, 'w') as f:
                    json.dump(cp, f, indent=2)
        except Exception as e:
            step_changes.append(f"WARNING: Could not update checkpoint: {e}")

    elif cp_action == "keep":
        step_changes.append("Checkpoint kept (will resume from last phase)")

    # -------------------------------------------------------------------------
    # Timestamp
    step['last_updated'] = datetime.now().isoformat()
    steps[step_id] = step
    changes_summary.append((step_id, step_changes))

# -------------------------------------------------------------------------
# Preview
# -------------------------------------------------------------------------
print(f"\n{'─'*60}")
print("  CHANGE PREVIEW")
print(f"{'─'*60}")
for step_id, ch_list in changes_summary:
    step = steps.get(step_id, {})
    print(f"\n  [{step_id}]  →  status={step.get('status')}  attempts={step.get('attempts', 0)}")
    for ch in ch_list:
        print(f"    • {ch}")
    if step.get('code_hints'):
        print(f"    • code_hints ({len(step['code_hints'])}):")
        for h in step['code_hints'][:3]:
            print(f"        "{h[:80]}"")
        if len(step['code_hints']) > 3:
            print(f"        ... and {len(step['code_hints'])-3} more")

print(f"\n{'─'*60}")

if dry_run:
    print("\n  DRY RUN — no files were written.")
    sys.exit(0)

# -------------------------------------------------------------------------
# Write
# -------------------------------------------------------------------------
state['steps'] = steps
state['last_updated'] = datetime.now().isoformat()

# Atomic write: write to temp then rename
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
    echo "  Run PARTIAL_CLEAN_PROJECT.sh first to clear old logs, then:"
    echo -e "  ${CYAN}sbatch RUN_AGI_PIPELINE_GPU.sh${NC}"
fi
echo ""

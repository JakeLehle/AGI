#!/bin/bash
# =============================================================================
# AGI Pipeline - FULL Clean Script (v1.2.5)
# =============================================================================
# Complete wipe of all pipeline artifacts for a fresh-start run.
# This is destructive — all generated state will be lost.
#
# REMOVES:
#   - logs/              Agent logs, Ollama logs, error logs
#   - logs/steps/        Per-step Phase 1/2/3 logs (v1.2.5) — fully wiped
#                        on FULL clean. Use PARTIAL clean to preserve these
#                        for diagnosing failures before resubmit.
#   - temp/checkpoints/  Step checkpoint JSONs
#   - slurm/logs/        SLURM subtask job stdout/stderr
#   - slurm_logs/        Master pipeline job stdout/stderr
#   - slurm/scripts/     Auto-generated sbatch scripts
#   - reports/           Pipeline status report + master_prompt_state.json
#   - prompts/prompt_*   Auto-generated decomposition JSONs (NOT .md files)
#   - envs/step_*.yml    Auto-generated conda env YAML specs
#   - conda envs         All conda environments matching agi_step_* prefix
#   - conda cache        All cached packages, tarballs, and index files
#
# PRESERVES:
#   - prompts/*.md       Original master prompt files (user-authored)
#   - scripts/           All user and example reference scripts
#   - data/              All input and output data
#   - config/            Pipeline configuration
#   - conda_env/         Base environment YAML specs (tracked)
#   - project.yaml, README.md, *.sh
#
# Usage:
#   bash FULL_CLEAN_PROJECT.sh              # interactive
#   bash FULL_CLEAN_PROJECT.sh --dry-run    # preview only, no changes
#   bash FULL_CLEAN_PROJECT.sh --yes        # skip confirmation
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
            echo "Usage: bash FULL_CLEAN_PROJECT.sh [--dry-run] [--yes]"
            echo ""
            echo "  --dry-run   Show what would be deleted without removing anything"
            echo "  --yes, -y   Skip confirmation prompt"
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
echo -e "${RED}============================================================${NC}"
echo -e "${RED}  AGI Pipeline — FULL Clean (v1.2.5)${NC}"
echo -e "${RED}============================================================${NC}"
echo -e "  Project: ${CYAN}${PROJECT_DIR}${NC}"
if [ "$DRY_RUN" = true ]; then
    echo -e "  Mode:    ${YELLOW}DRY RUN — nothing will be changed${NC}"
else
    echo -e "  Mode:    ${RED}DESTRUCTIVE — all generated state will be removed${NC}"
fi
echo ""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOTAL_FILES=0
TOTAL_SIZE=0

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

count_dir() {
    local label="$1"
    local dir="$2"
    if [ -d "$dir" ]; then
        local count size hs
        count=$(find "$dir" -type f 2>/dev/null | wc -l)
        size=$(du -sb "$dir" 2>/dev/null | awk '{print $1}')
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

count_conda_envs() {
    local label="$1"
    local prefix="$2"
    local count=0
    if command -v conda &>/dev/null; then
        while IFS= read -r line; do
            env_name=$(basename "$line")
            if [[ "$env_name" == ${prefix}* ]]; then
                count=$((count + 1))
            fi
        done < <(conda env list 2>/dev/null | grep -v '^#' | grep -v '^$' | awk '{print $NF}')
    fi
    if [ "$count" -gt 0 ]; then
        echo -e "  ${YELLOW}●${NC} ${label}: ${count} environment(s)"
        TOTAL_FILES=$((TOTAL_FILES + count))
    else
        echo -e "  ${GREEN}○${NC} ${label}: none found"
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

remove_dir_contents() {
    local label="$1"
    local dir="$2"
    if [ -d "$dir" ]; then
        local count
        count=$(find "$dir" -type f 2>/dev/null | wc -l)
        if [ "$count" -gt 0 ]; then
            if [ "$DRY_RUN" = false ]; then
                find "$dir" -type f -delete
                # Remove empty subdirs but keep the dir itself
                find "$dir" -mindepth 1 -type d -empty -delete 2>/dev/null || true
            fi
            echo -e "  ${GREEN}✓${NC} Cleared ${count} file(s): ${label}"
        fi
    fi
}

# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

echo "Scanning artifacts..."
echo ""

count_dir  "Agent/Ollama logs       (logs/ excl. steps/)"          "logs"
count_dir  "Phase logs — WIPED      (logs/steps/)"                 "logs/steps"
count_dir  "SLURM subtask logs      (slurm/logs/)"                 "slurm/logs"
count_dir  "SLURM master logs       (slurm_logs/)"                 "slurm_logs"
count_glob "Checkpoints             (temp/checkpoints/)"           "./temp/checkpoints/step_*_checkpoint.json"
count_dir  "Generated sbatch        (slurm/scripts/)"              "slurm/scripts"
count_glob "Reports                 (reports/)"                    "./reports/master_prompt_state.json" "./reports/pipeline_status.md"
count_glob "Generated prompt JSONs  (prompts/prompt_*.json)"       "./prompts/prompt_*.json"
count_glob "Step conda env YAMLs    (envs/step_*.yml)"             "./envs/step_*.yml"
count_conda_envs "Conda envs (agi_step_*)" "agi_step_"

echo ""

if [ "$TOTAL_FILES" -eq 0 ]; then
    echo -e "${GREEN}Project is already clean — nothing to remove.${NC}"
    exit 0
fi

TOTAL_HUMAN=$(numfmt --to=iec --suffix=B "$TOTAL_SIZE" 2>/dev/null || echo "${TOTAL_SIZE} bytes")
echo -e "  Total: ${YELLOW}${TOTAL_FILES} items${NC} (${TOTAL_HUMAN} on disk, plus conda envs and cache)"
echo ""

echo -e "${GREEN}Preserved (will NOT be touched):${NC}"
echo "  ● prompts/*.md          (master prompt files)"
echo "  ● scripts/              (all scripts)"
echo "  ● data/                 (inputs & outputs)"
echo "  ● config/               (pipeline configuration)"
echo "  ● conda_env/            (base environment YAMLs)"
echo "  ● project.yaml, README.md, *.sh"
echo ""
echo -e "${YELLOW}Also removing (not in file count above):${NC}"
echo "  ● All conda envs matching agi_step_* prefix"
echo "  ● conda package cache, tarballs, and index files"
echo ""

# ---------------------------------------------------------------------------
# Confirm
# ---------------------------------------------------------------------------

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}Dry run complete — no files were changed.${NC}"
    exit 0
fi

if [ "$AUTO_YES" = false ]; then
    echo -e "${RED}WARNING: This cannot be undone. All pipeline state will be lost.${NC}"
    read -p "Proceed with FULL clean? [y/N] " -n 1 -r
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
# File removal
# ---------------------------------------------------------------------------

# Logs
remove_dir_contents  "agent + ollama logs"      "logs"
remove_dir_contents  "SLURM subtask logs"        "slurm/logs"
remove_dir_contents  "SLURM master logs"         "slurm_logs"

# Checkpoints
remove_glob          "step checkpoints"          "./temp/checkpoints/step_*_checkpoint.json"

# Generated sbatch scripts
remove_dir_contents  "generated sbatch scripts"  "slurm/scripts"

# Reports
remove_glob          "pipeline status report"    "./reports/pipeline_status.md"
remove_glob          "master prompt state"       "./reports/master_prompt_state.json"

# Generated prompt JSONs (preserve .md files)
remove_glob          "generated prompt JSONs"    "./prompts/prompt_*.json"

# Auto-generated step env YAML specs
remove_glob          "step conda env YAMLs"      "./envs/step_*.yml"

# ---------------------------------------------------------------------------
# Conda environment removal
# ---------------------------------------------------------------------------

echo ""
echo "Removing conda environments (agi_step_*)..."

if command -v conda &>/dev/null; then
    ENV_COUNT=0
    while IFS= read -r line; do
        # conda env list output: "name   /path/to/env" or "* name   /path"
        # Get the last field (the path) or parse the name
        env_path=$(echo "$line" | awk '{print $NF}')
        env_name=$(basename "$env_path")

        if [[ "$env_name" == agi_step_* ]]; then
            echo -e "  ${YELLOW}→${NC} Removing: ${env_name}"
            conda env remove -n "$env_name" -y --quiet 2>/dev/null \
                && echo -e "  ${GREEN}✓${NC} Removed: ${env_name}" \
                || echo -e "  ${RED}✗${NC} Failed to remove: ${env_name} (may not exist or name mismatch)"
            ENV_COUNT=$((ENV_COUNT + 1))
        fi
    done < <(conda env list 2>/dev/null | grep -v '^#' | grep -v '^$')

    if [ "$ENV_COUNT" -eq 0 ]; then
        echo -e "  ${GREEN}○${NC} No agi_step_* environments found"
    fi
else
    echo -e "  ${YELLOW}⚠${NC} conda not found in PATH — skipping env removal"
    echo "     Activate your conda base environment and re-run to remove envs"
fi

# ---------------------------------------------------------------------------
# Conda cache purge
# ---------------------------------------------------------------------------

echo ""
echo "Purging conda package cache..."

if command -v conda &>/dev/null; then
    # Capture size before for reporting
    CACHE_BEFORE=0
    CONDA_PKGS_DIR=$(conda info --json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('pkgs_dirs',[''])[0])" 2>/dev/null || echo "")
    if [ -d "$CONDA_PKGS_DIR" ]; then
        CACHE_BEFORE=$(du -sb "$CONDA_PKGS_DIR" 2>/dev/null | awk '{print $1}')
    fi

    conda clean --all -y --quiet 2>/dev/null \
        && echo -e "  ${GREEN}✓${NC} conda cache cleared" \
        || echo -e "  ${YELLOW}⚠${NC} conda clean encountered warnings (non-fatal)"

    if [ "$CACHE_BEFORE" -gt 0 ]; then
        CACHE_HUMAN=$(numfmt --to=iec --suffix=B "$CACHE_BEFORE" 2>/dev/null || echo "${CACHE_BEFORE}B")
        echo -e "  ${GREEN}✓${NC} Freed approximately ${CACHE_HUMAN} of cached packages"
    fi
else
    echo -e "  ${YELLOW}⚠${NC} conda not found in PATH — skipping cache purge"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  FULL clean complete. Project is ready for a fresh run.${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "  Next: edit your master prompt if needed, then resubmit:"
echo -e "  ${CYAN}sbatch RUN_AGI_PIPELINE_GPU.sh${NC}"
echo ""

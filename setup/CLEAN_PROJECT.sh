#!/bin/bash
# =============================================================================
# AGI Pipeline - Project Cleanup Script
# =============================================================================
# Removes stale pipeline artifacts to prepare for a fresh run.
#
# REMOVES:
#   - logs/              Agent logs, Ollama logs, error logs
#   - temp/checkpoints/  Step checkpoint JSONs
#   - slurm_logs/        SLURM job stdout/stderr
#   - slurm/logs/        SLURM log directory
#   - reports/           Generated pipeline reports and master prompt state
#   - prompts/prompt_*   Auto-generated decomposition JSONs
#   - envs/step_*.yml    Auto-generated conda env specs
#   - slurm/scripts/     Auto-generated sbatch scripts
#
# PRESERVES:
#   - prompts/*.md       Original master prompt files (user-authored)
#   - scripts/           All user and example reference scripts
#   - data/              All input and output data (user manages separately)
#   - config/            Pipeline configuration
#   - *.sh, *.yaml       Project-level config and run scripts
#   - README.md, project.yaml
#
# Usage:
#   # From project directory:
#   bash /path/to/AGI/setup/clean_project.sh
#
#   # Or copy to project and run:
#   bash clean_project.sh
#
#   # Dry run (show what would be deleted):
#   bash clean_project.sh --dry-run
#
#   # Skip confirmation prompt:
#   bash clean_project.sh --yes
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

DRY_RUN=false
AUTO_YES=false

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --yes|-y) AUTO_YES=true ;;
        --help|-h)
            echo "Usage: bash clean_project.sh [--dry-run] [--yes]"
            echo ""
            echo "  --dry-run   Show what would be deleted without removing anything"
            echo "  --yes, -y   Skip confirmation prompt"
            echo ""
            exit 0
            ;;
    esac
done

# Verify we're in a project directory (look for telltale signs)
if [ ! -f "project.yaml" ] && [ ! -d "temp/checkpoints" ] && [ ! -d "logs" ]; then
    echo -e "${RED}ERROR: This doesn't look like an AGI project directory.${NC}"
    echo "  Expected to find project.yaml, temp/checkpoints/, or logs/"
    echo "  Run this script from your project root (e.g. slide-TCR-seq-working/)"
    exit 1
fi

PROJECT_DIR="$(pwd)"
echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  AGI Pipeline - Project Cleanup${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "  Project: ${PROJECT_DIR}"
if [ "$DRY_RUN" = true ]; then
    echo -e "  Mode:    ${YELLOW}DRY RUN (nothing will be deleted)${NC}"
fi
echo ""

# ============================================================================
# Inventory what will be cleaned
# ============================================================================

TOTAL_FILES=0
TOTAL_SIZE=0

count_files() {
    local pattern="$1"
    local label="$2"
    local count=0
    local size=0

    if compgen -G "$pattern" > /dev/null 2>&1; then
        count=$(find $pattern -maxdepth 0 -type f 2>/dev/null | wc -l)
        size=$(du -sb $pattern 2>/dev/null | awk '{s+=$1} END {print s+0}')
    fi

    if [ "$count" -gt 0 ]; then
        local human_size=$(numfmt --to=iec --suffix=B "$size" 2>/dev/null || echo "${size}B")
        echo -e "  ${YELLOW}●${NC} ${label}: ${count} files (${human_size})"
        TOTAL_FILES=$((TOTAL_FILES + count))
        TOTAL_SIZE=$((TOTAL_SIZE + size))
    else
        echo -e "  ${GREEN}○${NC} ${label}: clean"
    fi
}

echo "Scanning for stale artifacts..."
echo ""

# --- Logs ---
count_files "logs/agent_*.jsonl" "Agent logs (logs/agent_*.jsonl)"
count_files "logs/errors.jsonl" "Error log (logs/errors.jsonl)"
count_files "logs/ollama_*.log" "Ollama logs (logs/ollama_*.log)"

# --- Checkpoints ---
count_files "temp/checkpoints/step_*_checkpoint.json" "Checkpoints (temp/checkpoints/)"

# --- SLURM logs ---
count_files "slurm_logs/agi_*.out" "SLURM stdout (slurm_logs/agi_*.out)"
count_files "slurm_logs/agi_*.err" "SLURM stderr (slurm_logs/agi_*.err)"
count_files "slurm/logs/*" "SLURM logs (slurm/logs/)"

# --- Reports ---
count_files "reports/pipeline_status.md" "Pipeline status report"
count_files "reports/master_prompt_state.json" "Master prompt state"

# --- Auto-generated prompts (NOT .md master prompts) ---
count_files "prompts/prompt_*.json" "Generated prompt JSONs (prompts/prompt_*.json)"

# --- Auto-generated env specs ---
count_files "envs/step_*.yml" "Step env specs (envs/step_*.yml)"

# --- Auto-generated SLURM scripts ---
if [ -d "slurm/scripts" ]; then
    count_files "slurm/scripts/*" "Generated sbatch scripts (slurm/scripts/)"
fi

echo ""

if [ "$TOTAL_FILES" -eq 0 ]; then
    echo -e "${GREEN}Project is already clean. Nothing to remove.${NC}"
    exit 0
fi

TOTAL_HUMAN=$(numfmt --to=iec --suffix=B "$TOTAL_SIZE" 2>/dev/null || echo "${TOTAL_SIZE} bytes")
echo -e "  Total: ${YELLOW}${TOTAL_FILES} files${NC} (${TOTAL_HUMAN})"
echo ""

# --- Show what's preserved ---
echo -e "${GREEN}Preserved (will NOT be touched):${NC}"
echo "  ● prompts/*.md          (master prompt files)"
echo "  ● scripts/              (all user & example scripts)"
echo "  ● data/                 (inputs & outputs)"
echo "  ● config/               (pipeline configuration)"
echo "  ● project.yaml, README.md, *.sh"
echo ""

# ============================================================================
# Confirm and execute
# ============================================================================

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}Dry run complete. No files were deleted.${NC}"
    exit 0
fi

if [ "$AUTO_YES" = false ]; then
    read -p "Remove these ${TOTAL_FILES} files? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
fi

echo "Cleaning..."

# Helper: remove glob if exists
remove_glob() {
    local pattern="$1"
    local label="$2"
    local count=0
    for f in $pattern; do
        [ -e "$f" ] || continue
        rm -f "$f"
        count=$((count + 1))
    done
    if [ "$count" -gt 0 ]; then
        echo -e "  ${GREEN}✓${NC} Removed ${count} files: ${label}"
    fi
}

# Logs
remove_glob "logs/agent_*.jsonl" "agent logs"
remove_glob "logs/errors.jsonl" "error log"
remove_glob "logs/ollama_*.log" "ollama logs"

# Checkpoints
remove_glob "temp/checkpoints/step_*_checkpoint.json" "checkpoints"

# SLURM logs
remove_glob "slurm_logs/agi_*.out" "slurm stdout"
remove_glob "slurm_logs/agi_*.err" "slurm stderr"
if [ -d "slurm/logs" ]; then
    remove_glob "slurm/logs/*" "slurm/logs"
fi

# Reports
remove_glob "reports/pipeline_status.md" "pipeline status"
remove_glob "reports/master_prompt_state.json" "master prompt state"

# Auto-generated prompts (preserve .md files)
remove_glob "prompts/prompt_*.json" "generated prompt JSONs"

# Auto-generated env specs
remove_glob "envs/step_*.yml" "step env specs"

# Auto-generated SLURM scripts
if [ -d "slurm/scripts" ]; then
    remove_glob "slurm/scripts/*" "generated sbatch scripts"
fi

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  Cleanup complete. Project is ready for a fresh run.${NC}"
echo -e "${GREEN}============================================================${NC}"

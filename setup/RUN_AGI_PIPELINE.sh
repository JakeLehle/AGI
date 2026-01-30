#!/bin/bash
#SBATCH --job-name=slide-TCR-seq
#SBATCH --partition=normal
#SBATCH --account=jlehle
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm_logs/agi_%j.out
#SBATCH --error=slurm_logs/agi_%j.err

###############################################################################
# AGI Multi-Agent Pipeline - SLURM Submission Script
# 
# Usage:
#   sbatch run_agi_pipeline.sbatch
#
# Or with custom parameters:
#   sbatch --export=PROMPT_FILE=my_prompt.txt,PROJECT_DIR=my_project run_agi_pipeline.sbatch
#
# Configuration:
#   - Edit the CONFIGURATION section below for defaults
#   - Or pass environment variables via --export
###############################################################################

# ============================================================================
# CONFIGURATION - Edit these for your job
# ============================================================================

# Prompt file (relative to AGI_ROOT or absolute path)
PROMPT_FILE="/master/jlehle/WORKING/slide-TCR-seq-working/prompts/2026-01-29_prompt_1.md"

# Project directory (will be created if doesn't exist)
PROJECT_DIR="/master/jlehle/WORKING/slide-TCR-seq-working"

# Ollama model to use
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1:70b}"

# Maximum retries per subtask (enforced limit: 12)
MAX_RETRIES="${MAX_RETRIES:-12}"

# AGI repository root directory
AGI_ROOT="/master/jlehle/WORKING/AGI"

# Conda environment name
CONDA_ENV="${CONDA_ENV:-AGI}"

# ============================================================================
# VALIDATION
# ============================================================================

# Ensure PROJECT_DIR is specified
if [ -z "${PROJECT_DIR}" ]; then
    echo "ERROR: PROJECT_DIR must be specified"
    echo ""
    echo "Usage:"
    echo "  sbatch --export=PROJECT_DIR=/path/to/your/project run_agi_pipeline.sbatch"
    echo ""
    echo "Or set it in the script's CONFIGURATION section"
    exit 1
fi

# Resolve to absolute path
PROJECT_DIR=$(realpath "${PROJECT_DIR}")

# ============================================================================
# SETUP
# ============================================================================

echo "============================================================================"
echo "  AGI Multi-Agent Pipeline - Job Started"
echo "============================================================================"
echo "  Job ID:          ${SLURM_JOB_ID}"
echo "  Node:            $(hostname)"
echo "  Start Time:      $(date)"
echo "  AGI Root:        ${AGI_ROOT}"
echo "  Project Dir:     ${PROJECT_DIR}"
echo "  Prompt File:     ${PROMPT_FILE}"
echo "  Model:           ${OLLAMA_MODEL}"
echo "============================================================================"

# Validate directories exist
if [ ! -d "${AGI_ROOT}" ]; then
    echo "ERROR: AGI_ROOT does not exist: ${AGI_ROOT}"
    exit 1
fi

if [ ! -d "${PROJECT_DIR}" ]; then
    echo "ERROR: PROJECT_DIR does not exist: ${PROJECT_DIR}"
    echo "Run setup.sh in your project directory first."
    exit 1
fi

# Create SLURM logs directory in PROJECT (not AGI_ROOT)
mkdir -p "${PROJECT_DIR}/slurm/logs"

# ============================================================================
# LOAD ENVIRONMENT
# ============================================================================

echo ""
echo ">>> Loading modules..."

# Try to load git module (optional - for git tracking)
if module avail git 2>&1 | grep -q git; then
    module load git
    echo "    Loaded git module"
elif command -v git &> /dev/null; then
    echo "    git already in PATH: $(which git)"
else
    echo "    Note: git not available - git tracking will be disabled"
fi

# Try to load anaconda - adjust module name for your system
if module avail anaconda3 2>&1 | grep -q anaconda3; then
    module load anaconda3
    echo "    Loaded anaconda3 module"
elif [ -f "$HOME/anaconda3/bin/activate" ]; then
    source "$HOME/anaconda3/bin/activate"
    echo "    Sourced anaconda3 from HOME"
else
    echo "    Warning: No anaconda/conda module found, assuming conda is in PATH"
fi

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Activate the AGI environment
echo ">>> Activating conda environment: ${CONDA_ENV}"
conda activate "${CONDA_ENV}" || { echo "ERROR: Cannot activate conda env ${CONDA_ENV}"; exit 1; }

echo ">>> Python: $(which python)"
echo ">>> Python version: $(python --version)"

# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================

echo ""
echo ">>> Setting environment variables..."

# Ollama configuration
export OLLAMA_HOST="http://127.0.0.1:11434"
export OLLAMA_MODELS="${HOME}/.ollama/models"
export OLLAMA_DEBUG="INFO"
export OLLAMA_KEEP_ALIVE="10m"
export OLLAMA_CONTEXT_LENGTH="4096"

# Suppress GitPython errors if git isn't available
export GIT_PYTHON_REFRESH="quiet"

# Add AGI_ROOT to Python path so imports work
export PYTHONPATH="${AGI_ROOT}:${PYTHONPATH}"

# ============================================================================
# START OLLAMA SERVER
# ============================================================================

echo ""
echo ">>> Starting Ollama server..."

# Store Ollama logs in PROJECT directory
OLLAMA_LOG="${PROJECT_DIR}/logs/ollama_${SLURM_JOB_ID}.log"

if curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
    echo "    Ollama is already running"
else
    # Start Ollama server in background
    ollama serve > "${OLLAMA_LOG}" 2>&1 &
    OLLAMA_PID=$!
    echo "    Ollama started with PID: ${OLLAMA_PID}"
    echo "    Log: ${OLLAMA_LOG}"
    
    # Wait for Ollama to be ready
    echo "    Waiting for Ollama to initialize..."
    MAX_WAIT=60
    WAITED=0
    while ! curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; do
        sleep 2
        WAITED=$((WAITED + 2))
        if [ $WAITED -ge $MAX_WAIT ]; then
            echo "ERROR: Ollama failed to start within ${MAX_WAIT} seconds"
            echo "Check log: ${OLLAMA_LOG}"
            exit 1
        fi
    done
    echo "    Ollama is ready (waited ${WAITED}s)"
fi

# Verify model is available
echo ">>> Checking for model: ${OLLAMA_MODEL}"
if ollama list 2>/dev/null | grep -q "${OLLAMA_MODEL%%:*}"; then
    echo "    Model found"
else
    echo "    Model not found, pulling ${OLLAMA_MODEL}..."
    ollama pull "${OLLAMA_MODEL}"
fi

# ============================================================================
# RESOLVE PROMPT FILE PATH
# ============================================================================

# If PROMPT_FILE is not absolute, look for it in PROJECT_DIR
if [[ "${PROMPT_FILE}" != /* ]]; then
    if [ -f "${PROJECT_DIR}/${PROMPT_FILE}" ]; then
        PROMPT_FILE="${PROJECT_DIR}/${PROMPT_FILE}"
    elif [ -f "${AGI_ROOT}/${PROMPT_FILE}" ]; then
        # Fallback to AGI_ROOT for shared prompts
        PROMPT_FILE="${AGI_ROOT}/${PROMPT_FILE}"
    else
        echo "ERROR: Prompt file not found: ${PROMPT_FILE}"
        echo "Searched in:"
        echo "  - ${PROJECT_DIR}/${PROMPT_FILE}"
        echo "  - ${AGI_ROOT}/${PROMPT_FILE}"
        exit 1
    fi
fi

echo ""
echo ">>> Resolved prompt file: ${PROMPT_FILE}"

# ============================================================================
# RUN THE PIPELINE
# ============================================================================

echo ""
echo "============================================================================"
echo "  Running AGI Pipeline"
echo "============================================================================"
echo "  Working Directory:  ${AGI_ROOT}"
echo "  Project Directory:  ${PROJECT_DIR}"
echo "  Prompt File:        ${PROMPT_FILE}"
echo "  Model:              ${OLLAMA_MODEL}"
echo "  Max Retries:        ${MAX_RETRIES}"
echo "============================================================================"
echo ""
echo "  All output (logs, reports, data) will go to:"
echo "    ${PROJECT_DIR}/"
echo ""
echo "============================================================================"

# Change to AGI_ROOT to run Python (imports work correctly)
cd "${AGI_ROOT}" || { echo "ERROR: Cannot cd to ${AGI_ROOT}"; exit 1; }

# Run the pipeline
# KEY: --project-dir points to the external project directory
# This ensures all output goes there, not to AGI_ROOT
python main.py \
    --prompt-file "${PROMPT_FILE}" \
    --project-dir "${PROJECT_DIR}" \
    --model "${OLLAMA_MODEL}" \
    --max-iterations "${MAX_RETRIES}"

PIPELINE_EXIT_CODE=$?

# ============================================================================
# CLEANUP
# ============================================================================

echo ""
echo ">>> Pipeline finished with exit code: ${PIPELINE_EXIT_CODE}"

# Stop Ollama if we started it
if [ -n "${OLLAMA_PID}" ]; then
    echo ">>> Stopping Ollama server (PID: ${OLLAMA_PID})..."
    kill "${OLLAMA_PID}" 2>/dev/null || true
fi

# ============================================================================
# COPY SLURM OUTPUT TO PROJECT
# ============================================================================

# The SLURM output files go to AGI_ROOT/slurm_logs by default (see #SBATCH directives)
# Copy them to the project directory for completeness
echo ""
echo ">>> Copying SLURM output to project directory..."

SLURM_OUT="${AGI_ROOT}/slurm_logs/agi_${SLURM_JOB_ID}.out"
SLURM_ERR="${AGI_ROOT}/slurm_logs/agi_${SLURM_JOB_ID}.err"

if [ -f "${SLURM_OUT}" ]; then
    cp "${SLURM_OUT}" "${PROJECT_DIR}/slurm/logs/"
fi
if [ -f "${SLURM_ERR}" ]; then
    cp "${SLURM_ERR}" "${PROJECT_DIR}/slurm/logs/"
fi

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "============================================================================"
echo "  Job Complete"
echo "============================================================================"
echo "  End Time:       $(date)"
echo "  Exit Code:      ${PIPELINE_EXIT_CODE}"
echo ""
echo "  Project outputs are in: ${PROJECT_DIR}/"
echo "    ├── logs/           - Execution logs"
echo "    ├── reports/        - Generated reports"
echo "    ├── data/outputs/   - Output data files"
echo "    └── slurm/logs/     - SLURM job output"
echo ""
echo "  AGI code remains in: ${AGI_ROOT}/ (unchanged)"
echo "============================================================================"

exit ${PIPELINE_EXIT_CODE}


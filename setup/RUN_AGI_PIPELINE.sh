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
# Updated for v3 with Reflexion Memory support
#
# Usage:
#   sbatch RUN_AGI_PIPELINE.sh
#
# Or with custom parameters:
#   sbatch --export=PROMPT_FILE=my_prompt.txt,PROJECT_DIR=my_project RUN_AGI_PIPELINE.sh
#
# Configuration:
#   - Edit the CONFIGURATION section below for defaults
#   - Or pass environment variables via --export
###############################################################################

# ============================================================================
# CONFIGURATION - Edit these for your job
# ============================================================================

# Prompt file (relative to AGI_ROOT or absolute path)
PROMPT_FILE="${PROMPT_FILE:-/master/jlehle/WORKING/slide-TCR-seq-working/prompts/2026-01-29_prompt_1.md}"

# Project directory (will be created if doesn't exist)
PROJECT_DIR="${PROJECT_DIR:-/master/jlehle/WORKING/slide-TCR-seq-working}"

# Ollama model to use
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1:70b}"

# Embedding model for reflexion memory
EMBEDDING_MODEL="${EMBEDDING_MODEL:-nomic-embed-text}"

# =============================================================================
# TOKEN-BASED CONTEXT LIMITS (v3 Architecture)
# =============================================================================
# Each subtask gets its own context window that persists across ALL retries.
# The agent continues working on a subtask until:
#   1. Success
#   2. Context window exhausted (token limit reached)
#   3. Reflexion engine escalates (semantic duplicate or threshold hit)
#
# This replaces the old iteration-based limit with intelligent context management.
# The agent can try as many approaches as fit in the context, while reflexion
# memory prevents repeating semantically similar approaches.
# =============================================================================

# Maximum tokens per subtask context (default: 60K to leave room for response)
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-60000}"

# Maximum tokens for tool output before summarization
MAX_TOOL_OUTPUT_TOKENS="${MAX_TOOL_OUTPUT_TOKENS:-25000}"

# Minimum tokens remaining to continue (below this, force completion/escalation)
MIN_TOKENS_TO_CONTINUE="${MIN_TOKENS_TO_CONTINUE:-5000}"

# AGI repository root directory
AGI_ROOT="${AGI_ROOT:-/master/jlehle/WORKING/AGI}"

# AGI data directory (for Qdrant storage, memory history)
AGI_DATA_DIR="${AGI_DATA_DIR:-/master/jlehle/agi_data}"

# Conda environment name
CONDA_ENV="${CONDA_ENV:-AGI}"

# Enable reflexion memory (set to "false" to disable)
USE_REFLEXION_MEMORY="${USE_REFLEXION_MEMORY:-true}"

# ============================================================================
# VALIDATION
# ============================================================================

# Ensure PROJECT_DIR is specified
if [ -z "${PROJECT_DIR}" ]; then
    echo "ERROR: PROJECT_DIR must be specified"
    echo ""
    echo "Usage:"
    echo "  sbatch --export=PROJECT_DIR=/path/to/your/project RUN_AGI_PIPELINE.sh"
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
echo "  AGI Multi-Agent Pipeline v3 - Job Started"
echo "============================================================================"
echo "  Job ID:              ${SLURM_JOB_ID}"
echo "  Node:                $(hostname)"
echo "  Start Time:          $(date)"
echo "  AGI Root:            ${AGI_ROOT}"
echo "  AGI Data Dir:        ${AGI_DATA_DIR}"
echo "  Project Dir:         ${PROJECT_DIR}"
echo "  Prompt File:         ${PROMPT_FILE}"
echo "  Model:               ${OLLAMA_MODEL}"
echo "  Embedding Model:     ${EMBEDDING_MODEL}"
echo "  Reflexion Memory:    ${USE_REFLEXION_MEMORY}"
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

# Create required directories
mkdir -p "${PROJECT_DIR}/slurm/logs"
mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${AGI_DATA_DIR}/qdrant_storage"

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
export OLLAMA_BASE_URL="http://127.0.0.1:11434"  # For Mem0 config
export OLLAMA_MODELS="${HOME}/.ollama/models"
export OLLAMA_DEBUG="INFO"
export OLLAMA_KEEP_ALIVE="10m"
export OLLAMA_CONTEXT_LENGTH="4096"

# AGI data directory for reflexion memory
export AGI_DATA_DIR="${AGI_DATA_DIR}"

# Suppress GitPython errors if git isn't available
export GIT_PYTHON_REFRESH="quiet"

# Add AGI_ROOT to Python path so imports work
export PYTHONPATH="${AGI_ROOT}:${PYTHONPATH}"

echo "    OLLAMA_HOST=${OLLAMA_HOST}"
echo "    OLLAMA_BASE_URL=${OLLAMA_BASE_URL}"
echo "    AGI_DATA_DIR=${AGI_DATA_DIR}"
echo "    PYTHONPATH includes ${AGI_ROOT}"

# Token-based context limits (v3)
export AGI_MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS}"
export AGI_MAX_TOOL_OUTPUT_TOKENS="${MAX_TOOL_OUTPUT_TOKENS}"
export AGI_MIN_TOKENS_TO_CONTINUE="${MIN_TOKENS_TO_CONTINUE}"

echo "    AGI_MAX_CONTEXT_TOKENS=${AGI_MAX_CONTEXT_TOKENS}"
echo "    AGI_MAX_TOOL_OUTPUT_TOKENS=${AGI_MAX_TOOL_OUTPUT_TOKENS}"
echo "    AGI_MIN_TOKENS_TO_CONTINUE=${AGI_MIN_TOKENS_TO_CONTINUE}"

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

# ============================================================================
# VERIFY/PULL MODELS
# ============================================================================

echo ""
echo ">>> Checking for required models..."

# Check main LLM model
echo "    Checking: ${OLLAMA_MODEL}"
if ollama list 2>/dev/null | grep -q "${OLLAMA_MODEL%%:*}"; then
    echo "    ✓ ${OLLAMA_MODEL} found"
else
    echo "    Pulling ${OLLAMA_MODEL}..."
    ollama pull "${OLLAMA_MODEL}"
fi

# Check embedding model (required for reflexion memory)
if [ "${USE_REFLEXION_MEMORY}" = "true" ]; then
    echo "    Checking: ${EMBEDDING_MODEL} (for reflexion memory)"
    if ollama list 2>/dev/null | grep -q "${EMBEDDING_MODEL}"; then
        echo "    ✓ ${EMBEDDING_MODEL} found"
    else
        echo "    Pulling ${EMBEDDING_MODEL}..."
        ollama pull "${EMBEDDING_MODEL}"
    fi
fi

# ============================================================================
# VERIFY REFLEXION MEMORY SETUP
# ============================================================================

if [ "${USE_REFLEXION_MEMORY}" = "true" ]; then
    echo ""
    echo ">>> Verifying Reflexion Memory setup..."
    
    # Check Qdrant storage directory
    if [ -d "${AGI_DATA_DIR}/qdrant_storage" ]; then
        echo "    ✓ Qdrant storage directory exists"
    else
        mkdir -p "${AGI_DATA_DIR}/qdrant_storage"
        echo "    Created Qdrant storage directory"
    fi
    
    # Quick Python check for memory module
    python -c "
from memory import ReflexionMemory
from engines import ReflexionEngine
print('    ✓ Reflexion Memory modules loaded successfully')
" 2>&1 || {
        echo "    ⚠ Warning: Reflexion Memory modules failed to load"
        echo "    Pipeline will continue but memory features may be disabled"
    }
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
# CLEAR OLD TEST DATA (Optional - uncomment if needed)
# ============================================================================

# Uncomment to clear reflexion memory from previous failed runs
# echo ""
# echo ">>> Clearing old reflexion memory..."
# python -c "
# from memory import ReflexionMemory
# m = ReflexionMemory()
# m.reset_all(confirm=True)
# print('    Memory cleared')
# " 2>/dev/null || echo "    (no memory to clear)"

# ============================================================================
# RUN THE PIPELINE
# ============================================================================

echo ""
echo "============================================================================"
echo "  Running AGI Pipeline v3"
echo "============================================================================"
echo "  Working Directory:       ${AGI_ROOT}"
echo "  Project Directory:       ${PROJECT_DIR}"
echo "  Prompt File:             ${PROMPT_FILE}"
echo "  Model:                   ${OLLAMA_MODEL}"
echo "  Reflexion Memory:        ${USE_REFLEXION_MEMORY}"
echo ""
echo "  Context Limits (per subtask):"
echo "    Max Context Tokens:    ${MAX_CONTEXT_TOKENS}"
echo "    Max Tool Output:       ${MAX_TOOL_OUTPUT_TOKENS}"
echo "    Min Tokens to Continue:${MIN_TOKENS_TO_CONTINUE}"
echo "============================================================================"
echo ""
echo "  Each subtask gets a fresh context window that persists across retries."
echo "  The agent works until success, context exhaustion, or escalation."
echo "  Reflexion memory prevents repeating semantically similar approaches."
echo ""
echo "  All output (logs, reports, data) will go to:"
echo "    ${PROJECT_DIR}/"
echo ""
echo "  Reflexion memory stored in:"
echo "    ${AGI_DATA_DIR}/qdrant_storage/"
echo ""
echo "============================================================================"

# Change to AGI_ROOT to run Python (imports work correctly)
cd "${AGI_ROOT}" || { echo "ERROR: Cannot cd to ${AGI_ROOT}"; exit 1; }

# Run the pipeline
# v3: Token-based context limits instead of iteration counts
# Each subtask gets its own persistent context window
# Environment variables AGI_MAX_CONTEXT_TOKENS etc. are read by the workflow
python main.py \
    --prompt-file "${PROMPT_FILE}" \
    --project-dir "${PROJECT_DIR}" \
    --model "${OLLAMA_MODEL}"

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
echo "  Reflexion memory persists in: ${AGI_DATA_DIR}/"
echo "    └── qdrant_storage/ - Vector database"
echo ""
echo "  AGI code remains in: ${AGI_ROOT}/ (unchanged)"
echo "============================================================================"

exit ${PIPELINE_EXIT_CODE}

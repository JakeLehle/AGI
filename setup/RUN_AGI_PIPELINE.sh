#!/bin/bash
#SBATCH --job-name=agi_pipeline
#SBATCH --partition=compute2
#SBATCH --account=sdz852
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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
PROMPT_FILE="${PROMPT_FILE:-example_prompt.txt}"

# Project directory (will be created if doesn't exist)
PROJECT_DIR="${PROJECT_DIR:-pipeline_run_$(date +%Y%m%d_%H%M%S)}"

# Ollama model to use
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1:8b}"

# Maximum retries per subtask (enforced limit: 12)
MAX_RETRIES="${MAX_RETRIES:-12}"

# AGI repository root directory
AGI_ROOT="${AGI_ROOT:-/work/sdz852/WORKING/AGI}"

# Conda environment name
CONDA_ENV="${CONDA_ENV:-AGI}"

# ============================================================================
# SETUP - Don't edit below unless you know what you're doing
# ============================================================================

echo "============================================================================"
echo "  AGI Multi-Agent Pipeline - Job Started"
echo "============================================================================"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Node: $(hostname)"
echo "  Start Time: $(date)"
echo "  Working Directory: ${AGI_ROOT}"
echo "============================================================================"

# Create slurm logs directory if it doesn't exist
mkdir -p "${AGI_ROOT}/slurm_logs"

# Change to AGI root directory
cd "${AGI_ROOT}" || { echo "ERROR: Cannot cd to ${AGI_ROOT}"; exit 1; }

# Load required modules (adjust for your HPC system)
echo ""
echo ">>> Loading modules..."

# Try to load anaconda - adjust module name for your system
if module avail anaconda 2>&1 | grep -q anaconda; then
    module load anaconda
elif [ -f "$HOME/anaconda3/bin/activate" ]; then
    source "$HOME/anaconda3/bin/activate"
else
    echo "Warning: No anaconda/conda module found, assuming conda is in PATH"
fi

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Activate the AGI environment
echo ">>> Activating conda environment: ${CONDA_ENV}"
conda activate "${CONDA_ENV}" || { echo "ERROR: Cannot activate conda env ${CONDA_ENV}"; exit 1; }

# Verify Python
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

# Disable any GPU for CPU-only node (remove if using GPU partition)
# export CUDA_VISIBLE_DEVICES=""

# Python path
export PYTHONPATH="${AGI_ROOT}:${PYTHONPATH}"

# ============================================================================
# START OLLAMA SERVER
# ============================================================================

echo ""
echo ">>> Starting Ollama server..."

# Check if Ollama is already running
if curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
    echo "    Ollama is already running"
else
    # Start Ollama server in background
    ollama serve > "${AGI_ROOT}/slurm_logs/ollama_${SLURM_JOB_ID}.log" 2>&1 &
    OLLAMA_PID=$!
    echo "    Ollama started with PID: ${OLLAMA_PID}"
    
    # Wait for Ollama to be ready
    echo "    Waiting for Ollama to initialize..."
    MAX_WAIT=60
    WAITED=0
    while ! curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; do
        sleep 2
        WAITED=$((WAITED + 2))
        if [ $WAITED -ge $MAX_WAIT ]; then
            echo "ERROR: Ollama failed to start within ${MAX_WAIT} seconds"
            echo "Check log: ${AGI_ROOT}/slurm_logs/ollama_${SLURM_JOB_ID}.log"
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
# RUN THE PIPELINE
# ============================================================================

echo ""
echo "============================================================================"
echo "  Running AGI Pipeline"
echo "============================================================================"
echo "  Prompt File: ${PROMPT_FILE}"
echo "  Project Dir: ${PROJECT_DIR}"
echo "  Model: ${OLLAMA_MODEL}"
echo "  Max Retries: ${MAX_RETRIES}"
echo "============================================================================"
echo ""

# Create project directory
mkdir -p "${PROJECT_DIR}"

# Run the pipeline
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
# SUMMARY
# ============================================================================

echo ""
echo "============================================================================"
echo "  Job Complete"
echo "============================================================================"
echo "  End Time: $(date)"
echo "  Exit Code: ${PIPELINE_EXIT_CODE}"
echo "  Project Directory: ${AGI_ROOT}/${PROJECT_DIR}"
echo ""
echo "  Output files:"
echo "    - Logs: ${PROJECT_DIR}/logs/"
echo "    - Reports: ${PROJECT_DIR}/reports/"
echo "    - SLURM output: slurm_logs/agi_${SLURM_JOB_ID}.out"
echo "============================================================================"

exit ${PIPELINE_EXIT_CODE}


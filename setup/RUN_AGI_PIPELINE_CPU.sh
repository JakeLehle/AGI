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
#SBATCH --mem 900GB

###############################################################################
# AGI Multi-Agent Pipeline v3.1 - SLURM Submission Script
#
# With Cluster Profile Support for multi-cluster deployment
#
# Usage:
#   # CPU cluster (default)
#   sbatch RUN_AGI_PIPELINE.sh
#
#   # GPU cluster - use different sbatch directives AND set cluster var
#   sbatch --export=AGI_CLUSTER=gpu_v100 RUN_AGI_PIPELINE_GPU.sh
#
###############################################################################

# ============================================================================
# CLUSTER CONFIGURATION
# ============================================================================
# Which cluster are subtasks submitted to?
# This determines SLURM settings for the GENERATED subtask jobs.
#
# Available clusters (defined in config/cluster_config.yaml):
#   - zeus_cpu   : Normal partition, account=jlehle, up to 192 cores, 900GB
#   - gpu_v100   : gpu1v100 partition, account=sdz852, V100 GPUs
#   - gpu_a100   : gpu1a100 partition, account=sdz852, A100 GPUs
#   - dgx_a100   : dgxa100 partition, account=sdz852, DGX system
#
# Set via environment variable or edit default below:
AGI_CLUSTER="${AGI_CLUSTER:-zeus_cpu}"

# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================

# Prompt file (relative or absolute path)
PROMPT_FILE="${PROMPT_FILE:-/master/jlehle/WORKING/slide-TCR-seq-working/prompts/2026-01-29_prompt_1.md}"

# Project directory (where outputs go - separate from AGI code)
PROJECT_DIR="${PROJECT_DIR:-/master/jlehle/WORKING/slide-TCR-seq-working}"

# AGI repository root (the code)
AGI_ROOT="${AGI_ROOT:-/master/jlehle/WORKING/AGI}"

# AGI data directory (reflexion memory storage)
AGI_DATA_DIR="${AGI_DATA_DIR:-/master/jlehle/agi_data}"

# Conda environment for AGI pipeline
CONDA_ENV="${CONDA_ENV:-AGI}"

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Ollama model for task execution
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1:70b}"

# Embedding model for reflexion memory
EMBEDDING_MODEL="${EMBEDDING_MODEL:-nomic-embed-text}"

# Enable reflexion memory
USE_REFLEXION_MEMORY="${USE_REFLEXION_MEMORY:-true}"

# ============================================================================
# CONTEXT LIMITS (v3 Architecture)
# ============================================================================

MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-60000}"
MAX_TOOL_OUTPUT_TOKENS="${MAX_TOOL_OUTPUT_TOKENS:-25000}"
MIN_TOKENS_TO_CONTINUE="${MIN_TOKENS_TO_CONTINUE:-5000}"

# ============================================================================
# SUBTASK RESOURCE OVERRIDES (Optional)
# ============================================================================
# These override the cluster defaults for ALL subtasks.
# Leave empty to use cluster config defaults.
# Individual subtasks can still specify their own requirements.

# Override partition for subtasks (leave empty for cluster default)
SUBTASK_PARTITION="${SUBTASK_PARTITION:-}"

# Override account for subtasks
SUBTASK_ACCOUNT="${SUBTASK_ACCOUNT:-}"

# Override time limit for subtasks
SUBTASK_TIME="${SUBTASK_TIME:-}"

# Override memory for subtasks (leave empty for GPU clusters that don't use it)
SUBTASK_MEMORY="${SUBTASK_MEMORY:-}"

# Override CPUs for subtasks
SUBTASK_CPUS="${SUBTASK_CPUS:-}"

# Override GPU count for subtasks
SUBTASK_GPUS="${SUBTASK_GPUS:-}"

# ============================================================================
# VALIDATION
# ============================================================================

if [ -z "${PROJECT_DIR}" ]; then
    echo "ERROR: PROJECT_DIR must be specified"
    exit 1
fi

PROJECT_DIR=$(realpath "${PROJECT_DIR}")

if [ ! -d "${AGI_ROOT}" ]; then
    echo "ERROR: AGI_ROOT does not exist: ${AGI_ROOT}"
    exit 1
fi

if [ ! -d "${PROJECT_DIR}" ]; then
    echo "ERROR: PROJECT_DIR does not exist: ${PROJECT_DIR}"
    exit 1
fi

# ============================================================================
# STARTUP BANNER
# ============================================================================

echo "============================================================================"
echo "  AGI Multi-Agent Pipeline v3.1"
echo "============================================================================"
echo "  Job ID:              ${SLURM_JOB_ID:-local}"
echo "  Node:                $(hostname)"
echo "  Start Time:          $(date)"
echo ""
echo "  ┌────────────────────────────────────────────────────────────────────┐"
echo "  │  CLUSTER TARGET FOR SUBTASKS: ${AGI_CLUSTER}"
echo "  │  (Subtask jobs will be submitted to this cluster)"
echo "  └────────────────────────────────────────────────────────────────────┘"
echo ""
echo "  AGI Root:            ${AGI_ROOT}"
echo "  Project Dir:         ${PROJECT_DIR}"
echo "  Prompt File:         ${PROMPT_FILE}"
echo "  Model:               ${OLLAMA_MODEL}"
echo "============================================================================"

# Create directories
mkdir -p "${PROJECT_DIR}/slurm/logs"
mkdir -p "${PROJECT_DIR}/slurm/scripts"
mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/temp/checkpoints"
mkdir -p "${PROJECT_DIR}/envs"
mkdir -p "${PROJECT_DIR}/scripts"
mkdir -p "${AGI_DATA_DIR}/qdrant_storage"

# ============================================================================
# LOAD ENVIRONMENT
# ============================================================================

echo ""
echo ">>> Loading environment..."

# Try to load modules
if command -v module &> /dev/null; then
    module load git 2>/dev/null && echo "    Loaded git module" || true
    module load anaconda3 2>/dev/null && echo "    Loaded anaconda3 module" || true
fi

# Initialize conda
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    eval "$(conda shell.bash hook)"
fi

echo ">>> Activating conda environment: ${CONDA_ENV}"
conda activate "${CONDA_ENV}" || { echo "ERROR: Cannot activate ${CONDA_ENV}"; exit 1; }

echo ">>> Python: $(which python)"

# ============================================================================
# EXPORT ENVIRONMENT VARIABLES FOR SUBAGENT
# ============================================================================
# These are read by the subagent to generate appropriate sbatch scripts

echo ""
echo ">>> Setting environment variables..."

# Ollama
export OLLAMA_HOST="http://127.0.0.1:11434"
export OLLAMA_BASE_URL="http://127.0.0.1:11434"
export OLLAMA_MODELS="${HOME}/.ollama/models"
export OLLAMA_KEEP_ALIVE="10m"

# AGI paths
export AGI_DATA_DIR="${AGI_DATA_DIR}"
export PYTHONPATH="${AGI_ROOT}:${PYTHONPATH}"

# Context limits
export AGI_MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS}"
export AGI_MAX_TOOL_OUTPUT_TOKENS="${MAX_TOOL_OUTPUT_TOKENS}"
export AGI_MIN_TOKENS_TO_CONTINUE="${MIN_TOKENS_TO_CONTINUE}"

# ==========================================================================
# CLUSTER CONFIGURATION - Critical for subtask sbatch generation
# ==========================================================================
export AGI_CLUSTER="${AGI_CLUSTER}"
export AGI_CLUSTER_CONFIG="${AGI_ROOT}/config/cluster_config.yaml"

# Pass any overrides
export AGI_SUBTASK_PARTITION="${SUBTASK_PARTITION}"
export AGI_SUBTASK_ACCOUNT="${SUBTASK_ACCOUNT}"
export AGI_SUBTASK_TIME="${SUBTASK_TIME}"
export AGI_SUBTASK_MEMORY="${SUBTASK_MEMORY}"
export AGI_SUBTASK_CPUS="${SUBTASK_CPUS}"
export AGI_SUBTASK_GPUS="${SUBTASK_GPUS}"

echo "    AGI_CLUSTER=${AGI_CLUSTER}"
echo "    AGI_CLUSTER_CONFIG=${AGI_CLUSTER_CONFIG}"

# Display cluster info
echo ""
echo ">>> Cluster Configuration for Subtasks:"
python3 << 'PYEOF'
import yaml
import os

config_path = os.environ.get('AGI_CLUSTER_CONFIG')
cluster_name = os.environ.get('AGI_CLUSTER', 'zeus_cpu')

if config_path and os.path.exists(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    cluster = config.get('clusters', {}).get(cluster_name, {})
    slurm = cluster.get('slurm', {})
    gpu = cluster.get('gpu', {})
    
    print(f"    Name:        {cluster.get('name', 'Unknown')}")
    print(f"    Partition:   {slurm.get('partition', 'N/A')}")
    print(f"    Account:     {slurm.get('account', 'N/A')}")
    print(f"    Default CPUs: {slurm.get('cpus_per_task', 'N/A')}")
    print(f"    Default Mem:  {slurm.get('memory', 'N/A')}")
    print(f"    Default Time: {slurm.get('time', 'N/A')}")
    if gpu.get('available'):
        print(f"    GPU:         Yes ({gpu.get('default_count', 1)}x {gpu.get('type', 'GPU')})")
    else:
        print(f"    GPU:         No")
else:
    print(f"    Config not found, will use defaults")
PYEOF

# ============================================================================
# START OLLAMA
# ============================================================================

echo ""
echo ">>> Starting Ollama server..."

OLLAMA_LOG="${PROJECT_DIR}/logs/ollama_${SLURM_JOB_ID:-$$}.log"

if curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
    echo "    Ollama already running"
else
    ollama serve > "${OLLAMA_LOG}" 2>&1 &
    OLLAMA_PID=$!
    echo "    Started Ollama (PID: ${OLLAMA_PID})"
    
    MAX_WAIT=60
    WAITED=0
    while ! curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; do
        sleep 2
        WAITED=$((WAITED + 2))
        if [ $WAITED -ge $MAX_WAIT ]; then
            echo "ERROR: Ollama failed to start"
            exit 1
        fi
    done
    echo "    Ollama ready (${WAITED}s)"
fi

# ============================================================================
# VERIFY MODELS
# ============================================================================

echo ""
echo ">>> Checking models..."

if ! ollama list 2>/dev/null | grep -q "${OLLAMA_MODEL%%:*}"; then
    echo "    Pulling ${OLLAMA_MODEL}..."
    ollama pull "${OLLAMA_MODEL}"
else
    echo "    ✓ ${OLLAMA_MODEL}"
fi

if [ "${USE_REFLEXION_MEMORY}" = "true" ]; then
    if ! ollama list 2>/dev/null | grep -q "${EMBEDDING_MODEL}"; then
        echo "    Pulling ${EMBEDDING_MODEL}..."
        ollama pull "${EMBEDDING_MODEL}"
    else
        echo "    ✓ ${EMBEDDING_MODEL}"
    fi
fi

# ============================================================================
# RESOLVE PROMPT FILE
# ============================================================================

if [[ "${PROMPT_FILE}" != /* ]]; then
    if [ -f "${PROJECT_DIR}/${PROMPT_FILE}" ]; then
        PROMPT_FILE="${PROJECT_DIR}/${PROMPT_FILE}"
    elif [ -f "${AGI_ROOT}/${PROMPT_FILE}" ]; then
        PROMPT_FILE="${AGI_ROOT}/${PROMPT_FILE}"
    else
        echo "ERROR: Prompt file not found: ${PROMPT_FILE}"
        exit 1
    fi
fi

# ============================================================================
# RUN PIPELINE
# ============================================================================

echo ""
echo "============================================================================"
echo "  Running AGI Pipeline"
echo "============================================================================"
echo "  Subtask Cluster Target: ${AGI_CLUSTER}"
echo "  Project:                ${PROJECT_DIR}"
echo "  Model:                  ${OLLAMA_MODEL}"
echo "============================================================================"
echo ""

cd "${AGI_ROOT}"

# Pass cluster to main.py
python main.py \
    --prompt-file "${PROMPT_FILE}" \
    --project-dir "${PROJECT_DIR}" \
    --model "${OLLAMA_MODEL}" \
    --cluster "${AGI_CLUSTER}"

PIPELINE_EXIT_CODE=$?

# ============================================================================
# CLEANUP
# ============================================================================

echo ""
echo ">>> Pipeline finished (exit code: ${PIPELINE_EXIT_CODE})"

if [ -n "${OLLAMA_PID}" ]; then
    echo ">>> Stopping Ollama..."
    kill "${OLLAMA_PID}" 2>/dev/null || true
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
echo "  Cluster:        ${AGI_CLUSTER}"
echo ""
echo "  Outputs:        ${PROJECT_DIR}/"
echo "  Checkpoints:    ${PROJECT_DIR}/temp/checkpoints/"
echo "  SLURM Logs:     ${PROJECT_DIR}/slurm/logs/"
echo "============================================================================"

exit ${PIPELINE_EXIT_CODE}

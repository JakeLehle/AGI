#!/bin/bash
#SBATCH --job-name=agi-pipeline
#SBATCH --partition=gpu1v100
#SBATCH --account=sdz852
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gres/gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm_logs/agi_%j.out
#SBATCH --error=slurm_logs/agi_%j.err

###############################################################################
# AGI Multi-Agent Pipeline v1.2.2 — ARC GPU Submission Script
#
# Runs the MASTER PIPELINE on an ARC GPU node for fast LLM inference.
# Subtask scripts are submitted to CPU or GPU partitions as needed.
#
# v1.2.2 CHANGES:
# ----------------
#   - OLLAMA_NUM_PARALLEL: Enables concurrent LLM request processing so
#     multiple sub-agent threads can generate scripts simultaneously.
#     Default: 4 (matches max_parallel_agents thread pool size).
#     VRAM budget: 2×V100S-32GB (64 GiB) supports 4 parallel slots safely.
#   - Version bumps in display and summary sections.
#
# v1.2.0 FEATURES:
# ---------------------
#   - Diagnostic Agent: Cross-task error diagnosis with global solution memory
#   - Diagnostic Memory: Persistent knowledge base of validated fixes
#   - Disk Manager: Proactive disk monitoring and auto-cleanup
#   - Bootstrap Solutions: Pre-seed known fixes on first run
#   - Enhanced Context Manager: Diagnostic context injection
#
# MODEL RESOLUTION CHAIN (v3.2.1):
# ---------------------------------
# The model used by every component is resolved via a 4-level priority chain.
# This RUN script participates at TWO levels:
#
#   Level 1 (CLI):  --model "${OLLAMA_MODEL}" passed to main.py
#   Level 2 (env):  export OLLAMA_MODEL  (read by sub-agents, subprocesses)
#   Level 3 (file): config/config.yaml → ollama.model
#   Level 4 (code): FALLBACK_MODEL in utils/model_config.py
#
# To override the model for a single run without editing this file:
#   sbatch --export=ALL,OLLAMA_MODEL=llama3.1:70b setup/RUN_AGI_PIPELINE_GPU.sh
#
# The default below (qwen3-coder:latest) is chosen because:
#   - Weights ≈ 20 GiB (Q4_K_M) → fits on V100S-32GB with room for KV cache
#   - qwen3-coder-next ≈ 48 GiB → requires CPU offload, causes Ollama 500 errors
#
# Usage:
#   # Default (gpu1v100 for master, compute1 for CPU subtasks)
#   sbatch setup/RUN_AGI_PIPELINE_GPU.sh
#
#   # Override subtask cluster target
#   sbatch --export=ALL,AGI_CLUSTER=arc_compute2 setup/RUN_AGI_PIPELINE_GPU.sh
#
#   # Override model
#   sbatch --export=ALL,OLLAMA_MODEL=llama3.1:70b setup/RUN_AGI_PIPELINE_GPU.sh
#
#   # Disable diagnostic agent (faster, no Mem0 overhead)
#   sbatch --export=ALL,DIAGNOSTIC_AGENT_ENABLED=false setup/RUN_AGI_PIPELINE_GPU.sh
#
###############################################################################

# ============================================================================
# CLUSTER CONFIGURATION
# ============================================================================
# AGI_CLUSTER: Default target for subtask scripts (CPU partition)
# AGI_GPU_CLUSTER: Target for subtasks that need GPU resources
#
# Available clusters (defined in config/cluster_config.yaml):
#   CPU:
#     - arc_compute1 : compute1 partition, 65 nodes, 3-day max (DEFAULT)
#     - arc_compute2 : compute2 partition, 27 nodes, 10-day max
#     - arc_compute3 : compute3 partition, 6 nodes, 3-day max
#   GPU:
#     - arc_gpu1v100  : gpu1v100, 22 nodes, 1×V100 (DEFAULT GPU)
#     - arc_gpu2v100  : gpu2v100, 9 nodes, 2×V100
#     - arc_gpu4v100  : gpu4v100, 2 nodes, 4×V100
#     - arc_gpu1a100  : gpu1a100, 2 nodes, 1×A100
#     - arc_dgxa100   : dgxa100, 3 nodes, DGX A100
#
AGI_CLUSTER="${AGI_CLUSTER:-arc_compute1}"
AGI_GPU_CLUSTER="${AGI_GPU_CLUSTER:-arc_gpu2v100}"

# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================

# Prompt file (relative or absolute path)
PROMPT_FILE="${PROMPT_FILE:-/work/sdz852/WORKING/slide-TCR-seq-working/prompts/2026-01-29_prompt_1.md}"

# Project directory (where outputs go - separate from AGI code)
PROJECT_DIR="${PROJECT_DIR:-/work/sdz852/WORKING/slide-TCR-seq-working}"

# AGI repository root (the code)
AGI_ROOT="${AGI_ROOT:-/work/sdz852/WORKING/AGI}"

# AGI data directory (reflexion memory + diagnostic memory storage)
AGI_DATA_DIR="${AGI_DATA_DIR:-/work/sdz852/agi_data}"

# Conda environment for AGI pipeline
CONDA_ENV="${CONDA_ENV:-AGI}"

# ============================================================================
# MODEL CONFIGURATION (v3.2.1 — resolution chain level 1 + 2)
# ============================================================================
# OLLAMA_MODEL serves double duty:
#   1. Passed as --model to main.py  → resolve_model() level 1 (explicit CLI)
#   2. Exported as env var           → resolve_model() level 2 (env fallback)
#
# This means sub-agents and subprocesses that don't receive --model directly
# can still pick up the correct model from the environment.
#
# To override for a single run:
#   sbatch --export=ALL,OLLAMA_MODEL=llama3.1:70b setup/RUN_AGI_PIPELINE_GPU.sh
#
# Selection criteria for V100S-32GB:
#   qwen3-coder:32b  ≈ 20 GiB weights → fits with ~10 GiB for 32K KV cache ✓
#   qwen3-coder-next ≈ 48 GiB weights → CPU offload required, 500 errors   ✗
# ============================================================================

OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3-coder:latest}"

# Ollama context window — must match config.yaml → ollama.model_context_length
OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-32768}"

# Embedding model for reflexion memory + diagnostic memory
EMBEDDING_MODEL="${EMBEDDING_MODEL:-nomic-embed-text}"

# Enable reflexion memory (per-task retry loop prevention)
USE_REFLEXION_MEMORY="${USE_REFLEXION_MEMORY:-true}"

# ============================================================================
# OLLAMA PARALLEL CONFIGURATION (v1.2.2 — NEW)
# ============================================================================
# Number of concurrent LLM requests Ollama processes simultaneously.
# Each parallel slot consumes additional KV cache memory:
#
#   Per-slot KV cache: ~3-4 GiB (at 32K context with qwen3-coder:latest)
#   Model weights:     ~20 GiB (shared across all slots)
#
# VRAM budget by hardware:
#   1× V100S-32GB (32 GiB):         2 safe, 3 tight
#   2× V100S-32GB (64 GiB total):   4 safe, 6 possible
#   CPU-only (1 TiB RAM):           4-6 safe (RAM is not the constraint)
#
# Must match or exceed max_parallel_agents in config.yaml (default: 4).
# Requests exceeding this limit queue in OLLAMA_MAX_QUEUE (default 512)
# and are processed as slots free up — no requests are lost.
#
# To override for a single run:
#   sbatch --export=ALL,OLLAMA_NUM_PARALLEL=2 setup/RUN_AGI_PIPELINE_GPU.sh
# ============================================================================

OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-4}"

# ============================================================================
# CONTEXT LIMITS (v3.2 Architecture)
# ============================================================================

MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-25000}"
MAX_TOOL_OUTPUT_TOKENS="${MAX_TOOL_OUTPUT_TOKENS:-12000}"
MIN_TOKENS_TO_CONTINUE="${MIN_TOKENS_TO_CONTINUE:-3000}"

# ============================================================================
# DIAGNOSTIC AGENT CONFIGURATION (v1.2.0 — NEW)
# ============================================================================
# The Diagnostic Agent provides cross-task error diagnosis with a global
# solution memory. When a subtask fails, the diagnostic agent:
#   1. Checks DiagnosticMemory for known solutions (instant fix)
#   2. If no known fix, investigates the error systematically
#   3. Stores validated fixes for future reuse across ALL tasks/projects
#
# DIAGNOSTIC_AGENT_ENABLED: Master switch for the diagnostic agent.
#   - true (default): Enables diagnostic agent + diagnostic memory
#   - false: Disables entirely (falls back to standard reflexion-only retry)
#
# DIAGNOSTIC_MAX_INVOCATIONS: Max diagnostic agent calls per subtask.
#   Prevents runaway diagnostic loops. Each invocation = 1 LLM call.
#   Default: 2 (investigate once, verify once)
#
# DIAGNOSTIC_TOKEN_BUDGET: Max tokens the diagnostic agent can consume
#   per invocation from the subtask's context window.
#   Default: 4000 (enough for error analysis + fix generation)
#
# DIAGNOSTIC_MEMORY_BOOTSTRAP: Seed the diagnostic memory with known
#   solutions on first run (e.g., "celltypist is pip-only", "no --mem
#   on GPU partitions"). Only runs once — subsequent runs skip if
#   solutions already exist.
#   - true (default): Bootstrap on first run
#   - false: Start with empty knowledge base
#
# SOLUTION_MEMORY_THRESHOLD: Minimum semantic similarity (0.0–1.0) for
#   a stored solution to be considered a "confident match". Lower values
#   = more matches but less precision. Higher = fewer but more reliable.
#   Default: 0.85
# ============================================================================

DIAGNOSTIC_AGENT_ENABLED="${DIAGNOSTIC_AGENT_ENABLED:-true}"
DIAGNOSTIC_MAX_INVOCATIONS="${DIAGNOSTIC_MAX_INVOCATIONS:-2}"
DIAGNOSTIC_TOKEN_BUDGET="${DIAGNOSTIC_TOKEN_BUDGET:-4000}"
DIAGNOSTIC_MEMORY_BOOTSTRAP="${DIAGNOSTIC_MEMORY_BOOTSTRAP:-true}"
SOLUTION_MEMORY_THRESHOLD="${SOLUTION_MEMORY_THRESHOLD:-0.85}"

# ============================================================================
# DISK MANAGER CONFIGURATION (v1.2.0 — NEW)
# ============================================================================
# The Disk Manager monitors user quota and proactively cleans up stale
# conda environments, pip caches, and temp files before "Disk quota
# exceeded" errors kill running jobs.
#
# DISK_MONITOR_ENABLED: Enable proactive disk monitoring.
#   - true (default): Check disk before each subtask + auto-cleanup
#   - false: No monitoring (rely on OS errors)
#
# DISK_LOW_SPACE_THRESHOLD_GB: Trigger cleanup when free space drops
#   below this threshold (in GB). Set based on your /work quota.
#   Default: 10 (10 GB remaining triggers cleanup)
#
# DISK_CLEANUP_STALE_ENVS: Auto-remove agi_* conda environments from
#   previous pipeline runs that are no longer in use.
#   - true (default): Clean up stale environments
#   - false: Keep all environments (useful for debugging)
#
# DISK_CLEANUP_CONDA_CACHE: Run 'conda clean --all' when low on space.
#   - true (default): Clean conda package cache
#   - false: Preserve cache (faster re-installs but uses more space)
# ============================================================================

DISK_MONITOR_ENABLED="${DISK_MONITOR_ENABLED:-true}"
DISK_LOW_SPACE_THRESHOLD_GB="${DISK_LOW_SPACE_THRESHOLD_GB:-10}"
DISK_CLEANUP_STALE_ENVS="${DISK_CLEANUP_STALE_ENVS:-true}"
DISK_CLEANUP_CONDA_CACHE="${DISK_CLEANUP_CONDA_CACHE:-true}"

# ============================================================================
# SUBTASK RESOURCE OVERRIDES (Optional)
# ============================================================================
# These override the cluster defaults for ALL subtasks.
# Leave empty to use cluster config defaults.
# Individual subtasks can still specify their own requirements.

SUBTASK_PARTITION="${SUBTASK_PARTITION:-}"
SUBTASK_ACCOUNT="${SUBTASK_ACCOUNT:-}"
SUBTASK_TIME="${SUBTASK_TIME:-}"
SUBTASK_MEMORY="${SUBTASK_MEMORY:-}"
SUBTASK_CPUS="${SUBTASK_CPUS:-}"
SUBTASK_GPUS="${SUBTASK_GPUS:-}"

# ============================================================================
# CONDA ENVIRONMENT CLEANUP (v3.2)
# ============================================================================
# After a subtask completes successfully, its conda environment can be
# removed to free disk space. Set to "false" for debugging.

CLEANUP_ENV_ON_SUCCESS="${CLEANUP_ENV_ON_SUCCESS:-true}"

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

echo ""
echo "============================================================================"
echo "  AGI Multi-Agent Pipeline v1.2.2"
echo "============================================================================"
echo "  Job ID:              ${SLURM_JOB_ID:-interactive}"
echo "  Node:                $(hostname)"
echo "  Start Time:          $(date)"
echo "  Cluster (CPU):       ${AGI_CLUSTER}"
echo "  Cluster (GPU):       ${AGI_GPU_CLUSTER}"
echo "  AGI Root:            ${AGI_ROOT}"
echo "  Project Dir:         ${PROJECT_DIR}"
echo "  Prompt File:         ${PROMPT_FILE}"
echo "  Model:               ${OLLAMA_MODEL}"
echo "  Context Length:       ${OLLAMA_CONTEXT_LENGTH}"
echo "  Parallel Slots:      ${OLLAMA_NUM_PARALLEL}"
echo "  ---"
echo "  Diagnostic Agent:    ${DIAGNOSTIC_AGENT_ENABLED}"
echo "  Diagnostic Memory:   bootstrap=${DIAGNOSTIC_MEMORY_BOOTSTRAP}, threshold=${SOLUTION_MEMORY_THRESHOLD}"
echo "  Disk Monitor:        ${DISK_MONITOR_ENABLED} (threshold=${DISK_LOW_SPACE_THRESHOLD_GB}GB)"
echo "  Cleanup Envs:        ${CLEANUP_ENV_ON_SUCCESS}"
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
    module load cudatoolkit 2>/dev/null && echo "    Loaded cudatoolkit module" || true
    module load ollama 2>/dev/null && echo "     Loaded ollama module" || true
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
# GPU VERIFICATION
# ============================================================================

echo ""
echo ">>> Verifying GPU access..."

if command -v nvidia-smi &> /dev/null; then
    echo "    GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | while read line; do
        echo "      $line"
    done
else
    echo "    WARNING: nvidia-smi not found - Ollama will run on CPU"
    echo "    This will be SLOW. Consider requesting a GPU node."
fi

# Unset SLURM GPU variables that can interfere with Ollama
unset CUDA_VISIBLE_DEVICES 2>/dev/null
unset ROCR_VISIBLE_DEVICES 2>/dev/null
unset GPU_DEVICE_ORDINAL 2>/dev/null

# ============================================================================
# DISK SPACE CHECK (v1.2.0)
# ============================================================================

if [ "${DISK_MONITOR_ENABLED}" = "true" ]; then
    echo ""
    echo ">>> Checking disk space..."

    # Check /work quota (most common constraint on HPC)
    WORK_DIR=$(dirname "${PROJECT_DIR}")
    AVAIL_KB=$(df -k "${WORK_DIR}" 2>/dev/null | tail -1 | awk '{print $4}')
    if [ -n "${AVAIL_KB}" ]; then
        AVAIL_GB=$((AVAIL_KB / 1048576))
        echo "    Available: ~${AVAIL_GB} GB on $(df "${WORK_DIR}" 2>/dev/null | tail -1 | awk '{print $6}')"
        if [ "${AVAIL_GB}" -lt "${DISK_LOW_SPACE_THRESHOLD_GB}" ]; then
            echo "    ⚠ WARNING: Low disk space (${AVAIL_GB}GB < ${DISK_LOW_SPACE_THRESHOLD_GB}GB threshold)"
            if [ "${DISK_CLEANUP_CONDA_CACHE}" = "true" ]; then
                echo "    >>> Running conda clean to free space..."
                conda clean --all --yes 2>/dev/null
                echo "    >>> Cleaned conda cache"
            fi
            if [ "${DISK_CLEANUP_STALE_ENVS}" = "true" ]; then
                echo "    >>> Removing stale agi_* environments..."
                for env_dir in $(conda env list 2>/dev/null | grep "agi_" | awk '{print $NF}'); do
                    if [ -d "${env_dir}" ]; then
                        env_name=$(basename "${env_dir}")
                        echo "      Removing: ${env_name}"
                        conda env remove -n "${env_name}" --yes 2>/dev/null || true
                    fi
                done
                echo "    >>> Stale environments cleaned"
            fi
            # Re-check
            AVAIL_KB_POST=$(df -k "${WORK_DIR}" 2>/dev/null | tail -1 | awk '{print $4}')
            AVAIL_GB_POST=$((AVAIL_KB_POST / 1048576))
            echo "    Available after cleanup: ~${AVAIL_GB_POST} GB"
        else
            echo "    ✓ Disk space OK"
        fi
    else
        echo "    Could not determine disk space — skipping check"
    fi
fi

# ============================================================================
# EXPORT ENVIRONMENT VARIABLES FOR SUBAGENT
# ============================================================================

echo ""
echo ">>> Setting environment variables..."

# Ollama
export OLLAMA_HOST="http://127.0.0.1:11434"
export OLLAMA_BASE_URL="http://127.0.0.1:11434"
export OLLAMA_MODELS="${OLLAMA_MODELS:-/work/sdz852/ollama/models}"
export OLLAMA_KEEP_ALIVE="10m"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH}"

# v1.2.2: Parallel LLM request processing
# This must be exported BEFORE ollama serve starts so the server picks it up.
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL}"

# v3.2.1: Export OLLAMA_MODEL so resolve_model() picks it up at level 2 (env)
# for any component that doesn't receive it via CLI --model.
export OLLAMA_MODEL="${OLLAMA_MODEL}"

# AGI paths
export AGI_DATA_DIR="${AGI_DATA_DIR}"
export PYTHONPATH="${AGI_ROOT}:${PYTHONPATH}"
echo "    PYTHONPATH=${PYTHONPATH}"

# Context limits
export AGI_MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS}"
export AGI_MAX_TOOL_OUTPUT_TOKENS="${MAX_TOOL_OUTPUT_TOKENS}"
export AGI_MIN_TOKENS_TO_CONTINUE="${MIN_TOKENS_TO_CONTINUE}"

# ==========================================================================
# CLUSTER CONFIGURATION - Critical for subtask sbatch generation
# ==========================================================================
export AGI_CLUSTER="${AGI_CLUSTER}"
export AGI_GPU_CLUSTER="${AGI_GPU_CLUSTER}"
export AGI_CLUSTER_CONFIG="${AGI_ROOT}/config/cluster_config.yaml"

# Pass any overrides
export AGI_SUBTASK_PARTITION="${SUBTASK_PARTITION}"
export AGI_SUBTASK_ACCOUNT="${SUBTASK_ACCOUNT}"
export AGI_SUBTASK_TIME="${SUBTASK_TIME}"
export AGI_SUBTASK_MEMORY="${SUBTASK_MEMORY}"
export AGI_SUBTASK_CPUS="${SUBTASK_CPUS}"
export AGI_SUBTASK_GPUS="${SUBTASK_GPUS}"

# ==========================================================================
# v1.2.0: DIAGNOSTIC AGENT & DISK MANAGER CONFIGURATION
# ==========================================================================
export AGI_DIAGNOSTIC_AGENT_ENABLED="${DIAGNOSTIC_AGENT_ENABLED}"
export AGI_DIAGNOSTIC_MAX_INVOCATIONS="${DIAGNOSTIC_MAX_INVOCATIONS}"
export AGI_DIAGNOSTIC_TOKEN_BUDGET="${DIAGNOSTIC_TOKEN_BUDGET}"
export AGI_DIAGNOSTIC_MEMORY_BOOTSTRAP="${DIAGNOSTIC_MEMORY_BOOTSTRAP}"
export AGI_SOLUTION_MEMORY_THRESHOLD="${SOLUTION_MEMORY_THRESHOLD}"

export AGI_DISK_MONITOR_ENABLED="${DISK_MONITOR_ENABLED}"
export AGI_DISK_LOW_SPACE_THRESHOLD_GB="${DISK_LOW_SPACE_THRESHOLD_GB}"
export AGI_DISK_CLEANUP_STALE_ENVS="${DISK_CLEANUP_STALE_ENVS}"
export AGI_DISK_CLEANUP_CONDA_CACHE="${DISK_CLEANUP_CONDA_CACHE}"

export AGI_CLEANUP_ENV_ON_SUCCESS="${CLEANUP_ENV_ON_SUCCESS}"

echo "    AGI_CLUSTER=${AGI_CLUSTER}"
echo "    AGI_GPU_CLUSTER=${AGI_GPU_CLUSTER}"
echo "    AGI_CLUSTER_CONFIG=${AGI_CLUSTER_CONFIG}"
echo "    OLLAMA_MODEL=${OLLAMA_MODEL}"
echo "    OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL}"
echo "    AGI_DIAGNOSTIC_AGENT_ENABLED=${AGI_DIAGNOSTIC_AGENT_ENABLED}"
echo "    AGI_DISK_MONITOR_ENABLED=${AGI_DISK_MONITOR_ENABLED}"

# Display cluster info
echo ""
echo ">>> Cluster Configuration for Subtasks:"
python3 << 'PYEOF'
import yaml
import os

config_path = os.environ.get('AGI_CLUSTER_CONFIG')
cluster_name = os.environ.get('AGI_CLUSTER', 'arc_compute1')

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
echo "    OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL} (concurrent request slots)"

OLLAMA_LOG="${PROJECT_DIR}/logs/ollama_${SLURM_JOB_ID:-$$}.log"

if curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
    echo "    Ollama already running"
    echo "    ⚠ NOTE: OLLAMA_NUM_PARALLEL only takes effect on server start."
    echo "    If the running server has a different setting, restart Ollama."
else
    ollama serve > "${OLLAMA_LOG}" 2>&1 &
    OLLAMA_PID=$!
    echo "    Started Ollama (PID: ${OLLAMA_PID})"

    MAX_WAIT=120
    WAITED=0
    while ! curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; do
        sleep 2
        WAITED=$((WAITED + 2))
        if [ $WAITED -ge $MAX_WAIT ]; then
            echo "ERROR: Ollama failed to start after ${MAX_WAIT}s"
            echo "Check log: ${OLLAMA_LOG}"
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

if [ "${USE_REFLEXION_MEMORY}" = "true" ] || [ "${DIAGNOSTIC_AGENT_ENABLED}" = "true" ]; then
    if ! ollama list 2>/dev/null | grep -q "${EMBEDDING_MODEL}"; then
        echo "    Pulling ${EMBEDDING_MODEL}..."
        ollama pull "${EMBEDDING_MODEL}"
    else
        echo "    ✓ ${EMBEDDING_MODEL}"
    fi
fi

# ============================================================================
# BOOTSTRAP DIAGNOSTIC MEMORY (v1.2.0)
# ============================================================================

if [ "${DIAGNOSTIC_AGENT_ENABLED}" = "true" ] && [ "${DIAGNOSTIC_MEMORY_BOOTSTRAP}" = "true" ]; then
    echo ""
    echo ">>> Bootstrapping diagnostic memory (first-run seeding)..."
    python3 << 'BOOTSTRAP_EOF'
import sys
import os

# Ensure AGI root is on path
agi_root = os.environ.get('AGI_ROOT', '.')
if agi_root not in sys.path:
    sys.path.insert(0, agi_root)

try:
    from memory.diagnostic_memory import DiagnosticMemory, BOOTSTRAP_SOLUTIONS

    dm = DiagnosticMemory()
    stats = dm.get_stats()

    if stats.get("total_solutions", 0) == 0:
        print(f"    Seeding {len(BOOTSTRAP_SOLUTIONS)} known solutions...")
        result = dm.store_known_solutions(BOOTSTRAP_SOLUTIONS)
        print(f"    ✓ Stored {result['stored_count']} solutions "
              f"({result['failed_count']} failed)")
    else:
        print(f"    ✓ Diagnostic memory already has "
              f"{stats['total_solutions']} solutions — skipping bootstrap")
except ImportError as e:
    print(f"    ⚠ DiagnosticMemory not available: {e}")
    print(f"    Diagnostic agent will run without persistent memory")
except Exception as e:
    print(f"    ⚠ Bootstrap failed (non-fatal): {e}")
BOOTSTRAP_EOF
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
echo "  Running AGI Pipeline v1.2.2"
echo "============================================================================"
echo "  Master Node:            $(hostname) (GPU)"
echo "  Subtask CPU Target:     ${AGI_CLUSTER}"
echo "  Subtask GPU Target:     ${AGI_GPU_CLUSTER}"
echo "  Project:                ${PROJECT_DIR}"
echo "  Model:                  ${OLLAMA_MODEL}"
echo "  Context Length:          ${OLLAMA_CONTEXT_LENGTH}"
echo "  Parallel Slots:         ${OLLAMA_NUM_PARALLEL}"
echo "  ---"
echo "  Diagnostic Agent:       ${DIAGNOSTIC_AGENT_ENABLED} (max=${DIAGNOSTIC_MAX_INVOCATIONS}, budget=${DIAGNOSTIC_TOKEN_BUDGET}t)"
echo "  Solution Memory:        threshold=${SOLUTION_MEMORY_THRESHOLD}"
echo "  Disk Monitor:           ${DISK_MONITOR_ENABLED} (threshold=${DISK_LOW_SPACE_THRESHOLD_GB}GB)"
echo "  Cleanup Envs on Pass:   ${CLEANUP_ENV_ON_SUCCESS}"
echo "============================================================================"
echo ""

cd "${AGI_ROOT}"

# v3.2.1: --model passes OLLAMA_MODEL as level 1 (explicit CLI) to main.py.
# main.py calls resolve_model(args.model, config) which returns this value
# directly since it's non-None. The exported env var serves as level 2
# fallback for any subprocess that doesn't receive --model.
python main.py \
    --prompt-file "${PROMPT_FILE}" \
    --project-dir "${PROJECT_DIR}" \
    --model "${OLLAMA_MODEL}" \
    --cluster "${AGI_CLUSTER}" \
    --gpu-cluster "${AGI_GPU_CLUSTER}"

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
# POST-RUN DISK CLEANUP (v1.2.0)
# ============================================================================

if [ "${DISK_MONITOR_ENABLED}" = "true" ] && [ "${DISK_CLEANUP_STALE_ENVS}" = "true" ]; then
    echo ""
    echo ">>> Post-run disk cleanup..."
    STALE_COUNT=0
    for env_dir in $(conda env list 2>/dev/null | grep "agi_" | awk '{print $NF}'); do
        if [ -d "${env_dir}" ]; then
            env_name=$(basename "${env_dir}")
            # Only clean up if env is not actively used by a running job
            if ! squeue -u "$USER" 2>/dev/null | grep -q "agi"; then
                echo "    Removing stale env: ${env_name}"
                conda env remove -n "${env_name}" --yes 2>/dev/null || true
                STALE_COUNT=$((STALE_COUNT + 1))
            fi
        fi
    done
    if [ "${STALE_COUNT}" -gt 0 ]; then
        echo "    Cleaned ${STALE_COUNT} stale environments"
    else
        echo "    No stale environments found"
    fi
fi

# ============================================================================
# DIAGNOSTIC MEMORY SUMMARY (v1.2.0)
# ============================================================================

if [ "${DIAGNOSTIC_AGENT_ENABLED}" = "true" ]; then
    echo ""
    echo ">>> Diagnostic Memory Summary:"
    python3 << 'STATS_EOF' 2>/dev/null || echo "    (stats unavailable)"
import sys, os
agi_root = os.environ.get('AGI_ROOT', '.')
if agi_root not in sys.path:
    sys.path.insert(0, agi_root)
try:
    from memory.diagnostic_memory import DiagnosticMemory
    dm = DiagnosticMemory()
    stats = dm.get_stats()
    print(f"    Total solutions:   {stats.get('total_solutions', 0)}")
    by_type = stats.get('solutions_by_type', {})
    if by_type:
        top_types = sorted(by_type.items(), key=lambda x: x[1], reverse=True)[:5]
        for etype, count in top_types:
            print(f"      {etype}: {count}")
    top = stats.get('most_reused', [])
    if top:
        print(f"    Most reused fix:   {top[0].get('solution', 'N/A')[:60]}... "
              f"(used {top[0].get('success_count', 0)}x)")
except Exception as e:
    print(f"    Could not retrieve stats: {e}")
STATS_EOF
fi

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "============================================================================"
echo "  Job Complete — v1.2.2"
echo "============================================================================"
echo "  End Time:       $(date)"
echo "  Exit Code:      ${PIPELINE_EXIT_CODE}"
echo "  Cluster:        ${AGI_CLUSTER}"
echo "  GPU Cluster:    ${AGI_GPU_CLUSTER}"
echo "  Model:          ${OLLAMA_MODEL}"
echo "  Parallel Slots: ${OLLAMA_NUM_PARALLEL}"
echo "  Diagnostic:     ${DIAGNOSTIC_AGENT_ENABLED}"
echo ""
echo "  Outputs:        ${PROJECT_DIR}/"
echo "  Checkpoints:    ${PROJECT_DIR}/temp/checkpoints/"
echo "  SLURM Logs:     ${PROJECT_DIR}/slurm/logs/"
echo "  Diagnostic DB:  ${AGI_DATA_DIR}/qdrant_storage/"
echo "============================================================================"

exit ${PIPELINE_EXIT_CODE}

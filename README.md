# AGI Multi-Agent Pipeline

> Automated computational workflow orchestration for HPC clusters using local LLMs.

The AGI pipeline decomposes complex research prompts into executable scripts, submits them as SLURM jobs, monitors completion, and handles failures with reflexion-based retry logic — all driven by a local LLM running on a GPU node.

## Architecture (v3.2)

```
┌────────────────────────────────────────────────────────────┐
│  GPU Node (master pipeline)                                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Ollama Server │◄──│ Master Agent │───►│  Sub-Agents  │  │
│  │ (qwen3-coder)│    │ Decompose    │    │ Script Gen   │  │
│  └──────────────┘    │ Validate     │    │ Env Create   │  │
│                      │ Track        │    │ Submit Jobs  │  │
│                      └──────────────┘    └──────┬───────┘  │
└─────────────────────────────────────────────────┼──────────┘
                                                  │ sbatch
                          ┌───────────────────────┼──────────────────┐
                          │                       │                  │
                    ┌─────▼─────┐          ┌──────▼──────┐    ┌─────▼─────┐
                    │ compute1  │          │  gpu1v100   │    │ compute2  │
                    │ CPU tasks │          │  GPU tasks  │    │ Long jobs │
                    └───────────┘          └─────────────┘    └───────────┘
```

The master pipeline runs on a GPU node for fast LLM inference. Subtasks are automatically routed to CPU or GPU partitions based on package detection (scanpy → CPU, torch/scvi → GPU).

## Prerequisites

- Python 3.10+
- Conda (conda-forge channel)
- Ollama (system-wide install for GPU support)
- SLURM (HPC job scheduler)
- Git

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/AGI.git
cd AGI

# 2. Create the conda environment
conda create -n AGI python=3.10 -c conda-forge -y
conda activate AGI
pip install -r requirements.txt

# 3. Configure Ollama model storage
#    Models should be stored on your working directory (not home) to avoid
#    quota issues. Set this in your shell profile or RUN scripts:
export OLLAMA_MODELS="/work/$USER/ollama/models"

# 4. Pull models
ollama pull qwen3-coder-next     # Primary coding LLM (GPU, 32K context)
ollama pull nomic-embed-text     # Embeddings for reflexion memory

# 5. Create data directory for reflexion memory
mkdir -p /work/$USER/agi_data/qdrant_storage

# 6. Verify
python -c "from workflows.langgraph_workflow import MultiAgentWorkflow; print('OK')"
```

## Setting Up a New Project

Every analysis gets its own project directory, separate from this AGI repo. The setup script creates the full directory structure and copies pre-configured RUN scripts:

```bash
# Create and enter your project directory
mkdir /work/$USER/WORKING/my-new-project
cd /work/$USER/WORKING/my-new-project

# Run setup (auto-detects AGI repo location)
bash /path/to/AGI/setup/setup.sh

# Or with explicit AGI root
bash /path/to/AGI/setup/setup.sh --agi-root /work/$USER/WORKING/AGI
```

This creates:

```
my-new-project/
├── agents/                  # Python package stubs
├── conda_env/               # Conda env YAML specs (tracked)
├── config/                  # Project-specific config (tracked)
├── data/
│   ├── inputs/              # Your input data
│   └── outputs/             # Pipeline outputs
├── prompts/                 # Master prompt files (tracked)
├── scripts/                 # User + generated scripts (tracked)
│   └── example_reference_scripts/
├── logs/                    # Runtime logs (ignored)
├── reports/                 # Pipeline status (ignored)
├── slurm_logs/              # Master job stdout/stderr (ignored)
├── temp/checkpoints/        # Resume checkpoints (ignored)
├── project.yaml             # Project configuration
├── RUN_AGI_PIPELINE_GPU.sh  # ← Submit this to start the pipeline
├── RUN_AGI_PIPELINE_CPU.sh  # CPU-only alternative
└── CLEAN_PROJECT.sh         # Cleanup between runs
```

## Running the Pipeline

### 1. Write your master prompt

Create a detailed prompt in `prompts/`. The more specific you are, the better the decomposition:

```bash
vi prompts/my_analysis.md
```

### 2. Update the RUN script

Open `RUN_AGI_PIPELINE_GPU.sh` and set the `PROMPT_FILE` path:

```bash
PROMPT_FILE="${PROMPT_FILE:-/work/$USER/WORKING/my-project/prompts/my_analysis.md}"
```

The setup script fills in `PROJECT_DIR` and `AGI_ROOT` automatically.

### 3. Submit

```bash
sbatch RUN_AGI_PIPELINE_GPU.sh
```

### 4. Monitor

```bash
# Watch the master job output
tail -f slurm_logs/agi_*.out

# Check your jobs
squeue -u $USER

# Check pipeline status
cat reports/pipeline_status.md
```

### 5. Clean up between runs

```bash
bash CLEAN_PROJECT.sh --dry-run   # Preview
bash CLEAN_PROJECT.sh             # Remove stale artifacts
```

## Available Clusters (ARC)

| Cluster Key | Partition | Nodes | GPU | Max Time | Use Case |
|-------------|-----------|-------|-----|----------|----------|
| `arc_compute1` | compute1 | 65 | — | 3 days | Default CPU subtasks |
| `arc_compute2` | compute2 | 27 | — | 10 days | Long-running CPU |
| `arc_compute3` | compute3 | 6 | — | 3 days | Overflow |
| `arc_gpu1v100` | gpu1v100 | 22 | 1× V100 | 3 days | Default GPU subtasks |
| `arc_gpu2v100` | gpu2v100 | 9 | 2× V100 | 3 days | Multi-GPU |
| `arc_gpu1a100` | gpu1a100 | 2 | 1× A100 | 3 days | Large models |
| `arc_dgxa100` | dgxa100 | 3 | DGX A100 | 3 days | Heavy GPU |
| `zeus_cpu` | normal | — | — | 1 day | Legacy cluster |

Override cluster targets at submission time:

```bash
sbatch --export=AGI_CLUSTER=arc_compute2,AGI_GPU_CLUSTER=arc_gpu1a100 RUN_AGI_PIPELINE_GPU.sh
```

## Key Configuration

### Token Budget (v3.2)

Sized for `qwen3-coder-next:latest` with 32K context window:

| Setting | Value | Purpose |
|---------|-------|---------|
| `MAX_CONTEXT_TOKENS` | 25,000 | Working budget per subtask |
| `MAX_TOOL_OUTPUT_TOKENS` | 12,000 | Max tool output size |
| `MIN_TOKENS_TO_CONTINUE` | 3,000 | Minimum for one more exchange |

### Timeout Guards

| Timeout | Default | Purpose |
|---------|---------|---------|
| `STEP_EXPAND_TIMEOUT` | 300s (5 min) | Single LLM expansion call |
| `TOTAL_DECOMPOSITION_TIMEOUT` | 21,600s (6 hr) | Full prompt decomposition |
| `REVIEW_TIMEOUT` | 120s (2 min) | Failure review LLM call |
| `poll_interval` | 30s | SLURM job status check interval |
| `max_poll_attempts` | 8,640 | 3 days of polling coverage |

### Ollama Model Storage

Models are stored in `/work/$USER/ollama/models` (not `$HOME/.ollama/models`) to avoid home directory quota limits. This is set via `OLLAMA_MODELS` in the RUN scripts.

## Setup Scripts Reference

| Script | Location | Purpose |
|--------|----------|---------|
| `setup/setup.sh` | AGI repo | Initialize a new project directory |
| `setup/RUN_AGI_PIPELINE_GPU.sh` | AGI repo (template) | GPU pipeline submission |
| `setup/RUN_AGI_PIPELINE_CPU.sh` | AGI repo (template) | CPU pipeline submission |
| `setup/CLEAN_PROJECT.sh` | AGI repo | Project cleanup utility |
| `setup/setup_mem0.sh` | AGI repo | Mem0 / reflexion memory setup |

### setup.sh Options

```bash
bash setup.sh                        # Full interactive setup
bash setup.sh --force                 # Overwrite existing files
bash setup.sh --no-git                # Skip git initialization
bash setup.sh --verify                # Check directory structure
bash setup.sh --agi-root /path/to/AGI # Explicit AGI repo path
```

## Troubleshooting

### Ollama 404 (model not found)

The RUN scripts export `OLLAMA_MODELS` to tell Ollama where to find models. If you see 404 errors, check:

```bash
# Verify the path in your RUN script matches where models actually are
grep OLLAMA_MODELS RUN_AGI_PIPELINE_GPU.sh
ls /work/$USER/ollama/models/manifests/
```

### Reflexion memory import error

If you see `Failed to initialize reflexion memory: mem0ai package not installed`, the issue is usually PYTHONPATH, not a missing package. The AGI repo's `main.py` adds itself to `sys.path` automatically, but verify:

```bash
PYTHONPATH=/path/to/AGI:$PYTHONPATH python -c "from memory.reflexion_memory import ReflexionMemory; print('OK')"
```

### Decomposition timeout

With many prompt steps (50+), the total decomposition timeout may be reached before all steps get full LLM expansion. Remaining steps get basic fallback plans. Increase `TOTAL_DECOMPOSITION_TIMEOUT` in `master_agent.py` or reduce prompt granularity.

### GPU partition memory errors

Never specify `--mem` on ARC GPU partitions — it causes allocation failures. The RUN scripts handle this automatically. If submitting manually, omit the `--mem` flag.

### Disk quota on home directory

Keep `$HOME` for conda environments only. Everything else goes to `/work/$USER/`:
- Ollama models: `OLLAMA_MODELS=/work/$USER/ollama/models`
- AGI data: `AGI_DATA_DIR=/work/$USER/agi_data`
- Projects: `/work/$USER/WORKING/`

## File Structure (AGI Repo)

```
AGI/
├── agents/
│   ├── master_agent.py      # Task decomposition + validation
│   ├── sub_agent.py          # Script generation + SLURM submission
│   └── tool_creator.py       # Dynamic tool creation
├── config/
│   ├── config.yaml           # Pipeline configuration
│   └── cluster_config.yaml   # SLURM cluster definitions
├── memory/
│   ├── reflexion_memory.py   # Failure pattern storage
│   └── config.py             # Mem0 configuration
├── engines/
│   └── __init__.py           # Reflexion decision engine
├── setup/
│   ├── setup.sh              # Project directory initializer
│   ├── RUN_AGI_PIPELINE_GPU.sh
│   ├── RUN_AGI_PIPELINE_CPU.sh
│   ├── CLEAN_PROJECT.sh
│   └── setup_mem0.sh
├── tools/
│   ├── sandbox.py            # Execution sandbox
│   ├── slurm_tools.py        # SLURM job management
│   ├── execution_tools.py    # Script execution
│   └── conda_tools.py        # Environment management
├── utils/
│   ├── config_loader.py      # YAML config loading
│   ├── context_manager.py    # Token budget management
│   ├── logging_config.py     # Structured logging
│   └── documentation.py      # Report generation
├── workflows/
│   └── langgraph_workflow.py  # LangGraph state machine
├── main.py                    # Entry point
└── requirements.txt
```

## Version History

- **v3.2** — ARC dual-cluster routing, qwen3-coder-next default, JSON parsing resilience, GPU-aware task routing, token-based context management, decomposition timeouts
- **v3.1** — LangGraph workflow, reflexion memory, SLURM integration
- **v3.0** — Script-first architecture, master prompt living document

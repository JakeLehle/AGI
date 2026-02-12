# AGI Multi-Agent Pipeline

A sophisticated multi-agent AI automation system for research projects running on HPC cluster infrastructure. Built with LangGraph for workflow orchestration, Ollama for local LLM inference, and SLURM for job scheduling.

## Overview

AGI orchestrates multiple AI agents to decompose complex research tasks, generate executable scripts, and manage their execution on HPC clusters. Originally designed for bioinformatics workflows, it's adaptable to any domain requiring automated task execution with intelligent error recovery.

### Key Features

- **Token-Based Context Management**: Each subtask gets a persistent 25K token context window spanning all retry attempts — agents learn from failures without losing history
- **Reflexion Memory**: Semantic memory (Mem0 + Qdrant) prevents infinite loops by detecting semantically similar approaches
- **Script-First Architecture**: Agents generate executable scripts, submit to SLURM, and monitor completion
- **Dual-Cluster GPU Routing**: Automatic CPU/GPU partition selection — GPU tasks route to V100/A100/DGX, CPU tasks to compute partitions
- **Parallel SLURM Execution**: Independent tasks run concurrently with GPU-aware batch limits
- **Resilient JSON Parsing**: 5-strategy fallback chain handles code fences, truncated output, and malformed JSON from local LLMs
- **Automatic Error Classification**: 15 error types with intelligent escalation thresholds
- **Project Isolation**: Clean separation between AGI codebase and project artifacts
- **Fully Self-Hosted**: No external APIs — runs entirely on your infrastructure

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MASTER AGENT                                    │
│              (Task Decomposition, GPU Detection & Orchestration)              │
│                                                                              │
│    Prompt ──► Extract Steps ──► Expand Plans ──► Validate ──► Assign        │
│                                                                              │
│    v3.2: 3-phase decomposition preserves detailed prompt context             │
│          GPU metadata (requires_gpu) tagged per subtask                      │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REFLEXION ENGINE                                           │
│                                                                              │
│   ┌──────────────┐   ┌────────────────────┐   ┌───────────────────────┐    │
│   │    Error      │   │  Semantic Memory   │   │     Escalation        │    │
│   │  Classifier   │   │  (Mem0 + Qdrant)   │   │     Thresholds        │    │
│   └──────────────┘   └────────────────────┘   └───────────────────────┘    │
│                                                                              │
│   Prevents infinite loops via semantic similarity detection                  │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
            ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
            │  SubAgent   │ │  SubAgent   │ │  SubAgent   │  (Parallel)
            │  (Task 1)   │ │  (Task 2)   │ │  (Task 3)   │
            │  [CPU]      │ │  [GPU]      │ │  [CPU]      │
            │ 25K Context │ │ 25K Context │ │ 25K Context │
            │   Window    │ │   Window    │ │   Window    │
            └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
                   │               │               │
                   ▼               ▼               ▼
            ┌─────────────────────────────────────────────────┐
            │              ARC SLURM CLUSTERS                  │
            │                                                  │
            │  CPU: compute1-3        GPU: gpu1v100, gpu1a100  │
            │  (65+ nodes)            dgxa100, gpu2v100, etc.  │
            │                                                  │
            │  Auto-routed via get_cluster_for_task()          │
            └─────────────────────────────────────────────────┘
```

## Token Budget

All token limits are sized for **qwen3-coder-next:latest** with a 32K context window:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `MAX_CONTEXT_TOKENS` | 25,000 | Per-subtask context budget (leaves ~7K for system prompt + response) |
| `MAX_TOOL_OUTPUT_TOKENS` | 12,000 | Max single tool output before pagination |
| `MIN_TOKENS_TO_CONTINUE` | 3,000 | Minimum headroom to attempt another iteration |

Budget breakdown for 32K context:
```
~1K   system prompt + instructions
~25K  accumulated agent context (history, tool outputs, task state)
~3K   current turn input
~3K   response generation
```

Override via environment variables:
```bash
export AGI_MAX_CONTEXT_TOKENS=25000
export AGI_MAX_TOOL_OUTPUT_TOKENS=12000
export AGI_MIN_TOKENS_TO_CONTINUE=3000
```

## Cluster Configuration

### ARC Dual-Cluster Architecture (v3.2)

The pipeline automatically routes subtasks to the appropriate cluster partition based on GPU requirements:

| Cluster | Partition | Nodes | GPUs | Max Time | Use Case |
|---------|-----------|-------|------|----------|----------|
| `arc_compute1` | compute1 | 65 | — | 3 days | Default CPU tasks |
| `arc_compute2` | compute2 | 27 | — | 10 days | Long-running CPU |
| `arc_compute3` | compute3 | 6 | — | 3 days | Overflow CPU |
| `arc_gpu1v100` | gpu1v100 | 22 | 1×V100 | 3 days | Default GPU tasks |
| `arc_gpu2v100` | gpu2v100 | 9 | 2×V100 | 3 days | Multi-GPU |
| `arc_gpu4v100` | gpu4v100 | 2 | 4×V100 | 3 days | Large GPU |
| `arc_gpu1a100` | gpu1a100 | 2 | 1×A100 | 3 days | A100 tasks |
| `arc_dgxa100` | dgxa100 | 3 | DGX | 3 days | DGX system |

**Zeus fallback** (`zeus_cpu`) is available for the original zeus cluster (192 cores, 900GB RAM).

### GPU-Aware Task Routing

The master agent tags each subtask with `requires_gpu` based on package detection:

```python
GPU_PACKAGES = {'torch', 'pytorch', 'tensorflow', 'keras', 'cupy', 'rapids',
                'cuml', 'cugraph', 'jax', 'triton', 'scvi', 'scvi-tools', ...}
```

During parallel batching, GPU tasks are capped at `max_parallel_gpu_jobs` (default: 4) to prevent partition saturation, with remaining batch slots filled by CPU tasks.

### Routing Logic

```
Subtask tagged requires_gpu=True  ──► AGI_GPU_CLUSTER (arc_gpu1v100)
Subtask tagged requires_gpu=False ──► AGI_CLUSTER     (arc_compute1)

GPU memory guard: tasks requesting >40GB VRAM skip V100, route to A100/DGX
```

## Subtask Lifecycle

Each subtask continues until one of three outcomes:

1. **Success** — Task completed, outputs verified
2. **Context exhausted** — 25K token limit reached, no room for another exchange
3. **Escalation** — Reflexion engine detects too many similar failures

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SUBTASK CONTEXT WINDOW                            │
│                      (25,000 tokens default)                         │
│                                                                      │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐         ┌─────────┐     │
│  │ Attempt │───►│ Attempt │───►│ Attempt │───►...──►│ Attempt │     │
│  │   #1    │    │   #2    │    │   #3    │         │   #N    │     │
│  └─────────┘    └─────────┘    └─────────┘         └─────────┘     │
│                                                                      │
│  Full history preserved │ Reflexion prevents semantic duplicates    │
└─────────────────────────────────────────────────────────────────────┘
```

## How Reflexion Memory Works

The Reflexion Engine prevents infinite retry loops:

### Error Classification

```python
class FailureType(Enum):
    CODE_BUG = "code_bug"              # Logic errors
    MISSING_PACKAGE = "missing_package" # Import failures
    MISSING_FILE = "missing_file"       # FileNotFoundError
    OUT_OF_MEMORY = "out_of_memory"     # OOM errors
    GPU_ERROR = "gpu_error"             # CUDA errors
    TIMEOUT = "timeout"                 # Job timeouts
    SLURM_ERROR = "slurm_error"         # SLURM failures
    SYNTAX_ERROR = "syntax_error"       # Python syntax
    PERMISSION_ERROR = "permission_error"
    # ... and more
```

### Escalation Thresholds

| Error Type | Threshold | Rationale |
|------------|-----------|-----------|
| `missing_package` | 2 | Usually fixable with pip install |
| `code_bug` | 3 | May need multiple debugging attempts |
| `out_of_memory` | 2 | Likely needs architectural change |
| `design_flaw` | 1 | Immediate escalation |

### Semantic Duplicate Detection

```python
result = memory.check_if_tried(
    task_id="task_001",
    proposed_approach="Import pandas at the top"
)
# {"tried": True, "similarity": 0.89, "similar_approach": "..."}
```

### Decision Flow

```
Error Occurs
     │
     ▼
┌─────────────────┐
│ Classify Error   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     Yes    ┌─────────────────┐
│ Known Solution?  │───────────►│ APPLY_SOLUTION  │
└────────┬────────┘            └─────────────────┘
         │ No
         ▼
┌─────────────────┐     Yes    ┌─────────────────┐
│ Semantic Dup?    │───────────►│ REJECT_DUPLICATE│
└────────┬────────┘            └─────────────────┘
         │ No
         ▼
┌─────────────────┐     Yes    ┌─────────────────┐
│ Hit Threshold?   │───────────►│    ESCALATE     │
└────────┬────────┘            └─────────────────┘
         │ No
         ▼
┌─────────────────┐
│     RETRY        │
└─────────────────┘
```

## JSON Parsing Resilience (v3.2)

Local LLMs frequently produce malformed JSON. The pipeline uses `parse_json_resilient()` with a 5-strategy fallback chain:

1. **`\`\`\`json` code fence** — Extract and parse JSON from fenced blocks
2. **Bare `\`\`\`` code fence** — Same without language tag
3. **Greedy `{...}` regex** — Find first JSON object in raw text
4. **Trailing garbage strip** — Remove text after last valid `}`
5. **Brace balancing** — Force-close truncated JSON (missing closing braces)

This handles the most common failure modes from qwen3-coder-next:latest and similar local models.

## Installation

### Prerequisites

- Python 3.10+
- Conda (conda-forge channel — no Anaconda commercial license required)
- Ollama
- SLURM (optional, for HPC execution)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/AGI.git
cd AGI

# Create environment (open-source channels only)
conda create -n AGI python=3.10 -c conda-forge -y
conda activate AGI
pip install -r requirements.txt

# Pull Ollama models
ollama pull qwen3-coder-next:latest     # Default coding LLM (v3.2)
ollama pull nomic-embed-text     # Embeddings for reflexion memory

# Create data directory
mkdir -p ~/agi_data/qdrant_storage

# Verify installation
python setup/test_reflexion_integration.py
```

## Usage

### ARC GPU Submission (Recommended)

```bash
# Edit paths in the script, then:
sbatch setup/RUN_AGI_PIPELINE_ARC.sh

# Override cluster targets
sbatch --export=AGI_CLUSTER=arc_compute2,AGI_GPU_CLUSTER=arc_gpu1a100 \
    setup/RUN_AGI_PIPELINE_ARC.sh

# Override model
sbatch --export=OLLAMA_MODEL=llama3.1:70b setup/RUN_AGI_PIPELINE_ARC.sh
```

### Local Execution

```bash
conda activate AGI

python main.py \
    --prompt-file /path/to/prompt.md \
    --project-dir /path/to/project \
    --model qwen3-coder-next:latest
```

### CPU-Only Submission (Zeus)

```bash
# For zeus cluster (no GPU, uses llama3.1:70b by default)
sbatch setup/RUN_AGI_PIPELINE_CPU.sh
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AGI_CLUSTER` | `arc_compute1` | CPU partition for subtask jobs |
| `AGI_GPU_CLUSTER` | `arc_gpu1v100` | GPU partition for subtask jobs |
| `OLLAMA_MODEL` | `qwen3-coder-next:latest` | LLM model for all agents |
| `OLLAMA_CONTEXT_LENGTH` | `32768` | Model context window |
| `AGI_MAX_CONTEXT_TOKENS` | `25000` | Per-subtask token budget |
| `AGI_MAX_TOOL_OUTPUT_TOKENS` | `12000` | Max tool output tokens |
| `AGI_MIN_TOKENS_TO_CONTINUE` | `3000` | Min tokens for retry |
| `USE_REFLEXION_MEMORY` | `true` | Enable/disable Reflexion Memory |
| `AGI_DATA_DIR` | `~/agi_data` | Persistent data (Qdrant, logs) |

## Project Structure

```
AGI/
├── agents/
│   ├── master_agent.py          # Task decomposition + GPU detection
│   ├── sub_agent.py             # Script-first execution + cluster routing
│   └── tool_creator.py          # Dynamic tool generation
├── config/
│   ├── config.yaml              # Pipeline config (25K/12K/3K token budget)
│   ├── cluster_config.yaml      # ARC cluster definitions (8 partitions + zeus)
│   └── mem0_config.yaml         # Reflexion memory config
├── engines/
│   └── __init__.py              # Reflexion Engine
├── memory/
│   ├── reflexion_memory.py      # Mem0 wrapper
│   └── config.py                # Config loader
├── mcp_server/
│   ├── memory_server.py         # Optional HTTP server
│   └── client.py                # Memory client
├── tools/
│   ├── sandbox.py               # Safe file operations
│   ├── conda_tools.py           # Environment management
│   ├── slurm_tools.py           # SLURM integration + cluster routing
│   ├── execution_tools.py       # Script execution
│   └── base_tools.py            # Core file I/O tools
├── utils/
│   ├── config_loader.py         # Typed config access (v3.2 defaults)
│   ├── context_manager.py       # Token management
│   └── reflexion_integration.py # LangGraph helpers
├── workflows/
│   └── langgraph_workflow.py    # Main orchestration + GPU-aware batching
├── setup/
│   ├── RUN_AGI_PIPELINE_ARC.sh  # ARC GPU submission (recommended)
│   ├── RUN_AGI_PIPELINE_CPU.sh  # Zeus CPU submission
│   ├── RUN_AGI_PIPELINE_GPU.sh  # Legacy GPU submission
│   ├── setup.sh                 # Project init script
│   ├── setup_mem0.sh            # Mem0/Qdrant setup
│   └── test_*.py                # Test scripts
├── main.py
├── requirements.txt
└── README.md
```

## Writing Effective Prompts

The master agent preserves detailed prompt context through a 3-phase decomposition process (extract → expand → validate). Write prompts with maximum detail:

```markdown
# Project Goal
[High-level description]

## Input Data
- Location: /path/to/data/
- Format: [CSV, h5ad, FASTQ, etc.]

## Steps

1. **Load and QC the data**
   Load the h5ad file from `data/inputs/sample.h5ad`.
   Use scanpy to filter cells with `min_genes=200` and genes with `min_cells=3`.
   Save QC plots to `data/outputs/qc_plots/`.

2. **Normalize and cluster**
   Use `scanpy.pp.normalize_total(target_sum=1e4)` followed by `log1p`.
   Run leiden clustering with resolution=0.5.
   **This step needs GPU** for scvi-tools integration.

## Expected Outputs
- data/outputs/processed.h5ad
- data/outputs/qc_plots/violin.png
- reports/clustering_summary.md
```

**Tips:**
- Include specific function calls and parameters
- Mention packages explicitly (helps GPU detection)
- Reference input/output file paths
- Note GPU requirements when applicable
- Include code snippets for complex operations

## Changelog

### v3.2 (Current)
- **ARC dual-cluster architecture**: 8 ARC partitions + zeus fallback with automatic GPU/CPU routing
- **qwen3-coder-next:latest default**: Optimized token budgets (25K/12K/3K) for 32K context window
- **GPU-aware parallel batching**: GPU tasks capped at `max_parallel_gpu_jobs` to prevent partition saturation
- **Resilient JSON parsing**: 5-strategy `parse_json_resilient()` handles code fences, truncated output, brace balancing
- **Decomposition timeout infrastructure**: 3-tier timeouts (5min/step, 30min/total, 2min/validation)
- **Transition logging**: All 5 workflow routing points emit structured events
- **GPU metadata tagging**: `detect_requires_gpu()` scans packages/keywords per subtask
- **Config defaults aligned**: All files use consistent 25K/12K/3K, qwen3-coder-next:latest, arc_compute1

### v3.1
- Reflexion Memory integration (Mem0 + Qdrant)
- Script-first SubAgent architecture
- Token-based context management (replaced iteration counts)
- Parallel SLURM execution with dependency resolution

### v3.0
- LangGraph workflow orchestration
- Master agent 3-phase decomposition
- Living document prompt tracking
- Project isolation (AGI code separate from project artifacts)

## License

[Your license here]

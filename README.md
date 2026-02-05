# AGI Multi-Agent Pipeline

A sophisticated multi-agent AI automation system for research projects running on HPC cluster infrastructure. Built with LangGraph for workflow orchestration, Ollama for local LLM inference, and SLURM for job scheduling.

## Overview

AGI orchestrates multiple AI agents to decompose complex research tasks, generate executable scripts, and manage their execution on HPC clusters. Originally designed for bioinformatics workflows, it's adaptable to any domain requiring automated task execution with intelligent error recovery.

### Key Features

- **Token-Based Context Management**: Each subtask gets a persistent 60K token context window spanning all retry attempts—agents learn from failures without losing history
- **Reflexion Memory**: Semantic memory (Mem0 + Qdrant) prevents infinite loops by detecting semantically similar approaches
- **Script-First Architecture**: Agents generate executable scripts, submit to SLURM, and monitor completion
- **Parallel SLURM Execution**: Independent tasks run concurrently across cluster nodes
- **Automatic Error Classification**: 15 error types with intelligent escalation thresholds
- **Project Isolation**: Clean separation between AGI codebase and project artifacts
- **Fully Self-Hosted**: No external APIs—runs entirely on your infrastructure

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MASTER AGENT                                    │
│                     (Task Decomposition & Orchestration)                     │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           REFLEXION ENGINE                                   │
│                                                                              │
│   ┌──────────────┐   ┌────────────────────┐   ┌───────────────────────┐    │
│   │    Error     │   │  Semantic Memory   │   │     Escalation        │    │
│   │  Classifier  │   │  (Mem0 + Qdrant)   │   │     Thresholds        │    │
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
            │             │ │             │ │             │
            │ 60K Context │ │ 60K Context │ │ 60K Context │
            │   Window    │ │   Window    │ │   Window    │
            └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
                   │               │               │
                   ▼               ▼               ▼
            ┌─────────────────────────────────────────────────┐
            │                  SLURM CLUSTER                   │
            │        (Parallel Job Submission & Monitoring)    │
            └─────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.10+
- Conda
- Ollama
- SLURM (optional, for HPC execution)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/AGI.git
cd AGI

# Create environment
conda create -n AGI python=3.10 -y
conda activate AGI
pip install -r requirements.txt

# Pull Ollama models
ollama pull llama3.1:70b        # Main LLM
ollama pull nomic-embed-text    # Embeddings for reflexion memory

# Create data directory
mkdir -p ~/agi_data/qdrant_storage

# Verify installation
python setup/test_reflexion_integration.py
```

## Usage

### Local Execution

```bash
conda activate AGI

python main.py \
    --prompt-file /path/to/prompt.md \
    --project-dir /path/to/project \
    --model llama3.1:70b
```

### SLURM Submission

```bash
cd /path/to/your/project
sbatch RUN_AGI_PIPELINE.sh
```

### GPU vs CPU

Ollama auto-detects GPU availability. For GPU inference on HPC, simply change your SLURM partition:

```bash
# CPU partition
#SBATCH --partition=normal

# GPU partition
#SBATCH --partition=gpu1v100
#SBATCH --gres=gpu:1
```

No changes to the AGI code are required—Ollama handles GPU detection automatically.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `AGI_DATA_DIR` | `~/agi_data` | Qdrant and memory storage |
| `AGI_MAX_CONTEXT_TOKENS` | `60000` | Max tokens per subtask |
| `AGI_MAX_TOOL_OUTPUT_TOKENS` | `25000` | Max tool output before summarization |
| `AGI_MIN_TOKENS_TO_CONTINUE` | `5000` | Min tokens to continue (else escalate) |

### Token-Based Context Limits

Unlike iteration-based systems, AGI uses token budgets. Each subtask continues until:

1. **Success** - Task completed successfully
2. **Context exhausted** - Token limit reached
3. **Escalation** - Reflexion engine escalates (too many similar failures)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SUBTASK CONTEXT WINDOW                            │
│                      (60,000 tokens default)                         │
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
# Agent proposes: "Import pandas at the top of the file"
# Memory contains: "Added import pandas without installing"
# Similarity: 0.89 (threshold: 0.85)
# Decision: REJECT_DUPLICATE

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
│ Classify Error  │──────────────────────────────────┐
└────────┬────────┘                                  │
         │                                           │
         ▼                                           │
┌─────────────────┐     Yes    ┌─────────────────┐  │
│ Known Solution? │───────────►│ APPLY_SOLUTION  │  │
└────────┬────────┘            └─────────────────┘  │
         │ No                                        │
         ▼                                           │
┌─────────────────┐     Yes    ┌─────────────────┐  │
│ Semantic Dup?   │───────────►│ REJECT_DUPLICATE│  │
└────────┬────────┘            └─────────────────┘  │
         │ No                                        │
         ▼                                           │
┌─────────────────┐     Yes    ┌─────────────────┐  │
│ Hit Threshold?  │───────────►│    ESCALATE     │◄─┘
└────────┬────────┘            └─────────────────┘
         │ No
         ▼
┌─────────────────┐
│     RETRY       │
└─────────────────┘
```

## Project Structure

```
AGI/
├── agents/
│   ├── master_agent.py          # Task decomposition
│   └── sub_agent.py             # Script-first execution
├── config/
│   ├── config.yaml              # Pipeline config
│   └── mem0_config.yaml         # Memory config
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
│   └── slurm_tools.py           # SLURM integration
├── utils/
│   ├── reflexion_integration.py # LangGraph helpers
│   └── context_manager.py       # Token management
├── workflows/
│   └── langgraph_workflow.py    # Main orchestration
├── setup/
│   ├── setup.sh                 # Project init script
│   └── test_*.py                # Test scripts
├── main.py
└── requirements.txt
```

## Writing Effective Prompts

```markdown
# Project Goal
[High-level description]

## Input Data
- Location: /path/to/data/
- Format: [CSV, h5ad, FASTQ, etc.]

## Expected Outputs
- output_file.csv
- figures/plot.png

## Constraints
- Use scanpy for single-cell analysis
- Memory limit: 64GB per task

## Reference Scripts (Optional)
- scripts/example_analysis.py
```

## Troubleshooting

### Qdrant Lock Error
```
RuntimeError: Storage folder is already accessed by another instance
```
Kill any running memory processes:
```bash
pkill -f "memory_server.py"
```

### Ollama Not Responding
```bash
# Check status
curl http://localhost:11434/api/tags

# Restart if needed
pkill ollama
ollama serve &
```

### Context Exhausted Too Quickly
- Increase `AGI_MAX_CONTEXT_TOKENS`
- Break large tasks into smaller subtasks
- Reduce verbosity in agent prompts

## Running Tests

```bash
# All tests
python setup/test_reflexion_memory.py
python setup/test_reflexion_engine.py
python setup/test_reflexion_integration.py
```

## Memory Management

```python
from memory import ReflexionMemory

memory = ReflexionMemory()

# View stats
print(memory.get_stats())

# Clear task-specific memory
memory.clear_task_memory("task_001")

# Full reset (caution!)
memory.reset_all(confirm=True)
```

## License

MIT License

## Acknowledgments

- [LangGraph](https://github.com/langchain-ai/langgraph) - Workflow orchestration
- [Ollama](https://ollama.ai) - Local LLM inference  
- [Mem0](https://github.com/mem0ai/mem0) - Memory layer
- [Qdrant](https://qdrant.tech) - Vector database

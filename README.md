# Multi-Agent Automation System

A self-directed, locally-run multi-agent system that decomposes complex tasks into subtasks, executes them with reflection and retry capabilities, and produces comprehensive documentation of all actions taken. Designed for HPC environments with SLURM integration for parallel job execution on CPU and GPU clusters.

---

## Table of Contents

- [Vision \& Overview](#vision--overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Prompt File Format](#prompt-file-format)
- [SLURM Integration](#slurm-integration)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## Vision & Overview

### The Problem

Complex research and data analysis tasks often require:
- Multiple steps with dependencies
- Different tools and environments
- Iterative refinement when approaches fail
- Comprehensive documentation for reproducibility
- Efficient use of HPC resources

### The Solution

This system provides a **self-directed automation framework** that:

1. **Decomposes** high-level tasks into specific, actionable subtasks
2. **Executes** each subtask with appropriate tools and environments
3. **Reflects** on results and retries with improved strategies if needed
4. **Documents** every action for full traceability and reproducibility
5. **Scales** across HPC clusters via SLURM for parallel execution

### Design Philosophy

- **Local-First**: Uses Ollama for local LLM inference—no API keys or cloud dependencies
- **Sandboxed Execution**: All file operations are restricted to project directories
- **Self-Healing**: Agents can reflect on failures and try alternative approaches
- **Transparent**: Every decision and action is logged and committed to Git
- **HPC-Native**: First-class support for SLURM job submission on CPU and GPU clusters

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Task Decomposition** | Master agent breaks complex tasks into manageable subtasks |
| **Iterative Execution** | Sub-agents execute with up to 12 retry iterations per subtask |
| **Self-Reflection** | Agents analyze failures and develop improved strategies |
| **Dynamic Tool Creation** | Can generate new tools when existing ones are insufficient |
| **Sandboxed Environments** | Per-project conda environments with automatic setup |
| **SLURM Integration** | Submit jobs to CPU and GPU clusters with dependency management |
| **Parallel Execution** | Run independent subtasks concurrently for faster completion |
| **Git Tracking** | Every action creates commits for full audit trail |
| **Auto-Documentation** | Generates comprehensive README with execution history |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MULTI-AGENT SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌──────────────────────────────────────────────────┐  │
│  │   PROMPT    │────▶│              MASTER AGENT                        │  │
│  │   FILE      │     │  • Decomposes task into subtasks                 │  │
│  └─────────────┘     │  • Manages dependencies between subtasks         │  │
│                      │  • Reviews failures and decides next steps       │  │
│                      │  • Generates final report                        │  │
│                      └────────────────┬─────────────────────────────────┘  │
│                                       │                                     │
│                                       ▼                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐│
│  │                         LANGGRAPH WORKFLOW                             ││
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐            ││
│  │  │ Decompose│──▶│ Identify │──▶│ Execute  │──▶│ Reflect  │──┐         ││
│  │  │          │   │ Parallel │   │ Subtasks │   │ & Retry  │  │         ││
│  │  └──────────┘   └──────────┘   └──────────┘   └────┬─────┘  │         ││
│  │                                                     │        │         ││
│  │                                      ┌──────────────┘        │         ││
│  │                                      ▼                       │         ││
│  │                               ┌──────────┐                   │         ││
│  │                               │ Complete │◀──────────────────┘         ││
│  │                               └──────────┘                             ││
│  └────────────────────────────────────────────────────────────────────────┘│
│                                       │                                     │
│                                       ▼                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                          SUB-AGENTS                                   │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │  │  For each subtask:                                              │ │  │
│  │  │  1. Create execution plan using LLM                             │ │  │
│  │  │  2. Execute steps (write scripts, run commands, web search)     │ │  │
│  │  │  3. Reflect on results                                          │ │  │
│  │  │  4. If failed: improve strategy and retry (up to 12 iterations) │ │  │
│  │  │  5. If still failed: escalate to Master Agent                   │ │  │
│  │  └─────────────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                       │                                     │
│                    ┌──────────────────┼──────────────────┐                  │
│                    ▼                  ▼                  ▼                  │
│  ┌──────────────────────┐ ┌────────────────────┐ ┌────────────────────┐    │
│  │   EXECUTION TOOLS    │ │    SLURM TOOLS     │ │   CONDA TOOLS      │    │
│  │  • Write scripts     │ │  • Submit sbatch   │ │  • Create envs     │    │
│  │  • Run commands      │ │  • Monitor jobs    │ │  • Install pkgs    │    │
│  │  • Web search        │ │  • Collect output  │ │  • Manage deps     │    │
│  │  • File I/O          │ │  • GPU support     │ │  • Export YAML     │    │
│  └──────────────────────┘ └────────────────────┘ └────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OUTPUT                                         │
│  • Project directory with all generated files                               │
│  • Conda environment YAML for reproducibility                               │
│  • Git history with detailed commit messages                                │
│  • Execution logs (JSON format)                                             │
│  • Auto-generated README documenting all actions                            │
│  • Final report summarizing results                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## System Requirements

### Hardware

- **CPU**: Multi-core processor (recommended: 8+ cores for local LLM inference)
- **RAM**: 64GB+ recommended (for running llama3.1:70b locally)
- **Storage**: 50GB+ for models and project data

### Software

- **OS**: Linux (tested on Ubuntu 22.04/24.04, CentOS 7/8, Rocky Linux)
- **Python**: 3.10+
- **Conda**: Miniconda or Anaconda
- **Git**: 2.0+
- **Ollama**: Latest version (for local LLM inference)

### Optional (HPC)

- **SLURM**: For cluster job submission
- **CUDA**: 11.8+ (for GPU nodes)

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/JakeLehle/AGI.git
cd AGI
```

### Step 2: Create Conda Environment

```bash
# Create the AGI environment from the environment file
conda env create -f environment.yml

# Activate the environment
conda activate AGI
```

**Or manually:**

```bash
# Create environment
conda create -n AGI python=3.10 -y
conda activate AGI

# Install all packages from conda-forge
conda install -c conda-forge \
    langchain langchain-community langgraph ollama \
    pandas numpy requests beautifulsoup4 lxml \
    pyyaml gitpython loguru duckduckgo-search jsonschema -y
```

### Step 3: Install Ollama

Ollama runs the LLM locally. Install it separately (not a Python package):

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**macOS:**
```bash
# Download from https://ollama.com/download
# Or use Homebrew:
brew install ollama
```

**Verify installation:**
```bash
ollama --version
```

### Step 4: Pull the LLM Model

```bash
# Pull the recommended model (requires ~40GB disk space, ~64GB RAM to run)
ollama pull llama3.1:70b

# Or use a smaller model for testing (requires ~4GB disk, ~8GB RAM)
ollama pull llama3.1:8b
```

### Step 5: Start Ollama Server

```bash
# Start the Ollama server (runs on port 11434 by default)
ollama serve

# Or run in background
nohup ollama serve > /dev/null 2>&1 &
```

**For HPC systems**, you may want to run Ollama on a dedicated node:
```bash
# On the Ollama server node
ollama serve --host 0.0.0.0

# Then set the URL in config/config.yaml:
# ollama:
#   base_url: "http://ollama-node:11434"
```

### Step 6: Verify Installation

```bash
# Activate environment
conda activate AGI

# Test Ollama connection
python -c "import ollama; print(ollama.list())"

# Test the system (dry run)
python main.py --task "Test task" --project-dir ./test_project --dry-run
```

---

## Configuration

The system is configured via `config/config.yaml`. Key sections:

### Ollama Settings

```yaml
ollama:
  model: "llama3.1:70b"    # Model to use (must be pulled first)
  base_url: "http://127.0.0.1:11434"  # Ollama server URL
```

### Agent Settings

```yaml
agents:
  max_retries: 12          # Max iterations per subtask
  timeout_seconds: 300     # Timeout for individual operations
  enable_dynamic_tools: true  # Allow agents to create new tools
```

### SLURM Settings

```yaml
slurm:
  enabled: true
  default_cluster: "zeus"  # Default cluster to use
  poll_interval: 10        # Seconds between job status checks
```

### Cluster Configurations

The config includes two pre-configured clusters:

**Zeus (CPU cluster):**
```yaml
clusters:
  zeus:
    name: "zeus"
    cores_per_node: 192
    memory_per_node: "1000G"
    default_partition: "normal"
    default_cpus: 4
    default_memory: "16G"
    default_time: "04:00:00"
```

**GPU Cluster:**
```yaml
clusters:
  gpu_cluster:
    name: "gpu_cluster"
    cores_per_node: 80
    memory_per_node: "256G"
    has_gpu: true
    default_partition: "compute1"
    partitions:
      gpu1v100:
        max_gpus: 4
        gpu_type: "v100"
      gpu1a100:
        max_gpus: 4
        gpu_type: "a100"
      dgxa100:
        max_gpus: 8
        gpu_type: "a100"
```

---

## Usage

### Basic Usage

```bash
# Activate environment
conda activate AGI

# Run with inline task
python main.py --task "Analyze sales data and create visualizations" \
    --project-dir ./sales_analysis

# Run with prompt file
python main.py --prompt-file prompts/my_task.txt \
    --project-dir ./my_project
```

### Interactive Mode (No SLURM)

```bash
# Run locally without SLURM
python main.py --prompt-file prompts/analysis.txt \
    --project-dir ./analysis \
    --no-slurm
```

### SLURM Mode - CPU Cluster

```bash
# Submit to zeus cluster (CPU)
python main.py --prompt-file prompts/analysis.txt \
    --project-dir ./analysis \
    --slurm \
    --cluster zeus \
    --cpus 16 \
    --memory 64G \
    --time 08:00:00

# Use all cores on a node
python main.py --prompt-file prompts/heavy_compute.txt \
    --project-dir ./compute \
    --slurm \
    --cluster zeus \
    --cpus 192 \
    --memory 900G
```

### SLURM Mode - GPU Cluster

```bash
# V100 GPUs (4 GPUs)
python main.py --prompt-file prompts/ml_training.txt \
    --project-dir ./ml_project \
    --slurm \
    --cluster gpu_cluster \
    --partition gpu1v100 \
    --gpus 4 \
    --cpus 40 \
    --memory 256G

# A100 GPUs (higher memory, faster)
python main.py --prompt-file prompts/large_model.txt \
    --project-dir ./llm_training \
    --slurm \
    --cluster gpu_cluster \
    --partition gpu1a100 \
    --gpus 4 \
    --memory 512G

# DGX A100 (8 GPUs, premium)
python main.py --prompt-file prompts/distributed.txt \
    --project-dir ./distributed \
    --slurm \
    --cluster gpu_cluster \
    --partition dgxa100 \
    --gpus 8 \
    --cpus 128 \
    --memory 900G

# Specific node
python main.py --prompt-file prompts/debug.txt \
    --project-dir ./debug \
    --slurm \
    --cluster gpu_cluster \
    --partition gpu1v100 \
    --nodelist gpu004 \
    --gpus 1
```

### Utility Commands

```bash
# List available clusters
python main.py --list-clusters --project-dir ./test

# Check cluster status
python main.py --cluster-status --cluster gpu_cluster --project-dir ./test

# Check specific partition
python main.py --cluster-status --cluster gpu_cluster --partition gpu1a100 --project-dir ./test

# Dry run (validate without executing)
python main.py --prompt-file prompts/task.txt --project-dir ./test --dry-run
```

### CLI Options Reference

| Option | Description |
|--------|-------------|
| `--task` | Inline task description |
| `--prompt-file` | Path to prompt file |
| `--project-dir` | Project directory (required) |
| `--cluster` | Cluster name (zeus, gpu_cluster) |
| `--partition` | SLURM partition |
| `--slurm` / `--no-slurm` | Enable/disable SLURM |
| `--cpus` | CPUs per job |
| `--memory` | Memory per job (e.g., "64G") |
| `--time` | Time limit (e.g., "04:00:00") |
| `--gpus` | Number of GPUs |
| `--gpu-type` | GPU type (v100, a100) |
| `--nodelist` | Specific node(s) to use |
| `--exclude` | Node(s) to exclude |
| `--parallel` / `--no-parallel` | Enable/disable parallel execution |
| `--dry-run` | Validate without executing |
| `--verbose` | Enable verbose output |

---

## Prompt File Format

Prompt files use a simple markdown-like format:

```markdown
# Task Description

[Your main task description here. Be specific about what you want to accomplish.
The more detail you provide, the better the system can plan and execute.]

# Input Files

- data/inputs/file1.csv
- data/inputs/file2.json

# Expected Outputs

- data/outputs/results.csv
- reports/analysis_report.md

# Context

[Additional context, constraints, or requirements]

Focus areas:
- Specific area 1
- Specific area 2

Constraints:
- Constraint 1
- Constraint 2

Notes:
- Any other relevant information
```

### Example: Data Analysis Task

```markdown
# Task Description

Analyze the quarterly sales data for Q4 2024. Calculate total revenue by region,
identify top-performing products, and detect any anomalies in the data. Generate
visualizations for the executive summary.

# Input Files

- data/inputs/sales_q4_2024.csv
- data/inputs/product_catalog.json
- data/inputs/regional_targets.xlsx

# Expected Outputs

- data/outputs/regional_summary.csv
- data/outputs/top_products.json
- data/outputs/anomalies_detected.csv
- reports/q4_analysis.md
- reports/visualizations/revenue_by_region.png
- reports/visualizations/product_performance.png

# Context

Analysis requirements:
- Group sales by region (North, South, East, West)
- Calculate YoY growth where previous year data exists
- Flag any single-day revenue drops > 30% as anomalies

Tools to use:
- Python with pandas for data processing
- matplotlib or seaborn for visualizations

Output format:
- All monetary values in USD with 2 decimal places
- Dates in ISO format (YYYY-MM-DD)
- Final report should be suitable for executive presentation
```

### Example: Machine Learning Task (GPU)

```markdown
# Task Description

Fine-tune a BERT model for sentiment classification on the customer feedback dataset.
Evaluate model performance and save the best checkpoint.

# Input Files

- data/inputs/customer_feedback.csv
- data/inputs/labels.json

# Expected Outputs

- models/sentiment_bert_finetuned/
- data/outputs/predictions.csv
- reports/training_metrics.json
- reports/model_evaluation.md

# Context

Model specifications:
- Base model: bert-base-uncased
- Max sequence length: 256
- Batch size: 32 (adjust based on GPU memory)
- Learning rate: 2e-5
- Epochs: 3

GPU requirements:
- Minimum: 1x V100 (32GB)
- Recommended: 2x V100 or 1x A100

Evaluation metrics:
- Accuracy, Precision, Recall, F1
- Confusion matrix
- Per-class performance breakdown
```

---

## SLURM Integration

### How It Works

When SLURM is enabled, the system:

1. **Generates sbatch scripts** with proper resource requests
2. **Submits jobs** to the cluster queue
3. **Manages dependencies** between sequential tasks
4. **Monitors job status** via squeue/sacct
5. **Collects output** from SLURM log files
6. **Handles failures** with automatic retry or escalation

### Generated sbatch Script Example

```bash
#!/bin/bash
#SBATCH -J task_analysis              # Job name
#SBATCH -p gpu1v100                   # Partition
#SBATCH -N 1                          # 1 node
#SBATCH -n 1                          # 1 task (not MPI)
#SBATCH -c 40                         # 40 cores for threading
#SBATCH --mem=256G                    # Memory
#SBATCH -t 3-00:00:00                 # 3 days
#SBATCH --gres=gpu:v100:4             # 4 V100 GPUs
#SBATCH -o slurm/logs/task_%j.out
#SBATCH -e slurm/logs/task_%j.err

# GPU environment setup
echo "=== GPU Information ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate agi_project_20250124

# Run the task
python scripts/train_model.py
```

### Cluster Quick Reference

| Cluster | Partition | CPUs | Memory | GPUs | Max Time |
|---------|-----------|------|--------|------|----------|
| zeus | normal | 192 | 1TB | - | 7 days |
| zeus | interactive | 192 | 1TB | - | unlimited |
| gpu_cluster | compute1 | 80 | 256G | - | 3 days |
| gpu_cluster | compute2 | 80 | 256G | - | 10 days |
| gpu_cluster | bigmem | 80 | 1TB | - | 3 days |
| gpu_cluster | gpu1v100 | 40 | 256G | 4× V100 | 3 days |
| gpu_cluster | gpu1a100 | 64 | 512G | 4× A100 | 3 days |
| gpu_cluster | dgxa100 | 128 | 1TB | 8× A100 | unlimited |

---

## Project Structure

After running a task, your project directory will look like:

```
my_project/
├── data/
│   ├── inputs/           # Input files
│   └── outputs/          # Generated output files
├── scripts/              # Generated Python/R/bash scripts
├── reports/              # Generated reports and summaries
├── logs/                 # Execution logs (JSON format)
│   ├── execution_log.jsonl
│   ├── agent_activity.jsonl
│   └── errors.jsonl
├── envs/                 # Conda environment exports
│   └── environment.yml
├── slurm/                # SLURM-related files (if enabled)
│   ├── scripts/          # Generated sbatch scripts
│   └── logs/             # Job stdout/stderr
├── temp/                 # Temporary files
├── prompts/              # Archived prompt files
├── README.md             # Auto-generated documentation
├── workflow_state.db     # State checkpoint database
└── .git/                 # Git repository with full history
```

---

## Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve

# Check available models
ollama list
```

### SLURM Job Failures

```bash
# Check job status
squeue -u $USER

# View job output
cat slurm/logs/job_name_JOBID.out
cat slurm/logs/job_name_JOBID.err

# Check job accounting
sacct -j JOBID --format=JobID,State,ExitCode,MaxRSS,Elapsed
```

### Memory Issues

```bash
# For local execution, ensure enough RAM for the model
free -h

# For SLURM, request more memory
python main.py ... --memory 128G
```

### Conda Environment Issues

```bash
# Recreate environment
conda env remove -n AGI
conda env create -f environment.yml

# Update environment
conda env update -f environment.yml --prune
```

### Common Errors

| Error | Solution |
|-------|----------|
| `Connection refused` | Start Ollama server: `ollama serve` |
| `Model not found` | Pull model: `ollama pull llama3.1:70b` |
| `Out of memory` | Use smaller model or request more RAM |
| `SLURM not available` | Run without SLURM: `--no-slurm` |
| `Permission denied` | Check project directory permissions |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Submit a pull request

---

## License

[MIT License](LICENSE)

---

## Acknowledgments

- [LangChain](https://langchain.com/) - LLM application framework
- [LangGraph](https://langchain-ai.github.io/langgraph/) - Stateful agent workflows
- [Ollama](https://ollama.ai/) - Local LLM inference
- [Loguru](https://github.com/Delgan/loguru) - Python logging made simple

---

## Contact

For questions or support, please open an issue on GitHub.

# Quick Start Guide

Get the Multi-Agent Automation System running in 5 minutes.

---

## 1. Clone & Setup Environment

```bash
# Clone repository
git clone https://github.com/yourusername/multi-agent-system.git
cd multi-agent-system

# Create conda environment
conda env create -f environment.yml
conda activate AGI
```

## 2. Install Ollama

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**macOS:**
```bash
brew install ollama
# Or download from https://ollama.com/download
```

## 3. Pull LLM Model & Start Server

```bash
# Start Ollama server (in a separate terminal or background)
ollama serve &

# Pull the model (choose one)
ollama pull llama3.1:70b    # Full model (~40GB, needs 64GB RAM)
ollama pull llama3.1:8b     # Smaller model (~4GB, needs 8GB RAM)
```

## 4. Run Your First Task

```bash
# Verify setup
python main.py --task "Test task" --project-dir ./test --dry-run

# Run a real task (interactive mode)
python main.py --task "Analyze sample data and create a summary report" \
    --project-dir ./my_project

# Run with SLURM (HPC)
python main.py --prompt-file prompts/example_prompt.txt \
    --project-dir ./my_project \
    --slurm
```

---

## Common Commands

```bash
# List clusters (HPC)
python main.py --list-clusters --project-dir .

# Check cluster status
python main.py --cluster-status --cluster zeus --project-dir .

# GPU job submission
python main.py --prompt-file prompts/ml_task.txt \
    --project-dir ./ml_project \
    --slurm \
    --cluster gpu_cluster \
    --partition gpu1v100 \
    --gpus 4

# Verify installation
./setup.sh --verify
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Connection refused` | Start Ollama: `ollama serve` |
| `Model not found` | Pull model: `ollama pull llama3.1:70b` |
| `Out of memory` | Use smaller model: `llama3.1:8b` |
| Import errors | Recreate env: `conda env create -f environment.yml` |

---

See [README.md](README.md) for complete documentation.

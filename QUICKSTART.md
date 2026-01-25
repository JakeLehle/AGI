# Quick Start Guide (HPC Systems)

Get the Multi-Agent Automation System running on your HPC cluster.

---

## Prerequisites

- **Ollama** must be installed **system-wide** by your HPC administrators
- Request installation via ticket if not available:
  ```
  curl -fsSL https://ollama.com/install.sh | sh
  ```

---

## 1. Clone & Setup Environment

```bash
# Clone repository
git clone https://github.com/JakeLehle/AGI.git
cd AGI

# Create conda environment
conda env create -f environment.yml
conda activate AGI
```

## 2. Get a Compute Node

**GPU Node (recommended for 70b model):**
```bash
# V100 - do NOT specify --mem
srun --partition=gpu1v100 --gres=gpu:1 -N 1 -n 1 -c 10 --time=04:00:00 --pty bash

# Once on node:
module load anaconda3
module load cudatoolkit
conda activate AGI

# IMPORTANT: Unset SLURM GPU variables
unset CUDA_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL
```

**CPU Node (for testing with 8b model):**
```bash
srun --partition=compute2 -N 1 -n 1 -c 40 --time=08:00:00 --pty bash

# Once on node:
module load anaconda3
conda activate AGI
```

## 3. Start Ollama & Pull Model

```bash
# Start server in background
ollama serve > /dev/null 2>&1 &
sleep 5

# Pull model
ollama pull llama3.1:8b     # CPU testing (~4GB)
ollama pull llama3.1:70b    # Full model, needs GPU (~40GB)

# Verify
curl http://localhost:11434/api/tags
```

## 4. Run Your First Task

```bash
cd /path/to/AGI

# Dry run (validates setup)
python main.py --task "Test task" --project-dir ./test_project --dry-run

# Real task (interactive, no SLURM)
python main.py --task "Analyze sample data and create a summary" \
    --project-dir ./my_project \
    --no-slurm

# With SLURM job submission
python main.py --prompt-file prompts/example_prompt.txt \
    --project-dir ./my_project \
    --slurm
```

---

## Common Commands

```bash
# List clusters
python main.py --list-clusters --project-dir .

# Check cluster status
python main.py --cluster-status --cluster zeus --project-dir .

# GPU job
python main.py --prompt-file prompts/ml_task.txt \
    --project-dir ./ml_project \
    --slurm --cluster gpu_cluster --partition gpu1v100 --gpus 4

# Verify setup
./setup.sh --verify
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Connection refused` | Start Ollama: `ollama serve &` |
| `Model not found` | Pull model: `ollama pull llama3.1:8b` |
| `total vram = 0 B` | Unset CUDA vars, ensure cudatoolkit loaded |
| `Memory specification error` | Remove `--mem` from srun command |
| Import errors | Recreate env: `conda env create -f environment.yml` |

---

See [README.md](README.md) for complete documentation.

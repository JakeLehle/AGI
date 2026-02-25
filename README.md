# AGI Multi-Agent Pipeline

> Automated computational workflow orchestration for HPC clusters using local LLMs.

The AGI pipeline decomposes complex research prompts into executable scripts, submits them as SLURM jobs, monitors completion, and handles failures through a multi-layer diagnostic and retry system — all driven by a local LLM running on a GPU node.

## Architecture (v1.2.4)

```
┌────────────────────────────────────────────────────────────┐
│  GPU Node (master pipeline)                                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Ollama Server │◄──│ Master Agent │───►│  Sub-Agents  │  │
│  │ (qwen3-coder)│    │ Decompose    │    │ Phase 1-4    │  │
│  └──────────────┘    │ Validate     │    │ (threaded)   │  │
│                      │ Track State  │    └──────┬───────┘  │
│                      └──────────────┘           │          │
│                              ▲                  │ sbatch   │
│                    ┌─────────┴────────┐         │          │
│                    │ Diagnostic Agent │◄────────┘          │
│                    │ Error classify   │  (on failure)      │
│                    │ Fix prescribe    │                    │
│                    │ Solution memory  │                    │
│                    └──────────────────┘                    │
└─────────────────────────────────────────────────┬──────────┘
                                                  │
                          ┌───────────────────────┼──────────────────┐
                          │                       │                  │
                    ┌─────▼─────┐          ┌──────▼──────┐    ┌─────▼─────┐
                    │ compute1  │          │  gpu1v100   │    │ compute2  │
                    │ CPU tasks │          │  GPU tasks  │    │ Long jobs │
                    └───────────┘          └─────────────┘    └───────────┘
```

The master pipeline runs on a GPU node for fast LLM inference. Independent subtasks run concurrently in a thread pool (default 4 threads), each managing its own full 4-phase lifecycle. When a SLURM job fails, the Diagnostic Agent investigates, prescribes a fix, and the sub-agent applies it before retrying — without human intervention for known error patterns.

## Sub-Agent 4-Phase Lifecycle

Each subtask goes through four independent phases with its own checkpoint, allowing resumption from any phase after a crash or restart:

| Phase | What happens | Checkpoint key |
|-------|-------------|----------------|
| **Phase 1** — Script Generation | Two-pass outline → validation → full script → validation | `script_validated` |
| **Phase 2** — Environment Creation | LLM dependency analysis → conda YAML → env build + repair loop | `env_created` |
| **Phase 3** — Sbatch Generation | Language dispatch, GPU routing, input file argument injection | `sbatch_path` |
| **Phase 4** — Execution + Diagnostics | Submit → monitor → classify error → diagnose → fix → retry | `current_job_id` |

## Prerequisites

- Python 3.10+
- Conda (conda-forge/bioconda channels only — no defaults/main)
- Ollama (system-wide install for GPU CUDA support)
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
#    Models must be on /work (not $HOME) to avoid quota limits
export OLLAMA_MODELS="/work/$USER/ollama/models"

# 4. Pull models
ollama pull qwen3-coder:latest      # Primary coding LLM (~20 GiB, fits V100S-32GB)
ollama pull nomic-embed-text        # Embeddings for reflexion + diagnostic memory

# 5. Create data directories
mkdir -p /work/$USER/agi_data/qdrant_storage

# 6. Verify
python -c "from workflows.langgraph_workflow import MultiAgentWorkflow; print('OK')"
```

## Setting Up a New Project

Every analysis gets its own project directory, completely separate from this AGI repo. The setup script creates the full structure and copies pre-configured RUN scripts with paths filled in:

```bash
mkdir /work/$USER/WORKING/my-new-project
cd /work/$USER/WORKING/my-new-project
bash /path/to/AGI/setup/setup.sh
```

This creates:

```
my-new-project/
├── config/                  # Project-specific config (tracked)
├── conda_env/               # Base env YAML specs (tracked)
├── data/
│   ├── inputs/              # Your input data
│   └── outputs/             # Pipeline outputs
├── envs/                    # Auto-generated step env YAMLs (gitignored)
├── logs/                    # Agent + Ollama logs (gitignored)
├── prompts/                 # Master prompt .md files (tracked)
├── reports/                 # Pipeline state + status reports (gitignored)
├── scripts/                 # Generated + user scripts (tracked)
│   └── example_reference_scripts/
├── slurm/
│   ├── logs/                # Subtask SLURM stdout/stderr (gitignored)
│   └── scripts/             # Generated sbatch scripts (gitignored)
├── slurm_logs/              # Master job stdout/stderr (gitignored)
├── temp/checkpoints/        # Per-step resume checkpoints (gitignored)
├── project.yaml             # Project configuration (tracked)
├── RUN_AGI_PIPELINE_GPU.sh  # ← Submit this to run the pipeline
├── RUN_AGI_PIPELINE_CPU.sh  # CPU-only alternative
├── FULL_CLEAN_PROJECT.sh    # Complete wipe for a fresh start
├── PARTIAL_CLEAN_PROJECT.sh # Log cleanup preserving pipeline state
└── INJECT_HINTS.sh          # Human guidance injection between runs
```

## Running the Pipeline

### 1. Write your master prompt

Create a detailed `.md` file in `prompts/`. The more specific you are about packages, expected inputs/outputs, and implementation approach, the better the decomposition:

```bash
vi prompts/my_analysis.md
```

### 2. Update the RUN script

Open `RUN_AGI_PIPELINE_GPU.sh` and set the `PROMPT_FILE` path. The setup script fills in `PROJECT_DIR` and `AGI_ROOT` automatically:

```bash
PROMPT_FILE="${PROMPT_FILE:-/work/$USER/WORKING/my-project/prompts/my_analysis.md}"
```

### 3. Submit

```bash
sbatch RUN_AGI_PIPELINE_GPU.sh
```

### 4. Monitor

```bash
# Watch the master job live
tail -f slurm_logs/agi_*.out

# Check all your queued/running jobs
squeue -u $USER

# Human-readable pipeline state
cat reports/pipeline_status.md
```

---

## Between-Run Workflow

This is the most important operational section. After a run completes or stalls, you have three tools to manage the project state before resubmitting.

### Choosing the right cleanup

| Situation | Tool to use |
|-----------|-------------|
| Changing the master prompt significantly, starting completely fresh | `FULL_CLEAN_PROJECT.sh` |
| Some steps completed, want to resume and fix only what failed | `PARTIAL_CLEAN_PROJECT.sh` + `INJECT_HINTS.sh` |

---

### Full clean — complete fresh start

Wipes all generated state. Use this when you want to re-run everything from the beginning.

**Removes:**
- All logs (`logs/`, `slurm/logs/`, `slurm_logs/`)
- All checkpoints (`temp/checkpoints/`)
- Pipeline state (`reports/master_prompt_state.json`, `reports/pipeline_status.md`)
- All generated sbatch scripts (`slurm/scripts/`)
- All generated conda env YAMLs (`envs/step_*.yml`)
- All conda environments matching `agi_step_*`
- All conda package cache (`conda clean --all`)

**Preserves:** master prompts, scripts, data, config, conda_env/ base specs

```bash
bash FULL_CLEAN_PROJECT.sh --dry-run   # preview first
bash FULL_CLEAN_PROJECT.sh             # interactive with confirmation
bash FULL_CLEAN_PROJECT.sh --yes       # skip confirmation
```

---

### Partial clean — preserve state, remove logs

Removes only logs and safe intermediates. Pipeline state (completed steps, checkpoints, conda envs) is fully preserved so the next run resumes rather than restarts.

**Removes:**
- All logs (`logs/`, `slurm/logs/`, `slurm_logs/`)
- Generated sbatch scripts (will be regenerated from checkpoints on restart)
- Generated prompt JSON files (`prompts/prompt_*.json`)
- Pipeline status markdown (`reports/pipeline_status.md`)

**Preserves:**
- `reports/master_prompt_state.json` — which steps completed/failed/pending
- `temp/checkpoints/step_*_checkpoint.json` — phase progress per step
- `envs/step_*.yml` — conda env YAML specs
- All `agi_step_*` conda environments (reusing existing envs saves significant time)
- All scripts and data

```bash
bash PARTIAL_CLEAN_PROJECT.sh --dry-run   # shows pipeline state summary + what will be removed
bash PARTIAL_CLEAN_PROJECT.sh             # interactive
bash PARTIAL_CLEAN_PROJECT.sh --yes       # skip confirmation
```

The `--dry-run` output is particularly useful — it prints the current status of every step with error summaries so you can assess what failed before deciding what to do next.

> **Note on SLURM logs:** The diagnostic agent reads logs by SLURM job ID from the current execution only. It never looks at logs from previous runs. All SLURM logs are safe to delete between runs and should be deleted to manage disk quota.

---

### Injecting human guidance before restart

After a partial clean, use `INJECT_HINTS.sh` to inject corrective guidance into failed steps before resubmitting. This edits `reports/master_prompt_state.json` safely via Python — no manual JSON editing required.

```bash
bash INJECT_HINTS.sh              # interactive form
bash INJECT_HINTS.sh --dry-run    # preview all changes without writing
```

The tool shows the current pipeline state and error summaries, asks which steps to address (up to 4), then presents a menu for each:

| Option | What it does | When to use |
|--------|-------------|-------------|
| **1 — Add implementation hints** | Appends specific instructions to the script generation prompt | LLM chose wrong method, wrong key name, wrong file pattern |
| **2 — Fix input file paths** | Corrects `input_files` in state + optionally adds a load hint | Script received no input argument, paths were empty or wrong |
| **3 — Override approach** | Replaces `expanded_plan` entirely with your description | LLM chose a fundamentally wrong implementation strategy |
| **4 — Skip step** | Marks as completed to unblock downstream steps | Step is non-critical and blocking everything downstream |
| **5 — Reset and retry** | Clears failed status, resets attempt count | Transient error (disk full, node crash) that is now resolved |

For each change you also choose what happens to the step's checkpoint:

- **Delete checkpoint** — forces full restart from Phase 1 (script regeneration). Use after options 1, 2, 3.
- **Keep checkpoint** — resumes from the last phase reached. Use after option 5 for transient errors.
- **Delete only sbatch** — keeps the conda env and script, regenerates only the sbatch. Use after option 2 when the script is correct but the launch command was missing arguments.

**Example hints that work well:**

```
Use scanpy.read_h5ad(sys.argv[1]) to load the input file from the command line
The AnnData object uses obs['cell_type'] not obs['celltype']
Save output to data/outputs/processed/ not to scripts/
Use sc.pp.neighbors() before sc.tl.leiden() — neighbors must be computed first
The popV model requires the reference atlas to be loaded separately via popv.utils.load_reference()
```

---

### Standard restart workflow

```bash
# 1. Review what failed (no changes made)
bash INJECT_HINTS.sh --dry-run

# 2. Inject guidance for failed steps
bash INJECT_HINTS.sh

# 3. Clear logs, keep pipeline state
bash PARTIAL_CLEAN_PROJECT.sh --yes

# 4. Resubmit
sbatch RUN_AGI_PIPELINE_GPU.sh
```

For specific steps where you also want to force script regeneration (not just retry):

```bash
# Delete the checkpoint for a specific step to force Phase 1 restart
rm temp/checkpoints/step_5_checkpoint.json

# Then do the standard restart workflow above
```

---

## Diagnostic Agent

When a SLURM job fails, the Diagnostic Agent is automatically invoked before any retry. It classifies the error, checks its solution memory for known fixes, and if none are found, runs a type-specific investigation.

### Error types handled

| Error type | What triggers it | Typical fix |
|------------|-----------------|-------------|
| `missing_package` | ModuleNotFoundError, ImportError | Install package into env via conda/pip |
| `code_error` | Traceback with runtime/logic error | LLM diagnoses and rewrites the failing section |
| `runtime_argument_error` | "No input file provided", `sys.argv` IndexError | Deletes stale sbatch, regenerates with input files appended to exec cmd |
| `data_structure_error` | KeyError, shape mismatch, missing column | Runs diagnostic script to inspect data, then code fix |
| `memory_error` | MemoryError, OOM | Checks data sizes, recommends chunking or higher `--mem` |
| `disk_quota_error` | OSError: No space left | Runs conda clean, removes old logs, frees quota |
| `syntax_error` | SyntaxError | Sends script back for LLM rewrite |
| `sbatch_config_error` | Invalid sbatch directives | Repairs sbatch config against cluster rules |
| `binary_not_found` | command not found | Installs binary via bioconda or conda-forge |
| `gpu_error` | CUDA error, device assertion | Checks GPU availability, suggests CPU fallback or env fix |
| `permission_error` | PermissionError on file/dir | Reports permissions, suggests correct path |
| `network_error` | Download failure | Suggests retry logic or pre-downloading data |

The agent stores successful fixes in `DiagnosticMemory` (a persistent Qdrant collection). If the same error pattern appears in a future step or future run, the known fix is applied directly without re-investigation.

---

## Available Clusters (ARC)

| Cluster Key | Partition | GPU | Max Time | Use Case |
|-------------|-----------|-----|----------|----------|
| `arc_compute1` | compute1 | — | 3 days | Default CPU subtasks |
| `arc_compute2` | compute2 | — | 10 days | Long-running CPU |
| `arc_compute3` | compute3 | — | 3 days | Overflow |
| `arc_gpu1v100` | gpu1v100 | 1× V100S-32GB | 3 days | Default GPU subtasks |
| `arc_gpu2v100` | gpu2v100 | 2× V100S-32GB | 3 days | Multi-GPU |
| `arc_gpu1a100` | gpu1a100 | 1× A100 | 3 days | Large models |
| `arc_dgxa100` | dgxa100 | DGX A100 | 3 days | Heavy GPU |
| `zeus_cpu` | normal | — | 1 day | Legacy cluster |

Override at submission time:

```bash
sbatch --export=ALL,AGI_CLUSTER=arc_compute2,AGI_GPU_CLUSTER=arc_gpu1a100 RUN_AGI_PIPELINE_GPU.sh
```

GPU subtasks are auto-detected from package keywords (`torch`, `scvi`, `cuda`, `cellbender`, etc.) and routed to the GPU cluster automatically. Never specify `--mem` on GPU partitions — it causes allocation failures.

---

## Key Configuration

### Token budget (config/config.yaml)

Sized for `qwen3-coder:latest` with 32K context window:

| Setting | Value | Purpose |
|---------|-------|---------|
| `max_tokens_per_task` | 25,000 | Working budget per subtask context window |
| `max_tool_output_tokens` | 12,000 | Max size of any single tool output |
| `min_tokens_to_continue` | 3,000 | Minimum remaining before context is considered exhausted |

### Parallel execution (config/config.yaml)

| Setting | Default | Purpose |
|---------|---------|---------|
| `max_parallel_agents` | 4 | ThreadPoolExecutor threads (must match `OLLAMA_NUM_PARALLEL`) |
| `max_parallel_jobs` | 10 | Max SLURM jobs queued/running at once across all threads |
| `max_parallel_gpu_jobs` | 4 | GPU job concurrency limit (GPU nodes are scarce) |

### Model selection (RUN script → config → fallback)

The model is resolved through a 4-level priority chain:

1. `--model` CLI argument to `main.py`
2. `OLLAMA_MODEL` environment variable (set in RUN script)
3. `config/config.yaml` → `ollama.model`
4. `FALLBACK_MODEL` constant in `utils/model_config.py`

To override the model for a single run without editing files:

```bash
sbatch --export=ALL,OLLAMA_MODEL=llama3.1:70b RUN_AGI_PIPELINE_GPU.sh
```

The default `qwen3-coder:latest` (~20 GiB Q4_K_M) fits on a V100S-32GB with room for KV cache. The larger `qwen3-coder-next` (~48 GiB) requires CPU offload and causes Ollama 500 errors on V100 nodes.

---

## Project Management Scripts Reference

| Script | Purpose |
|--------|---------|
| `FULL_CLEAN_PROJECT.sh` | Complete wipe — removes all state, conda envs, and package cache. Use before a fresh start. |
| `PARTIAL_CLEAN_PROJECT.sh` | Log cleanup only — preserves pipeline state, checkpoints, and conda envs. Use between attempts on the same run. |
| `INJECT_HINTS.sh` | Form-based human guidance injection into failed steps. Edits `master_prompt_state.json` safely. |
| `RUN_AGI_PIPELINE_GPU.sh` | Submit the master pipeline to a GPU node. |
| `RUN_AGI_PIPELINE_CPU.sh` | Submit to CPU-only cluster (zeus). |

All scripts support `--dry-run` to preview what they would do, and `--yes` to skip confirmation prompts.

---

## AGI Repo File Structure

```
AGI/
├── agents/
│   ├── master_agent.py       # Task decomposition, validation, living document
│   ├── sub_agent.py          # 4-phase lifecycle: script → env → sbatch → execute
│   └── diagnostic_agent.py   # Error investigation and fix prescription
├── config/
│   ├── config.yaml           # Pipeline configuration (tokens, parallel, timeouts)
│   └── cluster_config.yaml   # SLURM cluster partition definitions
├── memory/
│   ├── reflexion_memory.py   # Per-task loop prevention (Mem0 + Qdrant)
│   ├── diagnostic_memory.py  # Cross-task solution storage (Qdrant)
│   └── config.py             # Mem0 configuration
├── setup/
│   ├── setup.sh              # New project directory initializer
│   ├── RUN_AGI_PIPELINE_GPU.sh  # Template: GPU submission script
│   ├── RUN_AGI_PIPELINE_CPU.sh  # Template: CPU submission script
│   ├── FULL_CLEAN_PROJECT.sh    # Template: full wipe script
│   ├── PARTIAL_CLEAN_PROJECT.sh # Template: partial log cleanup script
│   ├── INJECT_HINTS.sh          # Template: human guidance injection
│   └── setup_mem0.sh            # Mem0 / reflexion memory setup
├── tools/
│   ├── sandbox.py            # Path sandboxing and command validation
│   ├── slurm_tools.py        # SLURM job submission and monitoring
│   └── conda_tools.py        # Conda environment management
├── utils/
│   ├── config_loader.py      # YAML config loading
│   ├── context_manager.py    # Token budget management
│   ├── dependency_parser.py  # LLM-driven package dependency analysis
│   ├── disk_manager.py       # Disk quota monitoring and cleanup
│   ├── logging_config.py     # Structured JSONL logging
│   ├── model_config.py       # Model resolution chain
│   └── documentation.py      # Pipeline status report generation
├── workflows/
│   └── langgraph_workflow.py # LangGraph state machine + parallel orchestration
├── main.py                   # Entry point
└── requirements.txt
```

---

## Troubleshooting

### Step stalls at "running" status with 0 attempts

This is the ghost-running bug fixed in v1.2.3. The parallel execution path was not writing "failed" to the master document when a job failed, leaving the step stuck. Upgrade to v1.2.3+ and run a partial clean to reset the state.

### "No input file provided" error repeats indefinitely

The sbatch script was calling the Python script with no arguments. Fixed in v1.2.3: the `runtime_argument_error` classification now correctly identifies this pattern, and the diagnostic agent's `add_argument` fix regenerates the sbatch with `input_files` from the subtask definition appended to the exec command. If you are on v1.2.3+ and still seeing this, use `INJECT_HINTS.sh` option 2 to manually set the correct input file paths.

### Ollama 404 — model not found

```bash
grep OLLAMA_MODELS RUN_AGI_PIPELINE_GPU.sh
ls /work/$USER/ollama/models/manifests/
```

The `OLLAMA_MODELS` path in your RUN script must match where the models were actually pulled.

### Reflexion memory import error

If you see `Failed to initialize reflexion memory: mem0ai package not installed`, verify PYTHONPATH:

```bash
PYTHONPATH=/path/to/AGI:$PYTHONPATH python -c "from memory.reflexion_memory import ReflexionMemory; print('OK')"
```

### Decomposition timeout with long prompts

With 50+ steps, the total decomposition timeout (default 6 hours) may be reached before all steps get full LLM expansion. Remaining steps get basic fallback plans. Increase `TOTAL_DECOMPOSITION_TIMEOUT` in `master_agent.py` or break the prompt into smaller focused runs.

### Disk quota on home directory

Everything except conda environments should be on `/work/$USER/`:

```bash
OLLAMA_MODELS=/work/$USER/ollama/models
AGI_DATA_DIR=/work/$USER/agi_data
# Projects: /work/$USER/WORKING/project-name/
```

Between runs, use `PARTIAL_CLEAN_PROJECT.sh` to remove SLURM logs (which accumulate fast — one large run can generate thousands of `.out`/`.err` files). For a full quota recovery, `FULL_CLEAN_PROJECT.sh` also runs `conda clean --all` to remove cached packages.

---

## Version History

- **v1.2.4** — `code_hints` field now wired through to script generation prompts (`_build_python_script_prompt`, `_build_r_script_prompt`, `_build_generic_script_prompt`). `INJECT_HINTS.sh` form-based human guidance injection tool. `FULL_CLEAN_PROJECT.sh` (includes conda env removal + cache purge). `PARTIAL_CLEAN_PROJECT.sh` (preserves state, removes logs). Replaces `CLEAN_PROJECT.sh`.
- **v1.2.3** — Fixed 5-layer failure chain that caused steps to ghost-stall at "running" status. Added `runtime_argument_error` classification, diagnostic agent handler and `add_argument` fix type, sbatch input file argument injection (Phase 3), and master document sync in parallel execution path.
- **v1.2.2** — True parallel task execution via `ThreadPoolExecutor`. Progress-first routing prevents premature pipeline exit. `blocked` status for tasks with failed dependencies. `OLLAMA_NUM_PARALLEL` support for concurrent LLM requests.
- **v1.2.0** — Diagnostic Agent with 13 error type handlers. `DiagnosticMemory` persistent solution storage. `DiskManager` for proactive quota monitoring. Sub-agent refactored into 4-phase lifecycle. All LLM calls use `invoke_resilient` with exponential backoff.

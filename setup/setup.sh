#!/bin/bash
# =============================================================================
# AGI Multi-Agent Pipeline v1.2.8 - Project Directory Setup
# =============================================================================
# Initializes a NEW project directory for the AGI multi-agent system.
# Run this from inside the empty (or existing) project directory.
#
# What it does:
#   1. Creates the expected folder structure
#   2. Copies RUN and CLEAN scripts from the AGI repo (with paths filled in)
#   3. Creates project.yaml with v1.2.8 defaults
#   4. Creates .gitignore
#   5. Generates a project README.md
#   6. Optionally initializes a Git repository
#
# v1.2.8 directory structure changes:
#   - data/raw/       replaces data/inputs/ (protected input source)
#   - outputs/        new top-level protected dir for step scientific outputs
#   - logs/steps/     now created explicitly (was created on-the-fly by StepLogger)
#   - reports/        protected (contains output_manifest.json)
#   - data/outputs/   removed — step outputs now live in outputs/step_N/
#
# Usage:
#   # Auto-detect AGI repo (works when script is in AGI/setup/):
#   bash /path/to/AGI/setup/setup.sh
#
#   # Explicit AGI root:
#   bash setup.sh --agi-root /work/sdz852/WORKING/AGI
#
#   # Full setup with force overwrite:
#   bash /path/to/AGI/setup/setup.sh --force
#
#   # Verify existing project:
#   bash setup.sh --verify
# =============================================================================

set -e

# ─── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ─── Defaults ─────────────────────────────────────────────────────────────────
ENV_NAME="AGI"
PROJECT_NAME=$(basename "$(pwd)")
PROJECT_DIR="$(pwd)"
AGI_ROOT=""
SKIP_GIT=false
VERIFY_ONLY=false
FORCE=false

# ─── Helpers ──────────────────────────────────────────────────────────────────
print_header() {
    echo ""
    echo -e "${BLUE}==============================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}==============================================================================${NC}"
    echo ""
}
print_success() { echo -e "  ${GREEN}✓${NC} $1"; }
print_warning() { echo -e "  ${YELLOW}⚠${NC} $1"; }
print_error()   { echo -e "  ${RED}✗${NC} $1"; }
print_info()    { echo -e "  ${CYAN}→${NC} $1"; }

# ─── Parse Arguments ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --agi-root)
            AGI_ROOT="$2"
            shift 2
            ;;
        --no-git)
            SKIP_GIT=true
            shift
            ;;
        --verify)
            VERIFY_ONLY=true
            shift
            ;;
        --force|-f)
            FORCE=true
            shift
            ;;
        --help|-h)
            echo "Usage: bash setup.sh [OPTIONS]"
            echo ""
            echo "Initializes a project directory for the AGI multi-agent pipeline v3.2."
            echo "Run from inside your project directory."
            echo ""
            echo "Options:"
            echo "  --agi-root PATH   Path to the AGI repository (auto-detected if omitted)"
            echo "  --no-git          Skip Git repository initialization"
            echo "  --verify          Only verify the directory structure"
            echo "  --force, -f       Overwrite existing files without prompting"
            echo "  --help, -h        Show this help message"
            echo ""
            echo "Examples:"
            echo "  mkdir my-project && cd my-project"
            echo "  bash /work/sdz852/WORKING/AGI/setup/setup.sh"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1 (use --help)"
            exit 1
            ;;
    esac
done

# ─── Auto-detect AGI_ROOT ────────────────────────────────────────────────────
# If this script lives at AGI/setup/setup.sh, we can find AGI_ROOT automatically.
if [ -z "$AGI_ROOT" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    CANDIDATE="$(dirname "$SCRIPT_DIR")"
    if [ -f "$CANDIDATE/main.py" ] && [ -d "$CANDIDATE/agents" ]; then
        AGI_ROOT="$CANDIDATE"
    fi
fi

# Validate AGI_ROOT
if [ -n "$AGI_ROOT" ] && [ ! -f "$AGI_ROOT/main.py" ]; then
    echo -e "${RED}ERROR: AGI_ROOT ($AGI_ROOT) does not contain main.py${NC}"
    echo "  Provide the correct path with: --agi-root /path/to/AGI"
    exit 1
fi

# ─── Safety: don't run inside the AGI repo itself ─────────────────────────────
if [ -n "$AGI_ROOT" ]; then
    REAL_PROJECT="$(cd "$PROJECT_DIR" && pwd -P)"
    REAL_AGI="$(cd "$AGI_ROOT" && pwd -P)"
    if [ "$REAL_PROJECT" = "$REAL_AGI" ]; then
        echo -e "${RED}ERROR: You're inside the AGI repository, not a project directory.${NC}"
        echo "  Create a new directory first:"
        echo "    mkdir ../my-project && cd ../my-project"
        echo "    bash ${BASH_SOURCE[0]}"
        exit 1
    fi
fi

# =============================================================================
# VERIFICATION ONLY
# =============================================================================
if [ "$VERIFY_ONLY" = true ]; then
    print_header "Verifying Project Structure: ${PROJECT_NAME}"

    # v1.2.8: Updated to reflect canonical directory structure
    expected_dirs=(
        "agents"
        "config"
        "conda_env"
        "data/raw"
        "envs"
        "logs"
        "logs/steps"
        "outputs"
        "prompts"
        "reports"
        "scripts"
        "slurm/logs"
        "slurm/scripts"
        "slurm_logs"
        "temp/checkpoints"
        "tools/dynamic_tools"
        "utils"
        "workflows"
    )

    all_present=true
    for dir in "${expected_dirs[@]}"; do
        if [ -d "$dir" ]; then
            print_success "$dir/"
        else
            print_error "$dir/ (missing)"
            all_present=false
        fi
    done

    echo ""
    if [ "$all_present" = true ]; then
        print_success "All required directories present"
    else
        print_error "Some directories missing — re-run setup.sh to create them"
    fi

    echo ""
    for f in project.yaml RUN_AGI_PIPELINE_GPU.sh RUN_AGI_PIPELINE_CPU.sh \
              FULL_CLEAN_PROJECT.sh PARTIAL_CLEAN_PROJECT.sh INJECT_HINTS.sh; do
        if [ -f "$f" ]; then
            print_success "$f"
        else
            print_warning "$f (missing)"
        fi
    done

    # v1.2.8: Check that the manifest exists if the pipeline has run
    echo ""
    if [ -f "reports/output_manifest.json" ]; then
        print_success "reports/output_manifest.json (manifest present)"
    else
        print_info "reports/output_manifest.json (will be created on first pipeline run)"
    fi

    echo ""
    if [ -d ".git" ]; then
        print_success "Git initialized (branch: $(git branch --show-current 2>/dev/null || echo '?'))"
    else
        print_warning "Git not initialized"
    fi

    exit 0
fi

# =============================================================================
# MAIN SETUP
# =============================================================================
print_header "AGI Pipeline v3.2 — Project Setup"

echo -e "  Project:   ${CYAN}${PROJECT_NAME}${NC}"
echo -e "  Location:  ${CYAN}${PROJECT_DIR}${NC}"
if [ -n "$AGI_ROOT" ]; then
    echo -e "  AGI Root:  ${CYAN}${AGI_ROOT}${NC}"
else
    echo -e "  AGI Root:  ${YELLOW}not detected (script templates will be skipped)${NC}"
fi
echo ""
echo "  This script will:"
echo "    1. Create directory structure"
echo "    2. Copy RUN + CLEAN scripts from AGI repo"
echo "    3. Create project.yaml (v3.2 defaults)"
echo "    4. Create .gitignore"
echo "    5. Generate project README.md"
[ "$SKIP_GIT" = false ] && echo "    6. Initialize Git repository"
echo ""

if [ "$FORCE" = false ]; then
    read -p "  Continue? [Y/n] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]?$ ]]; then
        echo "  Cancelled."
        exit 0
    fi
fi

# =============================================================================
# Step 1: Directory Structure
# =============================================================================
print_header "Step 1: Creating Directory Structure"

# v1.2.8: Canonical directory structure.
#
# PROTECTED (never touched by any cleanup script):
#   data/raw/       — source input files, provided by user
#   outputs/        — scientific step outputs (outputs/step_N/ per step)
#   envs/           — conda YAML specs (overwritten on regen, never deleted)
#   scripts/        — generated analysis scripts (overwritten, never deleted)
#   slurm/scripts/  — generated sbatch files (overwritten, never deleted)
#   reports/        — pipeline state + output_manifest.json
#
# CLEANABLE (safe to remove between runs):
#   slurm/logs/     — SLURM job stdout/stderr (PARTIAL_CLEAN removes these)
#   logs/           — agent + pipeline logs (PARTIAL_CLEAN removes these)
#   logs/steps/     — per-step phase logs (PARTIAL_CLEAN removes these)
#   temp/           — checkpoints and ephemeral files

directories=(
    # Pipeline infrastructure (Python packages)
    "agents"
    "tools"
    "tools/dynamic_tools"
    "workflows"
    "utils"

    # Configuration — TRACKED
    "config"
    "prompts"
    "scripts"
    "scripts/example_reference_scripts"
    "conda_env"

    # Input data — PROTECTED, never cleaned
    # v1.2.8: data/raw/ replaces data/inputs/ as the canonical input location.
    # data/ is INPUT ONLY — the pipeline never writes outputs here.
    "data/raw"

    # Step scientific outputs — PROTECTED, never cleaned
    # v1.2.8: outputs/ is the canonical location for all step outputs.
    # Each step gets its own subdirectory: outputs/step_N/
    "outputs"

    # Runtime artifacts — IGNORED
    "logs"
    "logs/steps"
    "reports"
    "envs"

    # SLURM
    "slurm/scripts"
    "slurm/logs"
    "slurm_logs"

    # Temporary / checkpoints — IGNORED
    "temp/checkpoints"
)

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        print_info "Created: $dir/"
    else
        print_success "Exists:  $dir/"
    fi
done

# __init__.py for Python packages
for pkg in agents tools workflows utils; do
    init="${pkg}/__init__.py"
    [ ! -f "$init" ] && touch "$init" && print_info "Created: $init"
done

# .gitkeep for tracked empty directories that would otherwise be gitignored.
# reports/ gets a .gitkeep so the directory is committed even though its
# contents (logs, state JSON) are ignored — the manifest lives here and
# the directory must exist before the first pipeline run.
for dir in config prompts scripts slurm/scripts conda_env reports outputs; do
    gk="${dir}/.gitkeep"
    [ ! -f "$gk" ] && touch "$gk"
done

print_success "Directory structure created"

# =============================================================================
# Step 2: Copy Scripts from AGI Repo
# =============================================================================
print_header "Step 2: Copying Pipeline Scripts"

copy_and_patch() {
    local src="$1"
    local dst="$2"
    local desc="$3"

    if [ ! -f "$src" ]; then
        print_warning "Source not found: $src"
        return
    fi

    if [ -f "$dst" ] && [ "$FORCE" = false ]; then
        print_success "Exists: $dst (use --force to overwrite)"
        return
    fi

    cp "$src" "$dst"

    # Patch PROMPT_FILE, PROJECT_DIR, AGI_ROOT placeholders to point at this
    # project. Uses | delimiter to avoid / escaping issues.
    sed -i "s|PROMPT_FILE=\"\${PROMPT_FILE:-[^}]*}\"|PROMPT_FILE=\"\${PROMPT_FILE:-${PROJECT_DIR}/prompts/YOUR_PROMPT.md}\"|" "$dst" 2>/dev/null || true
    sed -i "s|PROJECT_DIR=\"\${PROJECT_DIR:-[^}]*}\"|PROJECT_DIR=\"\${PROJECT_DIR:-${PROJECT_DIR}}\"|" "$dst" 2>/dev/null || true
    sed -i "s|AGI_ROOT=\"\${AGI_ROOT:-[^}]*}\"|AGI_ROOT=\"\${AGI_ROOT:-${AGI_ROOT}}\"|" "$dst" 2>/dev/null || true

    print_success "Copied: $desc → $dst"
}

copy_straight() {
    local src="$1"
    local dst="$2"
    local desc="$3"

    if [ ! -f "$src" ]; then
        print_warning "Source not found: $src"
        return
    fi

    if [ -f "$dst" ] && [ "$FORCE" = false ]; then
        print_success "Exists: $dst (use --force to overwrite)"
        return
    fi

    cp "$src" "$dst"
    print_success "Copied: $desc → $dst"
}

if [ -n "$AGI_ROOT" ]; then
    # RUN scripts — patched with project-specific paths
    copy_and_patch \
        "$AGI_ROOT/setup/RUN_AGI_PIPELINE_GPU.sh" \
        "RUN_AGI_PIPELINE_GPU.sh" \
        "GPU pipeline script"

    copy_and_patch \
        "$AGI_ROOT/setup/RUN_AGI_PIPELINE_CPU.sh" \
        "RUN_AGI_PIPELINE_CPU.sh" \
        "CPU pipeline script"

    # v1.2.8: Copy all three project management scripts.
    # FULL and PARTIAL clean replace the old monolithic CLEAN_PROJECT.sh.
    # INJECT_HINTS is the human guidance injection tool.
    # These are straight copies — no path patching needed (they use $(pwd)).
    copy_straight \
        "$AGI_ROOT/setup/FULL_CLEAN_PROJECT.sh" \
        "FULL_CLEAN_PROJECT.sh" \
        "Full clean script"

    copy_straight \
        "$AGI_ROOT/setup/PARTIAL_CLEAN_PROJECT.sh" \
        "PARTIAL_CLEAN_PROJECT.sh" \
        "Partial clean script"

    copy_straight \
        "$AGI_ROOT/setup/INJECT_HINTS.sh" \
        "INJECT_HINTS.sh" \
        "Human guidance injection tool"

    # Backward-compat shim: if the old CLEAN_PROJECT.sh is still present in
    # the AGI repo, copy it too so existing muscle memory still works.
    if [ -f "$AGI_ROOT/setup/CLEAN_PROJECT.sh" ]; then
        copy_straight \
            "$AGI_ROOT/setup/CLEAN_PROJECT.sh" \
            "CLEAN_PROJECT.sh" \
            "Legacy clean script (use PARTIAL or FULL instead)"
    fi
else
    print_warning "AGI_ROOT not detected — skipping script copy"
    print_info "Re-run with: bash /path/to/AGI/setup/setup.sh"
fi

# =============================================================================
# Step 3: Project Metadata (project.yaml)
# =============================================================================
print_header "Step 3: Creating project.yaml"

PROJECT_YAML="project.yaml"

should_create() {
    [ ! -f "$1" ] && return 0
    [ "$FORCE" = true ] && return 0
    read -p "  $1 exists. Overwrite? [y/N] " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]]
}

if should_create "$PROJECT_YAML"; then
    cat > "$PROJECT_YAML" << EOF
# =============================================================================
# AGI Pipeline Project Configuration — v3.2
# =============================================================================
# Project: ${PROJECT_NAME}
# Created: $(date -I)
# =============================================================================

project:
  name: "${PROJECT_NAME}"
  description: ""
  created: "$(date -I)"
  version: "0.1.0"

# Environment
environment:
  conda_env: "${ENV_NAME}"
  python_version: "3.10"

# Ollama model (v3.2: qwen3-coder-next on GPU, llama3.1:70b on CPU-only)
ollama:
  model: "qwen3-coder-next:latest"
  fallback_model: "llama3.1:8b"
  base_url: "http://127.0.0.1:11434"
  context_length: 32768

# Agent settings
agents:
  max_retries: 3
  timeout_seconds: 300
  enable_dynamic_tools: true

# Workflow
workflow:
  enable_checkpointing: true
  checkpoint_frequency: "per_subtask"
  max_execution_time_minutes: 4320    # 3 days

# SLURM (v3.2: ARC dual-cluster architecture)
slurm:
  default_cluster: "arc_compute1"
  default_gpu_cluster: "arc_gpu1v100"
  poll_interval: 30
  max_poll_attempts: 8640             # 3 days @ 30s
  default_time: "1-00:00:00"
  default_cpus: 20

# Logging
logging:
  level: "INFO"
  json_format: true
  console_output: true
EOF
    print_success "Created project.yaml"
else
    print_success "Keeping existing project.yaml"
fi

# =============================================================================
# Step 4: .gitignore
# =============================================================================
print_header "Step 4: Creating .gitignore"

GITIGNORE=".gitignore"

if should_create "$GITIGNORE"; then
    cat > "$GITIGNORE" << 'EOF'
# =============================================================================
# AGI Pipeline .gitignore — v1.2.8
# Track: config, prompts, scripts, conda_env specs
# Ignore: data, logs, outputs (large scientific files), temp, runtime state
#
# Protected directories (never cleaned by scripts, but gitignored because
# they contain large binary files):
#   outputs/     — step scientific outputs (h5ad, RDS, BAM, etc.)
#   data/        — input source files
#
# reports/ is partially tracked: the directory itself is committed via
# .gitkeep, but its contents (state JSON, manifest) are gitignored.
# The exception is output_manifest.json which is explicitly unignored
# so the manifest can optionally be committed for reproducibility.
# =============================================================================

# ── Input data (large, sensitive) ─────────────────────────────────────────────
data/
!data/.gitkeep

# ── Step scientific outputs (large binary files) ──────────────────────────────
# v1.2.8: outputs/ is the canonical step output directory.
# The directory itself is tracked via .gitkeep; contents are not.
outputs/
!outputs/.gitkeep

# ── Logs ─────────────────────────────────────────────────────────────────────
logs/
*.log
*.jsonl
slurm/logs/
slurm_logs/

# ── Reports & runtime state ─────────────────────────────────────────────────
# The reports/ directory is tracked (via .gitkeep) but contents are not,
# except output_manifest.json which can optionally be committed.
reports/*
!reports/.gitkeep
!reports/output_manifest.json

# ── Temp / checkpoints ──────────────────────────────────────────────────────
temp/
*.tmp
*.temp
*.bak
*.swp
*.swo
*~

# ── Python ───────────────────────────────────────────────────────────────────
__pycache__/
*.py[cod]
*$py.class
*.so
*.egg-info/
.eggs/
dist/
build/
.pytest_cache/
.coverage
htmlcov/
venv/
.venv/

# ── Database & state ────────────────────────────────────────────────────────
*.db
*.sqlite
*.sqlite3

# ── Conda env installs (but keep YAML specs) ─────────────────────────────────
envs/*/

# ── IDE / OS ─────────────────────────────────────────────────────────────────
.idea/
.vscode/
*.sublime-*
.DS_Store
Thumbs.db

# ── Secrets ──────────────────────────────────────────────────────────────────
.env
.env.*
*.pem
*.key
secrets/
credentials/

# ── Large files / models ────────────────────────────────────────────────────
*.h5
*.hdf5
*.pkl
*.pickle
*.pt
*.pth
models/

# ── Bioinformatics ──────────────────────────────────────────────────────────
*.fastq
*.fastq.gz
*.bam
*.sam
*.vcf
*.vcf.gz
*.bed
*.gtf
*.gff

# ── Explicitly tracked (override ignores above) ───────────────────────────────
!.gitkeep
!config/
!config/**
!prompts/
!prompts/**
!scripts/
!scripts/**
!slurm/scripts/
!slurm/scripts/**
!conda_env/
!conda_env/**
!envs/*.yml
!envs/*.yaml
!project.yaml
!*.sh
!README.md
!requirements.txt
EOF
    print_success "Created .gitignore"
else
    print_success "Keeping existing .gitignore"
fi

# =============================================================================
# Step 5: README.md
# =============================================================================
print_header "Step 5: Creating README.md"

README="README.md"

if should_create "$README"; then
    cat > "$README" << 'READMEEOF'
# PROJECT_NAME_PLACEHOLDER

> AGI Multi-Agent Pipeline Project (v3.2 / v1.2.8)

## Quick Start

```bash
# 1. Activate the AGI environment
conda activate AGI

# 2. Place your input data
cp /path/to/your/data/* data/raw/

# 3. Write your master prompt
vi prompts/my_analysis.md

# 4. Update RUN_AGI_PIPELINE_GPU.sh with your prompt path
#    (setup.sh fills in PROJECT_DIR and AGI_ROOT automatically)
vi RUN_AGI_PIPELINE_GPU.sh

# 5. Submit to the GPU queue
sbatch RUN_AGI_PIPELINE_GPU.sh

# 6. Monitor
tail -f slurm_logs/agi_*.out
squeue -u $USER
```

## Directory Structure

```
PROJECT/
├── data/
│   └── raw/                 # Input source files — NEVER cleaned or written to
│
├── outputs/                 # Step scientific outputs — NEVER cleaned
│   ├── step_01/             # One subdirectory per pipeline step
│   ├── step_02/
│   └── ...
│
├── envs/                    # Conda YAML specs — NEVER deleted by cleanup
├── scripts/                 # Generated analysis scripts — NEVER deleted by cleanup
│
├── slurm/
│   ├── scripts/             # Generated sbatch files — NEVER deleted by cleanup
│   └── logs/                # SLURM job stdout/stderr — cleared by PARTIAL_CLEAN
│
├── logs/                    # Agent + pipeline logs — cleared by PARTIAL_CLEAN
│   └── steps/               # Per-step phase logs — cleared by PARTIAL_CLEAN
│
├── reports/                 # Pipeline state — NEVER cleaned
│   ├── master_prompt_state.json
│   ├── output_manifest.json # Canonical output registry (v1.2.8)
│   └── pipeline_report.md
│
├── temp/
│   └── checkpoints/         # Step checkpoint JSONs — cleared by PARTIAL_CLEAN
│
├── config/                  # Configuration files (tracked)
├── conda_env/               # Conda environment YAML specs (tracked)
├── prompts/                 # Master prompt .md files (tracked)
├── slurm_logs/              # Master job stdout/stderr (ignored)
├── project.yaml             # Project configuration (tracked)
├── RUN_AGI_PIPELINE_GPU.sh  # Submit pipeline to GPU node
├── RUN_AGI_PIPELINE_CPU.sh  # Submit to CPU-only cluster
├── FULL_CLEAN_PROJECT.sh    # Full wipe for fresh start
├── PARTIAL_CLEAN_PROJECT.sh # Log cleanup — preserves pipeline state
├── INJECT_HINTS.sh          # Human guidance injection between runs
└── README.md
```

## Project Management Scripts

| Script | Purpose |
|--------|---------|
| `RUN_AGI_PIPELINE_GPU.sh` | Submit master pipeline to GPU node (recommended) |
| `RUN_AGI_PIPELINE_CPU.sh` | Submit to CPU-only cluster (zeus) |
| `FULL_CLEAN_PROJECT.sh` | Complete wipe — removes all state, conda envs, cache. Use before a fresh start. |
| `PARTIAL_CLEAN_PROJECT.sh` | Log cleanup only — preserves pipeline state, checkpoints, conda envs. Use between attempts. |
| `INJECT_HINTS.sh` | Form-based human guidance injection for failed steps before restart. |

All scripts support `--dry-run` to preview and `--yes` to skip confirmation.

## Between-Run Workflow

```bash
# 1. Review what failed
bash INJECT_HINTS.sh --dry-run

# 2. Inject guidance for failed steps
bash INJECT_HINTS.sh

# 3. Clear logs, keep pipeline state
bash PARTIAL_CLEAN_PROJECT.sh --yes

# 4. Resubmit
sbatch RUN_AGI_PIPELINE_GPU.sh
```

Use `FULL_CLEAN_PROJECT.sh` only when you want to restart completely from scratch.

## Configuration

Edit `project.yaml` to customize model, SLURM defaults, and timeouts.
Edit the RUN scripts to change cluster targets or resource requests.

## Notes

- `data/raw/` and `outputs/` are NEVER touched by any cleanup script
- `reports/output_manifest.json` tracks all output paths and protects them
- Conda env YAML specs in `envs/` survive all cleanup operations
- Generated scripts in `scripts/` survive all cleanup operations
READMEEOF

    # Replace placeholder with actual project name
    sed -i "s/PROJECT_NAME_PLACEHOLDER/${PROJECT_NAME}/" "$README"

    print_success "Created README.md"
else
    print_success "Keeping existing README.md"
fi

# =============================================================================
# Step 6: Git Repository
# =============================================================================
if [ "$SKIP_GIT" = false ]; then
    print_header "Step 6: Initializing Git Repository"

    if [ -d ".git" ]; then
        print_warning "Git already initialized"
        print_info "Remote: $(git remote get-url origin 2>/dev/null || echo 'none')"
    else
        git init
        git branch -M main 2>/dev/null || true
        git add .
        git commit -m "Initial commit: ${PROJECT_NAME} — AGI Pipeline v3.2 / v1.2.8 project

v1.2.8 canonical directory structure:
  - data/raw/   protected input source directory
  - outputs/    protected step scientific output directory
  - reports/    protected pipeline state + output manifest
Includes RUN scripts, FULL/PARTIAL clean, INJECT_HINTS, and project config."
        print_success "Git initialized (branch: main)"
        echo ""
        echo "    To add a remote:"
        echo "    git remote add origin git@github.com:USERNAME/${PROJECT_NAME}.git"
        echo "    git push -u origin main"
    fi
else
    print_header "Step 6: Skipping Git (--no-git)"
    print_info "Run 'git init' later if needed"
fi

# =============================================================================
# Summary
# =============================================================================
print_header "Setup Complete!"

echo -e "  Project:     ${CYAN}${PROJECT_NAME}${NC}"
echo -e "  Location:    ${CYAN}${PROJECT_DIR}${NC}"
if [ -n "$AGI_ROOT" ]; then
    echo -e "  AGI Root:    ${CYAN}${AGI_ROOT}${NC}"
fi
echo ""
echo -e "  ${GREEN}PROTECTED${NC} (never touched by cleanup scripts):"
echo "    data/raw/         outputs/          envs/"
echo "    scripts/          slurm/scripts/    reports/"
echo ""
echo -e "  ${YELLOW}CLEANABLE${NC} (PARTIAL_CLEAN removes these between runs):"
echo "    slurm/logs/       logs/             logs/steps/"
echo "    temp/checkpoints/"
echo ""
echo -e "  ${CYAN}TRACKED${NC} (committed to git):"
echo "    config/           prompts/          scripts/"
echo "    conda_env/        slurm/scripts/    project.yaml"
echo "    RUN_*.sh          *_CLEAN_*.sh      INJECT_HINTS.sh"
echo ""
echo "  Next steps:"
echo ""
echo "    1. Place your input data:"
echo -e "       ${YELLOW}cp /path/to/data/* data/raw/${NC}"
echo ""
echo "    2. Write your master prompt:"
echo -e "       ${YELLOW}vi prompts/my_analysis.md${NC}"
echo ""
echo "    3. Update the prompt path in your RUN script:"
echo -e "       ${YELLOW}vi RUN_AGI_PIPELINE_GPU.sh${NC}"
echo "       (look for PROMPT_FILE near the top)"
echo ""
echo "    4. Submit:"
echo -e "       ${YELLOW}sbatch RUN_AGI_PIPELINE_GPU.sh${NC}"
echo ""

print_success "Happy automating!"

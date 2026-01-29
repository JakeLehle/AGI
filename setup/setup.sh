#!/bin/bash

# =============================================================================
# AGI Multi-Agent Pipeline - Project Directory Setup
# =============================================================================
# This script initializes a project directory for the AGI multi-agent system.
# It creates the expected folder structure, initializes git, and sets up
# proper .gitignore for tracking only configuration/scripts (not data/logs).
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh              # Full setup
#   ./setup.sh --no-git     # Skip git initialization
#   ./setup.sh --verify     # Verify directory structure
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
ENV_NAME="AGI"
PROJECT_NAME=$(basename "$(pwd)")

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

print_header() {
    echo ""
    echo -e "${BLUE}==============================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}==============================================================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${CYAN}→${NC} $1"
}

# -----------------------------------------------------------------------------
# Parse Arguments
# -----------------------------------------------------------------------------

SKIP_GIT=false
VERIFY_ONLY=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
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
            echo "Usage: ./setup.sh [OPTIONS]"
            echo ""
            echo "Initializes a project directory for the AGI multi-agent pipeline."
            echo ""
            echo "Options:"
            echo "  --no-git      Skip Git repository initialization"
            echo "  --verify      Only verify the directory structure"
            echo "  --force, -f   Overwrite existing files without prompting"
            echo "  --help, -h    Show this help message"
            echo ""
            echo "Project: ${PROJECT_NAME}"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Verification Only
# -----------------------------------------------------------------------------

if [ "$VERIFY_ONLY" = true ]; then
    print_header "Verifying Project Structure: ${PROJECT_NAME}"
    
    expected_dirs=(
        "agents"
        "config"
        "data/inputs"
        "data/outputs"
        "logs"
        "prompts"
        "reports"
        "scripts"
        "slurm/logs"
        "slurm/scripts"
        "temp"
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
        print_error "Some directories are missing - run setup.sh to create them"
        exit 1
    fi
    
    # Check for git
    echo ""
    if [ -d ".git" ]; then
        print_success "Git repository initialized"
        echo "  Remote: $(git remote get-url origin 2>/dev/null || echo 'none')"
        echo "  Branch: $(git branch --show-current 2>/dev/null || echo 'unknown')"
    else
        print_warning "Git repository not initialized"
    fi
    
    # Check for project metadata
    if [ -f "project.yaml" ]; then
        print_success "Project metadata file exists"
    else
        print_warning "project.yaml not found"
    fi
    
    exit 0
fi

# -----------------------------------------------------------------------------
# Main Setup
# -----------------------------------------------------------------------------

print_header "AGI Pipeline - Project Setup"

echo -e "Project Name:  ${CYAN}${PROJECT_NAME}${NC}"
echo -e "Location:      ${CYAN}$(pwd)${NC}"
echo ""
echo "This script will:"
echo "  1. Create AGI pipeline directory structure"
echo "  2. Create project metadata file (project.yaml)"
echo "  3. Set up .gitignore (tracks config/prompts/scripts, ignores data/logs)"
if [ "$SKIP_GIT" = false ]; then
    echo "  4. Initialize Git repository"
fi
echo ""

if [ "$FORCE" = false ]; then
    read -p "Continue? [Y/n] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]?$ ]]; then
        echo "Setup cancelled."
        exit 0
    fi
fi

# -----------------------------------------------------------------------------
# Step 1: Create Directory Structure
# -----------------------------------------------------------------------------

print_header "Step 1: Creating Directory Structure"

# Define directories
# These are organized by purpose:
#   - Pipeline infrastructure (agents, tools, workflows, utils)
#   - Configuration (config, prompts, scripts)
#   - Data I/O (data/inputs, data/outputs) - IGNORED by git
#   - Logging (logs) - IGNORED by git
#   - Reports (reports) - IGNORED by git (generated outputs)
#   - SLURM (slurm/scripts tracked, slurm/logs ignored)
#   - Temporary (temp) - IGNORED by git

directories=(
    # Pipeline infrastructure
    "agents"
    "tools"
    "tools/dynamic_tools"
    "workflows"
    "utils"
    
    # Configuration - TRACKED
    "config"
    "prompts"
    "scripts"
    
    # Data I/O - IGNORED
    "data/inputs"
    "data/outputs"
    
    # Logging - IGNORED
    "logs"
    
    # Reports - IGNORED (generated)
    "reports"
    
    # SLURM
    "slurm/scripts"
    "slurm/logs"
    
    # Temporary - IGNORED
    "temp"
    
    # Environment exports - TRACKED
    "envs"
)

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        print_info "Created: $dir/"
    else
        print_success "Exists:  $dir/"
    fi
done

# Create __init__.py files for Python packages
python_packages=("agents" "tools" "workflows" "utils")
for package in "${python_packages[@]}"; do
    init_file="${package}/__init__.py"
    if [ ! -f "$init_file" ]; then
        touch "$init_file"
        print_info "Created: $init_file"
    fi
done

# Create .gitkeep files ONLY for tracked empty directories
# (config, prompts, scripts, slurm/scripts, envs)
tracked_empty_dirs=("config" "prompts" "scripts" "slurm/scripts" "envs")
for dir in "${tracked_empty_dirs[@]}"; do
    gitkeep="${dir}/.gitkeep"
    if [ ! -f "$gitkeep" ]; then
        touch "$gitkeep"
    fi
done

print_success "Directory structure created"

# -----------------------------------------------------------------------------
# Step 2: Create Project Metadata
# -----------------------------------------------------------------------------

print_header "Step 2: Creating Project Metadata"

PROJECT_YAML="project.yaml"

if [ -f "$PROJECT_YAML" ] && [ "$FORCE" = false ]; then
    print_warning "project.yaml already exists"
    read -p "Overwrite? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Keeping existing project.yaml"
    else
        CREATE_PROJECT_YAML=true
    fi
else
    CREATE_PROJECT_YAML=true
fi

if [ "$CREATE_PROJECT_YAML" = true ] || [ ! -f "$PROJECT_YAML" ]; then
    cat > "$PROJECT_YAML" << EOF
# =============================================================================
# AGI Pipeline Project Configuration
# =============================================================================
# Project: ${PROJECT_NAME}
# Created: $(date -I)
# =============================================================================

project:
  name: "${PROJECT_NAME}"
  description: ""
  created: "$(date -I)"
  version: "0.1.0"

# Environment settings
environment:
  conda_env: "${ENV_NAME}"
  python_version: "3.10"

# Ollama model settings
ollama:
  model: "llama3.1:70b"
  fallback_model: "llama3.1:8b"
  base_url: "http://localhost:11434"

# Agent settings
agents:
  max_retries: 3
  timeout_seconds: 300
  enable_dynamic_tools: true

# Workflow settings
workflow:
  enable_checkpointing: true
  checkpoint_frequency: "per_subtask"
  max_execution_time_minutes: 60

# SLURM settings (HPC)
slurm:
  default_partition: "compute2"
  gpu_partition: "gpu1v100"
  default_time: "08:00:00"
  default_nodes: 1
  default_cpus: 40

# Logging
logging:
  level: "INFO"
  json_format: true
  console_output: true

# Documentation
documentation:
  auto_generate_readme: true
  update_frequency: "on_completion"
EOF
    print_success "Created project.yaml"
fi

# -----------------------------------------------------------------------------
# Step 3: Create .gitignore
# -----------------------------------------------------------------------------

print_header "Step 3: Creating .gitignore"

GITIGNORE=".gitignore"

if [ -f "$GITIGNORE" ] && [ "$FORCE" = false ]; then
    print_warning ".gitignore already exists"
    read -p "Overwrite? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Keeping existing .gitignore"
        CREATE_GITIGNORE=false
    else
        CREATE_GITIGNORE=true
    fi
else
    CREATE_GITIGNORE=true
fi

if [ "$CREATE_GITIGNORE" = true ]; then
    cat > "$GITIGNORE" << 'EOF'
# =============================================================================
# AGI Pipeline .gitignore
# =============================================================================
# Philosophy: Track configuration, prompts, and scripts.
#             Ignore data, logs, outputs, and temporary files.
# =============================================================================

# -----------------------------------------------------------------------------
# DATA - Never track data files (potentially large, sensitive)
# -----------------------------------------------------------------------------
data/
!data/.gitkeep

# -----------------------------------------------------------------------------
# LOGS - Never track log files
# -----------------------------------------------------------------------------
logs/
*.log
*.jsonl

# SLURM job output logs (but keep script templates)
slurm/logs/
*.out
*.err

# -----------------------------------------------------------------------------
# OUTPUTS & REPORTS - Generated files, don't track
# -----------------------------------------------------------------------------
reports/
!reports/.gitkeep

# -----------------------------------------------------------------------------
# TEMPORARY FILES
# -----------------------------------------------------------------------------
temp/
*.tmp
*.temp
*.bak
*.swp
*.swo
*~

# -----------------------------------------------------------------------------
# PYTHON
# -----------------------------------------------------------------------------
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
.eggs/
dist/
build/
.pytest_cache/
.coverage
htmlcov/

# Virtual environments (use conda instead)
venv/
.venv/
env/

# -----------------------------------------------------------------------------
# DATABASE & STATE FILES
# -----------------------------------------------------------------------------
*.db
*.sqlite
*.sqlite3
workflow_state.db

# -----------------------------------------------------------------------------
# ENVIRONMENT EXPORTS - Keep these for reproducibility
# -----------------------------------------------------------------------------
# Note: envs/ folder IS tracked for environment.yml exports
# But ignore actual conda env installations
envs/*/

# -----------------------------------------------------------------------------
# DYNAMIC TOOLS - Track the generated tools for reproducibility
# -----------------------------------------------------------------------------
# tools/dynamic_tools/ IS tracked

# -----------------------------------------------------------------------------
# IDE & EDITOR
# -----------------------------------------------------------------------------
.idea/
.vscode/
*.sublime-*
.spyderproject
.spyproject

# -----------------------------------------------------------------------------
# OS FILES
# -----------------------------------------------------------------------------
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
desktop.ini

# -----------------------------------------------------------------------------
# SECRETS & CREDENTIALS - Never track these
# -----------------------------------------------------------------------------
.env
.env.*
*.pem
*.key
secrets/
credentials/

# -----------------------------------------------------------------------------
# LARGE FILES & MODELS
# -----------------------------------------------------------------------------
*.h5
*.hdf5
*.pkl
*.pickle
*.joblib
*.model
*.ckpt
*.pt
*.pth
models/

# Bioinformatics specific large files
*.fastq
*.fastq.gz
*.fq
*.fq.gz
*.bam
*.sam
*.vcf
*.vcf.gz
*.bed
*.gtf
*.gff

# -----------------------------------------------------------------------------
# EXPLICITLY TRACKED (override ignores above)
# -----------------------------------------------------------------------------
# These patterns ensure key files are tracked even if parent folders are ignored
!.gitkeep
!config/
!config/**
!prompts/
!prompts/**
!scripts/
!scripts/**
!slurm/scripts/
!slurm/scripts/**
!envs/*.yml
!envs/*.yaml
!environment.yml
!project.yaml
!setup.sh
!README.md
!QUICKSTART.md
!requirements.txt
EOF
    print_success "Created .gitignore"
fi

# -----------------------------------------------------------------------------
# Step 4: Create README template
# -----------------------------------------------------------------------------

print_header "Step 4: Creating README"

README="README.md"

if [ ! -f "$README" ] || [ "$FORCE" = true ]; then
    cat > "$README" << EOF
# ${PROJECT_NAME}

> AGI Multi-Agent Pipeline Project

## Overview

This project uses the AGI multi-agent automation system for [describe your project].

## Quick Start

\`\`\`bash
# 1. Activate environment
conda activate ${ENV_NAME}

# 2. (On HPC) Get a compute node
srun --partition=compute2 -N 1 -n 1 -c 40 --time=08:00:00 --pty bash

# 3. Start Ollama (if using local LLM)
ollama serve > /dev/null 2>&1 &

# 4. Run a task
python main.py --task "Your task description" --project-dir .
\`\`\`

## Directory Structure

\`\`\`
${PROJECT_NAME}/
├── agents/              # Agent definitions
├── config/              # Configuration files (tracked)
├── data/
│   ├── inputs/          # Input data (NOT tracked)
│   └── outputs/         # Output data (NOT tracked)
├── envs/                # Environment exports (tracked)
├── logs/                # Execution logs (NOT tracked)
├── prompts/             # Task prompts (tracked)
├── reports/             # Generated reports (NOT tracked)
├── scripts/             # Utility scripts (tracked)
├── slurm/
│   ├── logs/            # SLURM job logs (NOT tracked)
│   └── scripts/         # SLURM submission scripts (tracked)
├── temp/                # Temporary files (NOT tracked)
├── tools/
│   └── dynamic_tools/   # Dynamically created tools (tracked)
├── utils/               # Utility modules
├── workflows/           # LangGraph workflows
├── project.yaml         # Project configuration
└── README.md            # This file
\`\`\`

## Configuration

Edit \`project.yaml\` to customize:
- Ollama model settings
- Agent retry limits
- SLURM partition defaults
- Logging preferences

## Usage

### Running Tasks

\`\`\`bash
# Interactive task
python main.py --task "Analyze data in data/inputs/" --project-dir .

# From prompt file
python main.py --prompt-file prompts/my_analysis.txt --project-dir .

# SLURM submission
python main.py --task "Long running analysis" --project-dir . --slurm
\`\`\`

### Adding Prompts

Create prompt files in \`prompts/\` directory:

\`\`\`bash
# prompts/analyze_samples.txt
Analyze the RNA-seq samples in data/inputs/ and generate a summary report.
Focus on differential expression between treatment groups.
\`\`\`

## Notes

- Data files are NOT tracked in git (too large, potentially sensitive)
- Logs are NOT tracked (generated during execution)
- Configuration, prompts, and scripts ARE tracked for reproducibility
- Environment exports in \`envs/\` ARE tracked

---

*Generated by AGI Pipeline setup.sh on $(date -I)*
EOF
    print_success "Created README.md"
else
    print_info "README.md already exists, skipping"
fi

# -----------------------------------------------------------------------------
# Step 5: Initialize Git Repository
# -----------------------------------------------------------------------------

if [ "$SKIP_GIT" = false ]; then
    print_header "Step 5: Initializing Git Repository"
    
    if [ -d ".git" ]; then
        print_warning "Git repository already initialized"
        print_info "Remote: $(git remote get-url origin 2>/dev/null || echo 'none configured')"
    else
        git init
        
        # Set the initial branch name to main
        git branch -M main 2>/dev/null || true
        
        # Stage all tracked files
        git add .
        
        # Initial commit with project name
        git commit -m "Initial commit: ${PROJECT_NAME} - AGI Pipeline project setup

Project: ${PROJECT_NAME}
Created: $(date -I)

Directory structure initialized with:
- Agent infrastructure (agents/, tools/, workflows/, utils/)
- Configuration (config/, prompts/, scripts/)
- Data directories (data/inputs/, data/outputs/) [gitignored]
- Logging (logs/) [gitignored]
- SLURM support (slurm/scripts/, slurm/logs/)

Tracked items:
- Configuration files (config/, project.yaml)
- Prompts (prompts/)
- Scripts (scripts/, slurm/scripts/)
- Environment exports (envs/*.yml)
- Dynamic tools (tools/dynamic_tools/)

Ignored items:
- All data files (data/)
- Log files (logs/, *.log, *.jsonl)
- Reports (reports/)
- Temporary files (temp/)
- SLURM job output (slurm/logs/)"
        
        print_success "Git repository initialized"
        print_info "Branch: main"
        print_info "Initial commit created"
        
        echo ""
        echo "To add a remote origin:"
        echo "  git remote add origin git@github.com:USERNAME/${PROJECT_NAME}.git"
        echo "  git push -u origin main"
    fi
else
    print_header "Step 5: Skipping Git (--no-git)"
    print_info "Run 'git init' later to initialize repository"
fi

# -----------------------------------------------------------------------------
# Final Summary
# -----------------------------------------------------------------------------

print_header "Setup Complete!"

echo -e "Project:     ${CYAN}${PROJECT_NAME}${NC}"
echo -e "Location:    ${CYAN}$(pwd)${NC}"
echo -e "Environment: ${CYAN}${ENV_NAME}${NC}"
echo ""
echo "Directory structure:"
echo ""
echo "  TRACKED (committed to git):"
echo "    config/         - Configuration files"
echo "    prompts/        - Task prompt files"
echo "    scripts/        - Utility scripts"
echo "    slurm/scripts/  - SLURM job templates"
echo "    envs/*.yml      - Environment exports"
echo "    tools/          - Pipeline tools"
echo "    project.yaml    - Project settings"
echo ""
echo "  NOT TRACKED (gitignored):"
echo "    data/           - Input/output data"
echo "    logs/           - Execution logs"
echo "    reports/        - Generated reports"
echo "    temp/           - Temporary files"
echo "    slurm/logs/     - SLURM job output"
echo ""
echo "Next steps:"
echo ""
echo "  1. Edit project.yaml with your project description"
echo ""
echo "  2. Add your input data to data/inputs/"
echo ""
echo "  3. Create a prompt file:"
echo "     ${YELLOW}echo 'Your task description' > prompts/my_task.txt${NC}"
echo ""
echo "  4. Run the pipeline:"
echo "     ${YELLOW}conda activate ${ENV_NAME}${NC}"
echo "     ${YELLOW}python main.py --prompt-file prompts/my_task.txt --project-dir .${NC}"
echo ""
if [ "$SKIP_GIT" = false ] && [ -d ".git" ]; then
    echo "  5. Push to GitHub:"
    echo "     ${YELLOW}git remote add origin git@github.com:USERNAME/${PROJECT_NAME}.git${NC}"
    echo "     ${YELLOW}git push -u origin main${NC}"
    echo ""
fi

print_success "Happy automating!"

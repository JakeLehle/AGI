#!/bin/bash

# =============================================================================
# Multi-Agent Automation System - Setup Script
# =============================================================================
# This script sets up the complete environment for the multi-agent system.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh              # Full setup
#   ./setup.sh --env-only   # Only create conda environment
#   ./setup.sh --verify     # Verify installation
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENV_NAME="AGI"
PYTHON_VERSION="3.10"
OLLAMA_MODEL="llama3.1:70b"
OLLAMA_MODEL_SMALL="llama3.1:8b"

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
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        print_success "$1 is installed"
        return 0
    else
        print_warning "$1 is not installed"
        return 1
    fi
}

# -----------------------------------------------------------------------------
# Parse Arguments
# -----------------------------------------------------------------------------

ENV_ONLY=false
VERIFY_ONLY=false
SKIP_OLLAMA=false
USE_SMALL_MODEL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --env-only)
            ENV_ONLY=true
            shift
            ;;
        --verify)
            VERIFY_ONLY=true
            shift
            ;;
        --skip-ollama)
            SKIP_OLLAMA=true
            shift
            ;;
        --small-model)
            USE_SMALL_MODEL=true
            OLLAMA_MODEL="$OLLAMA_MODEL_SMALL"
            shift
            ;;
        --help|-h)
            echo "Usage: ./setup.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --env-only      Only create the conda environment"
            echo "  --verify        Only verify the installation"
            echo "  --skip-ollama   Skip Ollama installation"
            echo "  --small-model   Use smaller model (llama3.1:8b) for testing"
            echo "  --help, -h      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Verification Only
# -----------------------------------------------------------------------------

if [ "$VERIFY_ONLY" = true ]; then
    print_header "Verifying Installation"
    
    echo "Checking required tools..."
    check_command conda
    check_command git
    check_command python
    
    echo ""
    echo "Checking Ollama..."
    if check_command ollama; then
        echo "  Ollama version: $(ollama --version 2>/dev/null || echo 'unknown')"
        
        # Check if server is running
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            print_success "Ollama server is running"
            echo "  Available models:"
            ollama list 2>/dev/null | head -10 || echo "    (could not list models)"
        else
            print_warning "Ollama server is not running"
            echo "  Start with: ollama serve"
        fi
    fi
    
    echo ""
    echo "Checking conda environment..."
    if conda env list | grep -q "^${ENV_NAME} "; then
        print_success "Conda environment '${ENV_NAME}' exists"
        
        # Check key packages
        echo "  Checking key packages..."
        if conda run -n ${ENV_NAME} python -c "import langchain" 2>/dev/null; then
            print_success "  langchain is installed"
        else
            print_warning "  langchain is NOT installed"
        fi
        
        if conda run -n ${ENV_NAME} python -c "import ollama" 2>/dev/null; then
            print_success "  ollama is installed"
        else
            print_warning "  ollama is NOT installed"
        fi
        
        if conda run -n ${ENV_NAME} python -c "import langgraph" 2>/dev/null; then
            print_success "  langgraph is installed"
        else
            print_warning "  langgraph is NOT installed"
        fi
    else
        print_error "Conda environment '${ENV_NAME}' does not exist"
    fi
    
    echo ""
    echo "Checking SLURM..."
    if check_command sinfo; then
        echo "  SLURM cluster info:"
        sinfo --summarize 2>/dev/null || echo "    (could not get cluster info)"
    else
        print_info "SLURM not available (not required for local execution)"
    fi
    
    exit 0
fi

# -----------------------------------------------------------------------------
# Main Setup
# -----------------------------------------------------------------------------

print_header "Multi-Agent Automation System Setup"

echo "This script will:"
echo "  1. Create conda environment '${ENV_NAME}'"
echo "  2. Install all Python dependencies"
if [ "$SKIP_OLLAMA" = false ]; then
    echo "  3. Check/install Ollama"
    echo "  4. Pull the LLM model (${OLLAMA_MODEL})"
fi
echo "  5. Create project directory structure"
echo "  6. Initialize Git repository"
echo ""
read -p "Continue? [Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]?$ ]]; then
    echo "Setup cancelled."
    exit 0
fi

# -----------------------------------------------------------------------------
# Step 1: Check Prerequisites
# -----------------------------------------------------------------------------

print_header "Step 1: Checking Prerequisites"

# Check conda
if ! check_command conda; then
    print_error "Conda is required but not installed."
    echo "Install Miniconda from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check git
if ! check_command git; then
    print_error "Git is required but not installed."
    echo "Install with: sudo apt install git (Ubuntu) or brew install git (macOS)"
    exit 1
fi

# -----------------------------------------------------------------------------
# Step 2: Create Conda Environment
# -----------------------------------------------------------------------------

print_header "Step 2: Creating Conda Environment"

# Check if environment exists
if conda env list | grep -q "^${ENV_NAME} "; then
    print_warning "Environment '${ENV_NAME}' already exists"
    read -p "Recreate it? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Keeping existing environment. Will update packages."
    fi
fi

# Create or update environment
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Updating existing environment..."
    conda env update -n ${ENV_NAME} -f environment.yml --prune
else
    echo "Creating new environment..."
    if [ -f "environment.yml" ]; then
        conda env create -f environment.yml
    else
        # Fallback to manual creation
        echo "environment.yml not found, creating manually..."
        conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
        
        echo "Installing conda packages..."
        conda install -n ${ENV_NAME} -c conda-forge \
            pandas numpy requests beautifulsoup4 lxml pyyaml gitpython -y
        
        echo "Installing pip packages..."
        conda run -n ${ENV_NAME} pip install \
            langchain langchain-community langgraph ollama loguru \
            duckduckgo-search jsonschema
    fi
fi

print_success "Conda environment '${ENV_NAME}' is ready"

if [ "$ENV_ONLY" = true ]; then
    echo ""
    echo "Environment-only setup complete!"
    echo "Activate with: conda activate ${ENV_NAME}"
    exit 0
fi

# -----------------------------------------------------------------------------
# Step 3: Install Ollama
# -----------------------------------------------------------------------------

if [ "$SKIP_OLLAMA" = false ]; then
    print_header "Step 3: Setting Up Ollama"
    
    if check_command ollama; then
        print_success "Ollama is already installed"
    else
        echo "Installing Ollama..."
        
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            curl -fsSL https://ollama.com/install.sh | sh
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            print_warning "On macOS, please install Ollama manually from:"
            echo "  https://ollama.com/download"
            echo ""
            read -p "Press Enter after installing Ollama..." 
        else
            print_error "Unsupported OS for automatic Ollama installation"
            echo "Please install Ollama manually from: https://ollama.com/download"
        fi
    fi
    
    # Check if Ollama server is running
    echo ""
    echo "Checking Ollama server..."
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        print_success "Ollama server is running"
    else
        print_warning "Ollama server is not running"
        echo "Starting Ollama server..."
        
        # Start in background
        nohup ollama serve > /dev/null 2>&1 &
        sleep 3
        
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            print_success "Ollama server started"
        else
            print_warning "Could not start Ollama server automatically"
            echo "Please start manually with: ollama serve"
        fi
    fi
    
    # -----------------------------------------------------------------------------
    # Step 4: Pull Model
    # -----------------------------------------------------------------------------
    
    print_header "Step 4: Pulling LLM Model"
    
    echo "Model: ${OLLAMA_MODEL}"
    if [ "$USE_SMALL_MODEL" = true ]; then
        echo "(Using smaller model for testing - ~4GB download)"
    else
        echo "(This may take a while - ~40GB download for 70b model)"
    fi
    echo ""
    
    # Check if model already exists
    if ollama list 2>/dev/null | grep -q "${OLLAMA_MODEL}"; then
        print_success "Model '${OLLAMA_MODEL}' is already downloaded"
    else
        read -p "Download model now? [Y/n] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]?$ ]]; then
            ollama pull ${OLLAMA_MODEL}
            print_success "Model downloaded"
        else
            print_warning "Model download skipped"
            echo "Download later with: ollama pull ${OLLAMA_MODEL}"
        fi
    fi
fi

# -----------------------------------------------------------------------------
# Step 5: Create Directory Structure
# -----------------------------------------------------------------------------

print_header "Step 5: Creating Directory Structure"

# Create standard directories
directories=(
    "agents"
    "tools"
    "tools/dynamic_tools"
    "workflows"
    "utils"
    "data/inputs"
    "data/outputs"
    "logs"
    "config"
    "prompts"
    "scripts"
    "reports"
    "envs"
    "temp"
    "slurm/scripts"
    "slurm/logs"
)

for dir in "${directories[@]}"; do
    mkdir -p "$dir"
    echo "  Created: $dir/"
done

# Create __init__.py files for Python packages
for package in agents tools workflows utils; do
    touch "${package}/__init__.py"
done

# Create .gitkeep files for empty directories
for dir in logs data/inputs data/outputs temp slurm/scripts slurm/logs; do
    touch "${dir}/.gitkeep"
done

print_success "Directory structure created"

# -----------------------------------------------------------------------------
# Step 6: Initialize Git Repository
# -----------------------------------------------------------------------------

print_header "Step 6: Initializing Git Repository"

if [ -d ".git" ]; then
    print_success "Git repository already initialized"
else
    git init
    
    # Create .gitignore if it doesn't exist
    if [ ! -f ".gitignore" ]; then
        cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
.eggs/
dist/
build/

# Conda environments
envs/

# Logs
logs/*.jsonl
logs/*.log

# Database files
*.db

# Temporary files
temp/
*.tmp

# SLURM output (keep scripts)
slurm/logs/*.out
slurm/logs/*.err
slurm/logs/*.complete

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Keep structure
!.gitkeep
!slurm/scripts/
!logs/.gitkeep
!temp/.gitkeep
EOF
        print_success "Created .gitignore"
    fi
    
    git add .
    git commit -m "Initial commit: Multi-Agent Automation System setup"
    print_success "Git repository initialized with initial commit"
fi

# -----------------------------------------------------------------------------
# Final Summary
# -----------------------------------------------------------------------------

print_header "Setup Complete!"

echo "Environment: ${ENV_NAME}"
echo ""
echo "To get started:"
echo ""
echo "  1. Activate the environment:"
echo "     ${YELLOW}conda activate ${ENV_NAME}${NC}"
echo ""
if [ "$SKIP_OLLAMA" = false ]; then
    echo "  2. Make sure Ollama is running:"
    echo "     ${YELLOW}ollama serve${NC}"
    echo ""
    echo "  3. Verify the setup:"
    echo "     ${YELLOW}./setup.sh --verify${NC}"
    echo ""
    echo "  4. Run a test task:"
    echo "     ${YELLOW}python main.py --task \"Test task\" --project-dir ./test_project --dry-run${NC}"
else
    echo "  2. Verify the setup:"
    echo "     ${YELLOW}./setup.sh --verify${NC}"
    echo ""
    echo "  3. Run a test task:"
    echo "     ${YELLOW}python main.py --task \"Test task\" --project-dir ./test_project --dry-run${NC}"
fi
echo ""
echo "For SLURM clusters:"
echo "  ${YELLOW}python main.py --prompt-file prompts/example_prompt.txt --project-dir ./my_project --slurm${NC}"
echo ""
echo "See README.md for full documentation."
echo ""
print_success "Happy automating!"

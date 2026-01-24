#!/bin/bash

# Setup script for Multi-Agent System
# Run this once to initialize the system

set -e  # Exit on error

echo "========================================"
echo "  Multi-Agent System Setup"
echo "========================================"
echo ""

# Check for conda
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Check for ollama
if ! command -v ollama &> /dev/null; then
    echo "WARNING: ollama not found. Make sure it's installed and running."
    echo "         Install from: https://ollama.ai"
fi

# Create directory structure
echo "Creating directory structure..."
mkdir -p agents tools/dynamic_tools workflows utils 
mkdir -p data/inputs data/outputs 
mkdir -p logs config prompts
mkdir -p envs reports temp

# Create __init__.py files
echo "Creating Python package files..."
touch agents/__init__.py
touch tools/__init__.py
touch workflows/__init__.py
touch utils/__init__.py

# Create .gitkeep files for empty directories
touch logs/.gitkeep
touch data/inputs/.gitkeep
touch data/outputs/.gitkeep
touch tools/dynamic_tools/.gitkeep
touch envs/.gitkeep
touch reports/.gitkeep
touch temp/.gitkeep
touch prompts/.gitkeep

# Create .gitignore
echo "Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
*.pyc
__pycache__/
*.py[cod]
*$py.class
.Python
*.so

# Virtual environments
venv/
ENV/
.env

# Agent system files
workflow_state.db
logs/*.jsonl
temp/

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Keep structure files
!logs/.gitkeep
!data/inputs/.gitkeep
!data/outputs/.gitkeep
!tools/dynamic_tools/.gitkeep
!envs/.gitkeep
!reports/.gitkeep
!temp/.gitkeep
!prompts/.gitkeep
EOF

# Check if AGI conda environment exists
echo ""
echo "Checking conda environment..."
if conda env list | grep -q "AGI"; then
    echo "AGI environment found. Activating..."
    eval "$(conda shell.bash hook)"
    conda activate AGI
else
    echo "Creating AGI conda environment..."
    conda create -n AGI python=3.10 -y
    eval "$(conda shell.bash hook)"
    conda activate AGI
fi

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Initialize Git repository if not exists
if [ ! -d .git ]; then
    echo ""
    echo "Initializing Git repository..."
    git init
    git add .
    git commit -m "Initial commit: Multi-agent system setup"
fi

# Check Ollama connection
echo ""
echo "Checking Ollama connection..."
if curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
    echo "✓ Ollama is running"
    echo "Available models:"
    curl -s http://127.0.0.1:11434/api/tags | python3 -c "import sys, json; data=json.load(sys.stdin); print('  ' + '\n  '.join([m['name'] for m in data.get('models', [])]))" 2>/dev/null || echo "  (could not parse models)"
else
    echo "✗ Ollama is not running or not accessible"
    echo "  Start it with: ollama serve"
fi

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "To run the system:"
echo ""
echo "  1. Activate the environment:"
echo "     conda activate AGI"
echo ""
echo "  2. Start Ollama (if not running):"
echo "     ollama serve"
echo ""
echo "  3. Run with a prompt file:"
echo "     python main.py --prompt-file prompts/your_task.txt --project-dir ./your_project"
echo ""
echo "  4. Or run with inline task:"
echo "     python main.py --task \"Your task description\" --project-dir ./your_project"
echo ""
echo "Example prompt file format: see prompts/example_prompt.txt"
echo ""

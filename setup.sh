#!/bin/bash

# Setup script for Multi-Agent System

echo "Setting up Multi-Agent System..."

# Create directory structure
mkdir -p agents tools/dynamic_tools workflows utils data/inputs data/outputs logs config

# Create __init__.py files
touch agents/__init__.py
touch tools/__init__.py
touch workflows/__init__.py
touch utils/__init__.py

# Create .gitkeep files
touch logs/.gitkeep
touch data/inputs/.gitkeep
touch data/outputs/.gitkeep
touch tools/dynamic_tools/.gitkeep

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Initialize Git repository
if [ ! -d .git ]; then
    echo "Initializing Git repository..."
    git init
    git add .
    git commit -m "Initial commit: Multi-agent system setup"
fi

echo "Setup complete!"
echo ""
echo "To run the system:"
echo '  python main.py --task "Your task description here"'
echo ""
echo "Example:"
echo '  python main.py --task "Analyze biotech research centers in San Antonio and create a partnership strategy"'

#!/bin/bash
# =============================================================================
# Mem0 Setup Script for AGI Pipeline v3
# =============================================================================
# Sets up embedded Qdrant + Ollama for ReflexionMemory
#
# Usage:
#   ./scripts/setup_mem0.sh
#
# Or with custom data directory:
#   AGI_DATA_DIR=/custom/path ./scripts/setup_mem0.sh
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "========================================"
echo "  Mem0 Setup for AGI Pipeline v3"
echo "  Embedded Qdrant + Ollama"
echo "========================================"
echo ""

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

AGI_DATA_DIR="${AGI_DATA_DIR:-$HOME/agi_data}"
OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"

echo -e "${BLUE}Configuration:${NC}"
echo "  AGI_DATA_DIR: $AGI_DATA_DIR"
echo "  OLLAMA_URL: $OLLAMA_URL"
echo ""

# -----------------------------------------------------------------------------
# Step 1: Create directories
# -----------------------------------------------------------------------------

echo -e "${BLUE}Step 1: Creating directories...${NC}"

mkdir -p "$AGI_DATA_DIR/qdrant_storage"
mkdir -p "$AGI_DATA_DIR/logs"

echo -e "  ${GREEN}✓${NC} Created $AGI_DATA_DIR/qdrant_storage"
echo -e "  ${GREEN}✓${NC} Created $AGI_DATA_DIR/logs"

# -----------------------------------------------------------------------------
# Step 2: Check Python environment
# -----------------------------------------------------------------------------

echo ""
echo -e "${BLUE}Step 2: Checking Python environment...${NC}"

if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    echo -e "  ${GREEN}✓${NC} $PYTHON_VERSION"
else
    echo -e "  ${RED}✗${NC} Python not found"
    exit 1
fi

# Check if in conda environment
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo -e "  ${GREEN}✓${NC} Conda environment: $CONDA_DEFAULT_ENV"
else
    echo -e "  ${YELLOW}!${NC} Not in a conda environment (using system Python)"
fi

# -----------------------------------------------------------------------------
# Step 3: Install Python dependencies
# -----------------------------------------------------------------------------

echo ""
echo -e "${BLUE}Step 3: Installing Python dependencies...${NC}"

# Check for pip
if ! command -v pip &> /dev/null; then
    echo -e "  ${RED}✗${NC} pip not found"
    exit 1
fi

# Install mem0ai
echo "  Installing mem0ai..."
if pip show mem0ai &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} mem0ai already installed"
else
    pip install mem0ai --quiet
    echo -e "  ${GREEN}✓${NC} mem0ai installed"
fi

# Install qdrant-client (should come with mem0ai, but just in case)
echo "  Checking qdrant-client..."
if pip show qdrant-client &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} qdrant-client available"
else
    pip install qdrant-client --quiet
    echo -e "  ${GREEN}✓${NC} qdrant-client installed"
fi

# Install pyyaml for config loading
echo "  Checking pyyaml..."
if pip show pyyaml &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} pyyaml available"
else
    pip install pyyaml --quiet
    echo -e "  ${GREEN}✓${NC} pyyaml installed"
fi

# -----------------------------------------------------------------------------
# Step 4: Check Ollama
# -----------------------------------------------------------------------------

echo ""
echo -e "${BLUE}Step 4: Checking Ollama...${NC}"

# Test Ollama connection
if curl -s --connect-timeout 5 "$OLLAMA_URL/api/tags" > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} Ollama is running at $OLLAMA_URL"
else
    echo -e "  ${RED}✗${NC} Cannot connect to Ollama at $OLLAMA_URL"
    echo ""
    echo "  Please start Ollama:"
    echo "    ollama serve"
    echo ""
    echo "  Or if on HPC, ensure Ollama is accessible"
    exit 1
fi

# Check for required models
MODELS=$(curl -s "$OLLAMA_URL/api/tags" | python -c "import sys,json; print(' '.join(m['name'] for m in json.load(sys.stdin).get('models',[])))" 2>/dev/null || echo "")

echo "  Available models: $MODELS"

# -----------------------------------------------------------------------------
# Step 5: Pull embedding model if needed
# -----------------------------------------------------------------------------

echo ""
echo -e "${BLUE}Step 5: Checking embedding model...${NC}"

if echo "$MODELS" | grep -q "nomic-embed-text"; then
    echo -e "  ${GREEN}✓${NC} nomic-embed-text is available"
else
    echo "  Pulling nomic-embed-text (this may take a few minutes)..."
    ollama pull nomic-embed-text
    echo -e "  ${GREEN}✓${NC} nomic-embed-text pulled"
fi

# -----------------------------------------------------------------------------
# Step 6: Verify embedding model works
# -----------------------------------------------------------------------------

echo ""
echo -e "${BLUE}Step 6: Testing embedding model...${NC}"

EMBED_DIMS=$(curl -s "$OLLAMA_URL/api/embeddings" \
    -d '{"model": "nomic-embed-text", "prompt": "test"}' \
    | python -c "import sys,json; e=json.load(sys.stdin).get('embedding',[]); print(len(e))" 2>/dev/null || echo "0")

if [ "$EMBED_DIMS" == "768" ]; then
    echo -e "  ${GREEN}✓${NC} Embedding model working (768 dimensions)"
else
    echo -e "  ${RED}✗${NC} Embedding test failed (got $EMBED_DIMS dimensions)"
    echo "  Try: ollama pull nomic-embed-text"
    exit 1
fi

# -----------------------------------------------------------------------------
# Step 7: Check for LLM model
# -----------------------------------------------------------------------------

echo ""
echo -e "${BLUE}Step 7: Checking LLM model...${NC}"

if echo "$MODELS" | grep -qE "(llama3|mistral|qwen)"; then
    echo -e "  ${GREEN}✓${NC} LLM model available"
else
    echo -e "  ${YELLOW}!${NC} No standard LLM found"
    echo "  Pulling llama3.1:70b (recommended for memory operations)..."
    ollama pull llama3.1:70b
    echo -e "  ${GREEN}✓${NC} llama3.1:70b pulled"
fi

# -----------------------------------------------------------------------------
# Step 8: Set environment variables
# -----------------------------------------------------------------------------

echo ""
echo -e "${BLUE}Step 8: Environment variables...${NC}"

echo ""
echo "  Add these to your shell profile (~/.bashrc or ~/.zshrc):"
echo ""
echo "    export AGI_DATA_DIR=\"$AGI_DATA_DIR\""
echo "    export OLLAMA_BASE_URL=\"$OLLAMA_URL\""
echo ""

# Create a sourceable file
ENV_FILE="$AGI_DATA_DIR/env.sh"
cat > "$ENV_FILE" << EOF
# Mem0 Environment Variables
# Source this file: source $ENV_FILE

export AGI_DATA_DIR="$AGI_DATA_DIR"
export OLLAMA_BASE_URL="$OLLAMA_URL"
EOF

echo -e "  ${GREEN}✓${NC} Created $ENV_FILE"
echo "  You can source it with: source $ENV_FILE"

# -----------------------------------------------------------------------------
# Step 9: Run test script
# -----------------------------------------------------------------------------

echo ""
echo -e "${BLUE}Step 9: Running memory test...${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$SCRIPT_DIR/test_memory.py" ]; then
    python "$SCRIPT_DIR/test_memory.py"
else
    echo -e "  ${YELLOW}!${NC} test_memory.py not found at $SCRIPT_DIR"
    echo "  Run it manually after copying to your repo"
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "Data directory: $AGI_DATA_DIR"
echo "Qdrant storage: $AGI_DATA_DIR/qdrant_storage (embedded, no container)"
echo ""
echo "Next steps:"
echo "  1. Source environment: source $ENV_FILE"
echo "  2. Run test: python scripts/test_memory.py"
echo "  3. Import in your code:"
echo ""
echo "     from agi.memory import ReflexionMemory, FailureType"
echo "     memory = ReflexionMemory()"
echo ""

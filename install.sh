#!/bin/bash
# Helix Memory Installer for Claude Code
# Usage: ./install.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILLS_DIR="${CLAUDE_SKILLS_DIR:-$HOME/.claude/skills}"

echo "=== Helix Memory Installer ==="
echo ""

# Check dependencies
echo "Checking dependencies..."

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is required but not installed."
    echo "Install Docker Desktop from: https://www.docker.com/products/docker-desktop"
    exit 1
fi
echo "  Docker: OK"

# Check for Helix CLI
if ! command -v helix &> /dev/null; then
    echo ""
    echo "Installing Helix CLI..."
    curl -fsSL https://www.helix-db.com/install.sh | bash

    # Add to PATH if needed
    if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        export PATH="$HOME/.local/bin:$PATH"
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
    fi
fi
echo "  Helix CLI: OK"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is required but not installed."
    exit 1
fi
echo "  Python 3: OK"

# Check for requests module
if ! python3 -c "import requests" 2>/dev/null; then
    echo "Installing Python requests module..."
    pip3 install requests
fi
echo "  Python requests: OK"

echo ""
echo "Installing Helix Memory..."

# Make scripts executable
chmod +x "$SCRIPT_DIR/memory"
chmod +x "$SCRIPT_DIR/hooks/"*.py 2>/dev/null || true

# Create skills directory and symlink
mkdir -p "$SKILLS_DIR"

if [ -L "$SKILLS_DIR/helix-memory" ]; then
    rm "$SKILLS_DIR/helix-memory"
fi

if [ -d "$SKILLS_DIR/helix-memory" ]; then
    echo "  Backing up existing skill to $SKILLS_DIR/helix-memory.bak..."
    mv "$SKILLS_DIR/helix-memory" "$SKILLS_DIR/helix-memory.bak"
fi

# Symlink skill (so updates are automatic)
ln -s "$SCRIPT_DIR/skills/helix-memory" "$SKILLS_DIR/helix-memory"
echo "  Linked skill to $SKILLS_DIR/helix-memory"

# Update paths in skill files to point to this repo
HELIX_BIN=$(which helix 2>/dev/null || echo "$HOME/.local/bin/helix")
sed -i '' "s|/Users/cminds/Tools/helix-memory|$SCRIPT_DIR|g" "$SCRIPT_DIR/skills/helix-memory/SKILL.md" 2>/dev/null || true
sed -i '' "s|/Users/cminds/.local/bin/helix|$HELIX_BIN|g" "$SCRIPT_DIR/memory" 2>/dev/null || true

# Update hook paths
for hook in "$SCRIPT_DIR/hooks/"*.py; do
    sed -i '' "s|/Users/cminds/Tools/helix-memory|$SCRIPT_DIR|g" "$hook" 2>/dev/null || true
done

# Initialize HelixDB
echo ""
echo "Initializing HelixDB..."
cd "$SCRIPT_DIR"

# Start Docker if needed
if ! docker info &>/dev/null; then
    echo "  Starting Docker Desktop..."
    open -a "Docker Desktop"
    sleep 10
fi

# Push schema
helix push dev || true

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Usage:"
echo "  $SCRIPT_DIR/memory start   # Start HelixDB"
echo "  $SCRIPT_DIR/memory status  # Check status"
echo "  $SCRIPT_DIR/memory help    # Show all commands"
echo ""
echo "Add to your shell profile for convenience:"
echo "  alias memory='$SCRIPT_DIR/memory'"
echo ""
echo "Optional: Configure hooks in ~/.claude/settings.json"
echo "(See README.md for hook configuration)"
echo ""
echo "Optional: Install Ollama for better semantic search:"
echo "  brew install ollama && ollama pull nomic-embed-text"

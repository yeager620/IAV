#!/bin/bash
# Unified environment setup script

set -e

echo "🚁 Setting up Drone VLA unified environment..."

# Detect which approach to use
if command -v pixi &> /dev/null; then
    echo "Using pixi for unified environment management"
    
    # Replace current pixi.toml with unified version
    cp pixi-unified.toml pixi.toml
    
    # Install all dependencies  
    pixi install
    
    echo "✅ Unified pixi environment ready!"
    echo "Usage:"
    echo "  pixi run build    # Build and validate"
    echo "  pixi run test     # Run tests"
    echo "  pixi run dev      # Start development server"
    echo "  pixi shell        # Enter environment"
    
elif command -v uv &> /dev/null; then
    echo "Using uv for Python, minimal config for Mojo"
    
    # Replace current pyproject.toml with streamlined version
    cp pyproject-streamlined.toml pyproject.toml
    
    # Install Python dependencies
    uv sync
    
    # Install optional ML dependencies if needed
    # uv sync --extra ml
    
    echo "✅ Streamlined uv environment ready!"
    echo "Usage:"
    echo "  uv run python main.py    # Run main application"
    echo "  uv run pytest tests/     # Run tests"
    echo "  ./scripts/build.sh       # Build and validate"
    
else
    echo "❌ Neither pixi nor uv found. Please install one of them:"
    echo "  curl -fsSL https://pixi.sh/install.sh | bash  # For pixi"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh  # For uv"
    exit 1
fi

# Clean up old lock files from other package managers
echo "Cleaning up old package manager files..."
if [ -f "magic.lock" ]; then
    echo "  Removing magic.lock"
    rm -f magic.lock
fi

# Keep the lock file for the chosen package manager
if [ -f "pixi.toml" ] && [ ! -f "pixi-unified.toml" ]; then
    # We're using the original pixi setup
    if [ -f "uv.lock" ]; then
        echo "  Moving uv.lock to uv.lock.backup"
        mv uv.lock uv.lock.backup
    fi
elif [ -f "pyproject-streamlined.toml" ]; then
    # We're using the streamlined uv setup
    if [ -f "pixi.lock" ]; then
        echo "  Moving pixi.lock to pixi.lock.backup"
        mv pixi.lock pixi.lock.backup
    fi
fi

echo "🎉 Environment setup complete!"
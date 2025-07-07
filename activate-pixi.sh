#!/bin/bash
# Convenience script to activate pixi environment
# Usage: source activate-pixi.sh

export PATH="$HOME/.pixi/bin:$PATH"

echo "🚁 Pixi environment activated!"
echo "Available commands:"
echo "  pixi run build     # Build and validate project"
echo "  pixi run test      # Run all tests"
echo "  pixi run dev       # Start development mode"
echo "  pixi run format    # Format code with ruff"
echo "  pixi run lint      # Lint code with ruff"
echo "  pixi shell         # Enter pixi shell"
echo ""
echo "Current environment: $(pixi info)"
#!/bin/bash
# Streamlined test script for cleaned codebase

set -e

echo "Running UAV system tests..."

# Run Python tests
echo "Running Python component tests..."
uv run python -m pytest tests/ -v

# Test main.py functionality
echo "Testing main.py functionality..."
uv run python test_main.py

# Test syntax validation
echo "Testing Mojo syntax..."
uv run python test_mojo_syntax.py

echo "✅ All tests passed"
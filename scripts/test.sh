#!/bin/bash
# Streamlined test script

set -e

export PATH="/Users/$(whoami)/.modular/bin:$PATH"

echo "Running UAV system tests..."

# Run Python tests
echo "Running Python component tests..."
magic run pytest tests/test_streamlined.py -v

echo "✅ All tests passed"
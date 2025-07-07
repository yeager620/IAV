#!/bin/bash
# Streamlined build script for cleaned codebase

set -e

echo "Building UAV system..."

# Test Python compilation
echo "Testing Python compilation..."
uv run python -c "import sys; sys.path.append('src/python'); from minimal_interface import SystemOrchestrator; print('✅ Python compilation successful')"

# Test Mojo syntax (without actual compilation since Mojo may not be installed)
echo "Testing Mojo syntax..."
uv run python test_mojo_syntax.py

# Create build directory for future Mojo builds
mkdir -p build

echo "Build structure:"
echo "  - Python interface: src/python/minimal_interface.py"
echo "  - Mojo components: src/mojo/*.mojo (9 files)"
echo "  - Configuration: config/minimal_config.json"
echo "  - Tests: tests/"

echo "✅ Build validation completed successfully"
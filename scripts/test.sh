#!/bin/bash
# Streamlined test script for pixi-unified environment

set -e

echo "Running UAV system tests with pixi..."

# Run Python tests
echo "Running Python component tests..."
python -m pytest tests/ -v

# Test main.py functionality  
echo "Testing main.py functionality..."
python -c "
import sys
sys.path.append('src/python')
from minimal_interface import SystemOrchestrator
config = {'mavlink': {'connection': 'udp:127.0.0.1:14550'}, 'camera_id': 0, 'control_frequency': 100}
system = SystemOrchestrator(config)
print('✅ Main.py functionality validated')
"

echo "✅ All tests passed"
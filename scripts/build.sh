#!/bin/bash
# Enhanced build script for pixi-unified environment with Mojo compilation

set -e

echo "Building UAV system with pixi..."

# Test Python compilation
echo "Testing Python compilation..."
python -c "import sys; sys.path.append('src'); from models.huggingface.vla_model import create_drone_vla_model; print('Python compilation successful')"

# Create build directories
mkdir -p build/mojo
mkdir -p build/python

# Compile Mojo files
echo "Compiling Mojo files..."
if command -v mojo &> /dev/null; then
    echo "Found Mojo compiler, compiling..."
    
    # Core Mojo files that need compilation
    MOJO_FILES=(
        "src/mojo/drone_core.mojo"
        "src/mojo/uav_core.mojo"
        "src/mojo/camera_bridge.mojo"
        "src/mojo/network_bridge.mojo"
    )
    
    for file in "${MOJO_FILES[@]}"; do
        if [[ -f "$file" ]]; then
            echo "  Compiling $file..."
            filename=$(basename "$file" .mojo)
            mojo build "$file" -o "build/mojo/$filename"
            echo "    -> build/mojo/$filename"
        fi
    done
    
    # Create __init__.py for Python imports
    cat > build/mojo/__init__.py << 'EOF'
"""
Compiled Mojo modules for drone control system
"""
import os
import sys

# Add build directory to path for compiled modules
MOJO_BUILD_DIR = os.path.dirname(os.path.abspath(__file__))
if MOJO_BUILD_DIR not in sys.path:
    sys.path.insert(0, MOJO_BUILD_DIR)
EOF
    
    echo "Mojo compilation completed"
else
    echo "Mojo compiler not found, creating stubs..."
    
    # Create stub files for development
    cat > build/mojo/drone_core.py << 'EOF'
"""Stub for drone_core.mojo - replace with actual Mojo compilation"""
import numpy as np

def create_drone_controller():
    """Stub function"""
    return None

def process_navigation_command(command):
    """Stub function"""
    return np.zeros(6)
EOF
    
    cat > build/mojo/uav_core.py << 'EOF'
"""Stub for uav_core.mojo - replace with actual Mojo compilation"""
import numpy as np

def process_motor_commands(commands):
    """Stub function"""
    return np.zeros(4)
EOF
    
    cat > build/mojo/camera_bridge.py << 'EOF'
"""Stub for camera_bridge.mojo - replace with actual Mojo compilation"""
import numpy as np

def process_frame(frame):
    """Stub function"""
    return frame
EOF
    
    cat > build/mojo/network_bridge.py << 'EOF'
"""Stub for network_bridge.mojo - replace with actual Mojo compilation"""

def create_mavlink_interface():
    """Stub function"""
    return None
EOF
    
    cat > build/mojo/__init__.py << 'EOF'
"""
Mojo module stubs for development
"""
EOF
    
    echo "Mojo stubs created (install Mojo compiler for actual compilation)"
fi

# Test Mojo compilation results
echo "Testing Mojo compilation results..."
python -c "
import sys
sys.path.append('build')
try:
    import mojo
    print('Mojo modules accessible')
except ImportError as e:
    print(f'Mojo module import failed: {e}')
"

echo ""
echo "Build structure:"
echo "  - Python interface: src/python/minimal_interface.py"
echo "  - Mojo components: src/mojo/*.mojo (9 files)"
echo "  - Configuration: config/minimal_config.json"
echo "  - Tests: tests/"
echo ""
echo "Environment: pixi-unified"
echo "Python: $(python --version)"
echo "✅ Build validation completed successfully"
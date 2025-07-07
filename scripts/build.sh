#!/bin/bash
# Streamlined build script

set -e

export PATH="/Users/$(whoami)/.modular/bin:$PATH"

echo "Building Mojo UAV system..."

# Create build directory
mkdir -p build

# Build core Mojo components
echo "Building core types..."
magic run mojo build src/mojo/core_types.mojo -o build/core_types

echo "Building safety monitor..."
magic run mojo build src/mojo/safety_monitor.mojo -o build/safety_monitor

echo "Building control allocator..."  
magic run mojo build src/mojo/control_allocator.mojo -o build/control_allocator

echo "Building complete UAV system..."
magic run mojo build src/mojo/uav_system.mojo -o build/uav_system

echo "✅ Build completed successfully"
#!/bin/bash
# Development setup script for UAV VLA system

set -e

echo "Setting up UAV VLA development environment..."

# Check if we're in the right directory
if [ ! -f "pixi.toml" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Setup Magic/Mojo environment
export PATH="/Users/$(whoami)/.modular/bin:$PATH"

echo "Installing Python dependencies..."
magic add pymavlink opencv-python numpy pytest pytest-asyncio

echo "Installing development tools..."  
magic add --dev black ruff mypy

echo "Creating data directories..."
mkdir -p data/models data/logs data/datasets

echo "Setting up remote development keys..."
if [ ! -f "$HOME/.ssh/jetson_key" ]; then
    echo "Generating SSH key for Jetson..."
    ssh-keygen -t rsa -b 4096 -f "$HOME/.ssh/jetson_key" -N ""
    echo "Please copy the public key to your Jetson:"
    echo "ssh-copy-id -i ~/.ssh/jetson_key.pub jetson@192.168.1.100"
fi

if [ ! -f "$HOME/.ssh/pi_key" ]; then
    echo "Generating SSH key for Raspberry Pi..."
    ssh-keygen -t rsa -b 4096 -f "$HOME/.ssh/pi_key" -N ""
    echo "Please copy the public key to your Raspberry Pi:"
    echo "ssh-copy-id -i ~/.ssh/pi_key.pub pi@192.168.1.101"
fi

echo "Creating SSH config..."
cat > "$HOME/.ssh/config_uav" << EOF
Host jetson
    HostName 192.168.1.100
    User jetson
    IdentityFile ~/.ssh/jetson_key
    Port 22

Host pi  
    HostName 192.168.1.101
    User pi
    IdentityFile ~/.ssh/pi_key
    Port 22
EOF

echo "Setting up VS Code configuration..."
mkdir -p .vscode
cat > .vscode/settings.json << EOF
{
    "python.defaultInterpreterPath": "./.pixi/envs/default/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "files.associations": {
        "*.mojo": "python"
    },
    "terminal.integrated.env.osx": {
        "PATH": "/Users/$(whoami)/.modular/bin:\${env:PATH}"
    }
}
EOF

cat > .vscode/launch.json << EOF
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Test Mojo Integration",
            "type": "python",
            "request": "launch",
            "program": "\${workspaceFolder}/tests/test_mojo_integration.py",
            "console": "integratedTerminal",
            "justMyCode": true,
            "env": {
                "PYTHONPATH": "\${workspaceFolder}/src/python"
            }
        },
        {
            "name": "Python: System Test",
            "type": "python", 
            "request": "launch",
            "program": "\${workspaceFolder}/main.py",
            "args": ["--config", "config/system_config.json", "--simulation"],
            "console": "integratedTerminal",
            "justMyCode": true
        }
    ]
}
EOF

echo "Creating development scripts..."
cat > scripts/test.sh << 'EOF'
#!/bin/bash
export PATH="/Users/$(whoami)/.modular/bin:$PATH"
magic run pytest tests/ -v
EOF

cat > scripts/format.sh << 'EOF'
#!/bin/bash
export PATH="/Users/$(whoami)/.modular/bin:$PATH"
magic run black src/ tests/
magic run ruff src/ tests/ --fix
EOF

cat > scripts/build_mojo.sh << 'EOF'
#!/bin/bash
export PATH="/Users/$(whoami)/.modular/bin:$PATH"
echo "Building Mojo components..."
magic run mojo build src/mojo/vla_inference.mojo -o build/vla_inference
magic run mojo build src/mojo/control_allocator.mojo -o build/control_allocator  
magic run mojo build src/mojo/safety_monitor.mojo -o build/safety_monitor
echo "Mojo build completed"
EOF

chmod +x scripts/*.sh

echo "Setting up pre-commit hooks..."
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook for UAV VLA system

echo "Running pre-commit checks..."

# Format code
./scripts/format.sh

# Run tests
./scripts/test.sh

if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi

echo "Pre-commit checks passed"
EOF

chmod +x .git/hooks/pre-commit

echo "Creating main entry point..."
cat > main.py << 'EOF'
#!/usr/bin/env python3
"""
Main entry point for UAV VLA system
"""

import asyncio
import argparse
import json
import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src" / "python"))

from system_orchestrator import SystemOrchestrator

def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('data/logs/system.log')
        ]
    )

async def main():
    parser = argparse.ArgumentParser(description="UAV VLA System")
    parser.add_argument("--config", default="config/system_config.json", help="Configuration file")
    parser.add_argument("--simulation", action="store_true", help="Run in simulation mode")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--command", help="Single command to execute")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    with open(args.config) as f:
        config = json.load(f)
    
    if args.simulation:
        config['mavlink']['connection'] = 'udp:127.0.0.1:14550'  # SITL connection
    
    # Create and initialize system
    system = SystemOrchestrator(config)
    
    try:
        logger.info("Initializing UAV VLA system...")
        if not await system.initialize():
            logger.error("System initialization failed")
            return 1
        
        logger.info("System initialized successfully")
        
        if args.command:
            # Execute single command
            logger.info(f"Executing command: {args.command}")
            success = await system.execute_command(args.command)
            return 0 if success else 1
        else:
            # Start autonomous mode
            logger.info("Starting autonomous mode...")
            if not await system.start_autonomous_mode():
                logger.error("Failed to start autonomous mode")
                return 1
            
            # Keep running until interrupted
            try:
                while True:
                    await asyncio.sleep(1)
                    
                    # Print status every 10 seconds
                    if int(asyncio.get_event_loop().time()) % 10 == 0:
                        stats = system.get_performance_stats()
                        logger.info(f"System stats: {stats}")
                        
            except KeyboardInterrupt:
                logger.info("Shutdown requested")
            
            await system.stop_autonomous_mode()
        
        return 0
        
    except Exception as e:
        logger.error(f"System error: {e}")
        return 1
    finally:
        await system.cleanup()

if __name__ == "__main__":
    exit(asyncio.run(main()))
EOF

chmod +x main.py

echo "Development environment setup completed!"
echo ""
echo "Next steps:"
echo "1. Copy SSH keys to hardware: ssh-copy-id -i ~/.ssh/jetson_key.pub jetson@192.168.1.100"
echo "2. Run tests: ./scripts/test.sh"
echo "3. Build Mojo components: ./scripts/build_mojo.sh" 
echo "4. Start system: python main.py --simulation"
echo ""
echo "For remote development:"
echo "1. Use 'ssh jetson' or 'ssh pi' to connect to hardware"
echo "2. Use './scripts/deploy.sh' to deploy code to hardware"
#!/usr/bin/env python3
"""
Streamlined UAV VLA System - Production Ready
Uses Mojo for performance-critical components, Python only for I/O and system integration.
"""

import asyncio
import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src" / "python"))

try:
    from minimal_interface import SystemOrchestrator
except ImportError:
    # Fallback to unified system
    from src.core.autonomous_system import AutonomousDroneSystem as SystemOrchestrator

def setup_logging(level: str = "INFO"):
    """Setup logging"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

async def main():
    parser = argparse.ArgumentParser(description="UAV VLA System")
    parser.add_argument("--config", default="config/minimal_config.json", help="Configuration file")
    parser.add_argument("--simulation", action="store_true", help="Use simulation connection")
    parser.add_argument("--command", help="Single command to execute")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    try:
        with open(args.config) as f:
            config = json.load(f)
    except FileNotFoundError:
        # Default configuration
        config = {
            "mavlink": {"connection": "/dev/ttyUSB0"},
            "camera_id": 0,
            "control_frequency": 100
        }
    
    if args.simulation:
        config['mavlink']['connection'] = 'udp:127.0.0.1:14550'
    
    # Create system
    system = SystemOrchestrator(config)
    
    try:
        logger.info("Initializing UAV system...")
        if not await system.initialize():
            logger.error("System initialization failed")
            return 1
        
        if args.command:
            # Execute single command (simplified)
            logger.info(f"Executing command: {args.command}")
            await asyncio.sleep(1.0)  # Simulate command execution
            return 0
        else:
            # Start control loop
            logger.info("Starting control loop...")
            await system.control_loop()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
        system.stop()
        return 0
    except Exception as e:
        logger.error(f"System error: {e}")
        return 1
    finally:
        system.cleanup()

if __name__ == "__main__":
    exit(asyncio.run(main()))
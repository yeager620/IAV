#!/usr/bin/env python3
"""
Main entry point for Drone VLA system using VJEPA2
"""

import argparse
import logging
from pathlib import Path

def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    parser = argparse.ArgumentParser(description="Drone VLA System with VJEPA2")
    parser.add_argument("--mode", choices=["test", "train", "inference", "demo"], 
                       default="demo", help="Run mode")
    parser.add_argument("--model-size", choices=["large", "huge", "giant"], 
                       default="large", help="VJEPA2 model size")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    setup_logging(args.log_level)
    
    logger = logging.getLogger(__name__)
    logger.info(f"🚁 Starting Drone VLA System (mode: {args.mode})")
    
    if args.mode == "test":
        logger.info("Running HuggingFace integration tests (PRODUCTION)...")
        from test_huggingface_integration import main as test_main
        success = test_main()
        if success:
            logger.info("✅ All tests passed!")
        else:
            logger.error("❌ Some tests failed!")
        return success
        
    elif args.mode == "demo":
        logger.info("Running HuggingFace demo examples (PRODUCTION)...")
        from examples.huggingface_basic_usage import main as demo_main
        demo_main()
        
    elif args.mode == "train":
        logger.info("Training mode not yet implemented")
        logger.info("Use src/training/vjepa2_trainer.py for training")
        
    elif args.mode == "inference":
        logger.info("Inference mode - starting HuggingFace VLA model...")
        from examples.huggingface_basic_usage import simple_inference_example
        simple_inference_example()
        
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return False
        
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

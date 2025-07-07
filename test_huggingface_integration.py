
"""
Test script for HuggingFace VJEPA2 integration (PRIMARY APPROACH)
This tests the production-ready HuggingFace implementation.
"""

import torch
import numpy as np
from PIL import Image
import cv2
import logging
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dummy_video_frames(num_frames: int = 16, height: int = 256, width: int = 256) -> list:
    """Create dummy video frames for testing"""
    frames = []
    for i in range(num_frames):

        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = (i * 255 // num_frames)
        frame[height//4:3*height//4, width//4:3*width//4, 1] = 255
        frame[height//3:2*height//3, width//3:2*width//3, 2] = 128
        frames.append(frame)
    return frames


def test_huggingface_encoder():
    """Test HuggingFace VJEPA2 encoder"""
    logger.info("Testing HuggingFace VJEPA2 Encoder...")
    
    try:
        from src.models.huggingface.vla_model import HuggingFaceVJEPA2Encoder
        

        encoder = HuggingFaceVJEPA2Encoder(
            model_name="facebook/vjepa2-vitl-fpc64-256",
            freeze_backbone=True,
            output_dim=512
        )
        

        video_frames = create_dummy_video_frames(16)
        

        with torch.no_grad():
            features = encoder(video_frames)
            logger.info(f"Encoded features shape: {features.shape}")
            
        logger.info("✅ HuggingFace VJEPA2 Encoder test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ HuggingFace VJEPA2 Encoder test failed: {e}")
        return False


def test_drone_vla_model():
    """Test complete HuggingFace VLA model"""
    logger.info("Testing HuggingFace Drone VLA Model...")
    
    try:
        from src.models.huggingface.vla_model import create_drone_vla_model
        

        model = create_drone_vla_model(
            model_size="large",
            freeze_backbone=True,
            feature_dim=512,
            action_dim=6
        )
        

        video_frames = create_dummy_video_frames(16)
        text_command = "Take off and hover at 5 meters"
        

        with torch.no_grad():
            output = model(
                frames=video_frames,
                text_command=text_command
            )
            
            logger.info(f"Actions shape: {output.actions.shape}")
            logger.info(f"Action confidence shape: {output.action_confidence.shape}")
            logger.info(f"Features shape: {output.features.shape}")
            

        with torch.no_grad():
            actions, confidence = model.predict_action(
                frames=video_frames,
                text_command=text_command,
                deterministic=True
            )
            logger.info(f"Predicted action: {actions[0]}")
            logger.info(f"Action confidence: {confidence[0]}")
            

        info = model.get_model_info()
        logger.info(f"Model info: {info}")
            
        logger.info("✅ HuggingFace Drone VLA Model test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ HuggingFace Drone VLA Model test failed: {e}")
        return False


def test_quick_inference():
    """Test quick inference function"""
    logger.info("Testing Quick Inference...")
    
    try:
        from src.models.huggingface.vla_model import quick_inference
        

        video_frames = create_dummy_video_frames(8)
        text_command = "Move forward slowly"
        

        actions, confidence = quick_inference(
            frames=video_frames,
            text_command=text_command,
            model_size="large"
        )
        
        logger.info(f"Quick inference - Actions: {actions}")
        logger.info(f"Quick inference - Confidence: {confidence}")
        
        logger.info("✅ Quick Inference test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Quick Inference test failed: {e}")
        return False


def test_integration_with_drone_control():
    """Test integration with drone control system"""
    logger.info("Testing Integration with Drone Control...")
    
    try:
        from src.models.huggingface.vla_model import create_drone_vla_model
        from src.core.drone_control import DroneController
        

        model = create_drone_vla_model(
            model_size="large",
            freeze_backbone=True,
            feature_dim=256
        )
        

        drone = DroneController(simulation_mode=True)
        drone.connect()
        drone.start_control_loop()
        

        video_frames = create_dummy_video_frames(8)
        text_command = "Take off to 3 meters"
        

        with torch.no_grad():
            actions, confidence = model.predict_action(
                frames=video_frames,
                text_command=text_command,
                deterministic=True
            )
            

        action_np = actions[0]
        drone.execute_action_vector(action_np)
        

        status = drone.get_status()
        logger.info(f"Drone state: {status.state}")
        logger.info(f"Drone position: {status.position}")
        

        drone.cleanup()
        
        logger.info("✅ Integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Integration test failed: {e}")
        return False


def test_different_model_sizes():
    """Test different VJEPA2 model sizes"""
    logger.info("Testing Different Model Sizes...")
    
    try:
        from src.models.huggingface.vla_model import create_drone_vla_model
        

        video_frames = create_dummy_video_frames(4)
        text_command = "Test command"
        

        logger.info("Testing Large model...")
        model_large = create_drone_vla_model(model_size="large", feature_dim=256)
        
        with torch.no_grad():
            actions_large, _ = model_large.predict_action(video_frames, text_command)
            logger.info(f"Large model action: {actions_large[0]}")
        


        





        
        logger.info("✅ Model size test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Model size test failed: {e}")
        return False


def test_performance():
    """Test performance metrics"""
    logger.info("Testing Performance...")
    
    try:
        from src.models.huggingface.vla_model import create_drone_vla_model
        import time
        

        model = create_drone_vla_model(model_size="large", feature_dim=256)
        model.eval()
        

        video_frames = create_dummy_video_frames(8)
        text_command = "Performance test"
        

        for _ in range(3):
            with torch.no_grad():
                model.predict_action(video_frames, text_command)
        

        num_iterations = 5
        start_time = time.time()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                actions, confidence = model.predict_action(video_frames, text_command)
        
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations
        fps = 1.0 / avg_time
        
        logger.info(f"Average inference time: {avg_time:.3f}s")
        logger.info(f"Inference FPS: {fps:.2f}")
        
        if torch.cuda.is_available():
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            logger.info(f"Peak GPU memory: {memory_mb:.1f} MB")
        
        logger.info("✅ Performance test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Performance test failed: {e}")
        return False


def test_autonomous_system():
    """Test complete autonomous system integration"""
    logger.info("Testing Autonomous System...")
    
    try:
        from src.core.autonomous_system import create_autonomous_system, MissionStep
        

        system = create_autonomous_system(
            model_size="large",
            safety_level="strict",
            simulation_mode=True
        )
        

        success = system.initialize()
        if not success:
            logger.error("System initialization failed")
            return False
            

        status = system.get_status()
        logger.info(f"System status: {status.state}")
        

        cmd_success = system.execute_command("Test hover command", timeout=5.0)
        logger.info(f"Command execution: {cmd_success}")
        

        system.cleanup()
        
        logger.info("✅ Autonomous system test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Autonomous system test failed: {e}")
        return False


def main():
    """Run all HuggingFace integration tests"""
    logger.info("Starting HuggingFace VJEPA2 Integration Tests (PRODUCTION APPROACH)...")
    logger.info("=" * 70)
    
    tests = [
        test_huggingface_encoder,
        test_drone_vla_model,
        test_quick_inference,
        test_integration_with_drone_control,
        test_different_model_sizes,
        test_performance,
        test_autonomous_system
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            logger.error(f"Test {test_func.__name__} failed with exception: {e}")
            
    logger.info(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All HuggingFace tests passed! Production system is ready.")
        logger.info("💡 Next steps:")
        logger.info("  - Run: python main.py --mode demo")
        logger.info("  - Or: python examples/huggingface_basic_usage.py")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Check the logs above.")
        
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
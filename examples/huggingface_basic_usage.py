
"""
Basic usage examples for HuggingFace VJEPA2-based drone VLA system.
This demonstrates the PRIMARY/PRODUCTION approach.
"""

import torch
import numpy as np
import cv2
import logging
from pathlib import Path
from typing import List


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_video_frames(num_frames: int = 16) -> List[np.ndarray]:
    """Create sample video frames for testing"""
    frames = []
    for i in range(num_frames):

        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        

        frame[:, :, 0] = (i * 255 // num_frames)
        

        x_pos = int(128 + 50 * np.sin(i * 0.5))
        y_pos = int(128 + 30 * np.cos(i * 0.3))
        cv2.circle(frame, (x_pos, y_pos), 20, (0, 255, 0), -1)
        

        cv2.rectangle(frame, (50, 50), (200, 200), (0, 0, 255), 2)
        
        frames.append(frame)
    
    logger.info(f"Created {len(frames)} sample video frames")
    return frames


def basic_model_loading_example():
    """Example 1: Basic model loading and info"""
    logger.info("=== Example 1: Basic Model Loading ===")
    
    try:
        from src.models.huggingface.vla_model import create_drone_vla_model
        

        logger.info("Loading VJEPA2 Large model...")
        model = create_drone_vla_model(model_size="large")
        

        info = model.get_model_info()
        logger.info("Model Information:")
        for key, value in info.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("Model loading successful")
        return True
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return False


def simple_inference_example():
    """Example 2: Simple inference with video and text"""
    logger.info("=== Example 2: Simple Inference ===")
    
    try:
        from src.models.huggingface.vla_model import create_drone_vla_model
        

        model = create_drone_vla_model(model_size="large", freeze_backbone=True)
        model.eval()
        

        video_frames = create_sample_video_frames(16)
        text_command = "Take off and hover at 5 meters altitude"
        
        logger.info(f"Input: {len(video_frames)} frames + text: '{text_command}'")
        

        with torch.no_grad():
            actions, confidence = model.predict_action(
                frames=video_frames,
                text_command=text_command,
                deterministic=True
            )
        

        logger.info("Predicted Actions:")
        for i, (action, conf) in enumerate(zip(actions[0], confidence[0])):
            action_names = ["vel_x", "vel_y", "vel_z", "roll_rate", "pitch_rate", "yaw_rate"]
            logger.info(f"  {action_names[i]}: {action:.4f} (confidence: {conf:.4f})")
        
        logger.info("Simple inference successful")
        return True
        
    except Exception as e:
        logger.error(f"Simple inference failed: {e}")
        return False


def multi_command_example():
    """Example 3: Multiple commands with different scenarios"""
    logger.info("=== Example 3: Multi-Command Scenarios ===")
    
    try:
        from src.models.huggingface.vla_model import create_drone_vla_model
        

        model = create_drone_vla_model(model_size="large")
        model.eval()
        

        scenarios = [
            {
                "description": "Takeoff",
                "command": "Take off to 3 meters height",
                "expected_action": "Positive vertical velocity"
            },
            {
                "description": "Landing", 
                "command": "Land safely on the ground",
                "expected_action": "Negative vertical velocity"
            },
            {
                "description": "Forward movement",
                "command": "Move forward slowly",
                "expected_action": "Positive forward velocity"
            },
            {
                "description": "Rotation",
                "command": "Turn right and scan the area", 
                "expected_action": "Positive yaw rate"
            },
            {
                "description": "Hover",
                "command": "Hover in place and maintain position",
                "expected_action": "Near-zero velocities"
            }
        ]
        
        logger.info(f"Testing {len(scenarios)} different scenarios...")
        
        for i, scenario in enumerate(scenarios):
            logger.info(f"\nScenario {i+1}: {scenario['description']}")
            logger.info(f"Command: '{scenario['command']}'")
            

            video_frames = create_sample_video_frames(8)
            

            with torch.no_grad():
                actions, confidence = model.predict_action(
                    frames=video_frames,
                    text_command=scenario["command"],
                    deterministic=True
                )
            

            action = actions[0]
            conf = confidence[0]
            

            logger.info(f"Key Actions:")
            logger.info(f"  Forward/Back: {action[0]:.3f}")
            logger.info(f"  Left/Right:   {action[1]:.3f}")
            logger.info(f"  Up/Down:      {action[2]:.3f}")
            logger.info(f"  Yaw rotation: {action[5]:.3f}")
            logger.info(f"Expected: {scenario['expected_action']}")
            logger.info(f"Avg confidence: {conf.mean():.3f}")
        
        logger.info("Multi-command testing successful")
        return True
        
    except Exception as e:
        logger.error(f"Multi-command testing failed: {e}")
        return False


def webcam_integration_example():
    """Example 4: Integration with webcam (if available)"""
    logger.info("=== Example 4: Webcam Integration ===")
    
    try:
        from src.models.huggingface.vla_model import create_drone_vla_model
        

        model = create_drone_vla_model(model_size="large")
        model.eval()
        

        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logger.warning("Webcam not available, using synthetic frames")
            video_frames = create_sample_video_frames(16)
        else:
            logger.info("Capturing frames from webcam...")
            video_frames = []
            
            for i in range(16):
                ret, frame = cap.read()
                if ret:

                    frame_resized = cv2.resize(frame, (256, 256))
                    video_frames.append(frame_resized)
                else:

                    video_frames.append(create_sample_video_frames(1)[0])
            
            cap.release()
            logger.info(f"Captured {len(video_frames)} frames from webcam")
        

        command = "Analyze the scene and approach any visible objects carefully"
        
        logger.info(f"Processing command: '{command}'")
        

        with torch.no_grad():
            actions, confidence = model.predict_action(
                frames=video_frames,
                text_command=command,
                deterministic=True
            )
        

        action = actions[0]
        logger.info("Real-world Action Prediction:")
        logger.info(f"  Movement: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}] m/s")
        logger.info(f"  Rotation: [{action[3]:.3f}, {action[4]:.3f}, {action[5]:.3f}] rad/s")
        logger.info(f"  Overall confidence: {confidence[0].mean():.3f}")
        
        logger.info("Webcam integration successful")
        return True
        
    except Exception as e:
        logger.error(f"Webcam integration failed: {e}")
        return False


def drone_simulation_integration():
    """Example 5: Integration with drone simulation"""
    logger.info("=== Example 5: Drone Simulation Integration ===")
    
    try:
        from src.models.huggingface.vla_model import create_drone_vla_model
        from src.core.drone_control import DroneController
        

        model = create_drone_vla_model(model_size="large")
        model.eval()
        

        drone = DroneController(simulation_mode=True)
        drone.connect()
        drone.start_control_loop()
        
        logger.info("Executing autonomous mission...")
        

        mission_steps = [
            ("Take off to 5 meters", 3),
            ("Look around for obstacles", 2), 
            ("Move forward carefully", 2),
            ("Return to starting position", 2),
            ("Land safely", 3)
        ]
        
        for step_command, duration in mission_steps:
            logger.info(f"Mission step: {step_command}")
            

            status = drone.get_status()
            logger.info(f"Drone position: {status.position}")
            

            video_frames = create_sample_video_frames(8)
            

            with torch.no_grad():
                actions, confidence = model.predict_action(
                    frames=video_frames,
                    text_command=step_command,
                    deterministic=True
                )
            

            action_vector = actions[0]
            drone.execute_action_vector(action_vector)
            
            logger.info(f"Executed action: {action_vector}")
            logger.info(f"Action confidence: {confidence[0].mean():.3f}")
            

            import time
            time.sleep(duration)
        

        final_status = drone.get_status()
        logger.info(f"Mission complete. Final position: {final_status.position}")
        

        drone.cleanup()
        
        logger.info("Drone simulation integration successful")
        return True
        
    except Exception as e:
        logger.error(f"Drone simulation integration failed: {e}")
        return False


def performance_benchmark():
    """Example 6: Performance benchmarking"""
    logger.info("=== Example 6: Performance Benchmark ===")
    
    try:
        from src.models.huggingface.vla_model import create_drone_vla_model
        import time
        

        model_sizes = ["large"]
        
        for model_size in model_sizes:
            logger.info(f"\nBenchmarking {model_size} model...")
            
            model = create_drone_vla_model(model_size=model_size)
            model.eval()
            

            video_frames = create_sample_video_frames(16)
            command = "Navigate to the target location"
            

            logger.info("Warming up...")
            for _ in range(3):
                with torch.no_grad():
                    model.predict_action(video_frames, command)
            

            num_iterations = 10
            logger.info(f"Running {num_iterations} iterations...")
            
            start_time = time.time()
            
            for i in range(num_iterations):
                with torch.no_grad():
                    actions, confidence = model.predict_action(video_frames, command)
            
            end_time = time.time()
            

            total_time = end_time - start_time
            avg_time = total_time / num_iterations
            fps = 1.0 / avg_time
            
            logger.info(f"Results for {model_size} model:")
            logger.info(f"  Total time: {total_time:.3f}s")
            logger.info(f"  Average time per inference: {avg_time:.3f}s")
            logger.info(f"  Inference FPS: {fps:.2f}")
            

            if torch.cuda.is_available():
                memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                logger.info(f"  Peak GPU memory: {memory_mb:.1f} MB")
                torch.cuda.reset_peak_memory_stats()
        
        logger.info("Performance benchmark successful")
        return True
        
    except Exception as e:
        logger.error(f"Performance benchmark failed: {e}")
        return False


def autonomous_system_example():
    """Example 7: Complete autonomous system"""
    logger.info("=== Example 7: Autonomous System ===")
    
    try:
        from src.core.autonomous_system import create_autonomous_system, MissionStep
        

        system = create_autonomous_system(
            model_size="large",
            safety_level="normal",
            simulation_mode=True
        )
        

        logger.info("Initializing autonomous system...")
        if not system.initialize():
            logger.error("Failed to initialize system")
            return False
            

        status = system.get_status()
        logger.info(f"System state: {status.state}")
        logger.info(f"Components ready: drone={status.drone_connected}, camera={status.camera_active}, model={status.model_loaded}")
        

        logger.info("Testing single command execution...")
        success = system.execute_command("Take off to 3 meters", timeout=10.0)
        logger.info(f"Command execution success: {success}")
        

        logger.info("Testing autonomous mission...")
        mission = [
            MissionStep("Take off to 5 meters", timeout=15.0),
            MissionStep("Look around for objects", timeout=10.0),
            MissionStep("Move forward slowly", timeout=10.0),
            MissionStep("Return to start position", timeout=15.0),
            MissionStep("Land safely", timeout=15.0)
        ]
        
        system.start_autonomous_mode()
        mission_success = system.execute_mission(mission)
        system.stop_autonomous_mode()
        
        logger.info(f"Mission completed successfully: {mission_success}")
        

        system.cleanup()
        
        logger.info("Autonomous system example successful")
        return True
        
    except Exception as e:
        logger.error(f"Autonomous system example failed: {e}")
        return False


def main():
    """Run all HuggingFace examples"""
    logger.info("HuggingFace VJEPA2 Drone VLA Examples")
    logger.info("=" * 60)
    logger.info("This demonstrates the PRIMARY/PRODUCTION approach using HuggingFace models")
    logger.info("=" * 60)
    
    examples = [
        ("Basic Model Loading", basic_model_loading_example),
        ("Simple Inference", simple_inference_example),
        ("Multi-Command Scenarios", multi_command_example),
        ("Webcam Integration", webcam_integration_example),
        ("Drone Simulation", drone_simulation_integration),
        ("Performance Benchmark", performance_benchmark),
        ("Autonomous System", autonomous_system_example)
    ]
    
    results = []
    
    for name, example_func in examples:
        logger.info(f"\nRunning {name}...")
        try:
            success = example_func()
            results.append((name, success))
            if success:
                logger.info(f"{name} completed successfully")
            else:
                logger.error(f"{name} failed")
        except Exception as e:
            logger.error(f"{name} failed with exception: {e}")
            results.append((name, False))
    

    logger.info("\n" + "=" * 60)
    logger.info("HuggingFace Example Results:")
    
    successful = 0
    for name, success in results:
        status = "PASS" if success else "FAIL"
        logger.info(f"  {name}: {status}")
        if success:
            successful += 1
            
    logger.info(f"\n{successful}/{len(results)} examples completed successfully")
    
    if successful == len(results):
        logger.info("All HuggingFace examples completed! Production system ready.")
    else:
        logger.warning("Some examples failed. Check logs and environment setup.")
    
    logger.info("\nNext steps:")
    logger.info("  - Fine-tune models with your drone data")
    logger.info("  - Integrate with real drone hardware")
    logger.info("  - Deploy in production environment")


if __name__ == "__main__":
    main()
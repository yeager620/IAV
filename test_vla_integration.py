#!/usr/bin/env python3
"""
Test script for VLA model integration with drone system
"""

import sys
import os
import numpy as np
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vla_model_creation():
    """Test VLA model creation and basic functionality"""
    print("=" * 60)
    print("Testing VLA Model Creation")
    print("=" * 60)
    
    try:
        from models.huggingface.vla_model import create_drone_vla_model
        
        # Test model creation
        print("Creating DroneVLAModel...")
        model = create_drone_vla_model(model_size="large", freeze_backbone=True)
        
        print(f"Model created successfully!")
        print(f"Model info: {model.get_model_info()}")
        
        return model
        
    except Exception as e:
        print(f"VLA model creation failed: {e}")
        return None

def test_action_space_conversion():
    """Test action space conversion utilities"""
    print("\n" + "=" * 60)
    print("Testing Action Space Conversion")
    print("=" * 60)
    
    try:
        from models.utils.action_space import (
            DroneActionSpace, 
            convert_manipulation_to_drone,
            extract_velocity_from_command
        )
        
        # Test action space
        action_space = DroneActionSpace()
        print(f"Action space bounds: {action_space.action_bounds}")
        
        # Test manipulation conversion
        manipulation_action = np.array([1.0, 0.5, -0.2, 0.1, 0.0, 0.3, 0.0])
        drone_action = convert_manipulation_to_drone(manipulation_action)
        print(f"Converted action: {drone_action}")
        
        # Test command extraction
        test_commands = [
            "takeoff to 5 meters",
            "move forward slowly",
            "rotate left quickly",
            "land now"
        ]
        
        for command in test_commands:
            base_action, scale = extract_velocity_from_command(command)
            print(f"Command: '{command}' -> Action: {base_action}, Scale: {scale}")
        
        print("Action space conversion tests passed!")
        return True
        
    except Exception as e:
        print(f"Action space conversion test failed: {e}")
        return False

def test_preprocessing():
    """Test preprocessing utilities"""
    print("\n" + "=" * 60)
    print("Testing Preprocessing")
    print("=" * 60)
    
    try:
        from models.utils.preprocessing import (
            preprocess_video_frames,
            preprocess_text_command
        )
        
        # Test video preprocessing
        dummy_frames = [
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            for _ in range(8)
        ]
        
        print(f"Processing {len(dummy_frames)} video frames...")
        video_tensor = preprocess_video_frames(dummy_frames)
        print(f"Video tensor shape: {video_tensor.shape}")
        
        # Test text preprocessing
        test_commands = [
            "takeoff to 5 meters",
            "navigate to the landing pad",
            "avoid the obstacle ahead"
        ]
        
        for command in test_commands:
            intent, params, tokens = preprocess_text_command(command)
            print(f"Command: '{command}'")
            print(f"  Intent: {intent}")
            print(f"  Parameters: {params}")
            print(f"  Tokens shape: {tokens['input_ids'].shape}")
        
        print("Preprocessing tests passed!")
        return True
        
    except Exception as e:
        print(f"Preprocessing test failed: {e}")
        return False

def test_mojo_bridge():
    """Test Mojo bridge functionality"""
    print("\n" + "=" * 60)
    print("Testing Mojo Bridge")
    print("=" * 60)
    
    try:
        from python.mojo_bridge import (
            get_mojo_bridge,
            mojo_drone_control,
            mojo_camera_process,
            is_mojo_available
        )
        
        # Test bridge availability
        mojo_available = is_mojo_available()
        print(f"Mojo bridge available: {mojo_available}")
        
        # Test drone control
        print("Testing drone control...")
        motor_commands = mojo_drone_control(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0)
        print(f"Motor commands: {motor_commands}")
        
        # Test camera processing
        print("Testing camera processing...")
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        processed_frame = mojo_camera_process(dummy_frame)
        print(f"Processed frame shape: {processed_frame.shape}")
        
        # Test performance metrics
        bridge = get_mojo_bridge()
        metrics = bridge.get_performance_metrics()
        print(f"Performance metrics: {metrics}")
        
        print("Mojo bridge tests passed!")
        return True
        
    except Exception as e:
        print(f"Mojo bridge test failed: {e}")
        return False

def test_vla_prediction():
    """Test VLA model prediction"""
    print("\n" + "=" * 60)
    print("Testing VLA Model Prediction")
    print("=" * 60)
    
    try:
        # Create model
        model = test_vla_model_creation()
        if model is None:
            return False
        
        # Test prediction
        dummy_frames = [
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            for _ in range(8)
        ]
        
        test_commands = [
            "takeoff to 5 meters",
            "move forward slowly",
            "rotate left and descend",
            "land safely"
        ]
        
        for command in test_commands:
            print(f"\nTesting command: '{command}'")
            actions, confidence = model.predict_action(dummy_frames, command)
            print(f"  Actions: {actions}")
            print(f"  Confidence: {confidence}")
            
            # Validate action bounds
            if np.any(np.abs(actions) > 10.0):
                print(f"  WARNING: Actions may be out of bounds")
            else:
                print(f"  Actions within expected bounds")
        
        print("\nVLA model prediction tests passed!")
        return True
        
    except Exception as e:
        print(f"VLA model prediction test failed: {e}")
        return False

def test_integration():
    """Test integration with autonomous system"""
    print("\n" + "=" * 60)
    print("Testing Integration with Autonomous System")
    print("=" * 60)
    
    try:
        from core.autonomous_system import create_autonomous_system
        
        # Create autonomous system
        print("Creating autonomous system...")
        system = create_autonomous_system(
            model_size="large",
            safety_level="normal",
            simulation_mode=True
        )
        
        # Test initialization
        print("Initializing system...")
        success = system.initialize()
        print(f"System initialization: {'SUCCESS' if success else 'FAILED'}")
        
        if success:
            # Test status
            status = system.get_status()
            print(f"System status: {status}")
            
            # Test command execution
            test_commands = [
                "hover in place",
                "move forward 2 meters"
            ]
            
            for command in test_commands:
                print(f"\nTesting command: '{command}'")
                result = system.execute_command(command, timeout=5.0)
                print(f"Execution result: {'SUCCESS' if result else 'FAILED'}")
        
        # Cleanup
        system.cleanup()
        
        print("Integration tests completed!")
        return success
        
    except Exception as e:
        print(f"Integration test failed: {e}")
        return False

def main():
    """Main test runner"""
    print("VLA Model Integration Tests")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    test_results.append(("VLA Model Creation", test_vla_model_creation() is not None))
    test_results.append(("Action Space Conversion", test_action_space_conversion()))
    test_results.append(("Preprocessing", test_preprocessing()))
    test_results.append(("Mojo Bridge", test_mojo_bridge()))
    test_results.append(("VLA Prediction", test_vla_prediction()))
    test_results.append(("Integration", test_integration()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! VLA integration is working correctly.")
        return 0
    else:
        print("Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
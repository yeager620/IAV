"""
Streamlined tests for the optimized UAV system
"""

import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

from minimal_interface import SystemState, LanguageEncoder, VisionCapture
from mojo_wrapper import MojoUAVController

class TestMojoController:
    """Test Mojo UAV controller"""
    
    @pytest.fixture
    def controller(self):
        return MojoUAVController()
    
    def test_initialization(self, controller):
        """Test controller initialization"""
        assert controller.mojo_available
    
    def test_control_processing(self, controller):
        """Test control processing"""
        # Test takeoff
        motors = controller.process_control(0.0, 0.0, 2.0, 0.0, 1.0)
        assert motors.shape == (4,)
        assert np.all(motors >= 0.0)
        assert np.all(motors <= 1.0)
        assert np.all(motors > 0.5)  # Should be higher for takeoff
        
        # Test hover
        hover_motors = controller.process_control(0.0, 0.0, 0.0, 0.0, 5.0)
        assert np.allclose(hover_motors, 0.5)  # Should be balanced
        
        # Test forward
        forward_motors = controller.process_control(1.0, 0.0, 0.0, 0.0, 5.0)
        assert forward_motors[0] > forward_motors[2]  # Front higher than rear
        assert forward_motors[1] > forward_motors[3]
    
    def test_emergency_stop(self, controller):
        """Test emergency stop"""
        emergency = controller.emergency_stop()
        assert np.allclose(emergency, 0.0)
    
    def test_safety_limits(self, controller):
        """Test safety limit enforcement"""
        # Test excessive velocity
        motors = controller.process_control(100.0, 0.0, 0.0, 0.0, 5.0)
        assert np.all(motors >= 0.0)
        assert np.all(motors <= 1.0)
        
        # Test altitude constraints
        low_alt_motors = controller.process_control(0.0, 0.0, -5.0, 0.0, 0.1)  # Below min altitude
        hover_motors = controller.process_control(0.0, 0.0, 0.0, 0.0, 0.1)
        assert np.allclose(low_alt_motors, hover_motors)  # Should prevent downward motion

class TestLanguageEncoder:
    """Test simplified language encoder"""
    
    @pytest.fixture
    def encoder(self):
        return LanguageEncoder()
    
    def test_basic_commands(self, encoder):
        """Test basic command encoding"""
        takeoff = encoder.encode("takeoff")
        assert len(takeoff) == 4
        assert takeoff[2] > 0  # Should have positive z component
        
        land = encoder.encode("land")
        assert land[2] < 0  # Should have negative z component
        
        hover = encoder.encode("hover")
        assert all(v == 0.0 for v in hover)  # Should be all zeros
    
    def test_directional_commands(self, encoder):
        """Test directional commands"""
        forward = encoder.encode("forward")
        assert forward[0] > 0  # Positive x velocity
        
        left = encoder.encode("left")
        assert left[1] < 0  # Negative y velocity
        
        rotate_right = encoder.encode("rotate_right")
        assert rotate_right[3] > 0  # Positive yaw rate
    
    def test_unknown_command(self, encoder):
        """Test unknown command defaults to hover"""
        unknown = encoder.encode("some unknown command")
        assert all(v == 0.0 for v in unknown)

class TestSystemState:
    """Test system state structure"""
    
    def test_initialization(self):
        """Test state initialization"""
        state = SystemState(
            position=np.array([1.0, 2.0, 3.0]),
            velocity=np.array([0.1, 0.2, 0.3]),
            attitude=np.array([0.0, 0.0, 1.57]),
            armed=True,
            timestamp=1234567890.0
        )
        
        assert state.position[2] == 3.0
        assert state.armed == True
        assert state.timestamp == 1234567890.0

class TestVisionCapture:
    """Test vision capture (without actual camera)"""
    
    def test_initialization_invalid_camera(self):
        """Test vision system initialization with invalid camera"""
        vision = VisionCapture(-1)  # Invalid camera ID
        assert not vision.initialize()  # Should fail gracefully
    
    def test_frame_sequence_no_camera(self):
        """Test frame sequence with no camera"""
        vision = VisionCapture(-1)
        frames = vision.get_frame_sequence(4)
        assert frames is None

# Performance tests
class TestPerformance:
    """Test performance characteristics"""
    
    def test_mojo_controller_speed(self):
        """Test Mojo controller performance"""
        controller = MojoUAVController()
        
        import time
        start_time = time.perf_counter()
        
        # Run 1000 control cycles
        for _ in range(1000):
            controller.process_control(1.0, 0.5, 0.0, 0.1, 5.0)
        
        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / 1000 * 1000  # ms
        
        assert avg_time < 1.0, f"Mojo controller too slow: {avg_time:.3f}ms per cycle"
        print(f"Mojo controller performance: {avg_time:.3f}ms per cycle")
    
    def test_language_encoding_speed(self):
        """Test language encoding performance"""
        encoder = LanguageEncoder()
        
        import time
        start_time = time.perf_counter()
        
        commands = ['takeoff', 'forward', 'left', 'hover', 'land'] * 200
        for cmd in commands:
            encoder.encode(cmd)
        
        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / len(commands) * 1000  # ms
        
        assert avg_time < 0.01, f"Language encoding too slow: {avg_time:.4f}ms"

# Integration test
def test_system_integration():
    """Test basic system integration"""
    from minimal_interface import SystemOrchestrator
    
    config = {
        "mavlink": {"connection": "udp:127.0.0.1:14550"},
        "camera_id": -1,  # Invalid camera for testing
        "control_frequency": 10  # Low frequency for testing
    }
    
    system = SystemOrchestrator(config)
    
    # Test basic functionality without actual hardware
    assert system.mojo_controller is not None
    assert system.language is not None
    
    # Test command setting
    system.set_command("takeoff")
    assert system.current_command == "takeoff"
    
    # Test emergency stop
    system.emergency_stop()
    assert system.current_command == "hover"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
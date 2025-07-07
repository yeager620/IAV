"""
Tests for Mojo component integration with Python.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

from mojo_interface import MojoVLAInterface, MojoControlInterface, MojoSystemInterface

class TestMojoVLAInterface:
    """Test VLA inference interface"""
    
    @pytest.fixture
    def vla_interface(self):
        """Create VLA interface for testing"""
        return MojoVLAInterface("data/models/test_model.mojo")
    
    def test_initialization(self, vla_interface):
        """Test VLA interface initialization"""
        assert vla_interface.initialize()
        assert vla_interface.is_initialized
    
    def test_warmup(self, vla_interface):
        """Test VLA warmup process"""
        vla_interface.initialize()
        assert vla_interface.warm_up()
        assert vla_interface.warmup_complete
    
    def test_inference_shape(self, vla_interface):
        """Test inference output shape"""
        vla_interface.initialize()
        vla_interface.warm_up()
        
        # Create test inputs
        frames = np.random.rand(16, 224, 224, 3).astype(np.float32)
        command = np.random.rand(512).astype(np.float32)
        
        result = vla_interface.predict(frames, command)
        
        assert result.actions.shape == (6,)
        assert isinstance(result.confidence, float)
        assert result.confidence >= 0.0 and result.confidence <= 1.0
        assert result.latency_ms > 0
    
    def test_inference_range(self, vla_interface):
        """Test inference output is in reasonable range"""
        vla_interface.initialize()
        vla_interface.warm_up()
        
        frames = np.random.rand(16, 224, 224, 3).astype(np.float32)
        command = np.random.rand(512).astype(np.float32)
        
        result = vla_interface.predict(frames, command)
        
        # Actions should be in reasonable range for velocities
        assert np.all(np.abs(result.actions) <= 5.0)  # Max 5 m/s
        assert np.all(np.isfinite(result.actions))
    
    def test_performance_tracking(self, vla_interface):
        """Test performance statistics tracking"""
        vla_interface.initialize()
        vla_interface.warm_up()
        
        frames = np.random.rand(16, 224, 224, 3).astype(np.float32)
        command = np.random.rand(512).astype(np.float32)
        
        # Run multiple inferences
        for _ in range(5):
            vla_interface.predict(frames, command)
        
        stats = vla_interface.get_performance_stats()
        
        assert stats['total_inferences'] == 5
        assert 'avg_latency_ms' in stats
        assert 'throughput_hz' in stats

class TestMojoControlInterface:
    """Test control allocation interface"""
    
    @pytest.fixture
    def control_interface(self):
        """Create control interface for testing"""
        interface = MojoControlInterface()
        interface.initialize()
        return interface
    
    def test_control_allocation(self, control_interface):
        """Test control allocation produces valid motor commands"""
        actions = np.array([1.0, 0.5, -0.3, 0.0, 0.1, 0.2])  # 6DOF actions
        altitude = 5.0
        velocity = np.array([0.0, 0.0, 0.0])
        
        result = control_interface.validate_and_allocate(actions, altitude, velocity)
        
        # Check motor commands
        assert result.motor_commands.shape == (4,)
        assert np.all(result.motor_commands >= 0.0)
        assert np.all(result.motor_commands <= 1.0)
        
        # Check control input
        assert result.control_input.shape == (4,)
        
        # Check safety violations list
        assert isinstance(result.safety_violations, list)
    
    def test_safety_limits(self, control_interface):
        """Test safety limit enforcement"""
        # Test excessive velocity
        excessive_actions = np.array([10.0, 10.0, 10.0, 5.0, 5.0, 5.0])
        altitude = 5.0
        velocity = np.array([0.0, 0.0, 0.0])
        
        result = control_interface.validate_and_allocate(excessive_actions, altitude, velocity)
        
        # Should have safety violations
        assert len(result.safety_violations) > 0
        assert any('limit exceeded' in violation for violation in result.safety_violations)
    
    def test_altitude_constraints(self, control_interface):
        """Test altitude-based constraints"""
        # Test low altitude with downward velocity
        actions = np.array([0.0, 0.0, -2.0, 0.0, 0.0, 0.0])  # Downward motion
        low_altitude = 0.3  # Below minimum
        velocity = np.array([0.0, 0.0, 0.0])
        
        result = control_interface.validate_and_allocate(actions, low_altitude, velocity)
        
        # Should prevent downward motion
        assert len(result.safety_violations) > 0
        assert any('altitude' in violation for violation in result.safety_violations)

class TestMojoSystemInterface:
    """Test complete system interface"""
    
    @pytest.fixture
    def system_interface(self):
        """Create system interface for testing"""
        return MojoSystemInterface("data/models/test_model.mojo")
    
    def test_system_initialization(self, system_interface):
        """Test complete system initialization"""
        assert system_interface.initialize()
        assert system_interface.is_initialized
    
    def test_end_to_end_processing(self, system_interface):
        """Test complete processing pipeline"""
        system_interface.initialize()
        
        # Create test inputs
        frames = np.random.rand(16, 224, 224, 3).astype(np.float32)
        command_embedding = np.random.rand(512).astype(np.float32)
        current_state = {
            'altitude': 5.0,
            'velocity': np.array([0.0, 0.0, 0.0]),
            'attitude': np.array([0.0, 0.0, 0.0]),
            'angular_velocity': np.array([0.0, 0.0, 0.0])
        }
        
        motor_commands, metadata = system_interface.process_frame_and_command(
            frames, command_embedding, current_state
        )
        
        # Check outputs
        assert motor_commands.shape == (4,)
        assert np.all(motor_commands >= 0.0)
        assert np.all(motor_commands <= 1.0)
        
        # Check metadata
        assert 'inference_latency_ms' in metadata
        assert 'confidence' in metadata
        assert 'raw_actions' in metadata
        assert 'safe_actions' in metadata
        assert 'safety_violations' in metadata
        assert 'timestamp' in metadata
    
    def test_system_stats(self, system_interface):
        """Test system statistics"""
        system_interface.initialize()
        
        stats = system_interface.get_system_stats()
        
        assert 'vla_performance' in stats
        assert 'system_initialized' in stats
        assert 'warmup_complete' in stats

class TestIntegrationPerformance:
    """Test performance characteristics"""
    
    def test_inference_latency(self):
        """Test that inference meets real-time requirements"""
        interface = MojoVLAInterface("data/models/test_model.mojo")
        interface.initialize()
        interface.warm_up()
        
        frames = np.random.rand(16, 224, 224, 3).astype(np.float32)
        command = np.random.rand(512).astype(np.float32)
        
        # Run multiple inferences
        latencies = []
        for _ in range(10):
            result = interface.predict(frames, command)
            latencies.append(result.latency_ms)
        
        avg_latency = np.mean(latencies)
        
        # Should be under 33ms for 30Hz operation
        assert avg_latency < 50.0, f"Average latency {avg_latency:.1f}ms too high"
    
    def test_control_loop_frequency(self):
        """Test that control allocation meets frequency requirements"""
        control_interface = MojoControlInterface()
        control_interface.initialize()
        
        import time
        
        actions = np.array([1.0, 0.5, -0.3, 0.0, 0.1, 0.2])
        altitude = 5.0
        velocity = np.array([0.0, 0.0, 0.0])
        
        # Time multiple control cycles
        start_time = time.perf_counter()
        num_cycles = 100
        
        for _ in range(num_cycles):
            control_interface.validate_and_allocate(actions, altitude, velocity)
        
        end_time = time.perf_counter()
        avg_cycle_time = (end_time - start_time) / num_cycles * 1000  # ms
        
        # Should be under 1ms for 100Hz operation
        assert avg_cycle_time < 2.0, f"Average cycle time {avg_cycle_time:.2f}ms too high"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
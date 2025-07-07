#!/usr/bin/env python3
"""
Integration tests for the entire system.
"""

import pytest
import asyncio
import numpy as np
import sys
import os
from unittest.mock import Mock, patch, AsyncMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

from minimal_interface import SystemOrchestrator

class TestSystemIntegration:
    """Test full system integration"""
    
    @pytest.fixture
    def config(self):
        return {
            "mavlink": {"connection": "udp:127.0.0.1:14550"},
            "camera_id": 0,
            "control_frequency": 50  # Lower frequency for testing
        }
    
    @pytest.fixture
    def system(self, config):
        return SystemOrchestrator(config)
    
    def test_system_initialization(self, system, config):
        """Test complete system initialization"""
        assert system.config == config
        assert isinstance(system.mavlink, type(system.mavlink))
        assert isinstance(system.vision, type(system.vision))
        assert isinstance(system.language, type(system.language))
        assert system.current_command == "hover"
        assert system.running is False
    
    def test_command_flow(self, system):
        """Test complete command processing flow"""
        # Test command encoding
        vx, vy, vz, wz = system.language.encode("takeoff")
        assert vx == 0.0 and vy == 0.0 and vz == 2.0 and wz == 0.0
        
        # Test Mojo control processing
        motors = system._call_mojo_control(vx, vy, vz, wz, 1.0)
        assert motors.shape == (4,)
        assert np.all(motors >= 0.0)
        assert np.all(motors <= 1.0)
        
        # Test command setting
        system.set_command("takeoff")
        assert system.current_command == "takeoff"
    
    def test_safety_integration(self, system):
        """Test integrated safety systems"""
        # Test velocity limits
        motors_high_vel = system._call_mojo_control(100.0, 0.0, 0.0, 0.0, 5.0)
        motors_normal_vel = system._call_mojo_control(1.0, 0.0, 0.0, 0.0, 5.0)
        
        # High velocity should be clamped, but both should be valid
        assert np.all(motors_high_vel >= 0.0)
        assert np.all(motors_high_vel <= 1.0)
        assert np.all(motors_normal_vel >= 0.0)
        assert np.all(motors_normal_vel <= 1.0)
        
        # Test altitude limits
        motors_low_alt = system._call_mojo_control(0.0, 0.0, -2.0, 0.0, 0.3)
        motors_high_alt = system._call_mojo_control(0.0, 0.0, 2.0, 0.0, 150.0)
        
        assert np.all(motors_low_alt >= 0.0)
        assert np.all(motors_high_alt >= 0.0)
    
    def test_emergency_procedures(self, system):
        """Test emergency stop procedures"""
        # Set system to active state
        system.current_command = "forward"
        
        # Trigger emergency stop
        system.emergency_stop()
        
        # Should revert to safe state
        assert system.current_command == "hover"
    
    @pytest.mark.asyncio
    @patch('minimal_interface.mavutil.mavlink_connection')
    async def test_control_loop_setup(self, mock_mavlink, system):
        """Test control loop initialization"""
        mock_conn = Mock()
        mock_mavlink.return_value = mock_conn
        
        with patch.object(system.vision, 'initialize', return_value=True):
            result = await system.initialize()
            assert result is True
    
    def test_motor_command_validation(self, system):
        """Test motor command validation and bounds"""
        test_cases = [
            # vx, vy, vz, wz, altitude
            (0.0, 0.0, 0.0, 0.0, 5.0),    # Hover
            (1.0, 0.0, 0.0, 0.0, 5.0),    # Forward
            (0.0, 1.0, 0.0, 0.0, 5.0),    # Right
            (0.0, 0.0, 1.0, 0.0, 5.0),    # Up
            (0.0, 0.0, 0.0, 1.0, 5.0),    # Rotate right
            (-1.0, -1.0, -1.0, -1.0, 5.0), # All negative
        ]
        
        for vx, vy, vz, wz, altitude in test_cases:
            motors = system._call_mojo_control(vx, vy, vz, wz, altitude)
            
            # All motor commands should be valid
            assert motors.shape == (4,)
            assert np.all(motors >= 0.0), f"Negative motor command for input ({vx}, {vy}, {vz}, {wz}, {altitude})"
            assert np.all(motors <= 1.0), f"Motor command > 1.0 for input ({vx}, {vy}, {vz}, {wz}, {altitude})"
    
    def test_language_command_coverage(self, system):
        """Test that all language commands produce valid motor outputs"""
        commands = [
            "takeoff", "land", "hover", "forward", "backward",
            "left", "right", "up", "down", "rotate_left", "rotate_right"
        ]
        
        for command in commands:
            # Encode command
            vx, vy, vz, wz = system.language.encode(command)
            
            # Process through control system
            motors = system._call_mojo_control(vx, vy, vz, wz, 5.0)
            
            # Validate output
            assert motors.shape == (4,)
            assert np.all(motors >= 0.0), f"Invalid motor output for command '{command}'"
            assert np.all(motors <= 1.0), f"Invalid motor output for command '{command}'"
    
    def test_system_state_consistency(self, system):
        """Test system state consistency across operations"""
        initial_command = system.current_command
        initial_running = system.running
        
        # Perform various operations
        system.set_command("takeoff")
        motors1 = system._call_mojo_control(0.0, 0.0, 2.0, 0.0, 1.0)
        
        system.set_command("hover")
        motors2 = system._call_mojo_control(0.0, 0.0, 0.0, 0.0, 5.0)
        
        system.emergency_stop()
        
        # Verify state consistency
        assert system.current_command == "hover"  # After emergency stop
        assert system.running == initial_running  # Should not change
        assert np.all(motors1 >= 0.0) and np.all(motors1 <= 1.0)
        assert np.all(motors2 >= 0.0) and np.all(motors2 <= 1.0)

@pytest.mark.asyncio
class TestAsyncIntegration:
    """Test async integration functionality"""
    
    @pytest.fixture
    def config(self):
        return {
            "mavlink": {"connection": "udp:127.0.0.1:14550"},
            "camera_id": 0,
            "control_frequency": 10  # Very low for testing
        }
    
    @pytest.fixture
    def system(self, config):
        return SystemOrchestrator(config)
    
    @patch('minimal_interface.mavutil.mavlink_connection')
    async def test_full_initialization_sequence(self, mock_mavlink, system):
        """Test complete async initialization sequence"""
        mock_conn = Mock()
        mock_conn.wait_heartbeat = Mock()
        mock_mavlink.return_value = mock_conn
        
        with patch.object(system.vision, 'initialize', return_value=True):
            # Test initialization
            result = await system.initialize()
            assert result is True
            
            # Verify connections were established
            mock_mavlink.assert_called_once()
            mock_conn.wait_heartbeat.assert_called_once_with(timeout=5)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
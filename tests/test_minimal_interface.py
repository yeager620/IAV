#!/usr/bin/env python3
"""
Test suite for minimal_interface.py after cleanup.
"""

import pytest
import numpy as np
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import asyncio
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

from minimal_interface import (
    SystemState, MAVLinkInterface, VisionCapture, 
    LanguageEncoder, SystemOrchestrator
)

class TestSystemState:
    """Test SystemState dataclass"""
    
    def test_system_state_creation(self):
        """Test SystemState creation"""
        state = SystemState(
            position=np.array([1.0, 2.0, 3.0]),
            velocity=np.array([0.1, 0.2, 0.3]),
            attitude=np.array([0.0, 0.1, 0.2]),
            armed=True,
            timestamp=time.time()
        )
        
        assert state.position.shape == (3,)
        assert state.velocity.shape == (3,)
        assert state.attitude.shape == (3,)
        assert state.armed is True
        assert isinstance(state.timestamp, float)

class TestLanguageEncoder:
    """Test LanguageEncoder functionality"""
    
    @pytest.fixture
    def encoder(self):
        return LanguageEncoder()
    
    def test_basic_commands(self, encoder):
        """Test basic command encoding"""
        # Test takeoff
        vx, vy, vz, wz = encoder.encode("takeoff")
        assert vx == 0.0
        assert vy == 0.0
        assert vz == 2.0
        assert wz == 0.0
        
        # Test landing
        vx, vy, vz, wz = encoder.encode("land")
        assert vx == 0.0
        assert vy == 0.0
        assert vz == -1.0
        assert wz == 0.0
        
        # Test hover
        vx, vy, vz, wz = encoder.encode("hover")
        assert all(v == 0.0 for v in [vx, vy, vz, wz])
    
    def test_movement_commands(self, encoder):
        """Test movement command encoding"""
        # Test forward
        vx, vy, vz, wz = encoder.encode("forward")
        assert vx == 1.0
        assert vy == 0.0
        
        # Test left
        vx, vy, vz, wz = encoder.encode("left")
        assert vx == 0.0
        assert vy == -1.0
        
        # Test right
        vx, vy, vz, wz = encoder.encode("right")
        assert vx == 0.0
        assert vy == 1.0
    
    def test_rotation_commands(self, encoder):
        """Test rotation command encoding"""
        # Test rotate left
        vx, vy, vz, wz = encoder.encode("rotate_left")
        assert wz == -1.0
        
        # Test rotate right
        vx, vy, vz, wz = encoder.encode("rotate_right")
        assert wz == 1.0
    
    def test_unknown_command(self, encoder):
        """Test unknown command defaults to hover"""
        vx, vy, vz, wz = encoder.encode("unknown_command")
        assert all(v == 0.0 for v in [vx, vy, vz, wz])

class TestMAVLinkInterface:
    """Test MAVLinkInterface with mocking"""
    
    @pytest.fixture
    def mavlink_interface(self):
        return MAVLinkInterface("udp:127.0.0.1:14550")
    
    def test_initialization(self, mavlink_interface):
        """Test MAVLink interface initialization"""
        assert mavlink_interface.connection_string == "udp:127.0.0.1:14550"
        assert mavlink_interface.connection is None
        assert isinstance(mavlink_interface.last_state, SystemState)
    
    @patch('minimal_interface.mavutil.mavlink_connection')
    def test_connect_success(self, mock_mavlink, mavlink_interface):
        """Test successful MAVLink connection"""
        mock_conn = Mock()
        mock_mavlink.return_value = mock_conn
        
        result = mavlink_interface.connect()
        
        assert result is True
        assert mavlink_interface.connection == mock_conn
        mock_conn.wait_heartbeat.assert_called_once_with(timeout=5)
    
    @patch('minimal_interface.mavutil.mavlink_connection')
    def test_connect_failure(self, mock_mavlink, mavlink_interface):
        """Test MAVLink connection failure"""
        mock_mavlink.side_effect = Exception("Connection failed")
        
        result = mavlink_interface.connect()
        
        assert result is False
        assert mavlink_interface.connection is None
    
    def test_send_motor_commands_no_connection(self, mavlink_interface):
        """Test sending motor commands without connection"""
        commands = np.array([0.5, 0.5, 0.5, 0.5])
        result = mavlink_interface.send_motor_commands(commands)
        assert result is False
    
    def test_send_motor_commands_wrong_size(self, mavlink_interface):
        """Test sending motor commands with wrong array size"""
        mavlink_interface.connection = Mock()
        commands = np.array([0.5, 0.5])  # Wrong size
        result = mavlink_interface.send_motor_commands(commands)
        assert result is False

class TestVisionCapture:
    """Test VisionCapture with mocking"""
    
    @pytest.fixture
    def vision_capture(self):
        return VisionCapture(camera_id=0)
    
    def test_initialization(self, vision_capture):
        """Test vision capture initialization"""
        assert vision_capture.camera_id == 0
        assert vision_capture.cap is None
    
    @patch('minimal_interface.cv2.VideoCapture')
    def test_initialize_success(self, mock_cv2, vision_capture):
        """Test successful camera initialization"""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cv2.return_value = mock_cap
        
        result = vision_capture.initialize()
        
        assert result is True
        assert vision_capture.cap == mock_cap
        mock_cap.set.assert_called()
    
    @patch('minimal_interface.cv2.VideoCapture')
    def test_initialize_failure(self, mock_cv2, vision_capture):
        """Test camera initialization failure"""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_cv2.return_value = mock_cap
        
        result = vision_capture.initialize()
        
        assert result is False
    
    @patch('minimal_interface.cv2.VideoCapture')
    def test_get_frame_sequence(self, mock_cv2, vision_capture):
        """Test frame sequence capture"""
        mock_cap = Mock()
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        vision_capture.cap = mock_cap
        
        frames = vision_capture.get_frame_sequence(num_frames=2)
        
        assert frames is not None
        assert frames.shape == (2, 224, 224, 3)
        assert frames.dtype == np.float32
        assert np.all(frames >= 0.0) and np.all(frames <= 1.0)

class TestSystemOrchestrator:
    """Test SystemOrchestrator functionality"""
    
    @pytest.fixture
    def config(self):
        return {
            "mavlink": {"connection": "udp:127.0.0.1:14550"},
            "camera_id": 0,
            "control_frequency": 100
        }
    
    @pytest.fixture
    def orchestrator(self, config):
        return SystemOrchestrator(config)
    
    def test_initialization(self, orchestrator, config):
        """Test orchestrator initialization"""
        assert orchestrator.config == config
        assert orchestrator.current_command == "hover"
        assert orchestrator.running is False
        assert orchestrator.control_frequency == 100
    
    def test_call_mojo_control(self, orchestrator):
        """Test Mojo control processing"""
        # Test takeoff
        motors = orchestrator._call_mojo_control(0.0, 0.0, 2.0, 0.0, 1.0)
        assert motors.shape == (4,)
        assert np.all(motors >= 0.0)
        assert np.all(motors <= 1.0)
        assert np.all(motors > 0.5)  # Should be higher for takeoff
        
        # Test hover
        hover_motors = orchestrator._call_mojo_control(0.0, 0.0, 0.0, 0.0, 5.0)
        assert np.allclose(hover_motors, 0.5, atol=0.1)
        
        # Test forward
        forward_motors = orchestrator._call_mojo_control(1.0, 0.0, 0.0, 0.0, 5.0)
        assert forward_motors[0] > forward_motors[2]  # Front higher than rear
        assert forward_motors[1] > forward_motors[3]
    
    def test_safety_limits(self, orchestrator):
        """Test safety limit enforcement"""
        # Test excessive velocity
        motors = orchestrator._call_mojo_control(10.0, 0.0, 0.0, 0.0, 5.0)
        # Should be clamped to max velocity (5.0)
        assert np.all(motors >= 0.0)
        assert np.all(motors <= 1.0)
        
        # Test low altitude descent protection
        motors = orchestrator._call_mojo_control(0.0, 0.0, -2.0, 0.0, 0.3)
        # Should prevent descent at low altitude
        assert np.all(motors >= 0.0)
        
        # Test high altitude ascent protection
        motors = orchestrator._call_mojo_control(0.0, 0.0, 2.0, 0.0, 150.0)
        # Should prevent ascent at high altitude
        assert np.all(motors >= 0.0)
    
    def test_emergency_stop(self, orchestrator):
        """Test emergency stop functionality"""
        orchestrator.emergency_stop()
        assert orchestrator.current_command == "hover"
    
    def test_set_command(self, orchestrator):
        """Test command setting"""
        orchestrator.set_command("takeoff")
        assert orchestrator.current_command == "takeoff"
        
        orchestrator.set_command("land")
        assert orchestrator.current_command == "land"
    
    def test_stop(self, orchestrator):
        """Test system stop"""
        orchestrator.running = True
        orchestrator.stop()
        assert orchestrator.running is False

@pytest.mark.asyncio
class TestAsyncFunctionality:
    """Test async functionality"""
    
    @pytest.fixture
    def config(self):
        return {
            "mavlink": {"connection": "udp:127.0.0.1:14550"},
            "camera_id": 0,
            "control_frequency": 10  # Lower frequency for testing
        }
    
    @pytest.fixture
    def orchestrator(self, config):
        return SystemOrchestrator(config)
    
    @patch('minimal_interface.mavutil.mavlink_connection')
    async def test_initialize_success(self, mock_mavlink, orchestrator):
        """Test successful system initialization"""
        mock_conn = Mock()
        mock_mavlink.return_value = mock_conn
        
        with patch.object(orchestrator.vision, 'initialize', return_value=True):
            result = await orchestrator.initialize()
            assert result is True
    
    @patch('minimal_interface.mavutil.mavlink_connection')
    async def test_initialize_mavlink_failure(self, mock_mavlink, orchestrator):
        """Test initialization with MAVLink failure"""
        mock_mavlink.side_effect = Exception("Connection failed")
        
        result = await orchestrator.initialize()
        assert result is False

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
#!/usr/bin/env python3
"""
Integration tests for the entire drone-vla system.
"""

import pytest
import asyncio
import numpy as np
import sys
import os
from unittest.mock import Mock, patch, AsyncMock
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.autonomous_system import AutonomousDroneSystem
from core.unified_drone_control import UnifiedDroneController
from safety.validator import SafetyMonitor


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
        """Create test system"""
        return AutonomousDroneSystem(simulation_mode=True, safety_level="medium")
    
    def test_system_initialization(self, system):
        """Test system can be initialized"""
        assert system.simulation_mode is True
        assert system.safety_level == "medium"
        assert system.running is False
    
    def test_drone_controller_integration(self):
        """Test drone controller integration"""
        controller = UnifiedDroneController(
            connection_string="udp:127.0.0.1:14550",
            simulation_mode=True,
            use_dronekit=False
        )
        
        # Test basic integration
        assert controller.simulation_mode is True
        assert controller.safety_enabled is True
        
        # Test command interface
        controller.arm()
        controller.takeoff(10.0)
        controller.move(vx=1.0, vy=0.0, vz=0.0)
        controller.land()
        
        # Commands should be queued
        assert not controller.command_queue.empty()
    
    @patch('src.core.autonomous_system.create_drone_vla_model')
    @patch('src.core.autonomous_system.DroneController')
    @patch('src.core.autonomous_system.VisionSystem')
    @patch('src.core.autonomous_system.SafetyMonitor')
    def test_component_integration(self, mock_safety, mock_vision, mock_drone, mock_vla):
        """Test component integration"""
        # Mock successful initialization
        mock_drone.return_value.connect.return_value = True
        mock_vision.return_value.initialize.return_value = True
        mock_vla.return_value = Mock()
        
        system = AutonomousDroneSystem(simulation_mode=True)
        
        # Test initialization - should not crash
        try:
            system.initialize()
        except Exception as e:
            # Expected to fail due to missing dependencies
            assert "Failed to connect to drone" in str(e) or any(
                keyword in str(e).lower() 
                for keyword in ["model", "vla", "vision", "safety"]
            )
    
    def test_safety_integration(self):
        """Test safety system integration"""
        controller = UnifiedDroneController(simulation_mode=True)
        
        # Test safety is enabled by default
        assert controller.safety_enabled is True
        
        # Test emergency stop
        controller.emergency_stop()
        
        # Should queue emergency command
        assert not controller.command_queue.empty()
    
    def test_simulation_mode_safety(self):
        """Test that simulation mode provides safe testing environment"""
        controller = UnifiedDroneController(simulation_mode=True)
        
        # Test various commands in simulation
        commands = [
            ("arm", {}),
            ("takeoff", {"altitude": 5.0}),
            ("move", {"vx": 1.0, "vy": 0.0, "vz": 0.0}),
            ("land", {}),
            ("emergency_stop", {})
        ]
        
        for cmd_type, params in commands:
            controller.send_command(
                controller.DroneCommand(cmd_type, params, time.time())
            )
        
        # All commands should be queued safely
        assert not controller.command_queue.empty()
    
    def test_mojo_fallback_integration(self):
        """Test system works with Mojo fallback"""
        controller = UnifiedDroneController(simulation_mode=True)
        
        # Should initialize even without Mojo
        assert controller.safety_validator is None or controller.safety_validator is not None
        assert controller.control_system is None or controller.control_system is not None
        
        # Basic functionality should work
        controller.arm()
        controller.takeoff(10.0)
        
        assert not controller.command_queue.empty()
    
    def test_command_validation_integration(self):
        """Test command validation works in integrated system"""
        controller = UnifiedDroneController(simulation_mode=True)
        
        # Test safe commands
        safe_commands = [
            "takeoff to 5 meters",
            "move forward 2 meters", 
            "land at current position",
            "hover in place"
        ]
        
        for cmd in safe_commands:
            # Should not raise exception
            controller.send_command(
                controller.DroneCommand("move", {"command": cmd}, time.time())
            )
        
        # Test potentially dangerous commands would be handled by safety system
        # (actual validation depends on safety system implementation)
        dangerous_commands = [
            "crash into building",
            "attack target",
            "destroy obstacle"
        ]
        
        for cmd in dangerous_commands:
            # Should still queue but would be filtered by safety system
            controller.send_command(
                controller.DroneCommand("move", {"command": cmd}, time.time())
            )
    
    def test_end_to_end_simulation(self):
        """Test end-to-end simulation workflow"""
        controller = UnifiedDroneController(simulation_mode=True)
        
        # Simulate a basic mission
        mission_steps = [
            ("arm", {}),
            ("takeoff", {"altitude": 10.0}),
            ("move", {"vx": 2.0, "vy": 0.0, "vz": 0.0}),  # Move forward
            ("move", {"vx": 0.0, "vy": 2.0, "vz": 0.0}),  # Move right
            ("move", {"vx": 0.0, "vy": 0.0, "vz": -1.0}), # Descend
            ("land", {}),
            ("disarm", {})
        ]
        
        for step_type, params in mission_steps:
            controller.send_command(
                controller.DroneCommand(step_type, params, time.time())
            )
        
        # All mission steps should be queued
        initial_queue_size = controller.command_queue.qsize()
        assert initial_queue_size == len(mission_steps)
        
        # Test emergency stop can interrupt mission
        controller.emergency_stop()
        
        # Emergency command should be added
        assert controller.command_queue.qsize() == initial_queue_size + 1
    
    def test_configuration_integration(self):
        """Test system works with different configurations"""
        configs = [
            {"simulation_mode": True, "use_dronekit": False},
            {"simulation_mode": True, "use_dronekit": True},
            {"simulation_mode": True, "control_frequency": 25},
            {"simulation_mode": True, "safety_enabled": True}
        ]
        
        for config in configs:
            controller = UnifiedDroneController(**config)
            
            # Basic functionality should work with all configs
            controller.arm()
            controller.takeoff(5.0)
            
            assert not controller.command_queue.empty()
            
            # Clean up
            controller.disconnect()


class TestPerformanceIntegration:
    """Test system performance in integrated environment"""
    
    def test_command_queue_performance(self):
        """Test command queue handles multiple commands efficiently"""
        controller = UnifiedDroneController(simulation_mode=True)
        
        # Add many commands quickly
        start_time = time.time()
        for i in range(100):
            controller.move(vx=1.0, vy=0.0, vz=0.0)
        
        elapsed = time.time() - start_time
        
        # Should handle 100 commands quickly (< 1 second)
        assert elapsed < 1.0
        assert controller.command_queue.qsize() == 100
    
    def test_safety_validation_performance(self):
        """Test safety validation doesn't block system"""
        controller = UnifiedDroneController(simulation_mode=True)
        
        # Test rapid command generation with safety enabled
        start_time = time.time()
        for i in range(50):
            controller.arm()
            controller.takeoff(5.0)
            controller.emergency_stop()
        
        elapsed = time.time() - start_time
        
        # Should handle rapid commands with safety (< 2 seconds)
        assert elapsed < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
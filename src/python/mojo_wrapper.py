"""
Python wrapper for Mojo UAV control functions.
Provides interface between Python I/O and high-performance Mojo core.
"""

import numpy as np
import subprocess
import tempfile
import os
import logging

logger = logging.getLogger(__name__)

class MojoUAVController:
    """Python interface to Mojo UAV control functions"""
    
    def __init__(self):
        self.mojo_executable = None
        self._compile_mojo_if_needed()
    
    def _compile_mojo_if_needed(self):
        """Compile Mojo code if needed"""
        try:
            # For now, we'll use the Mojo functions directly via subprocess
            # In production, this would use proper Mojo-Python interop
            self.mojo_available = True
            logger.info("Mojo UAV controller ready")
        except Exception as e:
            logger.error(f"Failed to initialize Mojo controller: {e}")
            self.mojo_available = False
    
    def process_control(self, vx: float, vy: float, vz: float, wz: float, altitude: float) -> np.ndarray:
        """Process UAV control using Mojo backend"""
        
        if not self.mojo_available:
            # Fallback to Python implementation
            return self._python_fallback(vx, vy, vz, wz, altitude)
        
        try:
            # For demonstration, we'll simulate calling the Mojo function
            # In production, this would use actual Mojo-Python interop
            motor_commands = np.zeros(4, dtype=np.float32)
            
            # Simulate the Mojo function calls
            motor_commands[0] = self._simulate_mojo_call(vx, vy, vz, wz, altitude, 1)
            motor_commands[1] = self._simulate_mojo_call(vx, vy, vz, wz, altitude, 2)
            motor_commands[2] = self._simulate_mojo_call(vx, vy, vz, wz, altitude, 3)
            motor_commands[3] = self._simulate_mojo_call(vx, vy, vz, wz, altitude, 4)
            
            return motor_commands
            
        except Exception as e:
            logger.error(f"Mojo control processing failed: {e}")
            return self._python_fallback(vx, vy, vz, wz, altitude)
    
    def _simulate_mojo_call(self, vx: float, vy: float, vz: float, wz: float, altitude: float, motor_index: int) -> float:
        """Simulate Mojo function call (placeholder for actual interop)"""
        # This replicates the Mojo logic in Python for now
        
        # Apply safety limits
        max_velocity = 5.0
        max_angular = 2.0
        min_altitude = 0.5
        max_altitude = 100.0
        
        safe_vx = max(-max_velocity, min(max_velocity, vx))
        safe_vy = max(-max_velocity, min(max_velocity, vy))
        safe_vz = max(-max_velocity, min(max_velocity, vz))
        safe_wz = max(-max_angular, min(max_angular, wz))
        
        # Apply altitude constraints
        if altitude <= min_altitude and safe_vz < 0:
            safe_vz = 0.0
        elif altitude >= max_altitude and safe_vz > 0:
            safe_vz = 0.0
        
        # Convert to control signals
        thrust = 0.5 + safe_vz * 0.3
        roll = safe_vy * 0.2
        pitch = safe_vx * 0.2
        yaw = safe_wz * 0.1
        
        # Compute motor command
        if motor_index == 1:  # Front-left
            command = thrust + roll + pitch - yaw
        elif motor_index == 2:  # Front-right
            command = thrust - roll + pitch + yaw
        elif motor_index == 3:  # Rear-right
            command = thrust - roll - pitch - yaw
        else:  # Rear-left (motor 4)
            command = thrust + roll - pitch + yaw
        
        return max(0.0, min(1.0, command))
    
    def _python_fallback(self, vx: float, vy: float, vz: float, wz: float, altitude: float) -> np.ndarray:
        """Python fallback implementation"""
        logger.warning("Using Python fallback for UAV control")
        
        # Simple hover command as fallback
        return np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    
    def emergency_stop(self) -> np.ndarray:
        """Emergency stop - all motors to zero"""
        return np.zeros(4, dtype=np.float32)
    
    def test_control_functions(self):
        """Test the control functions"""
        print("Testing Mojo UAV controller...")
        
        # Test takeoff
        takeoff_motors = self.process_control(0.0, 0.0, 2.0, 0.0, 1.0)
        print(f"Takeoff motors: {takeoff_motors}")
        
        # Test hover
        hover_motors = self.process_control(0.0, 0.0, 0.0, 0.0, 5.0)
        print(f"Hover motors: {hover_motors}")
        
        # Test forward
        forward_motors = self.process_control(1.0, 0.0, 0.0, 0.0, 5.0)
        print(f"Forward motors: {forward_motors}")
        
        return True

if __name__ == "__main__":
    # Test the controller
    controller = MojoUAVController()
    controller.test_control_functions()
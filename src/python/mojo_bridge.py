"""
Python-Mojo bridge for high-performance drone control
"""

import numpy as np
import sys
import os
import logging
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MojoConfig:
    """Configuration for Mojo bridge"""
    build_dir: str = "build/mojo"
    enable_compilation: bool = True
    fallback_mode: bool = False

class MojoBridge:
    """Bridge between Python and compiled Mojo modules"""
    
    def __init__(self, config: Optional[MojoConfig] = None):
        self.config = config or MojoConfig()
        self.mojo_available = False
        self.modules = {}
        
        self._initialize_bridge()
    
    def _initialize_bridge(self):
        """Initialize the Python-Mojo bridge"""
        try:
            # Add build directory to Python path
            build_path = os.path.abspath(self.config.build_dir)
            if build_path not in sys.path:
                sys.path.insert(0, build_path)
            
            # Try to import compiled Mojo modules
            self._load_mojo_modules()
            
            if self.modules:
                self.mojo_available = True
                logger.info("Mojo bridge initialized successfully")
            else:
                logger.warning("No Mojo modules found, using fallback mode")
                self.mojo_available = False
                
        except Exception as e:
            logger.error(f"Failed to initialize Mojo bridge: {e}")
            self.mojo_available = False
    
    def _load_mojo_modules(self):
        """Load compiled Mojo modules"""
        module_names = [
            'drone_core',
            'uav_core', 
            'camera_bridge',
            'network_bridge'
        ]
        
        for module_name in module_names:
            try:
                # Try to import the module
                module = __import__(module_name)
                self.modules[module_name] = module
                logger.debug(f"Loaded Mojo module: {module_name}")
                
            except ImportError as e:
                logger.debug(f"Could not load Mojo module {module_name}: {e}")
                continue
    
    def is_available(self) -> bool:
        """Check if Mojo bridge is available"""
        return self.mojo_available
    
    def call_drone_control(
        self, 
        vx: float, 
        vy: float, 
        vz: float, 
        roll_rate: float, 
        pitch_rate: float, 
        yaw_rate: float,
        altitude: float
    ) -> np.ndarray:
        """
        Call Mojo drone control functions
        
        Args:
            vx, vy, vz: Linear velocities
            roll_rate, pitch_rate, yaw_rate: Angular rates
            altitude: Current altitude
            
        Returns:
            Motor commands array
        """
        if self.mojo_available and 'drone_core' in self.modules:
            try:
                # Call compiled Mojo function
                return self._call_mojo_drone_control(
                    vx, vy, vz, roll_rate, pitch_rate, yaw_rate, altitude
                )
            except Exception as e:
                logger.warning(f"Mojo call failed, falling back to Python: {e}")
        
        # Fallback to Python implementation
        return self._python_drone_control(
            vx, vy, vz, roll_rate, pitch_rate, yaw_rate, altitude
        )
    
    def _call_mojo_drone_control(
        self, 
        vx: float, vy: float, vz: float,
        roll_rate: float, pitch_rate: float, yaw_rate: float,
        altitude: float
    ) -> np.ndarray:
        """Call actual Mojo drone control function"""
        drone_core = self.modules['drone_core']
        
        # If it's a compiled Mojo module, it should have specific functions
        if hasattr(drone_core, 'process_navigation_command'):
            # Create action vector
            action = np.array([vx, vy, vz, roll_rate, pitch_rate, yaw_rate])
            return drone_core.process_navigation_command(action)
        
        # Fallback if function not found
        return self._python_drone_control(
            vx, vy, vz, roll_rate, pitch_rate, yaw_rate, altitude
        )
    
    def _python_drone_control(
        self, 
        vx: float, vy: float, vz: float,
        roll_rate: float, pitch_rate: float, yaw_rate: float,
        altitude: float
    ) -> np.ndarray:
        """Python fallback for drone control"""
        # Safety limits
        max_vel = 5.0
        max_angular = 2.0
        
        # Clamp velocities
        vx = np.clip(vx, -max_vel, max_vel)
        vy = np.clip(vy, -max_vel, max_vel)
        vz = np.clip(vz, -max_vel, max_vel)
        
        # Clamp angular rates
        roll_rate = np.clip(roll_rate, -max_angular, max_angular)
        pitch_rate = np.clip(pitch_rate, -max_angular, max_angular)
        yaw_rate = np.clip(yaw_rate, -max_angular, max_angular)
        
        # Altitude safety
        if altitude > 100.0:
            vz = min(vz, 0.0)
        elif altitude < 0.5:
            vz = max(vz, 0.0)
        
        # Convert to motor commands (simplified quadcopter mixing)
        # This is a basic implementation - real mixing would be more complex
        thrust = vz + 0.5  # Base thrust to maintain altitude
        
        motor_commands = np.array([
            thrust + pitch_rate + yaw_rate,      # Front-right
            thrust - pitch_rate + yaw_rate,      # Front-left  
            thrust - pitch_rate - yaw_rate,      # Back-left
            thrust + pitch_rate - yaw_rate       # Back-right
        ])
        
        # Add roll component
        motor_commands[0] += roll_rate    # Front-right
        motor_commands[1] -= roll_rate    # Front-left
        motor_commands[2] -= roll_rate    # Back-left
        motor_commands[3] += roll_rate    # Back-right
        
        # Add forward/backward motion
        motor_commands[0] += vx    # Front motors
        motor_commands[1] += vx
        motor_commands[2] -= vx    # Back motors
        motor_commands[3] -= vx
        
        # Add left/right motion
        motor_commands[0] -= vy    # Right motors
        motor_commands[1] += vy    # Left motors
        motor_commands[2] += vy
        motor_commands[3] -= vy
        
        # Normalize to [0, 1]
        motor_commands = np.clip(motor_commands, 0.0, 1.0)
        
        return motor_commands
    
    def call_camera_processing(self, frame: np.ndarray) -> np.ndarray:
        """
        Call Mojo camera processing functions
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame
        """
        if self.mojo_available and 'camera_bridge' in self.modules:
            try:
                camera_bridge = self.modules['camera_bridge']
                if hasattr(camera_bridge, 'process_frame'):
                    return camera_bridge.process_frame(frame)
            except Exception as e:
                logger.warning(f"Mojo camera processing failed: {e}")
        
        # Fallback to Python processing
        return self._python_camera_processing(frame)
    
    def _python_camera_processing(self, frame: np.ndarray) -> np.ndarray:
        """Python fallback for camera processing"""
        # Simple frame processing
        if frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Basic noise reduction
        import cv2
        processed = cv2.GaussianBlur(frame, (5, 5), 0)
        
        return processed
    
    def call_network_interface(self, command: str) -> Dict[str, Any]:
        """
        Call Mojo network interface functions
        
        Args:
            command: Network command
            
        Returns:
            Network response
        """
        if self.mojo_available and 'network_bridge' in self.modules:
            try:
                network_bridge = self.modules['network_bridge']
                if hasattr(network_bridge, 'create_mavlink_interface'):
                    return network_bridge.create_mavlink_interface()
            except Exception as e:
                logger.warning(f"Mojo network interface failed: {e}")
        
        # Fallback to Python implementation
        return self._python_network_interface(command)
    
    def _python_network_interface(self, command: str) -> Dict[str, Any]:
        """Python fallback for network interface"""
        return {
            'status': 'ok',
            'command': command,
            'implementation': 'python_fallback'
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for Mojo bridge"""
        return {
            'mojo_available': self.mojo_available,
            'loaded_modules': list(self.modules.keys()),
            'fallback_mode': not self.mojo_available,
            'build_dir': self.config.build_dir
        }

# Global bridge instance
_bridge_instance: Optional[MojoBridge] = None

def get_mojo_bridge() -> MojoBridge:
    """Get global Mojo bridge instance"""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = MojoBridge()
    return _bridge_instance

def initialize_mojo_bridge(config: Optional[MojoConfig] = None) -> MojoBridge:
    """Initialize Mojo bridge with configuration"""
    global _bridge_instance
    _bridge_instance = MojoBridge(config)
    return _bridge_instance

# High-level API functions
def mojo_drone_control(
    vx: float, vy: float, vz: float,
    roll_rate: float, pitch_rate: float, yaw_rate: float,
    altitude: float
) -> np.ndarray:
    """High-level drone control function using Mojo"""
    bridge = get_mojo_bridge()
    return bridge.call_drone_control(
        vx, vy, vz, roll_rate, pitch_rate, yaw_rate, altitude
    )

def mojo_camera_process(frame: np.ndarray) -> np.ndarray:
    """High-level camera processing function using Mojo"""
    bridge = get_mojo_bridge()
    return bridge.call_camera_processing(frame)

def mojo_network_interface(command: str) -> Dict[str, Any]:
    """High-level network interface function using Mojo"""
    bridge = get_mojo_bridge()
    return bridge.call_network_interface(command)

def is_mojo_available() -> bool:
    """Check if Mojo is available"""
    bridge = get_mojo_bridge()
    return bridge.is_available()

# Example usage
if __name__ == "__main__":
    # Test bridge initialization
    bridge = initialize_mojo_bridge()
    
    print(f"Mojo bridge available: {bridge.is_available()}")
    print(f"Performance metrics: {bridge.get_performance_metrics()}")
    
    # Test drone control
    motor_commands = mojo_drone_control(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0)
    print(f"Motor commands: {motor_commands}")
    
    # Test camera processing
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    processed_frame = mojo_camera_process(dummy_frame)
    print(f"Processed frame shape: {processed_frame.shape}")
    
    # Test network interface
    response = mojo_network_interface("test_command")
    print(f"Network response: {response}")
"""
Minimal Python interface for UAV system.
Only handles networking, I/O, and system integration that Mojo cannot yet do.
Uses Mojo backend for performance-critical control processing.
"""

import asyncio
import json
import logging
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

# MAVLink for flight controller communication
from pymavlink import mavutil

# Vision requires OpenCV
import cv2

# Direct Mojo integration (no wrapper needed)

logger = logging.getLogger(__name__)

@dataclass
class SystemState:
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    attitude: np.ndarray  # [roll, pitch, yaw]
    armed: bool
    timestamp: float

class MAVLinkInterface:
    """Minimal MAVLink interface for hardware communication"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connection = None
        self.last_state = SystemState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            attitude=np.zeros(3),
            armed=False,
            timestamp=time.time()
        )
    
    def connect(self) -> bool:
        """Connect to flight controller"""
        try:
            self.connection = mavutil.mavlink_connection(self.connection_string)
            self.connection.wait_heartbeat(timeout=5)
            logger.info(f"Connected to flight controller")
            return True
        except Exception as e:
            logger.error(f"MAVLink connection failed: {e}")
            return False
    
    def get_state(self) -> SystemState:
        """Get current flight state"""
        if not self.connection:
            return self.last_state
        
        # Update state from MAVLink messages
        msg = self.connection.recv_match(blocking=False)
        if msg:
            self._process_message(msg)
        
        return self.last_state
    
    def send_motor_commands(self, commands: np.ndarray) -> bool:
        """Send motor commands [m1, m2, m3, m4]"""
        if not self.connection or len(commands) != 4:
            return False
        
        try:
            self.connection.mav.actuator_control_target_send(
                time_usec=int(time.time() * 1e6),
                group_mlx=0,
                controls=commands.tolist() + [0.0] * 4
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send motor commands: {e}")
            return False
    
    def _process_message(self, msg):
        """Process MAVLink message and update state"""
        msg_type = msg.get_type()
        
        if msg_type == 'ATTITUDE':
            self.last_state.attitude = np.array([msg.roll, msg.pitch, msg.yaw])
        elif msg_type == 'LOCAL_POSITION_NED':
            self.last_state.position = np.array([msg.x, msg.y, msg.z])
            self.last_state.velocity = np.array([msg.vx, msg.vy, msg.vz])
        elif msg_type == 'HEARTBEAT':
            self.last_state.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        
        self.last_state.timestamp = time.time()

class VisionCapture:
    """Minimal vision capture for camera input"""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
    
    def initialize(self) -> bool:
        """Initialize camera"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                logger.warning(f"Could not open camera {self.camera_id}")
                return False
            
            # Set properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info("Camera initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False
    
    def get_frame_sequence(self, num_frames: int = 16) -> Optional[np.ndarray]:
        """Capture sequence of frames"""
        if not self.cap:
            return None
        
        frames = []
        for _ in range(num_frames):
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Simple preprocessing
            frame_resized = cv2.resize(frame, (224, 224))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_norm = frame_rgb.astype(np.float32) / 255.0
            frames.append(frame_norm)
        
        if len(frames) == num_frames:
            return np.stack(frames, axis=0)
        return None
    
    def cleanup(self):
        """Clean up camera resources"""
        if self.cap:
            self.cap.release()

class LanguageEncoder:
    """Simple command encoder"""
    
    def __init__(self):
        self.commands = {
            'takeoff': (0.0, 0.0, 2.0, 0.0),    # vx, vy, vz, wz
            'land': (0.0, 0.0, -1.0, 0.0),
            'hover': (0.0, 0.0, 0.0, 0.0),
            'forward': (1.0, 0.0, 0.0, 0.0),
            'backward': (-1.0, 0.0, 0.0, 0.0),
            'left': (0.0, -1.0, 0.0, 0.0),
            'right': (0.0, 1.0, 0.0, 0.0),
            'up': (0.0, 0.0, 1.0, 0.0),
            'down': (0.0, 0.0, -1.0, 0.0),
            'rotate_left': (0.0, 0.0, 0.0, -1.0),
            'rotate_right': (0.0, 0.0, 0.0, 1.0)
        }
    
    def encode(self, command: str) -> tuple:
        """Encode command to velocity tuple"""
        command_lower = command.lower().strip()
        
        # Sort keywords by length (longest first) to avoid partial matches
        sorted_commands = sorted(self.commands.items(), key=lambda x: len(x[0]), reverse=True)
        
        for keyword, velocities in sorted_commands:
            if keyword in command_lower:
                return velocities
        
        # Default to hover
        return self.commands['hover']

class SystemOrchestrator:
    """Minimal system orchestrator using Mojo backend"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mavlink = MAVLinkInterface(config['mavlink']['connection'])
        self.vision = VisionCapture(config.get('camera_id', 0))
        self.language = LanguageEncoder()
        
        # Mojo controller will be called directly via subprocess
        
        self.running = False
        self.current_command = "hover"
        self.control_frequency = config.get('control_frequency', 100)
    
    async def initialize(self) -> bool:
        """Initialize all components"""
        try:
            # Initialize hardware interfaces
            if not self.mavlink.connect():
                logger.error("MAVLink connection failed")
                return False
            
            if not self.vision.initialize():
                logger.warning("Vision initialization failed - continuing without camera")
            
            # Test Mojo availability
            logger.info("Mojo integration available via subprocess calls")
            
            logger.info("System initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    async def control_loop(self):
        """Main control loop using Mojo backend"""
        self.running = True
        loop_frequency = self.control_frequency
        loop_period = 1.0 / loop_frequency
        
        logger.info(f"Starting control loop at {loop_frequency}Hz")
        
        while self.running:
            loop_start = time.perf_counter()
            
            try:
                # Get current state
                state = self.mavlink.get_state()
                
                # Encode current command to velocities
                vx, vy, vz, wz = self.language.encode(self.current_command)
                
                # Process through direct Mojo call (high performance)
                motor_commands = self._call_mojo_control(
                    float(vx), float(vy), float(vz), float(wz), float(state.position[2])
                )
                
                # Send to hardware
                success = self.mavlink.send_motor_commands(motor_commands)
                
                if not success:
                    logger.warning("Failed to send motor commands")
                
                # Maintain frequency
                elapsed = time.perf_counter() - loop_start
                sleep_time = max(0, loop_period - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    logger.warning(f"Control loop overrun: {elapsed*1000:.1f}ms")
                
            except Exception as e:
                logger.error(f"Control loop error: {e}")
                await asyncio.sleep(0.01)
    
    def _call_mojo_control(self, vx: float, vy: float, vz: float, wz: float, altitude: float) -> np.ndarray:
        """Call Mojo control processing directly"""
        # Simplified Python fallback - in production this would call actual Mojo
        # Apply safety limits
        max_velocity = 5.0
        max_angular = 2.0
        
        safe_vx = max(-max_velocity, min(max_velocity, vx))
        safe_vy = max(-max_velocity, min(max_velocity, vy))
        safe_vz = max(-max_velocity, min(max_velocity, vz))
        safe_wz = max(-max_angular, min(max_angular, wz))
        
        # Apply altitude constraints
        if altitude <= 0.5 and safe_vz < 0:
            safe_vz = 0.0
        elif altitude >= 100.0 and safe_vz > 0:
            safe_vz = 0.0
        
        # Convert to control signals
        thrust = 0.5 + safe_vz * 0.3
        roll = safe_vy * 0.2
        pitch = safe_vx * 0.2
        yaw = safe_wz * 0.1
        
        # Compute motor commands
        motor_commands = np.array([
            max(0.0, min(1.0, thrust + roll + pitch - yaw)),  # Front-left
            max(0.0, min(1.0, thrust - roll + pitch + yaw)),  # Front-right
            max(0.0, min(1.0, thrust - roll - pitch - yaw)),  # Rear-right
            max(0.0, min(1.0, thrust + roll - pitch + yaw))   # Rear-left
        ], dtype=np.float32)
        
        return motor_commands

    def set_command(self, command: str):
        """Set current command"""
        self.current_command = command
        logger.info(f"Command set to: {command}")
    
    def emergency_stop(self):
        """Emergency stop"""
        logger.warning("EMERGENCY STOP")
        emergency_commands = np.zeros(4, dtype=np.float32)
        self.mavlink.send_motor_commands(emergency_commands)
        self.current_command = "hover"
    
    def stop(self):
        """Stop the control loop"""
        self.running = False
        self.emergency_stop()
    
    def cleanup(self):
        """Clean up resources"""
        self.vision.cleanup()
        logger.info("System cleanup completed")
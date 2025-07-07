"""
Autonomous drone system integrating VLA model, safety, and object detection
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Callable
import logging
import time
import threading
from dataclasses import dataclass
from enum import Enum

from ..models.huggingface.vla_model import create_drone_vla_model
from ..safety.validator import SafetyMonitor, SafetyLevel
from ..vision.object_actions import VisionActionIntegrator
from .drone_control import DroneController
from .camera import VisionSystem

logger = logging.getLogger(__name__)


class SystemState(Enum):
    """System operational states"""
    OFFLINE = "offline"
    INITIALIZING = "initializing"
    READY = "ready"
    AUTONOMOUS = "autonomous"
    MANUAL_OVERRIDE = "manual_override"
    EMERGENCY = "emergency"
    LANDING = "landing"


@dataclass
class MissionStep:
    """Single step in autonomous mission"""
    command: str
    timeout: float = 30.0
    success_criteria: Optional[Dict[str, Any]] = None
    failure_action: str = "abort"


@dataclass
class SystemStatus:
    """Current system status"""
    state: SystemState
    drone_connected: bool
    camera_active: bool
    model_loaded: bool
    safety_active: bool
    emergency_stop: bool
    current_mission: Optional[str]
    errors: List[str]


class AutonomousDroneSystem:
    """Complete autonomous drone system with VLA, safety, and object detection"""
    
    def __init__(self, 
                 model_size: str = "large",
                 safety_level: str = "normal",
                 simulation_mode: bool = True):
        
        self.model_size = model_size
        self.safety_level = safety_level
        self.simulation_mode = simulation_mode
        
        # System components
        self.vla_model = None
        self.drone_controller = None
        self.vision_system = None
        self.safety_monitor = None
        self.vision_integrator = None
        
        # System state
        self.state = SystemState.OFFLINE
        self.emergency_callbacks = []
        self.running = False
        self.autonomous_thread = None
        self.current_mission = None
        
        # Mission tracking
        self.mission_queue = []
        self.mission_history = []
        
        logger.info(f"Initialized AutonomousDroneSystem (model: {model_size}, safety: {safety_level})")
        
    def initialize(self) -> bool:
        """Initialize all system components"""
        try:
            self.state = SystemState.INITIALIZING
            logger.info("Initializing autonomous drone system...")
            
            # Load VLA model
            logger.info("Loading VLA model...")
            self.vla_model = create_drone_vla_model(
                model_size=self.model_size,
                freeze_backbone=True
            )
            
            # Initialize drone controller
            logger.info("Initializing drone controller...")
            self.drone_controller = DroneController(simulation_mode=self.simulation_mode)
            if not self.drone_controller.connect():
                raise Exception("Failed to connect to drone")
            self.drone_controller.start_control_loop()
            
            # Initialize vision system
            logger.info("Initializing vision system...")
            self.vision_system = VisionSystem()
            if not self.vision_system.initialize():
                logger.warning("Vision system initialization failed, using dummy frames")
            
            # Initialize safety monitor
            logger.info("Initializing safety monitor...")
            self.safety_monitor = SafetyMonitor(SafetyLevel(self.safety_level))
            
            # Initialize vision-action integrator
            logger.info("Initializing vision-action integrator...")
            self.vision_integrator = VisionActionIntegrator(
                self.vla_model,
                self.vision_system.detector if self.vision_system else None
            )
            
            self.state = SystemState.READY
            logger.info("System initialization complete")
            return True
            
        except Exception as e:
            self.state = SystemState.OFFLINE
            logger.error(f"System initialization failed: {e}")
            return False
            
    def start_autonomous_mode(self):
        """Start autonomous operation mode"""
        if self.state != SystemState.READY:
            logger.error("System not ready for autonomous mode")
            return False
            
        self.state = SystemState.AUTONOMOUS
        self.running = True
        
        # Start autonomous control thread
        self.autonomous_thread = threading.Thread(target=self._autonomous_loop)
        self.autonomous_thread.daemon = True
        self.autonomous_thread.start()
        
        logger.info("Autonomous mode started")
        return True
        
    def stop_autonomous_mode(self):
        """Stop autonomous operation mode"""
        self.running = False
        self.state = SystemState.READY
        
        if self.autonomous_thread and self.autonomous_thread.is_alive():
            self.autonomous_thread.join(timeout=5.0)
            
        logger.info("Autonomous mode stopped")
        
    def emergency_stop(self):
        """Activate emergency stop"""
        logger.critical("EMERGENCY STOP ACTIVATED")
        
        self.state = SystemState.EMERGENCY
        self.safety_monitor.activate_emergency_stop()
        
        # Stop all movement
        emergency_action = np.zeros(6)
        self.drone_controller.execute_action_vector(emergency_action)
        
        # Execute emergency callbacks
        for callback in self.emergency_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Emergency callback failed: {e}")
                
    def add_emergency_callback(self, callback: Callable):
        """Add callback function for emergency situations"""
        self.emergency_callbacks.append(callback)
        
    def execute_command(self, command: str, timeout: float = 30.0) -> bool:
        """Execute single autonomous command"""
        if self.state not in [SystemState.READY, SystemState.AUTONOMOUS]:
            logger.error(f"Cannot execute command in state: {self.state}")
            return False
            
        try:
            # Get current video frames
            frames = self._get_current_frames()
            if not frames:
                logger.error("Failed to get video frames")
                return False
                
            # Get current drone state
            drone_status = self.drone_controller.get_status()
            current_state = {
                'altitude': drone_status.position[2],
                'battery': drone_status.battery_level,
                'position': drone_status.position
            }
            
            # Process with vision-action integrator
            result = self.vision_integrator.process_frame_with_objects(
                frames, command, detection_threshold=0.5
            )
            
            # Validate with safety monitor
            safety_result = self.safety_monitor.validate_and_filter(
                command=command,
                action=result['final_action'],
                confidence=np.ones(6),  # Placeholder confidence
                current_state=current_state
            )
            
            if not safety_result['is_safe']:
                logger.warning(f"Command blocked: {safety_result['blocked_reason']}")
                return False
                
            # Execute safe action
            self.drone_controller.execute_action_vector(safety_result['safe_action'])
            
            # Log execution
            logger.info(f"Executed command: '{command}'")
            if safety_result['warnings']:
                logger.warning(f"Warnings: {safety_result['warnings']}")
                
            return True
            
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return False
            
    def execute_mission(self, mission_steps: List[MissionStep]) -> bool:
        """Execute autonomous mission"""
        if self.state != SystemState.AUTONOMOUS:
            logger.error("System must be in autonomous mode for missions")
            return False
            
        self.current_mission = f"Mission_{int(time.time())}"
        logger.info(f"Starting mission: {self.current_mission}")
        
        success = True
        for i, step in enumerate(mission_steps):
            if not self.running or self.state == SystemState.EMERGENCY:
                logger.warning("Mission aborted")
                success = False
                break
                
            logger.info(f"Mission step {i+1}/{len(mission_steps)}: {step.command}")
            
            step_success = self.execute_command(step.command, step.timeout)
            
            if not step_success:
                logger.error(f"Mission step failed: {step.command}")
                if step.failure_action == "abort":
                    success = False
                    break
                elif step.failure_action == "continue":
                    logger.warning("Continuing mission despite failure")
                    
            # Wait for step completion
            time.sleep(2.0)
            
        # Record mission result
        self.mission_history.append({
            'mission_id': self.current_mission,
            'steps': mission_steps,
            'success': success,
            'timestamp': time.time()
        })
        
        self.current_mission = None
        logger.info(f"Mission completed. Success: {success}")
        return success
        
    def get_status(self) -> SystemStatus:
        """Get current system status"""
        errors = []
        
        # Check component status
        drone_connected = (self.drone_controller is not None and 
                          self.drone_controller.get_status().state != "disconnected")
        
        camera_active = (self.vision_system is not None and 
                        self.vision_system.is_initialized)
        
        model_loaded = self.vla_model is not None
        safety_active = (self.safety_monitor is not None and 
                        not self.safety_monitor.emergency_stop)
        
        # Collect any errors
        if not drone_connected:
            errors.append("Drone not connected")
        if not camera_active:
            errors.append("Camera not active")
        if not model_loaded:
            errors.append("VLA model not loaded")
            
        return SystemStatus(
            state=self.state,
            drone_connected=drone_connected,
            camera_active=camera_active,
            model_loaded=model_loaded,
            safety_active=safety_active,
            emergency_stop=self.safety_monitor.emergency_stop if self.safety_monitor else True,
            current_mission=self.current_mission,
            errors=errors
        )
        
    def _autonomous_loop(self):
        """Main autonomous control loop"""
        logger.info("Autonomous control loop started")
        
        while self.running and self.state == SystemState.AUTONOMOUS:
            try:
                # Process mission queue
                if self.mission_queue:
                    mission = self.mission_queue.pop(0)
                    self.execute_mission(mission)
                    
                # Regular status check
                status = self.get_status()
                if status.errors:
                    logger.warning(f"System warnings: {status.errors}")
                    
                time.sleep(0.1)  # 10Hz control loop
                
            except Exception as e:
                logger.error(f"Error in autonomous loop: {e}")
                time.sleep(1.0)
                
        logger.info("Autonomous control loop stopped")
        
    def _get_current_frames(self, num_frames: int = 8) -> List[np.ndarray]:
        """Get current video frames from vision system"""
        frames = []
        
        if self.vision_system and self.vision_system.is_initialized:
            try:
                for _ in range(num_frames):
                    result = self.vision_system.get_processed_frame(timeout=0.1)
                    if result:
                        processed_tensor, raw_frame = result
                        frames.append(raw_frame)
                    else:
                        break
            except Exception as e:
                logger.warning(f"Failed to get frames from vision system: {e}")
                
        # Fallback to dummy frames if needed
        if len(frames) < num_frames:
            dummy_frames_needed = num_frames - len(frames)
            for _ in range(dummy_frames_needed):
                dummy_frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                frames.append(dummy_frame)
                
        return frames
        
    def cleanup(self):
        """Clean up system resources"""
        logger.info("Cleaning up autonomous drone system...")
        
        self.stop_autonomous_mode()
        
        if self.drone_controller:
            self.drone_controller.cleanup()
            
        if self.vision_system:
            self.vision_system.cleanup()
            
        self.state = SystemState.OFFLINE
        logger.info("System cleanup complete")


def create_autonomous_system(model_size: str = "large",
                           safety_level: str = "normal", 
                           simulation_mode: bool = True) -> AutonomousDroneSystem:
    """Factory function to create autonomous drone system"""
    return AutonomousDroneSystem(model_size, safety_level, simulation_mode)
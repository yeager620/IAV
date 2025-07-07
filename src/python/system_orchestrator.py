"""
Main system orchestrator for the UAV VLA system.
Coordinates between vision, language, control, and hardware interfaces.
"""

import asyncio
import numpy as np
import time
import logging
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum

from .mavlink_interface import MAVLinkInterface, FlightState
from .mojo_interface import MojoSystemInterface
from .vision_pipeline import VisionPipeline
from .language_processor import LanguageProcessor

logger = logging.getLogger(__name__)

class SystemState(Enum):
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    ERROR = "error"
    EMERGENCY = "emergency"

@dataclass
class SystemStatus:
    state: SystemState
    components_ready: Dict[str, bool]
    last_update: float
    error_message: Optional[str] = None

class SystemOrchestrator:
    """Main orchestrator for the UAV VLA system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Core components
        self.mavlink: Optional[MAVLinkInterface] = None
        self.mojo_system: Optional[MojoSystemInterface] = None
        self.vision: Optional[VisionPipeline] = None
        self.language: Optional[LanguageProcessor] = None
        
        # System state
        self.system_state = SystemState.UNINITIALIZED
        self.status = SystemStatus(
            state=SystemState.UNINITIALIZED,
            components_ready={},
            last_update=time.time()
        )
        
        # Control loop settings
        self.control_frequency = config.get('control_frequency', 100)  # Hz
        self.control_loop_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Performance monitoring
        self.loop_times = []
        self.control_latencies = []
        self.total_cycles = 0
        
        # Safety and emergency handling
        self.emergency_callbacks: list[Callable] = []
        self.last_successful_command = time.time()
        self.command_timeout = config.get('command_timeout', 5.0)  # seconds
        
        # Current mission state
        self.current_command = "hover"
        self.mission_active = False
        
    async def initialize(self) -> bool:
        """Initialize all system components"""
        logger.info("Initializing UAV VLA system...")
        self.system_state = SystemState.INITIALIZING
        
        try:
            # Initialize MAVLink interface
            mavlink_config = self.config.get('mavlink', {})
            self.mavlink = MAVLinkInterface(
                connection_string=mavlink_config.get('connection', '/dev/ttyUSB0'),
                baud_rate=mavlink_config.get('baud_rate', 921600)
            )
            
            if not self.mavlink.connect():
                raise RuntimeError("Failed to connect to flight controller")
            
            if not self.mavlink.start():
                raise RuntimeError("Failed to start MAVLink interface")
            
            self.status.components_ready['mavlink'] = True
            
            # Initialize Mojo system
            model_path = self.config.get('model_path', 'data/models/vla_model.mojo')
            self.mojo_system = MojoSystemInterface(model_path)
            
            if not self.mojo_system.initialize():
                raise RuntimeError("Failed to initialize Mojo system")
            
            self.status.components_ready['mojo_system'] = True
            
            # Initialize vision pipeline
            vision_config = self.config.get('vision', {})
            self.vision = VisionPipeline(vision_config)
            
            if not self.vision.initialize():
                raise RuntimeError("Failed to initialize vision pipeline")
            
            self.status.components_ready['vision'] = True
            
            # Initialize language processor
            language_config = self.config.get('language', {})
            self.language = LanguageProcessor(language_config)
            
            if not self.language.initialize():
                raise RuntimeError("Failed to initialize language processor")
            
            self.status.components_ready['language'] = True
            
            # Setup callbacks
            self.mavlink.add_state_callback(self._on_flight_state_update)
            
            self.system_state = SystemState.READY
            self.status.state = SystemState.READY
            self.status.last_update = time.time()
            
            logger.info("System initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            self.system_state = SystemState.ERROR
            self.status.state = SystemState.ERROR
            self.status.error_message = str(e)
            return False
    
    async def start_autonomous_mode(self) -> bool:
        """Start autonomous control mode"""
        if self.system_state != SystemState.READY:
            logger.error("Cannot start autonomous mode - system not ready")
            return False
        
        try:
            logger.info("Starting autonomous control mode")
            self.running = True
            self.system_state = SystemState.ACTIVE
            self.status.state = SystemState.ACTIVE
            
            # Start main control loop
            self.control_loop_task = asyncio.create_task(self._control_loop())
            
            logger.info("Autonomous mode started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start autonomous mode: {e}")
            self.system_state = SystemState.ERROR
            return False
    
    async def stop_autonomous_mode(self):
        """Stop autonomous control mode"""
        logger.info("Stopping autonomous mode")
        self.running = False
        self.mission_active = False
        
        if self.control_loop_task:
            self.control_loop_task.cancel()
            try:
                await self.control_loop_task
            except asyncio.CancelledError:
                pass
        
        # Send hover command as safety measure
        if self.mavlink:
            self.mavlink.send_velocity_command(np.zeros(3), 0.0)
        
        self.system_state = SystemState.READY
        self.status.state = SystemState.READY
        logger.info("Autonomous mode stopped")
    
    async def execute_command(self, command: str, timeout: float = 10.0) -> bool:
        """Execute a single voice/text command"""
        if self.system_state not in [SystemState.READY, SystemState.ACTIVE]:
            logger.error("Cannot execute command - system not ready")
            return False
        
        try:
            logger.info(f"Executing command: '{command}'")
            self.current_command = command
            self.last_successful_command = time.time()
            
            # If not in autonomous mode, run a single control cycle
            if self.system_state == SystemState.READY:
                await self._single_control_cycle()
            
            return True
            
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return False
    
    async def emergency_stop(self):
        """Execute emergency stop procedure"""
        logger.warning("EMERGENCY STOP ACTIVATED")
        self.system_state = SystemState.EMERGENCY
        self.status.state = SystemState.EMERGENCY
        
        # Stop all motion
        if self.mavlink:
            self.mavlink.send_velocity_command(np.zeros(3), 0.0)
            
            # Try to land if possible
            if self.mavlink.get_flight_state().armed:
                self.mavlink.set_mode('LAND')
        
        # Stop autonomous mode
        await self.stop_autonomous_mode()
        
        # Notify emergency callbacks
        for callback in self.emergency_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Emergency callback failed: {e}")
    
    def add_emergency_callback(self, callback: Callable):
        """Add callback for emergency situations"""
        self.emergency_callbacks.append(callback)
    
    def get_system_status(self) -> SystemStatus:
        """Get current system status"""
        self.status.last_update = time.time()
        return self.status
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = {
            'system_state': self.system_state.value,
            'total_cycles': self.total_cycles,
            'control_frequency_actual': len(self.loop_times) / sum(self.loop_times) if self.loop_times else 0,
            'avg_loop_time_ms': np.mean(self.loop_times) * 1000 if self.loop_times else 0,
            'avg_control_latency_ms': np.mean(self.control_latencies) if self.control_latencies else 0,
        }
        
        if self.mojo_system:
            stats.update(self.mojo_system.get_system_stats())
        
        if self.mavlink:
            stats.update(self.mavlink.get_connection_stats())
        
        return stats
    
    async def _control_loop(self):
        """Main autonomous control loop"""
        logger.info(f"Starting control loop at {self.control_frequency}Hz")
        loop_period = 1.0 / self.control_frequency
        
        while self.running:
            loop_start = time.perf_counter()
            
            try:
                await self._single_control_cycle()
                
                # Performance tracking
                loop_end = time.perf_counter()
                loop_time = loop_end - loop_start
                
                self.loop_times.append(loop_time)
                if len(self.loop_times) > 100:
                    self.loop_times.pop(0)
                
                self.total_cycles += 1
                
                # Maintain control frequency
                sleep_time = max(0, loop_period - loop_time)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    logger.warning(f"Control loop overrun: {loop_time*1000:.1f}ms > {loop_period*1000:.1f}ms")
                
            except Exception as e:
                logger.error(f"Control loop error: {e}")
                await asyncio.sleep(0.01)  # Brief pause on error
    
    async def _single_control_cycle(self):
        """Execute a single control cycle"""
        control_start = time.perf_counter()
        
        try:
            # Get current flight state
            flight_state = self.mavlink.get_flight_state()
            
            # Get latest vision data
            frames = await self.vision.get_latest_frames()
            if frames is None:
                logger.warning("No vision data available")
                return
            
            # Process current command
            command_embedding = self.language.encode_command(self.current_command)
            
            # Prepare current state for Mojo system
            current_state = {
                'altitude': flight_state.position[2],
                'velocity': flight_state.velocity,
                'attitude': flight_state.attitude,
                'angular_velocity': flight_state.angular_velocity
            }
            
            # Generate motor commands via Mojo system
            motor_commands, metadata = self.mojo_system.process_frame_and_command(
                frames, command_embedding, current_state
            )
            
            # Send commands to flight controller
            success = self.mavlink.send_motor_commands(motor_commands)
            
            if success:
                self.last_successful_command = time.time()
            
            # Track control latency
            control_end = time.perf_counter()
            control_latency = (control_end - control_start) * 1000
            self.control_latencies.append(control_latency)
            if len(self.control_latencies) > 100:
                self.control_latencies.pop(0)
            
            # Check for timeouts and safety violations
            await self._check_safety_conditions(metadata)
            
        except Exception as e:
            logger.error(f"Control cycle error: {e}")
            raise
    
    async def _check_safety_conditions(self, metadata: Dict[str, Any]):
        """Check safety conditions and handle violations"""
        current_time = time.time()
        
        # Check command timeout
        if current_time - self.last_successful_command > self.command_timeout:
            logger.warning("Command timeout detected")
            await self.emergency_stop()
            return
        
        # Check for safety violations
        safety_violations = metadata.get('safety_violations', [])
        if safety_violations:
            logger.warning(f"Safety violations detected: {safety_violations}")
        
        # Check inference confidence
        confidence = metadata.get('confidence', 1.0)
        if confidence < 0.3:
            logger.warning(f"Low inference confidence: {confidence:.2f}")
            # Could implement confidence-based fallback here
        
        # Check control latency
        latency = metadata.get('inference_latency_ms', 0)
        if latency > 50:  # 50ms threshold
            logger.warning(f"High control latency: {latency:.1f}ms")
    
    def _on_flight_state_update(self, flight_state: FlightState):
        """Callback for flight state updates"""
        # Could implement state-based logic here
        pass
    
    async def cleanup(self):
        """Clean up all system resources"""
        logger.info("Cleaning up system resources")
        
        await self.stop_autonomous_mode()
        
        if self.vision:
            self.vision.cleanup()
        
        if self.mavlink:
            self.mavlink.stop()
        
        logger.info("System cleanup completed")
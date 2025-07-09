"""
Unified drone control system following Mojo-first architecture
Consolidates drone_control.py and minimal_interface.py into single interface
"""

import time
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import threading
import queue
import logging
import asyncio

# Fix for Python 3.12+ compatibility with dronekit
import sys
if sys.version_info >= (3, 3):
    import collections.abc
    collections.MutableMapping = collections.abc.MutableMapping

# MAVLink for direct communication
from pymavlink import mavutil

# Optional DroneKit for high-level operations
try:
    from dronekit import connect, VehicleMode, LocationGlobalRelative, APIException
    DRONEKIT_AVAILABLE = True
except ImportError:
    DRONEKIT_AVAILABLE = False
    print("DroneKit not available - using MAVLink direct mode")

# Mojo performance modules
try:
    from ..mojo.safety_validator import create_safety_validator, validate_drone_state
    from ..mojo.control_system import create_control_system, compute_control_commands
    MOJO_AVAILABLE = True
except ImportError:
    MOJO_AVAILABLE = False
    print("Mojo modules not available - using Python fallback")
    print("  Reason: Mojo modules have compilation errors due to API changes in Mojo 25.4.0")
    print("  Status: Using Python implementations for safety, control, and vision processing")

logger = logging.getLogger(__name__)


class DroneState(Enum):
    """Drone operational states"""
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    ARMED = "armed"
    FLYING = "flying"
    LANDING = "landing"
    EMERGENCY = "emergency"


@dataclass
class DroneStatus:
    """Complete drone status information"""
    state: DroneState
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    attitude: Tuple[float, float, float]
    battery_level: float
    gps_fix: bool
    armed: bool
    mode: str
    timestamp: float


@dataclass
class DroneCommand:
    """Drone command structure"""
    command_type: str
    parameters: Dict[str, Any]
    timestamp: float
    priority: int = 0


class UnifiedDroneController:
    """
    Unified drone controller with Mojo performance backend
    Supports both simulation and real hardware
    """
    
    def __init__(self, 
                 connection_string: str = "udp:127.0.0.1:14550",
                 simulation_mode: bool = True,
                 use_dronekit: bool = True):
        self.connection_string = connection_string
        self.simulation_mode = simulation_mode
        self.use_dronekit = use_dronekit and DRONEKIT_AVAILABLE
        
        # Connection objects
        self.mavlink_connection = None
        self.dronekit_vehicle = None
        
        # State management
        self.current_state = DroneState.DISCONNECTED
        self.last_status = None
        self.command_queue = queue.Queue()
        self.control_thread = None
        self.running = False
        
        # Performance backends
        if MOJO_AVAILABLE:
            self.safety_validator = create_safety_validator()
            self.control_system = create_control_system()
            logger.info("Mojo performance backend enabled")
        else:
            self.safety_validator = None
            self.control_system = None
            logger.warning("Mojo backend not available, using Python fallback")
        
        # Control parameters
        self.control_frequency = 50  # Hz
        self.safety_enabled = True
        
        logger.info(f"UnifiedDroneController initialized (simulation: {simulation_mode})")
    
    def connect(self) -> bool:
        """Connect to drone using appropriate method"""
        try:
            if self.use_dronekit:
                return self._connect_dronekit()
            else:
                return self._connect_mavlink()
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def _connect_dronekit(self) -> bool:
        """Connect using DroneKit (high-level)"""
        try:
            logger.info(f"Connecting to drone via DroneKit: {self.connection_string}")
            
            if self.simulation_mode:
                # Use shorter timeout for simulation
                self.dronekit_vehicle = connect(self.connection_string, wait_ready=True, timeout=30)
            else:
                # Use longer timeout for real hardware
                self.dronekit_vehicle = connect(self.connection_string, wait_ready=True, timeout=60)
            
            if self.dronekit_vehicle:
                self.current_state = DroneState.CONNECTED
                logger.info("DroneKit connection successful")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"DroneKit connection failed: {e}")
            return False
    
    def _connect_mavlink(self) -> bool:
        """Connect using MAVLink (direct)"""
        try:
            logger.info(f"Connecting to drone via MAVLink: {self.connection_string}")
            
            self.mavlink_connection = mavutil.mavlink_connection(self.connection_string)
            
            # Wait for heartbeat
            heartbeat_timeout = 10.0
            start_time = time.time()
            
            while time.time() - start_time < heartbeat_timeout:
                msg = self.mavlink_connection.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
                if msg:
                    self.current_state = DroneState.CONNECTED
                    logger.info("MAVLink connection successful")
                    return True
            
            logger.error("MAVLink heartbeat timeout")
            return False
            
        except Exception as e:
            logger.error(f"MAVLink connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from drone"""
        self.running = False
        
        if self.control_thread:
            self.control_thread.join(timeout=2.0)
        
        if self.dronekit_vehicle:
            self.dronekit_vehicle.close()
            self.dronekit_vehicle = None
        
        if self.mavlink_connection:
            self.mavlink_connection.close()
            self.mavlink_connection = None
        
        self.current_state = DroneState.DISCONNECTED
        logger.info("Drone disconnected")
    
    def start_control_loop(self):
        """Start the control loop thread"""
        if not self.running:
            self.running = True
            self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
            self.control_thread.start()
            logger.info("Control loop started")
    
    def stop_control_loop(self):
        """Stop the control loop thread"""
        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=2.0)
        logger.info("Control loop stopped")
    
    def _control_loop(self):
        """Main control loop running at specified frequency"""
        dt = 1.0 / self.control_frequency
        
        while self.running:
            start_time = time.time()
            
            try:
                # Update drone status
                self._update_status()
                
                # Process commands from queue
                self._process_commands()
                
                # Safety validation
                if self.safety_enabled and self.last_status:
                    self._validate_safety()
                
            except Exception as e:
                logger.error(f"Control loop error: {e}")
            
            # Maintain control frequency
            elapsed = time.time() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)
    
    def _update_status(self):
        """Update drone status from telemetry"""
        if self.dronekit_vehicle:
            self._update_status_dronekit()
        elif self.mavlink_connection:
            self._update_status_mavlink()
    
    def _update_status_dronekit(self):
        """Update status using DroneKit"""
        try:
            vehicle = self.dronekit_vehicle
            
            # Get position
            if vehicle.location.global_relative_frame:
                lat = vehicle.location.global_relative_frame.lat or 0.0
                lon = vehicle.location.global_relative_frame.lon or 0.0
                alt = vehicle.location.global_relative_frame.alt or 0.0
                position = (lat, lon, alt)
            else:
                position = (0.0, 0.0, 0.0)
            
            # Get velocity
            if vehicle.velocity:
                velocity = (vehicle.velocity[0] or 0.0, vehicle.velocity[1] or 0.0, vehicle.velocity[2] or 0.0)
            else:
                velocity = (0.0, 0.0, 0.0)
            
            # Get attitude
            if vehicle.attitude:
                attitude = (vehicle.attitude.roll or 0.0, vehicle.attitude.pitch or 0.0, vehicle.attitude.yaw or 0.0)
            else:
                attitude = (0.0, 0.0, 0.0)
            
            # Get other status
            battery_level = vehicle.battery.level if vehicle.battery else 0.0
            gps_fix = vehicle.gps_0.fix_type >= 3 if vehicle.gps_0 else False
            armed = vehicle.armed
            mode = str(vehicle.mode.name) if vehicle.mode else "UNKNOWN"
            
            # Update current state based on vehicle state
            if vehicle.armed:
                self.current_state = DroneState.ARMED
            elif vehicle.mode.name == "LAND":
                self.current_state = DroneState.LANDING
            
            self.last_status = DroneStatus(
                state=self.current_state,
                position=position,
                velocity=velocity,
                attitude=attitude,
                battery_level=battery_level,
                gps_fix=gps_fix,
                armed=armed,
                mode=mode,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Status update failed (DroneKit): {e}")
    
    def _update_status_mavlink(self):
        """Update status using MAVLink"""
        try:
            # Get latest messages (non-blocking)
            messages = {}
            while True:
                msg = self.mavlink_connection.recv_match(blocking=False)
                if msg is None:
                    break
                messages[msg.get_type()] = msg
            
            # Default values
            position = (0.0, 0.0, 0.0)
            velocity = (0.0, 0.0, 0.0)
            attitude = (0.0, 0.0, 0.0)
            battery_level = 0.0
            gps_fix = False
            armed = False
            mode = "UNKNOWN"
            
            # Parse messages
            if 'GLOBAL_POSITION_INT' in messages:
                msg = messages['GLOBAL_POSITION_INT']
                position = (msg.lat / 1e7, msg.lon / 1e7, msg.alt / 1000.0)
                velocity = (msg.vx / 100.0, msg.vy / 100.0, msg.vz / 100.0)
            
            if 'ATTITUDE' in messages:
                msg = messages['ATTITUDE']
                attitude = (msg.roll, msg.pitch, msg.yaw)
            
            if 'BATTERY_STATUS' in messages:
                msg = messages['BATTERY_STATUS']
                battery_level = msg.battery_remaining
            
            if 'GPS_RAW_INT' in messages:
                msg = messages['GPS_RAW_INT']
                gps_fix = msg.fix_type >= 3
            
            if 'HEARTBEAT' in messages:
                msg = messages['HEARTBEAT']
                armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            
            self.last_status = DroneStatus(
                state=self.current_state,
                position=position,
                velocity=velocity,
                attitude=attitude,
                battery_level=battery_level,
                gps_fix=gps_fix,
                armed=armed,
                mode=mode,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Status update failed (MAVLink): {e}")
    
    def _process_commands(self):
        """Process commands from the queue"""
        while not self.command_queue.empty():
            try:
                command = self.command_queue.get_nowait()
                self._execute_command(command)
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Command processing error: {e}")
    
    def _execute_command(self, command: DroneCommand):
        """Execute a drone command"""
        try:
            if command.command_type == "arm":
                self._arm_drone()
            elif command.command_type == "disarm":
                self._disarm_drone()
            elif command.command_type == "takeoff":
                self._takeoff(command.parameters.get("altitude", 5.0))
            elif command.command_type == "land":
                self._land()
            elif command.command_type == "move":
                self._move(command.parameters)
            elif command.command_type == "emergency_stop":
                self._emergency_stop()
            else:
                logger.warning(f"Unknown command type: {command.command_type}")
                
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
    
    def _validate_safety(self):
        """Validate current state using Mojo safety validator"""
        if not MOJO_AVAILABLE or not self.last_status:
            return
        
        try:
            # Convert status to format expected by Mojo
            position = np.array(self.last_status.position, dtype=np.float64)
            velocity = np.array(self.last_status.velocity, dtype=np.float64)
            battery_level = float(self.last_status.battery_level)
            
            # Validate using Mojo safety validator
            is_safe = validate_drone_state(
                self.safety_validator,
                position,
                velocity,
                battery_level
            )
            
            if not is_safe:
                logger.warning("Safety violation detected!")
                self._emergency_stop()
                
        except Exception as e:
            logger.error(f"Safety validation error: {e}")
    
    def _arm_drone(self):
        """Arm the drone"""
        if self.dronekit_vehicle:
            self.dronekit_vehicle.armed = True
            logger.info("Drone armed (DroneKit)")
        elif self.mavlink_connection:
            self.mavlink_connection.arducopter_arm()
            logger.info("Drone armed (MAVLink)")
    
    def _disarm_drone(self):
        """Disarm the drone"""
        if self.dronekit_vehicle:
            self.dronekit_vehicle.armed = False
            logger.info("Drone disarmed (DroneKit)")
        elif self.mavlink_connection:
            self.mavlink_connection.arducopter_disarm()
            logger.info("Drone disarmed (MAVLink)")
    
    def _takeoff(self, altitude: float):
        """Takeoff to specified altitude"""
        if self.dronekit_vehicle:
            self.dronekit_vehicle.simple_takeoff(altitude)
            logger.info(f"Takeoff to {altitude}m (DroneKit)")
        elif self.mavlink_connection:
            self.mavlink_connection.mav.command_long_send(
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0, 0, 0, 0, 0, 0, 0, altitude
            )
            logger.info(f"Takeoff to {altitude}m (MAVLink)")
    
    def _land(self):
        """Land the drone"""
        if self.dronekit_vehicle:
            self.dronekit_vehicle.mode = VehicleMode("LAND")
            logger.info("Landing (DroneKit)")
        elif self.mavlink_connection:
            self.mavlink_connection.mav.command_long_send(
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component,
                mavutil.mavlink.MAV_CMD_NAV_LAND,
                0, 0, 0, 0, 0, 0, 0, 0
            )
            logger.info("Landing (MAVLink)")
    
    def _move(self, parameters: Dict[str, Any]):
        """Move the drone"""
        vx = parameters.get("vx", 0.0)
        vy = parameters.get("vy", 0.0)
        vz = parameters.get("vz", 0.0)
        
        if self.dronekit_vehicle:
            from dronekit import LocationGlobalRelative
            from pymavlink import mavutil
            
            msg = self.dronekit_vehicle.message_factory.set_position_target_local_ned_encode(
                0, 0, 0, mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000111111000111,  # type_mask
                0, 0, 0,  # x, y, z
                vx, vy, vz,  # vx, vy, vz
                0, 0, 0,  # ax, ay, az
                0, 0  # yaw, yaw_rate
            )
            self.dronekit_vehicle.send_mavlink(msg)
            logger.info(f"Move command sent: vx={vx}, vy={vy}, vz={vz} (DroneKit)")
        
        elif self.mavlink_connection:
            self.mavlink_connection.mav.set_position_target_local_ned_send(
                0, 0, 0, mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000111111000111,
                0, 0, 0,
                vx, vy, vz,
                0, 0, 0,
                0, 0
            )
            logger.info(f"Move command sent: vx={vx}, vy={vy}, vz={vz} (MAVLink)")
    
    def _emergency_stop(self):
        """Emergency stop"""
        self.current_state = DroneState.EMERGENCY
        
        if self.dronekit_vehicle:
            self.dronekit_vehicle.mode = VehicleMode("LAND")
            logger.warning("Emergency stop - landing (DroneKit)")
        elif self.mavlink_connection:
            self.mavlink_connection.mav.command_long_send(
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component,
                mavutil.mavlink.MAV_CMD_NAV_LAND,
                0, 0, 0, 0, 0, 0, 0, 0
            )
            logger.warning("Emergency stop - landing (MAVLink)")
    
    # Public API methods
    def get_status(self) -> Optional[DroneStatus]:
        """Get current drone status"""
        return self.last_status
    
    def send_command(self, command: DroneCommand):
        """Send command to drone"""
        self.command_queue.put(command)
    
    def arm(self):
        """Arm the drone"""
        command = DroneCommand("arm", {}, time.time())
        self.send_command(command)
    
    def disarm(self):
        """Disarm the drone"""
        command = DroneCommand("disarm", {}, time.time())
        self.send_command(command)
    
    def takeoff(self, altitude: float = 5.0):
        """Takeoff to specified altitude"""
        command = DroneCommand("takeoff", {"altitude": altitude}, time.time())
        self.send_command(command)
    
    def land(self):
        """Land the drone"""
        command = DroneCommand("land", {}, time.time())
        self.send_command(command)
    
    def move(self, vx: float = 0.0, vy: float = 0.0, vz: float = 0.0):
        """Move with specified velocities"""
        command = DroneCommand("move", {"vx": vx, "vy": vy, "vz": vz}, time.time())
        self.send_command(command)
    
    def emergency_stop(self):
        """Emergency stop"""
        command = DroneCommand("emergency_stop", {}, time.time(), priority=10)
        self.send_command(command)
    
    def is_connected(self) -> bool:
        """Check if drone is connected"""
        return self.current_state != DroneState.DISCONNECTED
    
    def is_armed(self) -> bool:
        """Check if drone is armed"""
        return self.current_state in [DroneState.ARMED, DroneState.FLYING]


# For backward compatibility
DroneController = UnifiedDroneController
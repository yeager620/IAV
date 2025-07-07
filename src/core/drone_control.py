import time
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
import queue

# Fix Python 3.13 compatibility issue with collections
import sys
if sys.version_info >= (3, 10):
    import collections.abc
    collections.MutableMapping = collections.abc.MutableMapping

try:
    from dronekit import connect, VehicleMode, LocationGlobalRelative, APIException
    from pymavlink import mavutil
    DRONEKIT_AVAILABLE = True
except ImportError:
    DRONEKIT_AVAILABLE = False
    print("DroneKit not available - using simulation mode")


class DroneState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    ARMED = "armed"
    FLYING = "flying"
    LANDING = "landing"
    EMERGENCY = "emergency"


@dataclass
class DroneStatus:
    state: DroneState
    position: Tuple[float, float, float]  # lat, lon, alt
    velocity: Tuple[float, float, float]  # vx, vy, vz
    attitude: Tuple[float, float, float]  # roll, pitch, yaw
    battery_level: float
    gps_fix: bool
    armed: bool
    mode: str


@dataclass
class DroneCommand:
    command_type: str
    parameters: Dict
    timestamp: float
    priority: int = 0


class SimulatedDrone:
    """Simulated drone for testing without hardware"""
    def __init__(self):
        self.position = np.array([0.0, 0.0, 0.0])  # x, y, z in meters
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.attitude = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw
        self.armed = False
        self.mode = "STABILIZE"
        self.battery_level = 100.0
        self.last_update = time.time()
        
    def update_physics(self, dt: float, control_input: np.ndarray):
        """Simple physics simulation"""
        # Extract relevant components from 6DOF control input
        if len(control_input) >= 6:
            linear_accel = control_input[:3] * 2.0  # [vx, vy, vz] acceleration
            angular_accel = control_input[3:] * 1.0  # [roll, pitch, yaw] rates
        else:
            linear_accel = np.zeros(3)
            angular_accel = np.zeros(3)
        
        # Update velocity
        self.velocity += linear_accel * dt
        
        # Apply drag
        self.velocity *= 0.95
        
        # Update position
        self.position += self.velocity * dt
        
        # Update attitude (simplified)
        self.attitude += angular_accel * dt
        
        # Battery drain
        self.battery_level -= 0.01 * dt
        
    def get_status(self) -> DroneStatus:
        """Get current drone status"""
        return DroneStatus(
            state=DroneState.FLYING if self.armed else DroneState.CONNECTED,
            position=(self.position[0], self.position[1], self.position[2]),
            velocity=(self.velocity[0], self.velocity[1], self.velocity[2]),
            attitude=(self.attitude[0], self.attitude[1], self.attitude[2]),
            battery_level=self.battery_level,
            gps_fix=True,
            armed=self.armed,
            mode=self.mode
        )


class DroneController:
    def __init__(self, connection_string: Optional[str] = None, simulation_mode: bool = True):
        self.connection_string = connection_string
        self.simulation_mode = simulation_mode or not DRONEKIT_AVAILABLE
        self.vehicle = None
        self.simulated_drone = None
        
        if self.simulation_mode:
            self.simulated_drone = SimulatedDrone()
            
        self.command_queue = queue.Queue()
        self.status_history = []
        self.control_thread = None
        self.is_running = False
        
        # Safety limits
        self.max_velocity = 5.0  # m/s
        self.max_altitude = 50.0  # meters
        self.min_battery = 20.0  # percent
        
        # Emergency stop flag
        self.emergency_stop = False
        
    def connect(self) -> bool:
        """Connect to drone"""
        try:
            if self.simulation_mode:
                print("Connected to simulated drone")
                return True
                
            if not DRONEKIT_AVAILABLE:
                print("DroneKit not available - switching to simulation mode")
                self.simulation_mode = True
                self.simulated_drone = SimulatedDrone()
                return True
                
            self.vehicle = connect(self.connection_string, wait_ready=True)
            print(f"Connected to drone: {self.vehicle.version}")
            return True
            
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
            
    def disconnect(self):
        """Disconnect from drone"""
        if self.vehicle:
            self.vehicle.close()
            self.vehicle = None
            
    def start_control_loop(self):
        """Start the control loop in a separate thread"""
        if self.is_running:
            return
            
        self.is_running = True
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        
    def stop_control_loop(self):
        """Stop the control loop"""
        self.is_running = False
        if self.control_thread:
            self.control_thread.join(timeout=1.0)
            
    def _control_loop(self):
        """Main control loop"""
        last_time = time.time()
        
        while self.is_running:
            current_time = time.time()
            dt = current_time - last_time
            
            # Process commands
            self._process_commands()
            
            # Update simulation if in simulation mode
            if self.simulation_mode and self.simulated_drone:
                # Get current control input (simplified)
                control_input = np.zeros(6)  # [vx, vy, vz, roll_rate, pitch_rate, yaw_rate]
                self.simulated_drone.update_physics(dt, control_input)
                
            # Update status history
            status = self.get_status()
            self.status_history.append(status)
            
            # Keep only recent history
            if len(self.status_history) > 100:
                self.status_history.pop(0)
                
            # Check safety conditions
            self._check_safety()
            
            last_time = current_time
            time.sleep(0.1)  # 10Hz control loop
            
    def _process_commands(self):
        """Process commands from queue"""
        while not self.command_queue.empty():
            try:
                command = self.command_queue.get_nowait()
                self._execute_command(command)
            except queue.Empty:
                break
                
    def _execute_command(self, command: DroneCommand):
        """Execute a single command"""
        if self.emergency_stop:
            return
            
        try:
            if command.command_type == "takeoff":
                self._takeoff(command.parameters.get("altitude", 5.0))
            elif command.command_type == "land":
                self._land()
            elif command.command_type == "move":
                self._move(command.parameters)
            elif command.command_type == "rotate":
                self._rotate(command.parameters)
            elif command.command_type == "emergency_stop":
                self._emergency_stop()
            elif command.command_type == "arm":
                self._arm()
            elif command.command_type == "disarm":
                self._disarm()
                
        except Exception as e:
            print(f"Command execution failed: {e}")
            
    def _takeoff(self, altitude: float):
        """Take off to specified altitude"""
        if self.simulation_mode:
            self.simulated_drone.armed = True
            self.simulated_drone.mode = "GUIDED"
            print(f"Simulated takeoff to {altitude}m")
        else:
            self.vehicle.mode = VehicleMode("GUIDED")
            self.vehicle.armed = True
            self.vehicle.simple_takeoff(altitude)
            
    def _land(self):
        """Land the drone"""
        if self.simulation_mode:
            self.simulated_drone.mode = "LAND"
            print("Simulated landing")
        else:
            self.vehicle.mode = VehicleMode("LAND")
            
    def _move(self, params: Dict):
        """Move drone with velocity commands"""
        vx = params.get("vx", 0.0)
        vy = params.get("vy", 0.0)
        vz = params.get("vz", 0.0)
        
        # Apply safety limits
        vx = np.clip(vx, -self.max_velocity, self.max_velocity)
        vy = np.clip(vy, -self.max_velocity, self.max_velocity)
        vz = np.clip(vz, -self.max_velocity, self.max_velocity)
        
        if self.simulation_mode:
            control_input = np.array([vx, vy, vz, 0, 0, 0])
            self.simulated_drone.update_physics(0.1, control_input)
        else:
            # Send velocity command to real drone
            msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
                0, 0, 0, mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000111111000111, 0, 0, 0, vx, vy, vz, 0, 0, 0, 0, 0
            )
            self.vehicle.send_mavlink(msg)
            
    def _rotate(self, params: Dict):
        """Rotate drone"""
        yaw_rate = params.get("yaw_rate", 0.0)
        
        if self.simulation_mode:
            control_input = np.array([0, 0, 0, 0, 0, yaw_rate])
            self.simulated_drone.update_physics(0.1, control_input)
        else:
            # Send yaw rate command
            msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
                0, 0, 0, mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000111111111000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, yaw_rate
            )
            self.vehicle.send_mavlink(msg)
            
    def _arm(self):
        """Arm the drone"""
        if self.simulation_mode:
            self.simulated_drone.armed = True
        else:
            self.vehicle.armed = True
            
    def _disarm(self):
        """Disarm the drone"""
        if self.simulation_mode:
            self.simulated_drone.armed = False
        else:
            self.vehicle.armed = False
            
    def _emergency_stop(self):
        """Emergency stop - immediately stop all movement"""
        self.emergency_stop = True
        
        if self.simulation_mode:
            self.simulated_drone.velocity = np.zeros(3)
            self.simulated_drone.mode = "LAND"
        else:
            self.vehicle.mode = VehicleMode("LAND")
            
    def _check_safety(self):
        """Check safety conditions"""
        status = self.get_status()
        
        # Check battery level
        if status.battery_level < self.min_battery:
            print("Low battery - initiating emergency landing")
            self.add_command("land", {})
            
        # Check altitude
        if status.position[2] > self.max_altitude:
            print("Maximum altitude exceeded - stopping ascent")
            self.add_command("move", {"vx": 0, "vy": 0, "vz": -1})
            
    def get_status(self) -> DroneStatus:
        """Get current drone status"""
        if self.simulation_mode:
            return self.simulated_drone.get_status()
        else:
            if not self.vehicle:
                return DroneStatus(
                    state=DroneState.DISCONNECTED,
                    position=(0, 0, 0),
                    velocity=(0, 0, 0),
                    attitude=(0, 0, 0),
                    battery_level=0,
                    gps_fix=False,
                    armed=False,
                    mode="UNKNOWN"
                )
                
            return DroneStatus(
                state=DroneState.FLYING if self.vehicle.armed else DroneState.CONNECTED,
                position=(self.vehicle.location.global_frame.lat,
                         self.vehicle.location.global_frame.lon,
                         self.vehicle.location.global_frame.alt),
                velocity=(self.vehicle.velocity[0] if self.vehicle.velocity else 0,
                         self.vehicle.velocity[1] if self.vehicle.velocity else 0,
                         self.vehicle.velocity[2] if self.vehicle.velocity else 0),
                attitude=(self.vehicle.attitude.roll,
                         self.vehicle.attitude.pitch,
                         self.vehicle.attitude.yaw),
                battery_level=self.vehicle.battery.level if self.vehicle.battery else 0,
                gps_fix=self.vehicle.gps_0.fix_type >= 3,
                armed=self.vehicle.armed,
                mode=str(self.vehicle.mode)
            )
            
    def add_command(self, command_type: str, parameters: Dict, priority: int = 0):
        """Add command to execution queue"""
        command = DroneCommand(
            command_type=command_type,
            parameters=parameters,
            timestamp=time.time(),
            priority=priority
        )
        
        self.command_queue.put(command)
        
    def execute_action_vector(self, action: np.ndarray):
        """Execute action vector from neural network"""
        # Action vector: [vx, vy, vz, roll_rate, pitch_rate, yaw_rate]
        if len(action) != 6:
            raise ValueError("Action vector must have 6 elements")
            
        # Convert to command
        params = {
            "vx": float(action[0]),
            "vy": float(action[1]),
            "vz": float(action[2]),
            "yaw_rate": float(action[5])  # Only use yaw rate for now
        }
        
        self.add_command("move", params)
        
    def cleanup(self):
        """Clean up resources"""
        self.stop_control_loop()
        self.disconnect()
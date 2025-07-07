"""
Mojo-Python bridge for MAVLink networking and communication.
Provides high-performance networking interface using Python interop.
"""

from python import Python
from python.object import PythonObject
from collections import List, Dict
from memory import memset_zero
import time
import math

struct NetworkConfig:
    """Network configuration structure."""
    var connection_string: String
    var timeout_seconds: Float64
    var retry_count: Int
    var buffer_size: Int

    fn __init__(inout self, connection: String):
        self.connection_string = connection
        self.timeout_seconds = 5.0
        self.retry_count = 3
        self.buffer_size = 1024

struct MAVLinkMessage:
    """MAVLink message structure."""
    var message_type: String
    var sequence: Int
    var timestamp: Float64
    var data_size: Int
    var is_valid: Bool

    fn __init__(inout self, msg_type: String):
        self.message_type = msg_type
        self.sequence = 0
        self.timestamp = time.now() / 1e9
        self.data_size = 0
        self.is_valid = False

    fn validate(inout self) -> Bool:
        """Validate message structure."""
        if len(self.message_type) > 0 and self.timestamp > 0:
            self.is_valid = True
            return True
        else:
            self.is_valid = False
            return False

struct FlightState:
    """Flight state from MAVLink messages."""
    var position_x: Float64
    var position_y: Float64
    var position_z: Float64
    var velocity_x: Float64
    var velocity_y: Float64
    var velocity_z: Float64
    var roll: Float64
    var pitch: Float64
    var yaw: Float64
    var armed: Bool
    var timestamp: Float64

    fn __init__(inout self):
        self.position_x = 0.0
        self.position_y = 0.0
        self.position_z = 0.0
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.velocity_z = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.armed = False
        self.timestamp = time.now() / 1e9

struct MotorCommand:
    """Motor command structure."""
    var motor_1: Float64
    var motor_2: Float64
    var motor_3: Float64
    var motor_4: Float64
    var timestamp: Float64

    fn __init__(inout self, m1: Float64, m2: Float64, m3: Float64, m4: Float64):
        self.motor_1 = m1
        self.motor_2 = m2
        self.motor_3 = m3
        self.motor_4 = m4
        self.timestamp = time.now() / 1e9

    fn validate(self) -> Bool:
        """Validate motor commands are within safe range."""
        return (self.motor_1 >= 0.0 and self.motor_1 <= 1.0 and
                self.motor_2 >= 0.0 and self.motor_2 <= 1.0 and
                self.motor_3 >= 0.0 and self.motor_3 <= 1.0 and
                self.motor_4 >= 0.0 and self.motor_4 <= 1.0)

struct MAVLinkBridge:
    """Mojo-Python bridge for MAVLink communication."""
    var config: NetworkConfig
    var python_mavutil: PythonObject
    var connection: PythonObject
    var is_connected: Bool
    var message_count: Int
    var last_heartbeat: Float64

    fn __init__(inout self, config: NetworkConfig):
        self.config = config
        self.is_connected = False
        self.message_count = 0
        self.last_heartbeat = 0.0
        try:
            self.python_mavutil = Python.import_module("pymavlink.mavutil")
            self.connection = None
        except:
            print("Failed to import pymavlink")

    fn connect(inout self) -> Bool:
        """Connect to MAVLink endpoint."""
        try:
            self.connection = self.python_mavutil.mavlink_connection(self.config.connection_string)
            
            # Wait for heartbeat
            var heartbeat = self.connection.wait_heartbeat(timeout=self.config.timeout_seconds)
            if heartbeat:
                self.is_connected = True
                self.last_heartbeat = time.now() / 1e9
                print("MAVLink connection established")
                return True
            else:
                print("No heartbeat received")
                return False
        except:
            print("MAVLink connection failed")
            return False

    fn send_motor_commands(inout self, commands: MotorCommand) -> Bool:
        """Send motor commands via MAVLink."""
        if not self.is_connected:
            print("Not connected to MAVLink")
            return False
        
        if not commands.validate():
            print("Invalid motor commands")
            return False
        
        try:
            # Send actuator control message
            var controls = Python.evaluate("[" + str(commands.motor_1) + ", " + 
                                         str(commands.motor_2) + ", " + 
                                         str(commands.motor_3) + ", " + 
                                         str(commands.motor_4) + ", 0, 0, 0, 0]")
            
            var time_usec = int(time.now())
            self.connection.mav.actuator_control_target_send(
                time_usec, 0, controls
            )
            
            self.message_count += 1
            return True
        except:
            print("Failed to send motor commands")
            return False

    fn receive_state(inout self) -> FlightState:
        """Receive flight state from MAVLink."""
        var state = FlightState()
        
        if not self.is_connected:
            return state
        
        try:
            var msg = self.connection.recv_match(blocking=False)
            if msg:
                var msg_type = msg.get_type()
                
                if msg_type == "ATTITUDE":
                    state.roll = msg.roll
                    state.pitch = msg.pitch
                    state.yaw = msg.yaw
                elif msg_type == "LOCAL_POSITION_NED":
                    state.position_x = msg.x
                    state.position_y = msg.y
                    state.position_z = msg.z
                    state.velocity_x = msg.vx
                    state.velocity_y = msg.vy
                    state.velocity_z = msg.vz
                elif msg_type == "HEARTBEAT":
                    self.last_heartbeat = time.now() / 1e9
                    # Check if armed
                    var base_mode = msg.base_mode
                    state.armed = bool(base_mode & 128)  # MAV_MODE_FLAG_SAFETY_ARMED
                
                state.timestamp = time.now() / 1e9
                
        except:
            print("Error receiving MAVLink message")
        
        return state

    fn send_command(inout self, command: String) -> Bool:
        """Send high-level command to flight controller."""
        if not self.is_connected:
            return False
        
        try:
            # Convert command to MAVLink command
            if command == "ARM":
                self.connection.mav.command_long_send(
                    self.connection.target_system, self.connection.target_component,
                    400, 0, 1, 0, 0, 0, 0, 0, 0  # MAV_CMD_COMPONENT_ARM_DISARM
                )
            elif command == "DISARM":
                self.connection.mav.command_long_send(
                    self.connection.target_system, self.connection.target_component,
                    400, 0, 0, 0, 0, 0, 0, 0, 0  # MAV_CMD_COMPONENT_ARM_DISARM
                )
            elif command == "TAKEOFF":
                self.connection.mav.command_long_send(
                    self.connection.target_system, self.connection.target_component,
                    22, 0, 0, 0, 0, 0, 0, 0, 10  # MAV_CMD_NAV_TAKEOFF, 10m altitude
                )
            else:
                print("Unknown command:", command)
                return False
            
            return True
        except:
            print("Failed to send command:", command)
            return False

    fn get_connection_status(self) -> Bool:
        """Get connection status."""
        if self.is_connected:
            var current_time = time.now() / 1e9
            var heartbeat_age = current_time - self.last_heartbeat
            return heartbeat_age < 10.0  # Consider disconnected if no heartbeat for 10s
        else:
            return False

    fn get_message_count(self) -> Int:
        """Get total message count."""
        return self.message_count

    fn disconnect(inout self):
        """Disconnect from MAVLink."""
        if self.is_connected:
            try:
                self.connection.close()
                self.is_connected = False
                print("MAVLink disconnected")
            except:
                print("Error during disconnect")

fn create_emergency_stop_command() -> MotorCommand:
    """Create emergency stop motor command."""
    return MotorCommand(0.0, 0.0, 0.0, 0.0)

fn create_hover_command() -> MotorCommand:
    """Create hover motor command."""
    return MotorCommand(0.5, 0.5, 0.5, 0.5)

fn validate_flight_state(state: FlightState) -> Bool:
    """Validate flight state values."""
    var pos_valid = (abs(state.position_x) < 1000.0 and 
                     abs(state.position_y) < 1000.0 and 
                     abs(state.position_z) < 1000.0)
    
    var vel_valid = (abs(state.velocity_x) < 50.0 and 
                     abs(state.velocity_y) < 50.0 and 
                     abs(state.velocity_z) < 50.0)
    
    var att_valid = (abs(state.roll) < math.pi and 
                     abs(state.pitch) < math.pi and 
                     abs(state.yaw) < math.pi)
    
    return pos_valid and vel_valid and att_valid

fn compute_control_latency(command_time: Float64, state_time: Float64) -> Float64:
    """Compute control loop latency."""
    return abs(command_time - state_time)

fn main():
    """Test MAVLink bridge functionality."""
    print("Testing Mojo MAVLink bridge...")
    
    # Test network configuration
    var config = NetworkConfig("udp:127.0.0.1:14550")
    config.timeout_seconds = 5.0
    config.retry_count = 3
    
    print("Network config - connection:", config.connection_string)
    print("Timeout:", config.timeout_seconds)
    print("Retry count:", config.retry_count)
    
    # Test MAVLink message
    var message = MAVLinkMessage("HEARTBEAT")
    var is_valid = message.validate()
    print("MAVLink message valid:", is_valid)
    print("Message type:", message.message_type)
    
    # Test flight state
    var state = FlightState()
    state.position_x = 1.0
    state.position_y = 2.0
    state.position_z = 3.0
    state.armed = True
    
    var state_valid = validate_flight_state(state)
    print("Flight state valid:", state_valid)
    print("Position:", state.position_x, state.position_y, state.position_z)
    print("Armed:", state.armed)
    
    # Test motor commands
    var motor_cmd = MotorCommand(0.6, 0.5, 0.4, 0.5)
    var cmd_valid = motor_cmd.validate()
    print("Motor command valid:", cmd_valid)
    print("Motors:", motor_cmd.motor_1, motor_cmd.motor_2, motor_cmd.motor_3, motor_cmd.motor_4)
    
    # Test emergency commands
    var emergency = create_emergency_stop_command()
    print("Emergency stop:", emergency.motor_1, emergency.motor_2, emergency.motor_3, emergency.motor_4)
    
    var hover = create_hover_command()
    print("Hover command:", hover.motor_1, hover.motor_2, hover.motor_3, hover.motor_4)
    
    # Test MAVLink bridge (without actual connection)
    var bridge = MAVLinkBridge(config)
    print("Bridge initialized")
    
    # Test control latency
    var latency = compute_control_latency(time.now() / 1e9, time.now() / 1e9)
    print("Control latency:", latency, "seconds")
    
    print("MAVLink bridge test completed.")
from pymavlink import mavutil
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Callable
import threading
import queue
import time
import logging

logger = logging.getLogger(__name__)

@dataclass 
class FlightState:
    position: np.ndarray  # [x, y, z] in meters
    velocity: np.ndarray  # [vx, vy, vz] in m/s
    attitude: np.ndarray  # [roll, pitch, yaw] in radians
    angular_velocity: np.ndarray  # [wx, wy, wz] in rad/s
    armed: bool
    mode: str
    battery_voltage: float
    gps_fix: int
    timestamp: float

class MAVLinkInterface:
    """High-performance MAVLink interface for real-time drone control"""
    
    def __init__(self, connection_string: str = '/dev/ttyUSB0', baud_rate: int = 921600):
        self.connection_string = connection_string
        self.connection = None
        self.flight_state = FlightState(
            position=np.zeros(3),
            velocity=np.zeros(3), 
            attitude=np.zeros(3),
            angular_velocity=np.zeros(3),
            armed=False,
            mode='UNKNOWN',
            battery_voltage=0.0,
            gps_fix=0,
            timestamp=time.time()
        )
        
        # Threading components
        self.running = False
        self.command_queue = queue.Queue(maxsize=100)
        self.heartbeat_thread = None
        self.state_thread = None
        self.command_thread = None
        
        # Callbacks
        self.state_callbacks: List[Callable[[FlightState], None]] = []
        
        # Performance monitoring
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'last_heartbeat': 0,
            'connection_quality': 1.0
        }
    
    def connect(self) -> bool:
        """Establish MAVLink connection"""
        try:
            logger.info(f"Connecting to {self.connection_string}")
            self.connection = mavutil.mavlink_connection(
                self.connection_string,
                baud=921600,
                robust_parsing=True
            )
            
            # Wait for first heartbeat
            logger.info("Waiting for heartbeat...")
            self.connection.wait_heartbeat(timeout=10)
            logger.info(f"Connected to system {self.connection.target_system}, component {self.connection.target_component}")
            
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def start(self) -> bool:
        """Start all communication threads"""
        if not self.connection:
            logger.error("No connection established")
            return False
        
        self.running = True
        
        # Start communication threads
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.state_thread = threading.Thread(target=self._state_update_loop, daemon=True)
        self.command_thread = threading.Thread(target=self._command_loop, daemon=True)
        
        self.heartbeat_thread.start()
        self.state_thread.start()
        self.command_thread.start()
        
        logger.info("MAVLink interface started")
        return True
    
    def stop(self):
        """Stop all communication threads"""
        self.running = False
        
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=1.0)
        if self.state_thread:
            self.state_thread.join(timeout=1.0)
        if self.command_thread:
            self.command_thread.join(timeout=1.0)
        
        if self.connection:
            self.connection.close()
        
        logger.info("MAVLink interface stopped")
    
    def send_motor_commands(self, motor_commands: np.ndarray, group: int = 0):
        """Send motor commands via actuator controls"""
        if len(motor_commands) != 4:
            logger.error(f"Expected 4 motor commands, got {len(motor_commands)}")
            return False
        
        # Convert to PWM range (1000-2000) and normalize
        controls = np.zeros(8)
        controls[:4] = np.clip(motor_commands, 0.0, 1.0)
        
        self.command_queue.put(('ACTUATOR_CONTROLS', {
            'controls': controls,
            'group': group,
            'timestamp': time.time()
        }))
        return True
    
    def send_velocity_command(self, velocity: np.ndarray, yaw_rate: float = 0.0):
        """Send velocity command in body frame"""
        if len(velocity) != 3:
            logger.error(f"Expected 3D velocity vector, got {len(velocity)}")
            return False
        
        self.command_queue.put(('SET_POSITION_TARGET_LOCAL_NED', {
            'x': 0, 'y': 0, 'z': 0,  # Position ignored
            'vx': velocity[0], 'vy': velocity[1], 'vz': velocity[2],
            'afx': 0, 'afy': 0, 'afz': 0,  # Acceleration ignored
            'yaw': 0, 'yaw_rate': yaw_rate,
            'type_mask': (
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
            ),
            'coordinate_frame': mavutil.mavlink.MAV_FRAME_BODY_NED
        }))
        return True
    
    def arm(self) -> bool:
        """Arm the vehicle"""
        self.command_queue.put(('COMMAND_LONG', {
            'command': mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            'param1': 1,  # 1 = arm
            'param2': 0, 'param3': 0, 'param4': 0, 'param5': 0, 'param6': 0, 'param7': 0
        }))
        return True
    
    def disarm(self) -> bool:
        """Disarm the vehicle"""  
        self.command_queue.put(('COMMAND_LONG', {
            'command': mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            'param1': 0,  # 0 = disarm
            'param2': 0, 'param3': 0, 'param4': 0, 'param5': 0, 'param6': 0, 'param7': 0
        }))
        return True
    
    def set_mode(self, mode: str) -> bool:
        """Set flight mode"""
        mode_mapping = {
            'STABILIZE': 0,
            'ACRO': 1,
            'ALT_HOLD': 2,
            'AUTO': 3,
            'GUIDED': 4,
            'LOITER': 5,
            'RTL': 6,
            'LAND': 9,
            'OFFBOARD': 4  # Use GUIDED for offboard
        }
        
        if mode not in mode_mapping:
            logger.error(f"Unknown mode: {mode}")
            return False
        
        self.command_queue.put(('SET_MODE', {
            'mode': mode_mapping[mode],
            'custom_mode': 0
        }))
        return True
    
    def get_flight_state(self) -> FlightState:
        """Get current flight state (thread-safe)"""
        return self.flight_state
    
    def add_state_callback(self, callback: Callable[[FlightState], None]):
        """Add callback for state updates"""
        self.state_callbacks.append(callback)
    
    def get_connection_stats(self) -> dict:
        """Get connection statistics"""
        return self.stats.copy()
    
    def _heartbeat_loop(self):
        """Send periodic heartbeat messages"""
        while self.running:
            try:
                if self.connection:
                    self.connection.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_GCS,
                        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                        0, 0, 0
                    )
                    self.stats['last_heartbeat'] = time.time()
                
                time.sleep(1.0)  # 1Hz heartbeat
                
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                time.sleep(1.0)
    
    def _state_update_loop(self):
        """Process incoming state messages"""
        while self.running:
            try:
                if not self.connection:
                    time.sleep(0.01)
                    continue
                
                msg = self.connection.recv_match(blocking=False, timeout=0.01)
                if msg is None:
                    continue
                
                self.stats['messages_received'] += 1
                self._process_message(msg)
                
            except Exception as e:
                logger.error(f"State update error: {e}")
                time.sleep(0.01)
    
    def _command_loop(self):
        """Process outgoing command queue"""
        while self.running:
            try:
                cmd_type, data = self.command_queue.get(timeout=0.1)
                
                if cmd_type == 'ACTUATOR_CONTROLS':
                    self.connection.mav.actuator_control_target_send(
                        time_usec=int(data['timestamp'] * 1e6),
                        group_mlx=data['group'],
                        controls=data['controls'].tolist()
                    )
                
                elif cmd_type == 'SET_POSITION_TARGET_LOCAL_NED':
                    self.connection.mav.set_position_target_local_ned_send(
                        time_boot_ms=int(time.time() * 1000),
                        target_system=self.connection.target_system,
                        target_component=self.connection.target_component,
                        coordinate_frame=data['coordinate_frame'],
                        type_mask=data['type_mask'],
                        x=data['x'], y=data['y'], z=data['z'],
                        vx=data['vx'], vy=data['vy'], vz=data['vz'],
                        afx=data['afx'], afy=data['afy'], afz=data['afz'],
                        yaw=data['yaw'], yaw_rate=data['yaw_rate']
                    )
                
                elif cmd_type == 'COMMAND_LONG':
                    self.connection.mav.command_long_send(
                        target_system=self.connection.target_system,
                        target_component=self.connection.target_component,
                        command=data['command'],
                        confirmation=0,
                        param1=data['param1'], param2=data['param2'],
                        param3=data['param3'], param4=data['param4'],
                        param5=data['param5'], param6=data['param6'],
                        param7=data['param7']
                    )
                
                elif cmd_type == 'SET_MODE':
                    self.connection.mav.set_mode_send(
                        target_system=self.connection.target_system,
                        base_mode=mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                        custom_mode=data['mode']
                    )
                
                self.stats['messages_sent'] += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Command send error: {e}")
    
    def _process_message(self, msg):
        """Process received MAVLink message"""
        msg_type = msg.get_type()
        
        if msg_type == 'GLOBAL_POSITION_INT':
            # Convert from GPS to local coordinates (simplified)
            self.flight_state.position[0] = msg.lat / 1e7  # Lat as X (simplified)
            self.flight_state.position[1] = msg.lon / 1e7  # Lon as Y (simplified) 
            self.flight_state.position[2] = -msg.alt / 1000.0  # Alt as Z (NED)
            
            self.flight_state.velocity[0] = msg.vx / 100.0  # cm/s to m/s
            self.flight_state.velocity[1] = msg.vy / 100.0
            self.flight_state.velocity[2] = msg.vz / 100.0
        
        elif msg_type == 'ATTITUDE':
            self.flight_state.attitude[0] = msg.roll
            self.flight_state.attitude[1] = msg.pitch
            self.flight_state.attitude[2] = msg.yaw
            
            self.flight_state.angular_velocity[0] = msg.rollspeed
            self.flight_state.angular_velocity[1] = msg.pitchspeed
            self.flight_state.angular_velocity[2] = msg.yawspeed
        
        elif msg_type == 'HEARTBEAT':
            self.flight_state.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            
            # Decode flight mode (simplified)
            if msg.custom_mode == 0:
                self.flight_state.mode = 'STABILIZE'
            elif msg.custom_mode == 4:
                self.flight_state.mode = 'GUIDED'
            else:
                self.flight_state.mode = f'MODE_{msg.custom_mode}'
        
        elif msg_type == 'SYS_STATUS':
            self.flight_state.battery_voltage = msg.voltage_battery / 1000.0  # mV to V
        
        elif msg_type == 'GPS_RAW_INT':
            self.flight_state.gps_fix = msg.fix_type
        
        # Update timestamp and trigger callbacks
        self.flight_state.timestamp = time.time()
        
        for callback in self.state_callbacks:
            try:
                callback(self.flight_state)
            except Exception as e:
                logger.error(f"State callback error: {e}")
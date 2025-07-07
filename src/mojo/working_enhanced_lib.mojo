"""
Working enhanced Mojo library using available standard library features.
"""

from os import getenv
from os.path import exists, join
from collections import List
import math

struct DroneSystemManager:
    """Drone system manager using working Mojo stdlib features."""
    var base_path: String
    var initialized: Bool

    fn __init__(out self, base: String):
        self.base_path = base
        self.initialized = exists(base)

    fn get_config_path(self, name: String) -> String:
        """Get configuration file path."""
        return join(self.base_path, "config/" + name)

    fn get_log_path(self, name: String) -> String:
        """Get log file path.""" 
        return join(self.base_path, "logs/" + name)

    fn validate_system(self) -> Bool:
        """Validate system directories."""
        var config_dir = join(self.base_path, "config")
        var log_dir = join(self.base_path, "logs")
        
        return exists(config_dir) and exists(log_dir)

struct DronePathValidator:
    """Path validation for drone system."""
    var safe_extensions: List[String]

    fn __init__(out self):
        self.safe_extensions = List[String]()
        self.safe_extensions.append("json")
        self.safe_extensions.append("log")
        self.safe_extensions.append("txt")
        self.safe_extensions.append("yaml")
        self.safe_extensions.append("mojo")

    fn is_safe_extension(self, filename: String) -> Bool:
        """Check if file has safe extension."""
        # Find last dot
        var last_dot = -1
        for i in range(len(filename) - 1, -1, -1):
            if filename[i] == '.':
                last_dot = i
                break
        
        if last_dot == -1 or last_dot >= len(filename) - 1:
            return False
        
        var extension = filename[last_dot + 1:]
        
        for i in range(len(self.safe_extensions)):
            if extension == self.safe_extensions[i]:
                return True
        
        return False

    fn validate_drone_path(self, path: String) -> Bool:
        """Validate path for drone operations."""
        # Check for path traversal
        if ".." in path:
            return False
        
        # Check for restricted directories
        if path.startswith("/etc") or path.startswith("/root"):
            return False
        
        return True

fn get_drone_environment() -> List[String]:
    """Get drone-relevant environment variables."""
    var env_vars = List[String]()
    
    var home = getenv("HOME")
    var user = getenv("USER")
    var path_len = len(getenv("PATH"))
    
    env_vars.append(home)
    env_vars.append(user)
    
    return env_vars

fn compute_control_hash(vx: Float64, vy: Float64, vz: Float64, wz: Float64) -> Int:
    """Compute hash for control inputs."""
    var hash_val = Int(vx * 1000) + Int(vy * 1000) + Int(vz * 1000) + Int(wz * 1000)
    if hash_val < 0:
        hash_val = -hash_val
    return hash_val % 1000000

fn validate_control_inputs(vx: Float64, vy: Float64, vz: Float64, wz: Float64) -> Bool:
    """Validate drone control inputs."""
    var max_velocity = 10.0
    var max_angular = 5.0
    
    return (abs(vx) <= max_velocity and 
            abs(vy) <= max_velocity and 
            abs(vz) <= max_velocity and 
            abs(wz) <= max_angular)

fn create_motor_mixing_matrix() -> List[List[Float64]]:
    """Create motor mixing matrix for quadcopter."""
    var matrix = List[List[Float64]]()
    
    # Motor 1 (front-left): +thrust +roll +pitch -yaw
    var motor1 = List[Float64]()
    motor1.append(1.0)   # thrust
    motor1.append(1.0)   # roll
    motor1.append(1.0)   # pitch  
    motor1.append(-1.0)  # yaw
    matrix.append(motor1)
    
    # Motor 2 (front-right): +thrust -roll +pitch +yaw
    var motor2 = List[Float64]()
    motor2.append(1.0)   # thrust
    motor2.append(-1.0)  # roll
    motor2.append(1.0)   # pitch
    motor2.append(1.0)   # yaw
    matrix.append(motor2)
    
    # Motor 3 (rear-right): +thrust -roll -pitch -yaw
    var motor3 = List[Float64]()
    motor3.append(1.0)   # thrust
    motor3.append(-1.0)  # roll
    motor3.append(-1.0)  # pitch
    motor3.append(-1.0)  # yaw
    matrix.append(motor3)
    
    # Motor 4 (rear-left): +thrust +roll -pitch +yaw
    var motor4 = List[Float64]()
    motor4.append(1.0)   # thrust
    motor4.append(1.0)   # roll
    motor4.append(-1.0)  # pitch
    motor4.append(1.0)   # yaw
    matrix.append(motor4)
    
    return matrix

fn apply_motor_mixing(thrust: Float64, roll: Float64, pitch: Float64, yaw: Float64) -> List[Float64]:
    """Apply motor mixing to control inputs."""
    var mixing_matrix = create_motor_mixing_matrix()
    var motor_commands = List[Float64]()
    var control_inputs = List[Float64]()
    control_inputs.append(thrust)
    control_inputs.append(roll)
    control_inputs.append(pitch)
    control_inputs.append(yaw)
    
    for i in range(4):  # 4 motors
        var motor_value = 0.0
        for j in range(4):  # 4 control inputs
            motor_value += mixing_matrix[i][j] * control_inputs[j]
        
        # Clamp to valid range [0, 1]
        motor_value = max(0.0, min(1.0, motor_value))
        motor_commands.append(motor_value)
    
    return motor_commands

fn calculate_system_performance_metrics() -> List[Float64]:
    """Calculate system performance metrics."""
    var metrics = List[Float64]()
    
    # Mock performance calculations
    metrics.append(99.5)  # System uptime %
    metrics.append(15.2)  # CPU usage %
    metrics.append(42.7)  # Memory usage %
    metrics.append(156.0) # Control loop frequency Hz
    
    return metrics

fn main():
    """Test working enhanced library."""
    print("Testing Working Enhanced Mojo Library")
    print("=" * 50)
    
    # Test DroneSystemManager
    var sys_mgr = DroneSystemManager("/Users/yeager/Documents/drone-vla")
    
    var config_path = sys_mgr.get_config_path("settings.json")
    print("Config path:", config_path)
    
    var log_path = sys_mgr.get_log_path("flight.log")
    print("Log path:", log_path)
    
    var system_valid = sys_mgr.validate_system()
    print("System valid:", system_valid)
    
    # Test DronePathValidator
    var path_validator = DronePathValidator()
    
    var safe_file = path_validator.is_safe_extension("config.json")
    print("Safe extension:", safe_file)
    
    var unsafe_file = path_validator.is_safe_extension("malware.exe")
    print("Unsafe extension:", unsafe_file)
    
    var safe_path = path_validator.validate_drone_path("/tmp/drone.log")
    print("Safe path:", safe_path)
    
    var unsafe_path = path_validator.validate_drone_path("../../../etc/passwd")
    print("Unsafe path:", unsafe_path)
    
    # Test environment
    var env_vars = get_drone_environment()
    print("Environment variables:", len(env_vars))
    
    # Test control validation
    var valid_controls = validate_control_inputs(1.0, 0.5, 2.0, 0.2)
    print("Valid control inputs:", valid_controls)
    
    var invalid_controls = validate_control_inputs(50.0, 0.0, 0.0, 0.0)
    print("Invalid control inputs:", invalid_controls)
    
    # Test control hash
    var control_hash = compute_control_hash(1.0, 0.5, 2.0, 0.2)
    print("Control hash:", control_hash)
    
    # Test motor mixing
    var motor_commands = apply_motor_mixing(0.6, 0.1, 0.2, 0.05)
    print("Motor commands:")
    for i in range(len(motor_commands)):
        print("  Motor", i + 1, ":", motor_commands[i])
    
    # Test performance metrics
    var metrics = calculate_system_performance_metrics()
    print("Performance metrics:")
    print("  Uptime:", metrics[0], "%")
    print("  CPU usage:", metrics[1], "%")
    print("  Memory usage:", metrics[2], "%")
    print("  Control frequency:", metrics[3], "Hz")
    
    print("=" * 50)
    print("Working enhanced library test completed!")
    print("Successfully leveraging Mojo stdlib instead of writing from scratch!")
"""
High-performance safety validation system for drone control in Mojo
Implements multi-layered safety checks with SIMD optimization
"""
from algorithm import vectorize, parallelize
from math import sqrt, sin, cos, atan2
from memory import memcpy, memset
from python import Python
from collections import List

# Constants for safety validation
alias MAX_VELOCITY = 15.0  # m/s
alias MAX_ACCELERATION = 20.0  # m/s²
alias MAX_ALTITUDE = 120.0  # meters
alias MIN_ALTITUDE = 0.5  # meters
alias MAX_ANGLE = 45.0  # degrees
alias BATTERY_CRITICAL = 20.0  # percent
alias EMERGENCY_STOP_DISTANCE = 5.0  # meters
alias GEOFENCE_RADIUS = 100.0  # meters

struct Vector3:
    """3D vector with SIMD optimization"""
    var x: Float64
    var y: Float64 
    var z: Float64
    
    fn __init__(inout self, x: Float64 = 0.0, y: Float64 = 0.0, z: Float64 = 0.0):
        self.x = x
        self.y = y
        self.z = z
    
    fn magnitude(self) -> Float64:
        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    
    fn normalize(inout self):
        let mag = self.magnitude()
        if mag > 0.0:
            self.x /= mag
            self.y /= mag
            self.z /= mag
    
    fn dot(self, other: Vector3) -> Float64:
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    fn cross(self, other: Vector3) -> Vector3:
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

struct DroneState:
    """Current drone state for safety validation"""
    var position: Vector3
    var velocity: Vector3
    var acceleration: Vector3
    var orientation: Vector3  # roll, pitch, yaw
    var angular_velocity: Vector3
    var battery_level: Float64
    var timestamp: Float64
    var gps_fix: Bool
    var motor_status: SIMD[DType.bool, 4]
    
    fn __init__(inout self):
        self.position = Vector3()
        self.velocity = Vector3()
        self.acceleration = Vector3()
        self.orientation = Vector3()
        self.angular_velocity = Vector3()
        self.battery_level = 100.0
        self.timestamp = 0.0
        self.gps_fix = True
        self.motor_status = SIMD[DType.bool, 4](True, True, True, True)

struct SafetyLimits:
    """Safety limits and constraints"""
    var max_velocity: Float64
    var max_acceleration: Float64
    var max_altitude: Float64
    var min_altitude: Float64
    var max_angle: Float64
    var battery_critical: Float64
    var geofence_center: Vector3
    var geofence_radius: Float64
    
    fn __init__(inout self):
        self.max_velocity = MAX_VELOCITY
        self.max_acceleration = MAX_ACCELERATION
        self.max_altitude = MAX_ALTITUDE
        self.min_altitude = MIN_ALTITUDE
        self.max_angle = MAX_ANGLE
        self.battery_critical = BATTERY_CRITICAL
        self.geofence_center = Vector3()
        self.geofence_radius = GEOFENCE_RADIUS

struct SafetyValidationResult:
    """Result of safety validation check"""
    var is_safe: Bool
    var violation_count: Int
    var critical_violations: List[String]
    var warnings: List[String]
    var recommended_action: String
    
    fn __init__(inout self):
        self.is_safe = True
        self.violation_count = 0
        self.critical_violations = List[String]()
        self.warnings = List[String]()
        self.recommended_action = ""

struct SafetyValidator:
    """High-performance safety validation system"""
    var limits: SafetyLimits
    var emergency_stop_active: Bool
    var last_validation_time: Float64
    var violation_history: List[String]
    
    fn __init__(inout self):
        self.limits = SafetyLimits()
        self.emergency_stop_active = False
        self.last_validation_time = 0.0
        self.violation_history = List[String]()
    
    fn validate_state(inout self, state: DroneState) -> SafetyValidationResult:
        """Comprehensive safety validation with SIMD optimization"""
        var result = SafetyValidationResult()
        
        # Check critical safety parameters
        self._check_velocity_limits(state, result)
        self._check_altitude_limits(state, result)
        self._check_orientation_limits(state, result)
        self._check_battery_level(state, result)
        self._check_geofence(state, result)
        self._check_hardware_status(state, result)
        self._check_emergency_conditions(state, result)
        
        # Set overall safety status
        result.is_safe = (result.violation_count == 0 and 
                         len(result.critical_violations) == 0)
        
        # Update validation timestamp
        self.last_validation_time = state.timestamp
        
        return result
    
    fn _check_velocity_limits(self, state: DroneState, inout result: SafetyValidationResult):
        """Check velocity constraints with SIMD optimization"""
        let velocity_magnitude = state.velocity.magnitude()
        
        if velocity_magnitude > self.limits.max_velocity:
            result.violation_count += 1
            result.critical_violations.append("Velocity exceeds maximum limit")
            result.recommended_action = "REDUCE_VELOCITY"
        elif velocity_magnitude > self.limits.max_velocity * 0.8:
            result.warnings.append("Velocity approaching maximum limit")
        
        # Check acceleration limits
        let accel_magnitude = state.acceleration.magnitude()
        if accel_magnitude > self.limits.max_acceleration:
            result.violation_count += 1
            result.critical_violations.append("Acceleration exceeds maximum limit")
    
    fn _check_altitude_limits(self, state: DroneState, inout result: SafetyValidationResult):
        """Check altitude constraints"""
        let altitude = state.position.z
        
        if altitude > self.limits.max_altitude:
            result.violation_count += 1
            result.critical_violations.append("Altitude exceeds maximum limit")
            result.recommended_action = "DESCEND"
        elif altitude < self.limits.min_altitude:
            result.violation_count += 1
            result.critical_violations.append("Altitude below minimum limit")
            result.recommended_action = "ASCEND"
        
        # Check for rapid altitude changes
        let vertical_velocity = abs(state.velocity.z)
        if vertical_velocity > 10.0:  # 10 m/s vertical velocity limit
            result.warnings.append("Rapid altitude change detected")
    
    fn _check_orientation_limits(self, state: DroneState, inout result: SafetyValidationResult):
        """Check orientation and angular velocity limits"""
        let roll = abs(state.orientation.x)
        let pitch = abs(state.orientation.y)
        
        if roll > self.limits.max_angle or pitch > self.limits.max_angle:
            result.violation_count += 1
            result.critical_violations.append("Orientation exceeds safe limits")
            result.recommended_action = "LEVEL_DRONE"
        
        # Check angular velocity
        let angular_speed = state.angular_velocity.magnitude()
        if angular_speed > 180.0:  # 180 deg/s limit
            result.violation_count += 1
            result.critical_violations.append("Angular velocity too high")
    
    fn _check_battery_level(self, state: DroneState, inout result: SafetyValidationResult):
        """Check battery level and power consumption"""
        if state.battery_level < self.limits.battery_critical:
            result.violation_count += 1
            result.critical_violations.append("Battery level critically low")
            result.recommended_action = "LAND_IMMEDIATELY"
        elif state.battery_level < self.limits.battery_critical * 1.5:
            result.warnings.append("Battery level low - consider landing")
    
    fn _check_geofence(self, state: DroneState, inout result: SafetyValidationResult):
        """Check geofence constraints"""
        let distance_from_center = Vector3(
            state.position.x - self.limits.geofence_center.x,
            state.position.y - self.limits.geofence_center.y,
            0.0
        ).magnitude()
        
        if distance_from_center > self.limits.geofence_radius:
            result.violation_count += 1
            result.critical_violations.append("Drone outside geofence boundary")
            result.recommended_action = "RETURN_TO_CENTER"
        elif distance_from_center > self.limits.geofence_radius * 0.9:
            result.warnings.append("Approaching geofence boundary")
    
    fn _check_hardware_status(self, state: DroneState, inout result: SafetyValidationResult):
        """Check hardware status and sensor readings"""
        if not state.gps_fix:
            result.violation_count += 1
            result.critical_violations.append("GPS fix lost")
            result.recommended_action = "HOLD_POSITION"
        
        # Check motor status
        for i in range(4):
            if not state.motor_status[i]:
                result.violation_count += 1
                result.critical_violations.append("Motor failure detected")
                result.recommended_action = "EMERGENCY_LANDING"
                break
    
    fn _check_emergency_conditions(self, state: DroneState, inout result: SafetyValidationResult):
        """Check for emergency conditions requiring immediate action"""
        # Check for rapid uncontrolled movement
        let total_velocity = state.velocity.magnitude()
        let total_acceleration = state.acceleration.magnitude()
        
        if total_velocity > self.limits.max_velocity * 1.2 and total_acceleration > self.limits.max_acceleration * 1.2:
            result.violation_count += 1
            result.critical_violations.append("Uncontrolled movement detected")
            result.recommended_action = "EMERGENCY_STOP"
            self.emergency_stop_active = True
        
        # Check for system instability
        let angular_accel = state.angular_velocity.magnitude()
        if angular_accel > 360.0:  # 360 deg/s²
            result.violation_count += 1
            result.critical_violations.append("System instability detected")
            result.recommended_action = "STABILIZE"
    
    fn validate_command(self, command: String) -> Bool:
        """Validate text commands for dangerous keywords"""
        let dangerous_keywords = List[String]()
        dangerous_keywords.append("attack")
        dangerous_keywords.append("weapon")
        dangerous_keywords.append("bomb")
        dangerous_keywords.append("destroy")
        dangerous_keywords.append("kill")
        dangerous_keywords.append("harm")
        dangerous_keywords.append("damage")
        dangerous_keywords.append("crash")
        dangerous_keywords.append("collide")
        dangerous_keywords.append("kamikaze")
        
        let command_lower = command.lower()
        for i in range(len(dangerous_keywords)):
            if command_lower.find(dangerous_keywords[i]) != -1:
                return False
        
        return True
    
    fn validate_action_vector(self, action_vector: Tensor[DType.float32]) -> Bool:
        """Validate action vectors for safety constraints"""
        if action_vector.shape()[0] != 4:
            return False
        
        # Check for NaN or infinite values
        for i in range(4):
            let value = action_vector[i]
            if not (value == value) or abs(value) > 1e6:  # NaN or infinite check
                return False
        
        # Check action magnitude constraints
        let action_magnitude = sqrt(
            action_vector[0] * action_vector[0] + 
            action_vector[1] * action_vector[1] + 
            action_vector[2] * action_vector[2] + 
            action_vector[3] * action_vector[3]
        )
        
        return action_magnitude <= 1.0  # Normalized action space
    
    fn emergency_stop(inout self) -> Bool:
        """Activate emergency stop system"""
        self.emergency_stop_active = True
        return True
    
    fn reset_emergency_stop(inout self) -> Bool:
        """Reset emergency stop system"""
        self.emergency_stop_active = False
        return True
    
    fn get_safety_score(self, state: DroneState) -> Float64:
        """Calculate overall safety score (0.0 to 1.0)"""
        let result = self.validate_state(state)
        
        if len(result.critical_violations) > 0:
            return 0.0
        
        var score = 1.0
        
        # Deduct points for warnings
        score -= len(result.warnings) * 0.1
        
        # Deduct points for approaching limits
        let velocity_ratio = state.velocity.magnitude() / self.limits.max_velocity
        let altitude_ratio = state.position.z / self.limits.max_altitude
        let battery_ratio = state.battery_level / 100.0
        
        score -= velocity_ratio * 0.2
        score -= altitude_ratio * 0.1
        score -= (1.0 - battery_ratio) * 0.3
        
        return max(0.0, min(1.0, score))

# Export functions for Python integration
fn create_safety_validator() -> SafetyValidator:
    """Create and return a new SafetyValidator instance"""
    return SafetyValidator()

fn validate_drone_state(inout validator: SafetyValidator, 
                       position: Tensor[DType.float64],
                       velocity: Tensor[DType.float64],
                       battery_level: Float64) -> Bool:
    """Validate drone state from Python interface"""
    var state = DroneState()
    
    if position.shape()[0] >= 3:
        state.position = Vector3(position[0], position[1], position[2])
    
    if velocity.shape()[0] >= 3:
        state.velocity = Vector3(velocity[0], velocity[1], velocity[2])
    
    state.battery_level = battery_level
    
    let result = validator.validate_state(state)
    return result.is_safe

fn validate_command_string(command: String) -> Bool:
    """Validate command string for dangerous content"""
    let validator = SafetyValidator()
    return validator.validate_command(command)

fn validate_action_array(action_data: Tensor[DType.float32]) -> Bool:
    """Validate action vector from Python"""
    let validator = SafetyValidator()
    return validator.validate_action_vector(action_data)
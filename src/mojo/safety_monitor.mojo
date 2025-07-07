from tensor import Tensor, TensorSpec, TensorShape
from .math_utils import clamp

struct SafetyMonitor:
    var velocity_limits: Tensor[DType.float32]
    var angular_limits: Tensor[DType.float32] 
    var altitude_limits: Tensor[DType.float32]
    var acceleration_limits: Tensor[DType.float32]
    
    fn __init__(inout self):
        # Velocity limits [min, max] in m/s
        self.velocity_limits = Tensor[DType.float32](2)
        self.velocity_limits[0] = -5.0  # Max reverse velocity
        self.velocity_limits[1] = 5.0   # Max forward velocity
        
        # Angular rate limits [min, max] in rad/s
        self.angular_limits = Tensor[DType.float32](2)
        self.angular_limits[0] = -2.0   # Max angular rate
        self.angular_limits[1] = 2.0
        
        # Altitude limits [min, max] in meters
        self.altitude_limits = Tensor[DType.float32](2)
        self.altitude_limits[0] = 0.5   # Min altitude (ground clearance)
        self.altitude_limits[1] = 100.0 # Max altitude (regulatory limit)
        
        # Acceleration limits [min, max] in m/s²
        self.acceleration_limits = Tensor[DType.float32](2)
        self.acceleration_limits[0] = -10.0  # Max deceleration
        self.acceleration_limits[1] = 10.0   # Max acceleration

    fn validate(self, actions: Tensor[DType.float32], current_altitude: Float32, current_velocity: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Validate and constrain action vector for safety"""
        var safe_actions = Tensor[DType.float32](actions.shape())
        
        # Copy input actions
        for i in range(actions.num_elements()):
            safe_actions[i] = actions[i]
        
        # Velocity limits (actions[0:3] = [vx, vy, vz])
        safe_actions = self._clamp_velocity(safe_actions)
        
        # Angular rate limits (actions[3:6] = [wx, wy, wz])  
        safe_actions = self._clamp_angular(safe_actions)
        
        # Altitude-based constraints
        safe_actions = self._enforce_altitude_constraints(safe_actions, current_altitude)
        
        # Acceleration limits based on current velocity
        safe_actions = self._enforce_acceleration_limits(safe_actions, current_velocity)
        
        return safe_actions
    
    fn _clamp_velocity(self, actions: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Clamp linear velocity components"""
        var result = Tensor[DType.float32](actions.shape())
        
        # Copy all actions first
        for i in range(actions.num_elements()):
            result[i] = actions[i]
        
        # Clamp velocity components (0:3)
        for i in range(3):
            if result[i] < self.velocity_limits[0]:
                result[i] = self.velocity_limits[0]
            elif result[i] > self.velocity_limits[1]:
                result[i] = self.velocity_limits[1]
        
        return result
    
    fn _clamp_angular(self, actions: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Clamp angular velocity components"""
        var result = Tensor[DType.float32](actions.shape())
        
        # Copy all actions first
        for i in range(actions.num_elements()):
            result[i] = actions[i]
        
        # Clamp angular components (3:6)
        for i in range(3, 6):
            if result[i] < self.angular_limits[0]:
                result[i] = self.angular_limits[0]
            elif result[i] > self.angular_limits[1]:
                result[i] = self.angular_limits[1]
        
        return result
    
    fn _enforce_altitude_constraints(self, actions: Tensor[DType.float32], current_altitude: Float32) -> Tensor[DType.float32]:
        """Enforce altitude-based velocity constraints"""
        var result = Tensor[DType.float32](actions.shape())
        
        # Copy all actions first
        for i in range(actions.num_elements()):
            result[i] = actions[i]
        
        # Prevent going below minimum altitude
        if current_altitude <= self.altitude_limits[0]:
            if result[2] < 0:  # Negative vertical velocity (downward)
                result[2] = 0.0  # Force hover/upward motion
        
        # Prevent going above maximum altitude  
        if current_altitude >= self.altitude_limits[1]:
            if result[2] > 0:  # Positive vertical velocity (upward)
                result[2] = 0.0  # Force hover/downward motion
        
        return result
    
    fn _enforce_acceleration_limits(self, actions: Tensor[DType.float32], current_velocity: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Enforce acceleration limits to prevent aggressive maneuvers"""
        var result = Tensor[DType.float32](actions.shape())
        
        # Copy all actions first
        for i in range(actions.num_elements()):
            result[i] = actions[i]
        
        # Check acceleration for each velocity component
        for i in range(3):
            let velocity_change = result[i] - current_velocity[i]
            
            if velocity_change < self.acceleration_limits[0]:
                result[i] = current_velocity[i] + self.acceleration_limits[0]
            elif velocity_change > self.acceleration_limits[1]:
                result[i] = current_velocity[i] + self.acceleration_limits[1]
        
        return result
    
    fn is_safe_action(self, actions: Tensor[DType.float32], current_state: Tensor[DType.float32]) -> Bool:
        """Check if action is within safety bounds without modification"""
        let current_altitude = current_state[2]
        let current_velocity = Tensor[DType.float32](3)
        current_velocity[0] = current_state[3]
        current_velocity[1] = current_state[4]  
        current_velocity[2] = current_state[5]
        
        # Check velocity limits
        for i in range(3):
            if actions[i] < self.velocity_limits[0] or actions[i] > self.velocity_limits[1]:
                return False
        
        # Check angular limits
        for i in range(3, 6):
            if actions[i] < self.angular_limits[0] or actions[i] > self.angular_limits[1]:
                return False
        
        # Check altitude constraints
        if current_altitude <= self.altitude_limits[0] and actions[2] < 0:
            return False
        
        if current_altitude >= self.altitude_limits[1] and actions[2] > 0:
            return False
        
        return True
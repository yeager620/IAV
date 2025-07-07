from tensor import Tensor, TensorSpec, TensorShape
from .math_utils import matrix_multiply, clamp

struct ControlAllocator:
    """
    Converts high-level control commands to individual motor commands.
    Implements control allocation for a standard quadcopter configuration:
    
    Motor Layout (top view):
        1 ---- 2
        |      |
        |  +   |  (+ = center of mass)
        |      |
        4 ---- 3
    
    Motor rotation directions:
        1: CCW, 2: CW, 3: CCW, 4: CW
    """
    var allocation_matrix: Tensor[DType.float32]
    var motor_limits: Tensor[DType.float32]
    var arm_length: Float32
    var thrust_to_torque_ratio: Float32
    
    fn __init__(inout self, arm_length: Float32 = 0.225, thrust_to_torque_ratio: Float32 = 0.05):
        """
        Initialize control allocator
        
        Args:
            arm_length: Distance from center to motor (meters)
            thrust_to_torque_ratio: Ratio of motor torque to thrust
        """
        self.arm_length = arm_length
        self.thrust_to_torque_ratio = thrust_to_torque_ratio
        
        # Motor output limits [min, max] (normalized 0-1)
        self.motor_limits = Tensor[DType.float32](2)
        self.motor_limits[0] = 0.0    # Min throttle
        self.motor_limits[1] = 1.0    # Max throttle
        
        # Build allocation matrix [4 motors x 4 control inputs]
        # Control inputs: [thrust, roll_torque, pitch_torque, yaw_torque]
        self.allocation_matrix = Tensor[DType.float32](4, 4)
        
        # Thrust contribution (all motors contribute equally)
        self.allocation_matrix[0, 0] = 1.0  # Motor 1 thrust
        self.allocation_matrix[1, 0] = 1.0  # Motor 2 thrust
        self.allocation_matrix[2, 0] = 1.0  # Motor 3 thrust  
        self.allocation_matrix[3, 0] = 1.0  # Motor 4 thrust
        
        # Roll torque (motors 1,4 vs 2,3)
        self.allocation_matrix[0, 1] = -self.arm_length   # Motor 1: -roll
        self.allocation_matrix[1, 1] = self.arm_length    # Motor 2: +roll
        self.allocation_matrix[2, 1] = self.arm_length    # Motor 3: +roll
        self.allocation_matrix[3, 1] = -self.arm_length   # Motor 4: -roll
        
        # Pitch torque (motors 1,2 vs 3,4)
        self.allocation_matrix[0, 2] = self.arm_length    # Motor 1: +pitch
        self.allocation_matrix[1, 2] = self.arm_length    # Motor 2: +pitch
        self.allocation_matrix[2, 2] = -self.arm_length   # Motor 3: -pitch
        self.allocation_matrix[3, 2] = -self.arm_length   # Motor 4: -pitch
        
        # Yaw torque (CCW motors: 1,3 vs CW motors: 2,4)
        self.allocation_matrix[0, 3] = -self.thrust_to_torque_ratio  # Motor 1: -yaw (CCW)
        self.allocation_matrix[1, 3] = self.thrust_to_torque_ratio   # Motor 2: +yaw (CW)
        self.allocation_matrix[2, 3] = -self.thrust_to_torque_ratio  # Motor 3: -yaw (CCW)
        self.allocation_matrix[3, 3] = self.thrust_to_torque_ratio   # Motor 4: +yaw (CW)

    fn allocate(self, control_input: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """
        Convert control commands to motor outputs
        
        Args:
            control_input: [thrust, roll_torque, pitch_torque, yaw_torque]
            
        Returns:
            motor_commands: [motor1, motor2, motor3, motor4] (0-1 range)
        """
        # Matrix multiplication: motor_commands = allocation_matrix @ control_input
        var motor_commands = Tensor[DType.float32](4)
        
        for i in range(4):  # For each motor
            var command = Float32(0)
            for j in range(4):  # For each control input
                command += self.allocation_matrix[i, j] * control_input[j]
            motor_commands[i] = command
        
        # Apply motor limits and ensure feasibility
        return self._apply_limits_and_scale(motor_commands)
    
    fn velocity_to_control_input(self, velocity_cmd: Tensor[DType.float32], angular_cmd: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """
        Convert velocity commands to control torques
        
        Args:
            velocity_cmd: [vx, vy, vz] in m/s
            angular_cmd: [wx, wy, wz] in rad/s
            
        Returns:
            control_input: [thrust, roll_torque, pitch_torque, yaw_torque]
        """
        var control_input = Tensor[DType.float32](4)
        
        # Thrust from vertical velocity (with hover compensation)
        let base_thrust = Float32(0.5)  # Hover thrust baseline
        let thrust_gain = Float32(0.3)
        control_input[0] = base_thrust + velocity_cmd[2] * thrust_gain
        
        # Roll torque from lateral velocity (vy)
        let roll_gain = Float32(0.2)
        control_input[1] = velocity_cmd[1] * roll_gain
        
        # Pitch torque from forward velocity (vx)  
        let pitch_gain = Float32(0.2)
        control_input[2] = velocity_cmd[0] * pitch_gain
        
        # Yaw torque from angular velocity (wz)
        let yaw_gain = Float32(0.1)
        control_input[3] = angular_cmd[2] * yaw_gain
        
        return control_input
    
    fn _apply_limits_and_scale(self, motor_commands: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """Apply motor limits and scale to maintain control authority"""
        var limited_commands = Tensor[DType.float32](4)
        
        # First, clamp to basic limits
        for i in range(4):
            limited_commands[i] = motor_commands[i]
            if limited_commands[i] < self.motor_limits[0]:
                limited_commands[i] = self.motor_limits[0]
            elif limited_commands[i] > self.motor_limits[1]:
                limited_commands[i] = self.motor_limits[1]
        
        # Check if any motor is saturated and scale if necessary
        var max_cmd = Float32(0)
        var min_cmd = Float32(1)
        
        for i in range(4):
            if motor_commands[i] > max_cmd:
                max_cmd = motor_commands[i]
            if motor_commands[i] < min_cmd:
                min_cmd = motor_commands[i]
        
        # If commands exceed limits, scale proportionally
        if max_cmd > self.motor_limits[1] or min_cmd < self.motor_limits[0]:
            let scale_factor = min(
                self.motor_limits[1] / max_cmd if max_cmd > 0 else 1.0,
                self.motor_limits[0] / min_cmd if min_cmd < 0 else 1.0
            )
            
            for i in range(4):
                limited_commands[i] = motor_commands[i] * scale_factor
                
                # Final clamp after scaling
                if limited_commands[i] < self.motor_limits[0]:
                    limited_commands[i] = self.motor_limits[0]
                elif limited_commands[i] > self.motor_limits[1]:
                    limited_commands[i] = self.motor_limits[1]
        
        return limited_commands
    
    fn get_motor_mixing_info(self) -> Tensor[DType.float32]:
        """Return the allocation matrix for debugging/analysis"""
        return self.allocation_matrix
    
    fn set_motor_limits(inout self, min_throttle: Float32, max_throttle: Float32):
        """Update motor throttle limits"""
        self.motor_limits[0] = min_throttle
        self.motor_limits[1] = max_throttle
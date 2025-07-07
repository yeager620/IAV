"""
Simplified UAV core functions in Mojo for maximum performance.
"""

fn clamp_value(value: Float32, min_val: Float32, max_val: Float32) -> Float32:
    """Clamp value between min and max."""
    if value < min_val:
        return min_val
    elif value > max_val:
        return max_val
    else:
        return value

fn apply_safety_limits(velocity: Float32, is_angular: Bool) -> Float32:
    """Apply safety limits to velocity or angular rate."""
    if is_angular:
        return clamp_value(velocity, -2.0, 2.0)  # Angular rate limit
    else:
        return clamp_value(velocity, -5.0, 5.0)  # Linear velocity limit

fn apply_altitude_constraint(vz: Float32, altitude: Float32) -> Float32:
    """Apply altitude constraints to vertical velocity."""
    var min_altitude = Float32(0.5)
    var max_altitude = Float32(100.0)
    
    if altitude <= min_altitude and vz < 0:
        return 0.0  # Prevent going below minimum
    elif altitude >= max_altitude and vz > 0:
        return 0.0  # Prevent going above maximum
    else:
        return vz

fn compute_motor_command(thrust: Float32, roll: Float32, pitch: Float32, yaw: Float32, motor_index: Int) -> Float32:
    """Compute individual motor command based on control inputs."""
    var command: Float32
    
    if motor_index == 1:  # Front-left
        command = thrust + roll + pitch - yaw
    elif motor_index == 2:  # Front-right
        command = thrust - roll + pitch + yaw
    elif motor_index == 3:  # Rear-right
        command = thrust - roll - pitch - yaw
    else:  # Rear-left (motor 4)
        command = thrust + roll - pitch + yaw
    
    return clamp_value(command, 0.0, 1.0)

fn process_uav_control_single(vx: Float32, vy: Float32, vz: Float32, wz: Float32, altitude: Float32, motor_index: Int) -> Float32:
    """Process UAV control for a single motor."""
    
    # 1. Apply safety limits
    var safe_vx = apply_safety_limits(vx, False)
    var safe_vy = apply_safety_limits(vy, False) 
    var safe_vz = apply_safety_limits(vz, False)
    var safe_wz = apply_safety_limits(wz, True)
    
    # 2. Apply altitude constraints
    var constrained_vz = apply_altitude_constraint(safe_vz, altitude)
    
    # 3. Convert to control signals
    var thrust = 0.5 + constrained_vz * 0.3
    var roll = safe_vy * 0.2
    var pitch = safe_vx * 0.2
    var yaw = safe_wz * 0.1
    
    # 4. Compute motor command
    return compute_motor_command(thrust, roll, pitch, yaw, motor_index)

fn main():
    """Test the UAV control functions."""
    print("Testing UAV control functions...")
    
    # Test input: takeoff command
    var vx = Float32(0.0)
    var vy = Float32(0.0)
    var vz = Float32(2.0)  # 2 m/s up
    var wz = Float32(0.0)
    var altitude = Float32(1.0)
    
    print("Motor commands for takeoff (vz=2.0 m/s):")
    print("M1:", process_uav_control_single(vx, vy, vz, wz, altitude, 1))
    print("M2:", process_uav_control_single(vx, vy, vz, wz, altitude, 2))
    print("M3:", process_uav_control_single(vx, vy, vz, wz, altitude, 3))
    print("M4:", process_uav_control_single(vx, vy, vz, wz, altitude, 4))
    
    # Test hover command
    print("\nTesting hover command:")
    print("M1:", process_uav_control_single(0.0, 0.0, 0.0, 0.0, 5.0, 1))
    print("M2:", process_uav_control_single(0.0, 0.0, 0.0, 0.0, 5.0, 2))
    print("M3:", process_uav_control_single(0.0, 0.0, 0.0, 0.0, 5.0, 3))
    print("M4:", process_uav_control_single(0.0, 0.0, 0.0, 0.0, 5.0, 4))
    
    # Test forward motion
    print("\nTesting forward motion (vx=1.0 m/s):")
    print("M1:", process_uav_control_single(1.0, 0.0, 0.0, 0.0, 5.0, 1))
    print("M2:", process_uav_control_single(1.0, 0.0, 0.0, 0.0, 5.0, 2))
    print("M3:", process_uav_control_single(1.0, 0.0, 0.0, 0.0, 5.0, 3))
    print("M4:", process_uav_control_single(1.0, 0.0, 0.0, 0.0, 5.0, 4))
"""
Optimized UAV core functions using SIMD and advanced Mojo standard library.
"""

from algorithm import vectorize, parallelize
from collections import List, Dict
from memory import memset_zero
from benchmark import Benchmark
from math import sqrt, sin, cos, pi, abs, min, max

# SIMD-optimized constants and types
alias MOTOR_COUNT = 4
alias MotorSIMD = SIMD[DType.float32, MOTOR_COUNT]
alias ControlSIMD = SIMD[DType.float32, MOTOR_COUNT]

# Vectorized safety limits
alias VELOCITY_LIMITS = SIMD[DType.float32, 4](5.0, 5.0, 5.0, 2.0)  # vx, vy, vz, yaw_rate
alias MOTOR_LIMITS = SIMD[DType.float32, 4](0.0, 1.0, 0.0, 1.0)    # min, max, min, max

@always_inline
fn clamp_simd[width: Int](values: SIMD[DType.float32, width], min_vals: SIMD[DType.float32, width], max_vals: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    """SIMD-optimized clamping function"""
    return min(max(values, min_vals), max_vals)

fn clamp_value(value: Float32, min_val: Float32, max_val: Float32) -> Float32:
    """Single value clamp - kept for compatibility"""
    return min(max(value, min_val), max_val)

fn apply_safety_limits_simd(velocities: ControlSIMD) -> ControlSIMD:
    """SIMD-optimized safety limits for all control inputs at once"""
    var limits_min = SIMD[DType.float32, 4](-5.0, -5.0, -5.0, -2.0)
    var limits_max = SIMD[DType.float32, 4](5.0, 5.0, 5.0, 2.0)
    return clamp_simd(velocities, limits_min, limits_max)

fn apply_safety_limits(velocity: Float32, is_angular: Bool) -> Float32:
    """Apply safety limits to velocity or angular rate - legacy compatibility"""
    if is_angular:
        return clamp_value(velocity, -2.0, 2.0)  # Angular rate limit
    else:
        return clamp_value(velocity, -5.0, 5.0)  # Linear velocity limit

# SIMD-optimized motor mixing matrix
struct MotorMixingMatrix:
    """Optimized motor mixing using SIMD operations"""
    var matrix_data: List[ControlSIMD]
    
    fn __init__(inout self):
        self.matrix_data = List[ControlSIMD]()
        
        # Motor mixing matrix for quadcopter X configuration
        # Each row: [thrust_factor, roll_factor, pitch_factor, yaw_factor]
        self.matrix_data.append(ControlSIMD(1.0, 1.0, 1.0, -1.0))   # Motor 1: Front-left
        self.matrix_data.append(ControlSIMD(1.0, -1.0, 1.0, 1.0))   # Motor 2: Front-right  
        self.matrix_data.append(ControlSIMD(1.0, -1.0, -1.0, -1.0)) # Motor 3: Rear-right
        self.matrix_data.append(ControlSIMD(1.0, 1.0, -1.0, 1.0))   # Motor 4: Rear-left
    
    fn compute_all_motors(self, control_inputs: ControlSIMD) -> MotorSIMD:
        """Compute all motor commands using SIMD operations"""
        var motor_commands = MotorSIMD(0.0, 0.0, 0.0, 0.0)
        
        # Vectorized matrix multiplication
        for i in range(MOTOR_COUNT):
            var motor_mix = self.matrix_data[i]
            # Dot product using SIMD
            var command = (motor_mix * control_inputs).reduce_add()
            motor_commands[i] = clamp_value(command, 0.0, 1.0)
        
        return motor_commands

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

fn apply_altitude_constraint_simd(control_inputs: ControlSIMD, altitude: Float32) -> ControlSIMD:
    """SIMD version of altitude constraint"""
    var constrained = control_inputs
    var min_altitude = Float32(0.5)
    var max_altitude = Float32(100.0)
    
    if altitude <= min_altitude and control_inputs[2] < 0:
        constrained[2] = 0.0  # Zero vertical velocity
    elif altitude >= max_altitude and control_inputs[2] > 0:
        constrained[2] = 0.0  # Zero vertical velocity
        
    return constrained

fn compute_motor_command(thrust: Float32, roll: Float32, pitch: Float32, yaw: Float32, motor_index: Int) -> Float32:
    """Legacy motor command computation - kept for compatibility"""
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

# Optimized batch processing functions
fn process_uav_control_simd(vx: Float32, vy: Float32, vz: Float32, wz: Float32, altitude: Float32) -> MotorSIMD:
    """SIMD-optimized UAV control processing for all motors at once"""
    # 1. Pack input velocities into SIMD vector
    var input_velocities = ControlSIMD(vx, vy, vz, wz)
    
    # 2. Apply safety limits using SIMD
    var safe_velocities = apply_safety_limits_simd(input_velocities)
    
    # 3. Apply altitude constraints
    var constrained_velocities = apply_altitude_constraint_simd(safe_velocities, altitude)
    
    # 4. Convert to control signals (thrust, roll, pitch, yaw)
    var control_gains = ControlSIMD(0.3, 0.2, 0.2, 0.1)  # Scaling factors
    var control_signals = constrained_velocities * control_gains
    
    # Add base thrust
    control_signals[0] += 0.5  # Base thrust offset
    
    # 5. Compute all motor commands using optimized mixing matrix
    var mixer = MotorMixingMatrix()
    return mixer.compute_all_motors(control_signals)

fn process_uav_control_batch(velocities: List[ControlSIMD], altitudes: List[Float32]) -> List[MotorSIMD]:
    """Batch process multiple control inputs"""
    var results = List[MotorSIMD]()
    
    # Process in parallel batches for maximum performance
    for i in range(len(velocities)):
        var altitude = altitudes[i] if i < len(altitudes) else 5.0
        var vel = velocities[i]
        var motors = process_uav_control_simd(vel[0], vel[1], vel[2], vel[3], altitude)
        results.append(motors)
    
    return results

fn process_uav_control_single(vx: Float32, vy: Float32, vz: Float32, wz: Float32, altitude: Float32, motor_index: Int) -> Float32:
    """Legacy single motor processing - kept for compatibility"""
    
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

# Performance monitoring and benchmarking
fn benchmark_motor_computation(iterations: Int) -> Float64:
    """Benchmark motor computation performance"""
    var benchmark = Benchmark()
    
    @parameter
    fn test_simd_version():
        var _ = process_uav_control_simd(1.0, 0.5, 2.0, 0.1, 5.0)
    
    @parameter 
    fn test_legacy_version():
        var _ = process_uav_control_single(1.0, 0.5, 2.0, 0.1, 5.0, 1)
        var _ = process_uav_control_single(1.0, 0.5, 2.0, 0.1, 5.0, 2)
        var _ = process_uav_control_single(1.0, 0.5, 2.0, 0.1, 5.0, 3)
        var _ = process_uav_control_single(1.0, 0.5, 2.0, 0.1, 5.0, 4)
    
    var simd_report = benchmark.run[test_simd_version]()
    var legacy_report = benchmark.run[test_legacy_version]()
    
    print("SIMD version:", Float64(simd_report.mean()), "ns")
    print("Legacy version:", Float64(legacy_report.mean()), "ns")
    
    return Float64(legacy_report.mean()) / Float64(simd_report.mean())

fn validate_motor_outputs(motors: MotorSIMD) -> Bool:
    """Validate all motor outputs are within safe range"""
    var min_val = SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)
    var max_val = SIMD[DType.float32, 4](1.0, 1.0, 1.0, 1.0)
    var within_min = motors >= min_val
    var within_max = motors <= max_val
    return all(within_min) and all(within_max)

fn create_test_scenarios() -> List[ControlSIMD]:
    """Create test scenarios for validation"""
    var scenarios = List[ControlSIMD]()
    
    # Hover
    scenarios.append(ControlSIMD(0.0, 0.0, 0.0, 0.0))
    
    # Forward flight
    scenarios.append(ControlSIMD(2.0, 0.0, 0.0, 0.0))
    
    # Ascending turn
    scenarios.append(ControlSIMD(1.0, 0.5, 1.0, 0.3))
    
    # Emergency descent
    scenarios.append(ControlSIMD(0.0, 0.0, -3.0, 0.0))
    
    # Aggressive maneuver (should be clamped)
    scenarios.append(ControlSIMD(10.0, 5.0, 8.0, 4.0))
    
    return scenarios

fn main():
    """Enhanced UAV control system test with SIMD optimizations"""
    print("Enhanced UAV Control System Test")
    print("=" * 50)
    
    # Test SIMD-optimized control processing
    print("Testing SIMD-optimized motor control:")
    var motors_simd = process_uav_control_simd(1.0, 0.5, 2.0, 0.1, 5.0)
    print("  All motors SIMD:", motors_simd[0], motors_simd[1], motors_simd[2], motors_simd[3])
    print("  Motors valid:", validate_motor_outputs(motors_simd))
    
    # Compare with legacy implementation
    print("\nComparing with legacy implementation:")
    print("  Legacy M1:", process_uav_control_single(1.0, 0.5, 2.0, 0.1, 5.0, 1))
    print("  Legacy M2:", process_uav_control_single(1.0, 0.5, 2.0, 0.1, 5.0, 2))
    print("  Legacy M3:", process_uav_control_single(1.0, 0.5, 2.0, 0.1, 5.0, 3))
    print("  Legacy M4:", process_uav_control_single(1.0, 0.5, 2.0, 0.1, 5.0, 4))
    
    # Test safety clamping
    print("\nTesting safety limits:")
    var unsafe_input = ControlSIMD(10.0, 8.0, 12.0, 5.0)
    var safe_output = apply_safety_limits_simd(unsafe_input)
    print("  Unsafe input:", unsafe_input[0], unsafe_input[1], unsafe_input[2], unsafe_input[3])
    print("  Safe output:", safe_output[0], safe_output[1], safe_output[2], safe_output[3])
    
    # Test altitude constraints
    print("\nTesting altitude constraints:")
    var low_altitude_test = apply_altitude_constraint_simd(ControlSIMD(0.0, 0.0, -2.0, 0.0), 0.3)
    var high_altitude_test = apply_altitude_constraint_simd(ControlSIMD(0.0, 0.0, 3.0, 0.0), 150.0)
    print("  Low altitude constraint:", low_altitude_test[2])
    print("  High altitude constraint:", high_altitude_test[2])
    
    # Test motor mixing matrix
    print("\nTesting motor mixing matrix:")
    var mixer = MotorMixingMatrix()
    var test_controls = ControlSIMD(0.6, 0.1, 0.2, 0.05)
    var mixed_motors = mixer.compute_all_motors(test_controls)
    print("  Control inputs:", test_controls[0], test_controls[1], test_controls[2], test_controls[3])
    print("  Mixed motors:", mixed_motors[0], mixed_motors[1], mixed_motors[2], mixed_motors[3])
    
    # Test batch processing
    print("\nTesting batch processing:")
    var test_scenarios = create_test_scenarios()
    var altitudes = List[Float32]()
    for i in range(len(test_scenarios)):
        altitudes.append(Float32(5.0 + i))
    
    var batch_results = process_uav_control_batch(test_scenarios, altitudes)
    print("  Processed", len(batch_results), "scenarios")
    
    for i in range(len(batch_results)):
        var valid = validate_motor_outputs(batch_results[i])
        print("  Scenario", i + 1, "valid:", valid)
    
    # Performance benchmark
    print("\nPerformance benchmark:")
    var speedup = benchmark_motor_computation(1000)
    print("  SIMD speedup factor:", speedup, "x")
    
    # Test specific flight scenarios
    print("\nFlight scenario tests:")
    
    # Takeoff
    var takeoff = process_uav_control_simd(0.0, 0.0, 2.0, 0.0, 1.0)
    print("  Takeoff motors:", takeoff[0], takeoff[1], takeoff[2], takeoff[3])
    
    # Hover
    var hover = process_uav_control_simd(0.0, 0.0, 0.0, 0.0, 5.0)
    print("  Hover motors:", hover[0], hover[1], hover[2], hover[3])
    
    # Forward flight
    var forward = process_uav_control_simd(3.0, 0.0, 0.0, 0.0, 5.0)
    print("  Forward motors:", forward[0], forward[1], forward[2], forward[3])
    
    # Coordinated turn
    var turn = process_uav_control_simd(2.0, 1.0, 0.0, 0.5, 5.0)
    print("  Turn motors:", turn[0], turn[1], turn[2], turn[3])
    
    print("=" * 50)
    print("Enhanced UAV control test completed!")
    print("SIMD optimizations and vectorized processing active!")
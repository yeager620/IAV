"""
Comprehensive tests for Mojo control system implementation
"""
from testing import assert_equal, assert_true, assert_false, assert_raises
from math import abs, sqrt
from tensor import Tensor
from collections import List

from src.mojo.control_system import (
    ControlSystem, PIDController, QuaternionAttitude, DroneState, ControlCommand,
    create_control_system, compute_control_commands
)

fn test_pid_controller():
    """Test PID controller functionality"""
    print("Testing PID controller...")
    
    var pid = PIDController(1.0, 0.1, 0.05)
    
    # Test initial state
    assert_equal(pid.kp, 1.0)
    assert_equal(pid.ki, 0.1)
    assert_equal(pid.kd, 0.05)
    assert_equal(pid.integral, 0.0)
    assert_equal(pid.previous_error, 0.0)
    
    # Test update with positive error
    var output = pid.update(1.0, 0.1)
    assert_true(output > 0.0)
    
    # Test integral accumulation
    var output2 = pid.update(1.0, 0.1)
    assert_true(output2 > output)  # Should be larger due to integral
    
    # Test derivative response
    var output3 = pid.update(0.5, 0.1)  # Smaller error
    assert_true(output3 < output2)  # Should be smaller due to derivative
    
    # Test reset
    pid.reset()
    assert_equal(pid.integral, 0.0)
    assert_equal(pid.previous_error, 0.0)
    
    print("✅ PID controller passed")

fn test_quaternion_attitude():
    """Test quaternion attitude calculations"""
    print("Testing quaternion attitude...")
    
    var quat = QuaternionAttitude(1.0, 0.0, 0.0, 0.0)
    
    # Test identity quaternion
    assert_equal(quat.w, 1.0)
    assert_equal(quat.x, 0.0)
    assert_equal(quat.y, 0.0)
    assert_equal(quat.z, 0.0)
    
    # Test normalization
    var quat2 = QuaternionAttitude(0.5, 0.5, 0.5, 0.5)
    quat2.normalize()
    var norm = sqrt(quat2.w*quat2.w + quat2.x*quat2.x + quat2.y*quat2.y + quat2.z*quat2.z)
    assert_true(abs(norm - 1.0) < 1e-10)
    
    # Test Euler conversion
    var euler = quat.to_euler()
    assert_true(abs(euler.x) < 1e-10)  # Roll should be 0
    assert_true(abs(euler.y) < 1e-10)  # Pitch should be 0
    assert_true(abs(euler.z) < 1e-10)  # Yaw should be 0
    
    print("✅ Quaternion attitude passed")

fn test_drone_state():
    """Test drone state management"""
    print("Testing drone state...")
    
    var state = DroneState()
    
    # Test default initialization
    assert_equal(state.position.x, 0.0)
    assert_equal(state.position.y, 0.0)
    assert_equal(state.position.z, 0.0)
    assert_equal(state.velocity.x, 0.0)
    assert_equal(state.velocity.y, 0.0)
    assert_equal(state.velocity.z, 0.0)
    
    # Test state updates
    state.position.x = 10.0
    state.position.y = 20.0
    state.position.z = 30.0
    state.velocity.x = 1.0
    state.velocity.y = 2.0
    state.velocity.z = 3.0
    
    assert_equal(state.position.x, 10.0)
    assert_equal(state.position.y, 20.0)
    assert_equal(state.position.z, 30.0)
    assert_equal(state.velocity.x, 1.0)
    assert_equal(state.velocity.y, 2.0)
    assert_equal(state.velocity.z, 3.0)
    
    print("✅ Drone state passed")

fn test_control_command():
    """Test control command structure"""
    print("Testing control command...")
    
    var command = ControlCommand()
    
    # Test default initialization
    assert_equal(command.throttle, 0.0)
    assert_equal(command.roll, 0.0)
    assert_equal(command.pitch, 0.0)
    assert_equal(command.yaw, 0.0)
    assert_true(command.is_valid)
    
    # Test command updates
    command.throttle = 0.5
    command.roll = 0.1
    command.pitch = -0.2
    command.yaw = 0.3
    
    assert_equal(command.throttle, 0.5)
    assert_equal(command.roll, 0.1)
    assert_equal(command.pitch, -0.2)
    assert_equal(command.yaw, 0.3)
    
    # Test command validation
    command.throttle = 1.5  # Out of range
    command.validate()
    assert_false(command.is_valid)
    
    print("✅ Control command passed")

fn test_control_system_initialization():
    """Test control system initialization"""
    print("Testing control system initialization...")
    
    var control_system = ControlSystem()
    
    # Test default configuration
    assert_true(control_system.enabled)
    assert_false(control_system.emergency_stop)
    assert_equal(control_system.control_frequency, 100.0)
    
    # Test PID controller initialization
    assert_equal(control_system.position_pid_x.kp, 1.0)
    assert_equal(control_system.position_pid_y.kp, 1.0)
    assert_equal(control_system.position_pid_z.kp, 1.0)
    
    assert_equal(control_system.velocity_pid_x.kp, 2.0)
    assert_equal(control_system.velocity_pid_y.kp, 2.0)
    assert_equal(control_system.velocity_pid_z.kp, 2.0)
    
    assert_equal(control_system.attitude_pid_roll.kp, 3.0)
    assert_equal(control_system.attitude_pid_pitch.kp, 3.0)
    assert_equal(control_system.attitude_pid_yaw.kp, 1.5)
    
    print("✅ Control system initialization passed")

fn test_position_control():
    """Test position control functionality"""
    print("Testing position control...")
    
    var control_system = ControlSystem()
    var current_state = DroneState()
    var target_state = DroneState()
    
    # Set up test scenario
    current_state.position.x = 0.0
    current_state.position.y = 0.0
    current_state.position.z = 0.0
    
    target_state.position.x = 10.0
    target_state.position.y = 5.0
    target_state.position.z = 15.0
    
    # Test position control
    var command = control_system.compute_position_control(current_state, target_state, 0.1)
    
    # Should generate positive velocity commands to reach target
    assert_true(command.throttle > 0.0)  # Should climb
    
    print("✅ Position control passed")

fn test_velocity_control():
    """Test velocity control functionality"""
    print("Testing velocity control...")
    
    var control_system = ControlSystem()
    var current_state = DroneState()
    var target_state = DroneState()
    
    # Set up test scenario
    current_state.velocity.x = 0.0
    current_state.velocity.y = 0.0
    current_state.velocity.z = 0.0
    
    target_state.velocity.x = 5.0
    target_state.velocity.y = 2.0
    target_state.velocity.z = 1.0
    
    # Test velocity control
    var command = control_system.compute_velocity_control(current_state, target_state, 0.1)
    
    # Should generate attitude commands to achieve target velocity
    assert_true(abs(command.pitch) > 0.0)  # Should pitch for forward velocity
    assert_true(abs(command.roll) > 0.0)   # Should roll for lateral velocity
    assert_true(command.throttle > 0.0)    # Should throttle for upward velocity
    
    print("✅ Velocity control passed")

fn test_attitude_control():
    """Test attitude control functionality"""
    print("Testing attitude control...")
    
    var control_system = ControlSystem()
    var current_state = DroneState()
    var target_state = DroneState()
    
    # Set up test scenario
    current_state.attitude.x = 0.0  # Roll
    current_state.attitude.y = 0.0  # Pitch
    current_state.attitude.z = 0.0  # Yaw
    
    target_state.attitude.x = 0.1  # Target roll
    target_state.attitude.y = 0.2  # Target pitch
    target_state.attitude.z = 0.3  # Target yaw
    
    # Test attitude control
    var command = control_system.compute_attitude_control(current_state, target_state, 0.1)
    
    # Should generate commands to achieve target attitude
    assert_true(command.roll > 0.0)   # Should roll positive
    assert_true(command.pitch > 0.0)  # Should pitch positive
    assert_true(command.yaw > 0.0)    # Should yaw positive
    
    print("✅ Attitude control passed")

fn test_trajectory_following():
    """Test trajectory following capability"""
    print("Testing trajectory following...")
    
    var control_system = ControlSystem()
    var current_state = DroneState()
    
    # Create a simple trajectory
    var trajectory = List[DroneState]()
    for i in range(5):
        var waypoint = DroneState()
        waypoint.position.x = Float64(i) * 2.0
        waypoint.position.y = Float64(i) * 1.0
        waypoint.position.z = 10.0
        trajectory.append(waypoint)
    
    # Test trajectory following
    var command = control_system.follow_trajectory(current_state, trajectory, 0.1)
    
    # Should generate commands to follow first waypoint
    assert_true(command.is_valid)
    assert_true(command.throttle > 0.0)  # Should climb to altitude
    
    print("✅ Trajectory following passed")

fn test_emergency_handling():
    """Test emergency handling functionality"""
    print("Testing emergency handling...")
    
    var control_system = ControlSystem()
    var current_state = DroneState()
    
    # Test normal operation
    assert_false(control_system.emergency_stop)
    assert_true(control_system.enabled)
    
    # Activate emergency stop
    control_system.activate_emergency_stop()
    assert_true(control_system.emergency_stop)
    
    # Test emergency command generation
    var emergency_command = control_system.compute_emergency_response(current_state)
    
    # Should generate safe landing command
    assert_true(emergency_command.is_valid)
    assert_true(emergency_command.throttle < 0.5)  # Should reduce throttle
    
    # Test emergency reset
    control_system.reset_emergency_stop()
    assert_false(control_system.emergency_stop)
    
    print("✅ Emergency handling passed")

fn test_control_limits():
    """Test control command limits and saturation"""
    print("Testing control limits...")
    
    var control_system = ControlSystem()
    var current_state = DroneState()
    var target_state = DroneState()
    
    # Set up extreme target to test limits
    target_state.position.x = 1000.0  # Very far target
    target_state.position.y = 1000.0
    target_state.position.z = 1000.0
    
    var command = control_system.compute_position_control(current_state, target_state, 0.1)
    
    # Commands should be limited
    assert_true(command.throttle <= 1.0)
    assert_true(command.throttle >= 0.0)
    assert_true(command.roll <= 1.0)
    assert_true(command.roll >= -1.0)
    assert_true(command.pitch <= 1.0)
    assert_true(command.pitch >= -1.0)
    assert_true(command.yaw <= 1.0)
    assert_true(command.yaw >= -1.0)
    
    print("✅ Control limits passed")

fn test_control_system_performance():
    """Test control system performance characteristics"""
    print("Testing control system performance...")
    
    from benchmark import Benchmark
    
    var control_system = ControlSystem()
    var current_state = DroneState()
    var target_state = DroneState()
    
    target_state.position.x = 10.0
    target_state.position.y = 5.0
    target_state.position.z = 15.0
    
    # Benchmark control computation
    @parameter
    fn control_performance():
        _ = control_system.compute_position_control(current_state, target_state, 0.1)
    
    var bench = Benchmark()
    var report = bench.run[control_performance]()
    
    # Just verify it runs without crashing
    print("Control performance test completed")
    
    print("✅ Control system performance passed")

fn test_python_interface():
    """Test Python interface functions"""
    print("Testing Python interface...")
    
    # Test create control system
    var control_system = create_control_system()
    # Just verify it doesn't crash
    
    # Test compute control commands
    var position = Tensor[DType.float64](3)
    position[0] = 0.0
    position[1] = 0.0
    position[2] = 0.0
    
    var target = Tensor[DType.float64](3)
    target[0] = 10.0
    target[1] = 5.0
    target[2] = 15.0
    
    var commands = compute_control_commands(control_system, position, target)
    
    # Should return valid commands
    assert_equal(commands.shape()[0], 4)  # [throttle, roll, pitch, yaw]
    
    print("✅ Python interface passed")

fn test_control_stability():
    """Test control system stability"""
    print("Testing control stability...")
    
    var control_system = ControlSystem()
    var current_state = DroneState()
    var target_state = DroneState()
    
    # Set target position
    target_state.position.x = 5.0
    target_state.position.y = 5.0
    target_state.position.z = 10.0
    
    # Simulate control loop
    var dt = 0.01
    var stable_count = 0
    
    for i in range(100):
        var command = control_system.compute_position_control(current_state, target_state, dt)
        
        # Simple integration to simulate drone response
        current_state.velocity.x += command.pitch * dt
        current_state.velocity.y += command.roll * dt
        current_state.velocity.z += command.throttle * dt
        
        current_state.position.x += current_state.velocity.x * dt
        current_state.position.y += current_state.velocity.y * dt
        current_state.position.z += current_state.velocity.z * dt
        
        # Check if getting closer to target
        var distance = sqrt(
            (current_state.position.x - target_state.position.x) ** 2 +
            (current_state.position.y - target_state.position.y) ** 2 +
            (current_state.position.z - target_state.position.z) ** 2
        )
        
        if distance < 0.5:  # Within 0.5m of target
            stable_count += 1
    
    # Should achieve stability
    assert_true(stable_count > 10)  # Should be stable for at least 10 iterations
    
    print("✅ Control stability passed")

fn main():
    """Run all control system tests"""
    print("🧪 Running Mojo Control System Tests")
    print("=" * 50)
    
    test_pid_controller()
    test_quaternion_attitude()
    test_drone_state()
    test_control_command()
    test_control_system_initialization()
    test_position_control()
    test_velocity_control()
    test_attitude_control()
    test_trajectory_following()
    test_emergency_handling()
    test_control_limits()
    test_control_system_performance()
    test_python_interface()
    test_control_stability()
    
    print("=" * 50)
    print("✅ All control system tests passed!")
    print("🎮 Control system verified")
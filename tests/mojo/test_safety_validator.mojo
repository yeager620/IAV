"""
Comprehensive tests for Mojo safety validation system
"""
from testing import assert_equal, assert_true, assert_false, assert_raises
from math import abs
from tensor import Tensor
from collections import List

from src.mojo.safety_validator import (
    SafetyValidator, DroneState, Vector3, SafetyLimits,
    create_safety_validator, validate_command_string, validate_action_array
)

fn test_vector3_operations():
    """Test Vector3 mathematical operations"""
    print("Testing Vector3 operations...")
    
    var v1 = Vector3(3.0, 4.0, 0.0)
    var v2 = Vector3(1.0, 0.0, 0.0)
    
    # Test magnitude
    assert_equal(v1.magnitude(), 5.0)
    assert_equal(v2.magnitude(), 1.0)
    
    # Test dot product
    assert_equal(v1.dot(v2), 3.0)
    
    # Test cross product
    var cross = v1.cross(v2)
    assert_equal(cross.x, 0.0)
    assert_equal(cross.y, 0.0)
    assert_equal(cross.z, -4.0)
    
    # Test normalization
    var v3 = Vector3(3.0, 4.0, 0.0)
    v3.normalize()
    assert_true(abs(v3.magnitude() - 1.0) < 1e-10)
    
    print("✅ Vector3 operations passed")

fn test_drone_state_creation():
    """Test DroneState initialization and manipulation"""
    print("Testing DroneState creation...")
    
    var state = DroneState()
    
    # Test default values
    assert_equal(state.position.x, 0.0)
    assert_equal(state.position.y, 0.0)
    assert_equal(state.position.z, 0.0)
    assert_equal(state.battery_level, 100.0)
    assert_true(state.gps_fix)
    
    # Test state modification
    state.position = Vector3(10.0, 20.0, 30.0)
    state.velocity = Vector3(1.0, 2.0, 3.0)
    state.battery_level = 75.0
    
    assert_equal(state.position.x, 10.0)
    assert_equal(state.velocity.magnitude(), sqrt(14.0))
    assert_equal(state.battery_level, 75.0)
    
    print("✅ DroneState creation passed")

fn test_safety_limits():
    """Test SafetyLimits configuration"""
    print("Testing SafetyLimits...")
    
    var limits = SafetyLimits()
    
    # Test default limits
    assert_equal(limits.max_velocity, 15.0)
    assert_equal(limits.max_altitude, 120.0)
    assert_equal(limits.min_altitude, 0.5)
    assert_equal(limits.battery_critical, 20.0)
    
    # Test limits modification
    limits.max_velocity = 10.0
    limits.geofence_radius = 50.0
    
    assert_equal(limits.max_velocity, 10.0)
    assert_equal(limits.geofence_radius, 50.0)
    
    print("✅ SafetyLimits passed")

fn test_velocity_validation():
    """Test velocity limit validation"""
    print("Testing velocity validation...")
    
    var validator = SafetyValidator()
    var state = DroneState()
    
    # Test safe velocity
    state.velocity = Vector3(5.0, 5.0, 2.0)
    var result = validator.validate_state(state)
    assert_true(result.is_safe)
    assert_equal(result.violation_count, 0)
    
    # Test excessive velocity
    state.velocity = Vector3(20.0, 20.0, 5.0)  # > 15 m/s limit
    result = validator.validate_state(state)
    assert_false(result.is_safe)
    assert_true(result.violation_count > 0)
    
    print("✅ Velocity validation passed")

fn test_altitude_validation():
    """Test altitude limit validation"""
    print("Testing altitude validation...")
    
    var validator = SafetyValidator()
    var state = DroneState()
    
    # Test safe altitude
    state.position = Vector3(0.0, 0.0, 50.0)
    var result = validator.validate_state(state)
    assert_true(result.is_safe)
    
    # Test excessive altitude
    state.position = Vector3(0.0, 0.0, 150.0)  # > 120m limit
    result = validator.validate_state(state)
    assert_false(result.is_safe)
    assert_true(result.violation_count > 0)
    
    # Test minimum altitude violation
    state.position = Vector3(0.0, 0.0, 0.2)  # < 0.5m limit
    result = validator.validate_state(state)
    assert_false(result.is_safe)
    assert_true(result.violation_count > 0)
    
    print("✅ Altitude validation passed")

fn test_battery_validation():
    """Test battery level validation"""
    print("Testing battery validation...")
    
    var validator = SafetyValidator()
    var state = DroneState()
    
    # Test safe battery level
    state.battery_level = 50.0
    var result = validator.validate_state(state)
    assert_true(result.is_safe)
    
    # Test critical battery level
    state.battery_level = 15.0  # < 20% critical
    result = validator.validate_state(state)
    assert_false(result.is_safe)
    assert_true(result.violation_count > 0)
    
    print("✅ Battery validation passed")

fn test_geofence_validation():
    """Test geofence boundary validation"""
    print("Testing geofence validation...")
    
    var validator = SafetyValidator()
    var state = DroneState()
    
    # Test within geofence
    state.position = Vector3(50.0, 50.0, 10.0)
    var result = validator.validate_state(state)
    assert_true(result.is_safe)
    
    # Test outside geofence
    state.position = Vector3(150.0, 150.0, 10.0)  # > 100m radius
    result = validator.validate_state(state)
    assert_false(result.is_safe)
    assert_true(result.violation_count > 0)
    
    print("✅ Geofence validation passed")

fn test_hardware_status_validation():
    """Test hardware status validation"""
    print("Testing hardware status validation...")
    
    var validator = SafetyValidator()
    var state = DroneState()
    
    # Test normal hardware status
    state.gps_fix = True
    state.motor_status = SIMD[DType.bool, 4](True, True, True, True)
    var result = validator.validate_state(state)
    assert_true(result.is_safe)
    
    # Test GPS fix lost
    state.gps_fix = False
    result = validator.validate_state(state)
    assert_false(result.is_safe)
    assert_true(result.violation_count > 0)
    
    # Test motor failure
    state.gps_fix = True
    state.motor_status = SIMD[DType.bool, 4](True, False, True, True)
    result = validator.validate_state(state)
    assert_false(result.is_safe)
    assert_true(result.violation_count > 0)
    
    print("✅ Hardware status validation passed")

fn test_command_validation():
    """Test text command validation"""
    print("Testing command validation...")
    
    var validator = SafetyValidator()
    
    # Test safe commands
    assert_true(validator.validate_command("takeoff to 10 meters"))
    assert_true(validator.validate_command("land at current position"))
    assert_true(validator.validate_command("move forward 5 meters"))
    assert_true(validator.validate_command("rotate 90 degrees clockwise"))
    
    # Test dangerous commands
    assert_false(validator.validate_command("attack the target"))
    assert_false(validator.validate_command("crash into building"))
    assert_false(validator.validate_command("destroy the obstacle"))
    assert_false(validator.validate_command("kamikaze mission"))
    
    print("✅ Command validation passed")

fn test_action_vector_validation():
    """Test action vector validation"""
    print("Testing action vector validation...")
    
    var validator = SafetyValidator()
    
    # Test valid action vector
    var valid_action = Tensor[DType.float32](4)
    valid_action[0] = 0.5
    valid_action[1] = -0.3
    valid_action[2] = 0.2
    valid_action[3] = 0.1
    
    assert_true(validator.validate_action_vector(valid_action))
    
    # Test invalid action vector (too large magnitude)
    var invalid_action = Tensor[DType.float32](4)
    invalid_action[0] = 2.0
    invalid_action[1] = 2.0
    invalid_action[2] = 2.0
    invalid_action[3] = 2.0
    
    assert_false(validator.validate_action_vector(invalid_action))
    
    # Test wrong dimensions
    var wrong_dims = Tensor[DType.float32](3)
    wrong_dims[0] = 0.1
    wrong_dims[1] = 0.2
    wrong_dims[2] = 0.3
    
    assert_false(validator.validate_action_vector(wrong_dims))
    
    print("✅ Action vector validation passed")

fn test_safety_score_calculation():
    """Test safety score calculation"""
    print("Testing safety score calculation...")
    
    var validator = SafetyValidator()
    var state = DroneState()
    
    # Test perfect safety score
    state.position = Vector3(0.0, 0.0, 50.0)
    state.velocity = Vector3(1.0, 1.0, 0.0)
    state.battery_level = 100.0
    state.gps_fix = True
    
    var score = validator.get_safety_score(state)
    assert_true(score > 0.8)  # Should be high safety score
    
    # Test degraded safety score
    state.velocity = Vector3(10.0, 10.0, 5.0)  # Higher velocity
    state.battery_level = 30.0  # Lower battery
    
    score = validator.get_safety_score(state)
    assert_true(score < 0.8)  # Should be lower safety score
    
    # Test critical safety score
    state.velocity = Vector3(20.0, 20.0, 10.0)  # Excessive velocity
    state.battery_level = 10.0  # Critical battery
    
    score = validator.get_safety_score(state)
    assert_equal(score, 0.0)  # Should be zero due to critical violations
    
    print("✅ Safety score calculation passed")

fn test_emergency_stop():
    """Test emergency stop functionality"""
    print("Testing emergency stop...")
    
    var validator = SafetyValidator()
    
    # Test emergency stop activation
    assert_false(validator.emergency_stop_active)
    assert_true(validator.emergency_stop())
    assert_true(validator.emergency_stop_active)
    
    # Test emergency stop reset
    assert_true(validator.reset_emergency_stop())
    assert_false(validator.emergency_stop_active)
    
    print("✅ Emergency stop passed")

fn test_python_interface_functions():
    """Test Python interface functions"""
    print("Testing Python interface functions...")
    
    # Test command string validation
    assert_true(validate_command_string("takeoff to 10 meters"))
    assert_false(validate_command_string("attack the target"))
    
    # Test action array validation
    var action = Tensor[DType.float32](4)
    action[0] = 0.1
    action[1] = 0.2
    action[2] = 0.3
    action[3] = 0.4
    
    assert_true(validate_action_array(action))
    
    # Test create safety validator
    var validator = create_safety_validator()
    # Just verify it doesn't crash
    
    print("✅ Python interface functions passed")

fn test_performance_benchmarks():
    """Test performance characteristics"""
    print("Testing performance benchmarks...")
    
    from benchmark import Benchmark
    
    var validator = SafetyValidator()
    var state = DroneState()
    
    # Benchmark state validation
    @parameter
    fn validate_performance():
        _ = validator.validate_state(state)
    
    var bench = Benchmark()
    var report = bench.run[validate_performance]()
    
    # Just verify it runs without crashing
    print("Performance test completed")
    
    print("✅ Performance benchmarks passed")

fn main():
    """Run all safety validator tests"""
    print("🧪 Running Mojo Safety Validator Tests")
    print("=" * 50)
    
    test_vector3_operations()
    test_drone_state_creation()
    test_safety_limits()
    test_velocity_validation()
    test_altitude_validation()
    test_battery_validation()
    test_geofence_validation()
    test_hardware_status_validation()
    test_command_validation()
    test_action_vector_validation()
    test_safety_score_calculation()
    test_emergency_stop()
    test_python_interface_functions()
    test_performance_benchmarks()
    
    print("=" * 50)
    print("✅ All safety validator tests passed!")
    print("🔒 Safety validation system verified")
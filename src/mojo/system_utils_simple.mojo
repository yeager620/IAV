"""
Simplified Mojo system utilities library for drone-vla system.
Focuses on core functionality that works with current Mojo syntax.
"""

import math

fn clamp_int(value: Int, min_val: Int, max_val: Int) -> Int:
    """Clamp integer value between min and max."""
    if value < min_val:
        return min_val
    elif value > max_val:
        return max_val
    else:
        return value

fn lerp(a: Float64, b: Float64, t: Float64) -> Float64:
    """Linear interpolation between two values."""
    return a + t * (b - a)

fn degrees_to_radians(degrees: Float64) -> Float64:
    """Convert degrees to radians."""
    return degrees * math.pi / 180.0

fn radians_to_degrees(radians: Float64) -> Float64:
    """Convert radians to degrees."""
    return radians * 180.0 / math.pi

fn calculate_checksum(data: String) -> Int:
    """Calculate simple checksum for data integrity."""
    var checksum = 0
    for i in range(len(data)):
        checksum += ord(data[i])
    return checksum

fn validate_data_integrity(data: String, expected_checksum: Int) -> Bool:
    """Validate data integrity using checksum."""
    return calculate_checksum(data) == expected_checksum

fn compute_distance(x1: Float64, y1: Float64, x2: Float64, y2: Float64) -> Float64:
    """Compute 2D distance between two points."""
    var dx = x2 - x1
    var dy = y2 - y1
    return math.sqrt(dx * dx + dy * dy)

fn compute_bearing(x1: Float64, y1: Float64, x2: Float64, y2: Float64) -> Float64:
    """Compute bearing from point 1 to point 2 in radians."""
    var dx = x2 - x1
    var dy = y2 - y1
    return math.atan2(dy, dx)

fn normalize_angle(angle: Float64) -> Float64:
    """Normalize angle to [-pi, pi] range."""
    var normalized = angle
    while normalized > math.pi:
        normalized -= 2.0 * math.pi
    while normalized < -math.pi:
        normalized += 2.0 * math.pi
    return normalized

fn compute_control_effort(error: Float64, kp: Float64, ki: Float64, kd: Float64, 
                         integral: Float64, derivative: Float64) -> Float64:
    """Compute PID control effort."""
    return kp * error + ki * integral + kd * derivative

fn saturate_control(control: Float64, max_value: Float64) -> Float64:
    """Saturate control signal to prevent actuator limits."""
    return clamp_value(control, -max_value, max_value)

fn clamp_value(value: Float64, min_val: Float64, max_val: Float64) -> Float64:
    """Clamp float value between min and max."""
    if value < min_val:
        return min_val
    elif value > max_val:
        return max_val
    else:
        return value

fn compute_rms(values: List[Float64]) -> Float64:
    """Compute root mean square of values."""
    var sum_squares = 0.0
    for i in range(len(values)):
        var val = values[i]
        sum_squares += val * val
    return math.sqrt(sum_squares / len(values))

fn find_maximum(values: List[Float64]) -> Float64:
    """Find maximum value in list."""
    if len(values) == 0:
        return 0.0
    
    var max_val = values[0]
    for i in range(1, len(values)):
        if values[i] > max_val:
            max_val = values[i]
    return max_val

fn find_minimum(values: List[Float64]) -> Float64:
    """Find minimum value in list."""
    if len(values) == 0:
        return 0.0
    
    var min_val = values[0]
    for i in range(1, len(values)):
        if values[i] < min_val:
            min_val = values[i]
    return min_val

fn main():
    """Test system utilities."""
    print("Testing simplified Mojo system utilities...")
    
    # Test clamping
    var clamped_int = clamp_int(15, 0, 10)
    print("Clamp int (15 -> 10):", clamped_int)
    
    var clamped_float = clamp_value(15.5, 0.0, 10.0)
    print("Clamp float (15.5 -> 10.0):", clamped_float)
    
    # Test interpolation
    var lerped = lerp(0.0, 10.0, 0.5)
    print("Lerp (0,10,0.5):", lerped)
    
    # Test angle conversion
    var radians = degrees_to_radians(90.0)
    print("90 degrees to radians:", radians)
    
    var degrees = radians_to_degrees(radians)
    print("Back to degrees:", degrees)
    
    # Test checksum
    var test_data = "test_data_123"
    var checksum = calculate_checksum(test_data)
    print("Checksum for 'test_data_123':", checksum)
    
    var is_valid = validate_data_integrity(test_data, checksum)
    print("Data integrity valid:", is_valid)
    
    # Test distance and bearing
    var distance = compute_distance(0.0, 0.0, 3.0, 4.0)
    print("Distance from (0,0) to (3,4):", distance)
    
    var bearing = compute_bearing(0.0, 0.0, 1.0, 1.0)
    print("Bearing from (0,0) to (1,1):", bearing, "radians")
    
    # Test angle normalization
    var normalized = normalize_angle(4.0 * math.pi)
    print("Normalized angle (4π):", normalized)
    
    # Test control functions
    var pid_output = compute_control_effort(1.0, 0.5, 0.1, 0.05, 0.2, 0.3)
    print("PID control output:", pid_output)
    
    var saturated = saturate_control(15.0, 10.0)
    print("Saturated control (15.0 -> 10.0):", saturated)
    
    # Test list operations
    var test_values = List[Float64]()
    test_values.append(1.0)
    test_values.append(5.0)
    test_values.append(3.0)
    test_values.append(8.0)
    test_values.append(2.0)
    
    var max_val = find_maximum(test_values)
    print("Maximum value:", max_val)
    
    var min_val = find_minimum(test_values)
    print("Minimum value:", min_val)
    
    var rms = compute_rms(test_values)
    print("RMS value:", rms)
    
    print("System utilities test completed.")
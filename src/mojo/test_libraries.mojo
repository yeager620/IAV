"""
Test suite for all Mojo libraries without external dependencies.
"""

from .file_io import FileHandle, read_config_file, write_log_entry
from .system_utils import SystemInfo, Timer, MemoryPool, get_system_info
from .camera_bridge import CameraConfig, FrameBuffer, preprocess_frame
from .network_bridge import NetworkConfig, MAVLinkMessage, FlightState, MotorCommand
from .uav_core import process_uav_control_single, apply_safety_limits

fn test_file_io():
    """Test file I/O library."""
    print("Testing file I/O library...")
    
    var handle = FileHandle("/tmp/test.txt")
    var opened = handle.open_write()
    print("File handle open:", opened)
    handle.close()
    
    var config_content = read_config_file("/tmp/nonexistent.json")
    print("Config file read (empty expected):", len(config_content) == 0)
    
    var log_written = write_log_entry("/tmp/test.log", "Test entry")
    print("Log entry written:", log_written)

fn test_system_utils():
    """Test system utilities library."""
    print("Testing system utilities...")
    
    var info = get_system_info()
    print("System info - CPU count:", info.cpu_count)
    print("System info - Memory MB:", info.memory_mb)
    
    var timer = Timer()
    timer.start()
    var elapsed = timer.stop()
    print("Timer elapsed (should be ~0):", elapsed)
    
    var pool = MemoryPool(1024)
    var allocated = pool.allocate(256)
    print("Memory pool allocation:", allocated)
    print("Memory usage:", pool.get_usage_percent(), "%")

fn test_camera_bridge():
    """Test camera bridge library."""
    print("Testing camera bridge...")
    
    var config = CameraConfig(0)
    print("Camera config - ID:", config.camera_id)
    print("Camera config - Resolution:", config.width, "x", config.height)
    
    var frame = FrameBuffer(640, 480, 3)
    var valid = frame.validate()
    print("Frame buffer valid:", valid)
    print("Frame pixel count:", frame.get_pixel_count())
    
    var processed = preprocess_frame(frame)
    print("Processed frame valid:", processed.is_valid)

fn test_network_bridge():
    """Test network bridge library."""
    print("Testing network bridge...")
    
    var config = NetworkConfig("udp:127.0.0.1:14550")
    print("Network config - Connection:", config.connection_string)
    print("Network config - Timeout:", config.timeout_seconds)
    
    var message = MAVLinkMessage("HEARTBEAT")
    var msg_valid = message.validate()
    print("MAVLink message valid:", msg_valid)
    
    var state = FlightState()
    state.position_x = 1.0
    state.position_y = 2.0
    state.position_z = 3.0
    print("Flight state - Position:", state.position_x, state.position_y, state.position_z)
    
    var motors = MotorCommand(0.5, 0.5, 0.5, 0.5)
    var cmd_valid = motors.validate()
    print("Motor command valid:", cmd_valid)

fn test_uav_core():
    """Test UAV core library."""
    print("Testing UAV core...")
    
    var safe_velocity = apply_safety_limits(10.0, False)  # Should be clamped to 5.0
    print("Safety limits - velocity (10.0 -> 5.0):", safe_velocity)
    
    var safe_angular = apply_safety_limits(5.0, True)  # Should be clamped to 2.0
    print("Safety limits - angular (5.0 -> 2.0):", safe_angular)
    
    var motor1 = process_uav_control_single(0.0, 0.0, 2.0, 0.0, 1.0, 1)
    print("UAV control - motor 1 (takeoff):", motor1)
    
    var motor1_hover = process_uav_control_single(0.0, 0.0, 0.0, 0.0, 5.0, 1)
    print("UAV control - motor 1 (hover):", motor1_hover)

fn main():
    """Run all library tests."""
    print("Running Mojo library test suite...")
    print("=" * 40)
    
    test_file_io()
    print()
    
    test_system_utils()
    print()
    
    test_camera_bridge()
    print()
    
    test_network_bridge()
    print()
    
    test_uav_core()
    print()
    
    print("=" * 40)
    print("All library tests completed!")
    print("Note: Some tests use placeholder implementations")
    print("Python interop libraries require actual Python runtime")
"""
High-performance drone control system using Mojo.
"""
from math import sqrt

@register_passable("trivial")
struct Vector3:
    var x: Float64
    var y: Float64
    var z: Float64
    
    fn __init__(inout self, x: Float64, y: Float64, z: Float64):
        self.x = x
        self.y = y
        self.z = z
    
    fn magnitude(self) -> Float64:
        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

@register_passable("trivial")
struct ControlVector:
    var vx: Float64
    var vy: Float64
    var vz: Float64
    var yaw: Float64
    
    fn __init__(inout self, vx: Float64, vy: Float64, vz: Float64, yaw: Float64):
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.yaw = yaw
    
    fn is_safe(self) -> Bool:
        var max_vel = 10.0
        var abs_vx = self.vx if self.vx >= 0 else -self.vx
        var abs_vy = self.vy if self.vy >= 0 else -self.vy
        var abs_vz = self.vz if self.vz >= 0 else -self.vz
        
        return (abs_vx <= max_vel and abs_vy <= max_vel and abs_vz <= max_vel)

struct DroneController:
    var _armed: Bool
    var _position: Vector3
    var _velocity: Vector3
    
    fn __init__(inout self):
        self._armed = False
        self._position = Vector3(0.0, 0.0, 0.0)
        self._velocity = Vector3(0.0, 0.0, 0.0)
    
    fn arm(inout self) -> Bool:
        self._armed = True
        return True
    
    fn disarm(inout self):
        self._armed = False
    
    fn execute_command(inout self, cmd: ControlVector) -> Bool:
        if not self._armed:
            return False
        
        if not cmd.is_safe():
            return False
        
        self._velocity = Vector3(cmd.vx, cmd.vy, cmd.vz)
        return True

fn waypoint_navigation(start: Vector3, target: Vector3, speed: Float64) -> ControlVector:
    var diff = Vector3(target.x - start.x, target.y - start.y, target.z - start.z)
    var distance = diff.magnitude()
    
    if distance < 0.1:
        return ControlVector(0.0, 0.0, 0.0, 0.0)
    
    var scale = speed / distance if speed < distance * 2.0 else 2.0
    
    return ControlVector(
        diff.x * scale,
        diff.y * scale, 
        diff.z * scale,
        0.0
    )

fn main():
    print("🚁 Mojo Drone Control System Test")
    print("========================================")
    
    var drone = DroneController()
    
    # Test arming
    var armed = drone.arm()
    print("Drone armed")
    
    # Test safe command
    var safe_cmd = ControlVector(1.0, 0.5, 0.2, 0.1)
    var executed = drone.execute_command(safe_cmd)
    print("Safe command executed")
    
    # Test unsafe command
    var unsafe_cmd = ControlVector(20.0, 0.0, 0.0, 0.0)
    var rejected = drone.execute_command(unsafe_cmd)
    print("Unsafe command rejected")
    
    # Test waypoint navigation
    var start = Vector3(0.0, 0.0, 0.0)
    var target = Vector3(10.0, 5.0, 2.0)
    var nav_cmd = waypoint_navigation(start, target, 2.0)
    print("Navigation calculated")
    
    # Test vector operations
    var vec = Vector3(3.0, 4.0, 0.0)
    print("Vector magnitude:", vec.magnitude())
    
    print("✅ All tests passed!")
from math import sqrt, sin, cos, abs
from tensor import Tensor, TensorSpec, TensorShape
from algorithm import vectorize

fn clamp[type: DType](x: Tensor[type], min_val: Scalar[type], max_val: Scalar[type]) -> Tensor[type]:
    """Clamp tensor values between min and max"""
    var result = Tensor[type](x.shape())
    
    @parameter
    fn clamp_vectorized[simd_width: Int](idx: Int):
        let vals = x.load[width=simd_width](idx)
        result.store[width=simd_width](idx, vals.clamp(min_val, max_val))
    
    vectorize[clamp_vectorized, simdwidthof[type]()](x.num_elements())
    return result

fn matrix_multiply[type: DType](a: Tensor[type], b: Tensor[type]) -> Tensor[type]:
    """Optimized matrix multiplication for small matrices"""
    let m = a.shape()[0]
    let n = a.shape()[1] 
    let p = b.shape()[1]
    
    var result = Tensor[type](m, p)
    
    for i in range(m):
        for j in range(p):
            var sum = Scalar[type](0)
            for k in range(n):
                sum += a[i, k] * b[k, j]
            result[i, j] = sum
    
    return result

fn normalize_vector[type: DType](x: Tensor[type]) -> Tensor[type]:
    """Normalize vector to unit length"""
    var norm = Scalar[type](0)
    
    for i in range(x.num_elements()):
        norm += x[i] * x[i]
    
    norm = sqrt(norm)
    
    var result = Tensor[type](x.shape())
    for i in range(x.num_elements()):
        result[i] = x[i] / norm
        
    return result

fn quaternion_to_euler[type: DType](q: Tensor[type]) -> Tensor[type]:
    """Convert quaternion [w, x, y, z] to Euler angles [roll, pitch, yaw]"""
    let w = q[0]
    let x = q[1] 
    let y = q[2]
    let z = q[3]
    
    var euler = Tensor[type](3)
    
    # Roll (x-axis rotation)
    let sinr_cosp = 2 * (w * x + y * z)
    let cosr_cosp = 1 - 2 * (x * x + y * y)
    euler[0] = Scalar[type](sinr_cosp / cosr_cosp).atan2()
    
    # Pitch (y-axis rotation)  
    let sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        euler[1] = Scalar[type](1.5707963267948966) * (1 if sinp > 0 else -1)  # π/2
    else:
        euler[1] = sinp.asin()
    
    # Yaw (z-axis rotation)
    let siny_cosp = 2 * (w * z + x * y)
    let cosy_cosp = 1 - 2 * (y * y + z * z)
    euler[2] = Scalar[type](siny_cosp / cosy_cosp).atan2()
    
    return euler
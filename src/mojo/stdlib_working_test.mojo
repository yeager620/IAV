"""
Test working Mojo standard library features.
"""

from os import getenv
from os.path import exists, join
from pathlib import Path
from collections import List
import math

fn test_basic_operations():
    """Test basic operations that work."""
    print("Testing basic operations...")
    
    # Test path operations
    var test_path = "/tmp"
    var file_exists = exists(test_path)
    print("Path /tmp exists:", file_exists)
    
    # Test path joining
    var joined_path = join("/tmp", "test.txt")
    print("Joined path:", joined_path)

fn test_environment_simple():
    """Test environment variable access."""
    print("Testing environment...")
    
    # Test getting environment variable
    var home_dir = getenv("HOME")
    print("HOME directory length:", len(home_dir))

fn test_collections_simple():
    """Test collection types."""
    print("Testing collections...")
    
    # Test List
    var test_list = List[Float64]()
    test_list.append(1.0)
    test_list.append(2.0)
    test_list.append(3.0)
    print("List size:", len(test_list))

fn test_math_operations():
    """Test math operations."""
    print("Testing math operations...")
    
    var pi_value = math.pi
    var sqrt_value = math.sqrt(16.0)
    var sin_value = math.sin(pi_value / 2.0)
    
    print("Pi:", pi_value)
    print("Sqrt(16):", sqrt_value)
    print("Sin(π/2):", sin_value)

fn test_pathlib():
    """Test pathlib operations."""
    print("Testing pathlib...")
    
    var p = Path("/usr/local/bin")
    # Just test that Path can be created
    print("Path object created successfully")

fn main():
    """Test working standard library features."""
    print("Testing Working Mojo Standard Library Features")
    print("=" * 50)
    
    test_basic_operations()
    print()
    
    test_environment_simple()
    print()
    
    test_collections_simple()
    print()
    
    test_math_operations()
    print()
    
    test_pathlib()
    print()
    
    print("=" * 50)
    print("Standard library has sufficient features!")
    print("We can use built-in modules instead of writing from scratch.")
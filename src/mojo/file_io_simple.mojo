"""
Simplified Mojo file I/O library for drone-vla system.
Basic file operations using current Mojo syntax.
"""

from os.path import exists

fn check_file_exists(path: String) -> Bool:
    """Check if file exists."""
    try:
        return exists(path)
    except:
        return False

fn validate_file_path(path: String) -> Bool:
    """Validate file path format."""
    return len(path) > 0 and path != ""

fn get_file_extension(path: String) -> String:
    """Get file extension from path."""
    var dot_index = -1
    for i in range(len(path) - 1, -1, -1):
        if path[i] == '.':
            dot_index = i
            break
    
    if dot_index >= 0 and dot_index < len(path) - 1:
        return path[dot_index + 1:]
    else:
        return ""

fn validate_config_extension(path: String) -> Bool:
    """Validate that file has config extension."""
    var ext = get_file_extension(path)
    return ext == "json" or ext == "toml" or ext == "yaml"

fn validate_log_extension(path: String) -> Bool:
    """Validate that file has log extension."""
    var ext = get_file_extension(path)
    return ext == "log" or ext == "txt"

fn compute_path_depth(path: String) -> Int:
    """Compute depth of path (number of directory separators)."""
    var depth = 0
    for i in range(len(path)):
        if path[i] == '/':
            depth += 1
    return depth

fn is_absolute_path(path: String) -> Bool:
    """Check if path is absolute."""
    return len(path) > 0 and path[0] == '/'

fn join_paths(base: String, relative: String) -> String:
    """Join two paths together."""
    if len(base) == 0:
        return relative
    elif len(relative) == 0:
        return base
    elif base[len(base) - 1] == '/':
        return base + relative
    else:
        return base + "/" + relative

fn get_directory_name(path: String) -> String:
    """Get directory name from path."""
    var last_slash = -1
    for i in range(len(path) - 1, -1, -1):
        if path[i] == '/':
            last_slash = i
            break
    
    if last_slash >= 0:
        return path[:last_slash]
    else:
        return "."

fn get_file_name(path: String) -> String:
    """Get file name from path."""
    var last_slash = -1
    for i in range(len(path) - 1, -1, -1):
        if path[i] == '/':
            last_slash = i
            break
    
    if last_slash >= 0:
        return path[last_slash + 1:]
    else:
        return path

fn create_backup_path(original_path: String) -> String:
    """Create backup path by appending .bak."""
    return original_path + ".bak"

fn create_temp_path(original_path: String) -> String:
    """Create temporary path by appending .tmp."""
    return original_path + ".tmp"

fn main():
    """Test file I/O functions."""
    print("Testing simplified Mojo file I/O library...")
    
    # Test path validation
    var valid_path = validate_file_path("/tmp/test.log")
    print("Valid path '/tmp/test.log':", valid_path)
    
    var invalid_path = validate_file_path("")
    print("Invalid path '':", invalid_path)
    
    # Test extension checking
    var config_ext = validate_config_extension("/config/settings.json")
    print("Config extension valid:", config_ext)
    
    var log_ext = validate_log_extension("/logs/drone.log")
    print("Log extension valid:", log_ext)
    
    # Test path operations
    var extension = get_file_extension("/path/to/file.txt")
    print("Extension of '/path/to/file.txt':", extension)
    
    var depth = compute_path_depth("/usr/local/bin/program")
    print("Path depth:", depth)
    
    var is_abs = is_absolute_path("/usr/local/bin")
    print("Is absolute path:", is_abs)
    
    var is_rel = is_absolute_path("relative/path")
    print("Is relative path:", is_rel)
    
    # Test path joining
    var joined = join_paths("/home/user", "documents/file.txt")
    print("Joined path:", joined)
    
    var joined_with_slash = join_paths("/home/user/", "documents/file.txt")
    print("Joined with slash:", joined_with_slash)
    
    # Test path components
    var dir_name = get_directory_name("/usr/local/bin/program")
    print("Directory name:", dir_name)
    
    var file_name = get_file_name("/usr/local/bin/program")
    print("File name:", file_name)
    
    # Test backup and temp paths
    var backup_path = create_backup_path("/config/settings.json")
    print("Backup path:", backup_path)
    
    var temp_path = create_temp_path("/logs/drone.log")
    print("Temp path:", temp_path)
    
    # Test file existence (with actual file)
    var exists_check = check_file_exists("/tmp")
    print("'/tmp' exists:", exists_check)
    
    var missing_check = check_file_exists("/nonexistent/path")
    print("'/nonexistent/path' exists:", missing_check)
    
    print("File I/O library test completed.")
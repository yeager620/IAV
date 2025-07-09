# IAV: Intelligent Aerial Vehicle

## Project Goals 

### Functionality-related goals:

- **State-of-the-art AI capabilities:** World modeling, natural language understanding, ability to reason and plan
- **Robust and flexible multimodel data processing:** Video / image data through camera(s), Audio data through microphone, LIDAR, etc.
- **Blazingly fast inference and communication with sensors and actuators:** production-ready ai product, not a research toy or proof of concept
- **Lightweight and scalable hardware needs:** Ability to be mass produced and inexpensive for consumer usage
- **Robust system programming:** Fault tolerance and resilience, data consistency and integrity, high availability and responsiveness

### Internal development-related goals:

- **Streamlined development and deployment cycle:** package / environment management, automated workflows, version control, pre / post training pipelines, unit / integration testing, containerization and deployment
- **Flexible, modular, extensible APIs**
- **Ability to iterate, prototype, and test quickly**
- **Precise and correct program specifications:** 

### System Components Flowchart

```mermaid
flowchart TD
    %% Input Layer (Python)
    CAM[Camera System<br/>OpenCV + YOLOv8]
    CMD[Natural Language<br/>Commands]
    MAV[MAVLink Interface<br/>Flight Controller]
    
    %% AI/ML Layer (Python)
    VLA[VLA Model<br/>V-JEPA2 Vision-Language-Action<br/>PyTorch/ONNX]
    
    %% Processing Layer (Python Orchestration)
    VISION[Vision System<br/>Frame Processing<br/>Object Actions]
    SAFETY_PY[Safety Monitor<br/>High-level Validation]
    AUTON[Autonomous System<br/>Mission Planning<br/>State Management]
    
    %% Performance Layer (Mojo)
    SAFETY_MOJO[Safety Validator<br/>Real-time SIMD Validation]
    CTRL_MOJO[Control System<br/>PID Controllers<br/>Quaternion Math]
    VISION_MOJO[Vision Pipeline<br/>SIMD Image Processing]
    
    %% Hardware Interface (Python)
    UNIFIED[Unified Drone Controller<br/>DroneKit + MAVLink]
    MOTORS[Motor Commands<br/>Flight Controller]
    
    %% High-Performance Bridges (Mojo)
    CAM_BRIDGE[Camera Bridge<br/>Mojo SIMD Processing]
    NET_BRIDGE[Network Bridge<br/>Mojo MAVLink Interface]
    
    %% Data Flow - Input Processing
    CAM --> CAM_BRIDGE
    CAM_BRIDGE --> VISION_MOJO
    CMD --> VLA
    VISION --> VLA
    VLA --> AUTON
    
    %% Data Flow - Safety and Control
    AUTON --> SAFETY_PY
    SAFETY_PY --> SAFETY_MOJO
    SAFETY_MOJO --> CTRL_MOJO
    CTRL_MOJO --> UNIFIED
    UNIFIED --> MOTORS
    
    %% Data Flow - Telemetry
    MAV --> NET_BRIDGE
    NET_BRIDGE --> UNIFIED
    UNIFIED --> NET_BRIDGE
    NET_BRIDGE --> MAV
    
    %% Safety Override Paths
    SAFETY_MOJO -.-> UNIFIED
    SAFETY_PY -.-> MOTORS
    
    %% Styling
    classDef input fill:#e1f5fe
    classDef ai fill:#f3e5f5
    classDef processing fill:#fff3e0
    classDef control fill:#e8f5e8
    classDef output fill:#ffebee
    classDef mojo fill:#fff8e1
    
    class CAM,CMD,MAV input
    class VLA ai
    class VISION,SAFETY_PY,AUTON processing
    class UNIFIED control
    class MOTORS output
    class SAFETY_MOJO,CTRL_MOJO,VISION_MOJO,CAM_BRIDGE,NET_BRIDGE mojo
```

### Process Lifetime Diagram

```mermaid
gantt
    title UAV System Process Lifecycle (Continuous Operation)
    dateFormat YYYY-MM-DD
    axisFormat %H:%M:%S
    
    section System Initialization
    Config Loading         :2024-01-01, 1s
    MAVLink Connection     :after Config Loading, 1s
    Camera Initialization  :after MAVLink Connection, 1s
    Safety System Init     :after Camera Initialization, 1s
    System Ready          :milestone, after Safety System Init, 0s
    
    section Continuous Processes
    Main Control Loop (100Hz)    :active, main_ctrl, after System Ready, 116s
    Camera Capture (30Hz)        :active, camera, after System Ready, 116s
    Drone Control (10Hz)         :active, drone_ctrl, after System Ready, 116s
    Safety Monitor (10Hz)        :active, safety, after System Ready, 116s
    
    section Event-Driven Processes
    User Command Processing      :crit, cmd1, after System Ready, 2s
    VLA Model Inference         :crit, vla1, after cmd1, 2s
    Mission Planning            :crit, mission1, after vla1, 2s
    
    section Autonomous Mode
    Autonomous Loop (10Hz)      :active, auto_loop, after mission1, 40s
    Object Detection           :active, obj_detect, after mission1, 40s
    Action Generation          :active, action_gen, after mission1, 40s
    
    section Emergency Scenarios
    Emergency Detection        :crit, emergency1, after auto_loop, 1s
    Emergency Landing          :crit, landing1, after emergency1, 4s
    System Recovery           :recovery1, after landing1, 5s
    
    section Second Mission Cycle
    User Command 2            :crit, cmd2, after recovery1, 2s
    VLA Inference 2           :crit, vla2, after cmd2, 2s
    Mission Execution 2       :crit, mission2, after vla2, 7s
    
    section Continuous Operation
    Background Monitoring     :active, monitor, after System Ready, 116s
    Telemetry Logging        :active, telemetry, after System Ready, 116s
    Health Checks            :active, health, after System Ready, 116s
```

### Process Interaction Timeline

```mermaid
sequenceDiagram
    participant U as User/Environment
    participant M as Main Process
    participant C as Camera Thread
    participant D as Drone Controller
    participant A as Autonomous System
    participant S as Safety Monitor
    participant H as Hardware (MAVLink)
    
    Note over M: System Startup (t=0)
    M->>C: Initialize Camera
    M->>D: Initialize Drone Controller
    M->>A: Initialize Autonomous System
    M->>S: Initialize Safety Monitor
    M->>H: Connect MAVLink
    
    Note over M: Continuous Operation Loop
    
    loop Every 33ms (30Hz)
        C->>C: Capture Frame
        C->>A: Send Frame to Vision Queue
    end
    
    loop Every 10ms (100Hz)
        M->>D: Process Control Commands
        M->>S: Safety Validation
        D->>H: Send Motor Commands
        H->>D: Receive Telemetry
    end
    
    loop Every 100ms (10Hz)
        A->>A: Process Autonomous Loop
        D->>D: Update Control State
        S->>S: Monitor Safety Conditions
    end
    
    Note over U: User Interaction Event
    U->>M: Natural Language Command
    M->>A: Route Command to Autonomous System
    A->>A: VLA Model Inference (1-2s)
    A->>S: Validate Actions
    S->>A: Safety Approval/Rejection
    A->>D: Execute Flight Commands
    D->>H: Motor Control Signals
    
    Note over S: Emergency Scenario
    S->>S: Detect Safety Violation
    S->>A: Emergency Stop Signal
    S->>D: Emergency Landing Command
    D->>H: Emergency Motor Commands
    A->>A: Transition to Emergency State
    
    Note over M: Recovery Process
    M->>A: Reset Autonomous System
    A->>S: Request Safety Clearance
    S->>A: Safety System Ready
    A->>M: System Ready for Commands
    
    Note over M: Continuous Loop Resumes
```

### Core Components

- **Mojo Core**: Performance-critical flight control, safety monitoring, and VLA inference
- **Python Interface**: Hardware I/O, MAVLink communication, and vision capture
- **Safety System**: Real-time constraint validation and emergency procedures
- **Control Allocator**: 6DOF action mapping to motor commands


##  Quick Start

```bash
# Activate pixi environment
source activate-pixi.sh

# Build and validate
pixi run build

# Run tests
pixi run test

# Start development (simulation mode)
pixi run dev
```

### Basic Flight

```bash
# Start with default configuration
pixi run python main.py

# Use simulation mode  
pixi run python main.py --simulation

# Execute single command
python main.py --command "takeoff to 5 meters"

# Custom configuration
python main.py --config config/custom_config.json
```

### Configuration

Edit `config/minimal_config.json`:

```json
{
  "mavlink": {
    "connection": "/dev/ttyUSB0"
  },
  "camera_id": 0,
  "control_frequency": 100
}
```

### Simulation

For SITL testing:

```bash
# Start SITL simulator first
python main.py --simulation
```

### Setup & Installation

```bash
# Prerequisites: Install pixi
curl -fsSL https://pixi.sh/install.sh | bash

# Clone and setup
git clone <repository-url>
cd drone-vla

# Install unified environment
pixi install

# Activate environment (optional)
source activate-pixi.sh
```

### Building

```bash
# Build and validate all components
pixi run build

# Run comprehensive tests
pixi run test

# Development workflow
pixi run dev
```

### Testing

```bash
# Run full test suite
pixi run test

# Or run pytest directly in pixi environment
pixi run python -m pytest tests/ -v

# Test specific component
pixi run python -m pytest tests/test_minimal_interface.py -v
```


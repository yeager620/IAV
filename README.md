# IAV 

Building an intelligent, robust, and lightweight aerial vehicle with the ability to plan and execute missions and interact with environment through a flexible set of actuators and end-affectors

## Project Goals 

### Functionality-related goals:

- **State-of-the-art AI capabilities:** World modeling, natural language understanding, ability to reason and plan
- **Robust and flexible multimodel data processing:** Video / image data through camera(s), Audio data through microphone, LIDAR, etc.
- **Blazingly fast inference and communication with sensors and actuators:** production-ready ai product, not a research toy or proof of concept
- **Lightweight and scalable hardware needs:** Ability to be mass produced and inexpensive for consumer usage
- **Robust system programming:** Fault tolerance and resilience, data consistency and integrity, high availability and responsiveness

### Internal development-related goals:

- **Streamlined development and deployment cycle:** package / environment management, automated workflows, version control, unit / integration testing, containerization and deployment
- **Flexible, modular, extensible APIs**
- **Ability to iterate, prototype, and test quickly**
- **Precise and correct program specifications:** 

## Current Architecture

```
Hardware I/O (Python) ; Mojo Core ; Hardware I/O (Python)
  MAVLink              UAV System      Motor Commands
  Camera               VLA + Safety    
  Commands             + Control       
```

### System Components Flowchart

```mermaid
flowchart TD
    %% Input Layer
    CAM[Camera System<br/>YOLOv8 Object Detection]
    CMD[Natural Language<br/>Commands]
    MAV[MAVLink Interface<br/>Flight Controller]
    
    %% AI/ML Layer
    VLA[VLA Model<br/>Vision-Language-Action<br/>VJEPA2 Neural Network]
    
    %% Processing Layer
    VISION[Vision System<br/>Frame Processing<br/>Object Actions]
    SAFETY[Safety Monitor<br/>Command Validation<br/>Constraint Enforcement]
    AUTON[Autonomous System<br/>Mission Planning<br/>State Management]
    
    %% Control Layer
    CTRL[Flight Controller<br/>DroneController Class<br/>6DOF Control]
    MOJO_CORE[Mojo Core<br/>Performance-Critical<br/>Control Logic]
    MOJO_UAV[UAV Core Functions<br/>Motor Mixing<br/>Control Allocation]
    
    %% Output Layer
    MOTORS[Motor Commands<br/>Quadcopter Control]
    
    %% High-Performance Bridges
    CAM_BRIDGE[Camera Bridge<br/>Mojo]
    NET_BRIDGE[Network Bridge<br/>Mojo]
    
    %% Data Flow
    CAM --> CAM_BRIDGE
    CAM_BRIDGE --> VISION
    CMD --> VLA
    VISION --> VLA
    VLA --> AUTON
    AUTON --> SAFETY
    SAFETY --> CTRL
    CTRL --> MOJO_CORE
    MOJO_CORE --> MOJO_UAV
    MOJO_UAV --> MOTORS
    
    MAV --> NET_BRIDGE
    NET_BRIDGE --> CTRL
    CTRL --> NET_BRIDGE
    NET_BRIDGE --> MAV
    
    %% Safety Override
    SAFETY -.-> MOTORS
    SAFETY -.-> MAV
    
    %% Styling
    classDef input fill:#e1f5fe
    classDef ai fill:#f3e5f5
    classDef processing fill:#fff3e0
    classDef control fill:#e8f5e8
    classDef output fill:#ffebee
    classDef mojo fill:#fff8e1
    
    class CAM,CMD,MAV input
    class VLA ai
    class VISION,SAFETY,AUTON processing
    class CTRL control
    class MOTORS output
    class MOJO_CORE,MOJO_UAV,CAM_BRIDGE,NET_BRIDGE mojo
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


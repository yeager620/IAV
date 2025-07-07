#!/bin/bash
# Deployment script for UAV VLA system

set -e

echo "Deploying UAV VLA system to hardware..."

# Configuration
JETSON_HOST="jetson@192.168.1.100"
PI_HOST="pi@192.168.1.101"
DEPLOY_DIR="/tmp/uav_deploy"
JETSON_DEPLOY_PATH="/home/jetson/uav_system"
PI_DEPLOY_PATH="/home/pi/uav_system"

# Check if we're in the right directory
if [ ! -f "pixi.toml" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Build Mojo components for ARM64
echo "Building Mojo components for ARM64..."
export PATH="/Users/$(whoami)/.modular/bin:$PATH"

mkdir -p build/arm64

# Cross-compile Mojo components (placeholder - actual cross-compilation would be different)
echo "Cross-compiling Mojo components..."
magic run mojo build src/mojo/vla_inference.mojo -o build/arm64/vla_inference
magic run mojo build src/mojo/control_allocator.mojo -o build/arm64/control_allocator
magic run mojo build src/mojo/safety_monitor.mojo -o build/arm64/safety_monitor

echo "Mojo components built successfully"

# Create deployment package
echo "Creating deployment package..."
rm -rf $DEPLOY_DIR
mkdir -p $DEPLOY_DIR

# Copy source code
cp -r src/ $DEPLOY_DIR/
cp -r config/ $DEPLOY_DIR/
cp -r build/ $DEPLOY_DIR/
cp main.py $DEPLOY_DIR/
cp pixi.toml $DEPLOY_DIR/

# Copy deployment scripts
mkdir -p $DEPLOY_DIR/scripts
cat > $DEPLOY_DIR/scripts/install.sh << 'EOF'
#!/bin/bash
# Installation script for hardware deployment

set -e

echo "Installing UAV VLA system..."

# Update system
sudo apt update

# Install Python dependencies
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.cargo/env
fi

# Install system dependencies
sudo apt install -y python3-opencv python3-serial

# Install Magic/Mojo (if on Jetson)
if command -v nvidia-smi &> /dev/null; then
    echo "Installing Mojo on Jetson..."
    curl -ssL https://magic.modular.com/$(cat ~/.magic_key) | bash
    export PATH="$HOME/.modular/bin:$PATH"
fi

# Install Python packages
uv add pymavlink opencv-python numpy

# Make scripts executable
chmod +x scripts/*.sh

# Setup systemd service
sudo cp scripts/uav-system.service /etc/systemd/system/
sudo systemctl daemon-reload

echo "Installation completed"
EOF

# Create hardware-specific configurations
cat > $DEPLOY_DIR/config/jetson_config.json << EOF
{
  "system": {
    "control_frequency": 100,
    "command_timeout": 5.0,
    "model_path": "build/arm64/vla_inference"
  },
  "mavlink": {
    "connection": "/dev/ttyUSB0",
    "baud_rate": 921600
  },
  "vision": {
    "camera_id": 0,
    "frame_width": 640,
    "frame_height": 480,
    "fps": 30,
    "target_width": 224,
    "target_height": 224,
    "sequence_length": 16
  },
  "hardware_role": "compute_primary"
}
EOF

cat > $DEPLOY_DIR/config/pi_config.json << EOF
{
  "system": {
    "control_frequency": 100,
    "command_timeout": 5.0
  },
  "mavlink": {
    "connection": "/dev/ttyUSB0", 
    "baud_rate": 921600
  },
  "sensors": {
    "lidar_port": "/dev/ttyUSB1",
    "imu_i2c": "/dev/i2c-1"
  },
  "hardware_role": "io_controller"
}
EOF

# Create systemd service file
cat > $DEPLOY_DIR/scripts/uav-system.service << 'EOF'
[Unit]
Description=UAV VLA System
After=network.target

[Service]
Type=simple
User=jetson
WorkingDirectory=/home/jetson/uav_system
ExecStart=/home/jetson/uav_system/.pixi/envs/default/bin/python main.py --config config/jetson_config.json
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Create startup scripts
cat > $DEPLOY_DIR/scripts/start_jetson.sh << 'EOF'
#!/bin/bash
export PATH="$HOME/.modular/bin:$PATH"
python main.py --config config/jetson_config.json
EOF

cat > $DEPLOY_DIR/scripts/start_pi.sh << 'EOF'  
#!/bin/bash
python main.py --config config/pi_config.json
EOF

chmod +x $DEPLOY_DIR/scripts/*.sh

# Create archive
echo "Creating deployment archive..."
cd /tmp
tar -czf uav_system.tar.gz -C uav_deploy .
cd - > /dev/null

# Deploy to Jetson
echo "Deploying to Jetson..."
scp -i ~/.ssh/jetson_key /tmp/uav_system.tar.gz $JETSON_HOST:~/

ssh -i ~/.ssh/jetson_key $JETSON_HOST << 'JETSON_SCRIPT'
set -e
echo "Installing on Jetson..."

# Backup existing installation
if [ -d "uav_system" ]; then
    mv uav_system uav_system.backup.$(date +%Y%m%d_%H%M%S)
fi

# Extract new installation
tar -xzf uav_system.tar.gz
mv uav_deploy uav_system
cd uav_system

# Run installation script
./scripts/install.sh

echo "Jetson deployment completed"
JETSON_SCRIPT

# Deploy to Raspberry Pi
echo "Deploying to Raspberry Pi..."
scp -i ~/.ssh/pi_key /tmp/uav_system.tar.gz $PI_HOST:~/

ssh -i ~/.ssh/pi_key $PI_HOST << 'PI_SCRIPT'
set -e
echo "Installing on Raspberry Pi..."

# Backup existing installation
if [ -d "uav_system" ]; then
    mv uav_system uav_system.backup.$(date +%Y%m%d_%H%M%S)
fi

# Extract new installation
tar -xzf uav_system.tar.gz
mv uav_deploy uav_system
cd uav_system

# Run installation script (Pi version)
./scripts/install.sh

echo "Raspberry Pi deployment completed"
PI_SCRIPT

# Cleanup
rm -rf $DEPLOY_DIR
rm /tmp/uav_system.tar.gz

echo ""
echo "Deployment completed successfully!"
echo ""
echo "To start the system:"
echo "  Jetson: ssh jetson './uav_system/scripts/start_jetson.sh'"
echo "  Pi:     ssh pi './uav_system/scripts/start_pi.sh'"
echo ""
echo "To enable auto-start:"
echo "  ssh jetson 'sudo systemctl enable uav-system'"
echo ""
echo "To view logs:"
echo "  ssh jetson 'tail -f uav_system/data/logs/system.log'"
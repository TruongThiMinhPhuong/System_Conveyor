#!/bin/bash

#============================================
# Raspberry Pi Setup Script
# AI Fruit Sorting Conveyor System
#============================================

echo "======================================"
echo "AI Fruit Sorting System - Installation"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Raspberry Pi
if [ ! -f /proc/device-tree/model ] || ! grep -q "Raspberry Pi" /proc/device-tree/model; then
    echo -e "${YELLOW}Warning: This script is designed for Raspberry Pi${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Step 0: Setting up swap space (4GB)..."
echo "This is CRITICAL for training AI models on Pi!"

# Check current swap
current_swap=$(free -m | awk '/Swap:/ {print $2}')
if [ "$current_swap" -lt 4000 ]; then
    echo "Current swap is ${current_swap}MB, increasing to 4GB..."
    
    sudo dphys-swapfile swapoff
    sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=4096/' /etc/dphys-swapfile
    sudo dphys-swapfile setup
    sudo dphys-swapfile swapon
    
    new_swap=$(free -m | awk '/Swap:/ {print $2}')
    echo -e "${GREEN}✓ Swap increased to ${new_swap}MB${NC}"
else
    echo -e "${GREEN}✓ Swap already configured (${current_swap}MB)${NC}"
fi

echo ""
echo "Step 1: Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

echo ""
echo "Step 2: Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    libopencv-dev \
    python3-opencv \
    libatlas-base-dev \
    libjasper-dev \
    libqt4-test \
    libqtgui4 \
    libhdf5-dev \
    libhdf5-103 \
    libcap-dev \
    libffi-dev \
    git \
    cmake

echo ""
echo "Step 3: Installing camera dependencies..."
sudo apt-get install -y \
    libcamera-dev \
    python3-libcamera \
    python3-kms++ \
    python3-picamera2

echo ""
echo "Step 4: Enabling camera and GPIO..."
# Enable camera
sudo raspi-config nonint do_camera 0

# Enable I2C and SPI (useful for sensors)
sudo raspi-config nonint do_i2c 0
sudo raspi-config nonint do_spi 0

echo ""
echo "Step 5: Creating virtual environment..."
cd "$(dirname "$0")"
python3 -m venv venv
source venv/bin/activate

echo ""
echo "Step 6: Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo ""
echo "Step 7: Installing Python packages..."
pip install -r requirements.txt

echo ""
echo "Step 8: Creating directories..."
mkdir -p models
mkdir -p logs
mkdir -p data
mkdir -p datasets

echo ""
echo "Step 9: Setting GPIO permissions..."
sudo usermod -a -G gpio $USER
sudo usermod -a -G i2c $USER

echo ""
echo "Step 10: Testing installations..."
python3 - << EOF
import sys
print("\n🧪 Testing Python packages...")

try:
    import cv2
    print(f"✅ OpenCV: {cv2.__version__}")
except ImportError as e:
    print(f"❌ OpenCV: {e}")

try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy: {e}")

try:
    from ultralytics import YOLO
    print(f"✅ Ultralytics YOLOv8: Installed")
except ImportError as e:
    print(f"❌ Ultralytics: {e}")

try:
    import RPi.GPIO as GPIO
    print(f"✅ RPi.GPIO: Available")
    GPIO.setmode(GPIO.BCM)
    GPIO.cleanup()
except ImportError as e:
    print(f"⚠️ RPi.GPIO: {e} (OK if not on Raspberry Pi)")

try:
    from picamera2 import Picamera2
    print(f"✅ Picamera2: Available")
except ImportError as e:
    print(f"⚠️ Picamera2: {e} (OK if not on Raspberry Pi)")

print("\n✅ Installation test complete!")
EOF

echo ""
echo -e "${GREEN}======================================"
echo "Installation Complete!"
echo "======================================${NC}"

echo ""
echo "📝 Next Steps:"
echo "   1. Activate virtual environment: source venv/bin/activate"
echo "   2. Train YOLO model (on PC with GPU)"
echo "   3. Train MobileNetV2 model (on PC with GPU)"
echo "   4. Transfer models to ./models/ directory"
echo "   5. Run system: python fruit_sorter.py"

echo ""
echo "🔄 Reboot recommended to apply GPIO permissions"
read -p "Reboot now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo reboot
fi

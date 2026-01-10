#!/bin/bash
# Complete Virtual Environment Setup for Raspberry Pi

echo "=========================================="
echo "🔧 SETTING UP VIRTUAL ENVIRONMENT"
echo "=========================================="

cd ~/System_Conveyor

# Remove old venv
if [ -d "venv" ]; then
    echo "Removing old venv..."
    rm -rf venv
fi

# Create new venv
echo "Creating venv..."
python3 -m venv venv --system-site-packages
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install compatible versions
echo "Installing packages (10-15 min)..."
pip install 'numpy<2.0,>=1.23.0'
pip install flask==3.0.0 flask-socketio==5.3.0 flask-cors==4.0.0
pip install python-socketio==5.9.0 eventlet==0.33.3
pip install opencv-python==4.8.0.76 tflite-runtime==2.14.0 Pillow==10.0.0
pip install RPi.GPIO==0.7.1 gpiozero==2.0.1

echo ""
echo "✅ VENV SETUP COMPLETE!"
echo "Activate: source ~/System_Conveyor/venv/bin/activate"

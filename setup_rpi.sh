#!/bin/bash
# Complete System Setup - Raspberry Pi Deployment
# Run this on Raspberry Pi

echo "========================================"
echo "AI Fruit Sorter - Raspberry Pi Setup"
echo "========================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check Python
echo -e "\n${YELLOW}[1/6] Checking Python installation...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓ Python found: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}✗ Python 3 not found! Installing...${NC}"
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip
fi

# Install Raspberry Pi requirements
echo -e "\n${YELLOW}[2/6] Installing Raspberry Pi requirements...${NC}"

# Create minimal requirements for Raspberry Pi
cat > requirements-rpi.txt << 'EOF'
# Raspberry Pi Runtime Requirements (Inference Only)
tflite-runtime>=2.14.0
opencv-python>=4.8.0
numpy>=1.23.0
Pillow>=9.5.0
picamera2>=0.3.12
gpiozero>=2.0.0
RPi.GPIO>=0.7.1
ultralytics>=8.0.0
flask>=2.3.0
flask-socketio>=5.3.0
python-socketio>=5.9.0
EOF

echo -e "${CYAN}Created requirements-rpi.txt${NC}"

# Upgrade pip
python3 -m pip install --upgrade pip

# Install packages
pip3 install -r requirements-rpi.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Packages installed successfully${NC}"
else
    echo -e "${RED}✗ Package installation failed${NC}"
    exit 1
fi

# Verify TFLite Runtime
echo -e "\n${YELLOW}[3/6] Verifying TFLite Runtime...${NC}"
python3 -c "import tflite_runtime.interpreter as tflite; print('TFLite Runtime OK')" 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ TFLite Runtime verified${NC}"
else
    echo -e "${YELLOW}⚠ TFLite Runtime not found, installing from source...${NC}"
    pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
fi

# Create necessary directories
echo -e "\n${YELLOW}[4/6] Creating directories...${NC}"
mkdir -p models logs data raw_images/{fresh,spoiled}
echo -e "${GREEN}✓ Directories created${NC}"

# Check for models
echo -e "\n${YELLOW}[5/6] Checking models...${NC}"

YOLO_MODEL="models/yolov8n_fruit.pt"
MOBILENET_MODEL="models/mobilenet_classifier.tflite"

if [ -f "$YOLO_MODEL" ]; then
    echo -e "${GREEN}✓ YOLO model found: $YOLO_MODEL${NC}"
else
    echo -e "${YELLOW}⚠ YOLO model not found${NC}"
    echo -e "  Copy from PC: scp models/yolov8n_fruit.pt pi@raspberrypi:~/System_Conveyor/models/"
fi

if [ -f "$MOBILENET_MODEL" ]; then
    echo -e "${GREEN}✓ MobileNet model found: $MOBILENET_MODEL${NC}"
else
    echo -e "${YELLOW}⚠ MobileNet model not found${NC}"
    echo -e "  Train on PC first, then copy:"
    echo -e "  scp models/mobilenet_classifier.tflite pi@raspberrypi:~/System_Conveyor/models/"
fi

# Test hardware
echo -e "\n${YELLOW}[6/6] Testing hardware...${NC}"

# Check camera
if python3 -c "from picamera2 import Picamera2; cam = Picamera2(); cam.close()" 2>&1 | grep -q "error"; then
    echo -e "${YELLOW}⚠ Camera not detected (may need to enable in raspi-config)${NC}"
else
    echo -e "${GREEN}✓ Camera available${NC}"
fi

# Check GPIO
if python3 -c "import RPi.GPIO as GPIO; GPIO.setmode(GPIO.BCM)" 2>&1 | grep -q "error"; then
    echo -e "${YELLOW}⚠ GPIO access may require sudo${NC}"
else
    echo -e "${GREEN}✓ GPIO accessible${NC}"
fi

# Summary
echo -e "\n========================================"
echo -e "${GREEN}Raspberry Pi Setup Complete!${NC}"
echo -e "========================================"

echo -e "\n${YELLOW}System Status:${NC}"
echo -e "  Python: ${GREEN}✓${NC}"
echo -e "  Packages: ${GREEN}✓${NC}"
echo -e "  Directories: ${GREEN}✓${NC}"

if [ -f "$YOLO_MODEL" ] && [ -f "$MOBILENET_MODEL" ]; then
    echo -e "  Models: ${GREEN}✓ Ready${NC}"
    echo -e "\n${YELLOW}Next Steps:${NC}"
    echo -e "1. Test the system:"
    echo -e "   ${CYAN}python3 fruit_sorter.py${NC}"
    echo -e "\n2. Start web interface:"
    echo -e "   ${CYAN}python3 run_web.py${NC}"
else
    echo -e "  Models: ${YELLOW}⚠ Missing${NC}"
    echo -e "\n${YELLOW}Next Steps:${NC}"
    echo -e "1. Train models on PC"
    echo -e "2. Copy models to Raspberry Pi:"
    echo -e "   ${CYAN}scp models/*.tflite pi@$(hostname):~/System_Conveyor/models/${NC}"
    echo -e "   ${CYAN}scp models/*.pt pi@$(hostname):~/System_Conveyor/models/${NC}"
fi

echo -e "\n${GREEN}✓ Raspberry Pi is configured!${NC}"

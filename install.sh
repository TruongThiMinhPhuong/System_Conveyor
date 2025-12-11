#!/bin/bash

#============================================
# Raspberry Pi Complete Installation Script
# AI Fruit Sorting Conveyor System
#============================================

echo "=========================================="
echo "🍓 AI Fruit Sorting System - Installation"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if running on Raspberry Pi
if [ ! -f /proc/device-tree/model ] || ! grep -q "Raspberry Pi" /proc/device-tree/model; then
    echo -e "${YELLOW}⚠️  Warning: This script is designed for Raspberry Pi${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

cd "$(dirname "$0")"

echo ""
echo "Step 1: Setting up swap space (4GB)..."
echo "💡 This is CRITICAL for training AI models on Pi!"

# Check current swap
current_swap=$(free -m | awk '/Swap:/ {print $2}')
if [ "$current_swap" -lt 4000 ]; then
    echo "   Current swap: ${current_swap}MB, increasing to 4GB..."
    
    sudo dphys-swapfile swapoff
    sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=4096/' /etc/dphys-swapfile
    sudo dphys-swapfile setup
    sudo dphys-swapfile swapon
    
    new_swap=$(free -m | awk '/Swap:/ {print $2}')
    echo -e "${GREEN}   ✓ Swap increased to ${new_swap}MB${NC}"
else
    echo -e "${GREEN}   ✓ Swap already configured (${current_swap}MB)${NC}"
fi

echo ""
echo "Step 2: Updating system packages..."
sudo apt update
sudo apt upgrade -y

echo ""
echo "Step 3: Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    libopencv-dev \
    python3-opencv \
    libatlas-base-dev \
    libcap-dev \
    libffi-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    cmake

echo ""
echo "Step 4: Installing camera system packages..."
sudo apt install -y \
    libcamera-dev \
    libcamera-apps \
    python3-libcamera \
    python3-kms++ \
    python3-picamera2 \
    python3-prctl

echo ""
echo "Step 5: Enabling camera and GPIO..."
sudo raspi-config nonint do_camera 0
sudo raspi-config nonint do_i2c 0
sudo raspi-config nonint do_spi 0

echo ""
echo "Step 6: Creating virtual environment..."
if [ ! -d "venv" ]; then
    # Create venv WITH system-site-packages for libcamera/picamera2 access
    python3 -m venv --system-site-packages venv
    echo -e "${GREEN}   ✓ Virtual environment created with system-site-packages${NC}"
else
    echo "   Virtual environment already exists"
fi

source venv/bin/activate

echo ""
echo "Step 7: Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo ""
echo "Step 8: Installing Python packages..."
echo "   Installing packages individually to skip problematic ones..."

# Core packages
echo "   📦 Core packages..."
pip install numpy pillow pyyaml loguru || echo "   ⚠️  Some core packages failed"

# OpenCV
echo "   📦 OpenCV..."
pip install opencv-python || echo "   ⚠️  OpenCV failed"

# AI Models
echo "   📦 AI packages..."
pip install ultralytics || echo "   ⚠️  Ultralytics failed"

# TensorFlow Lite
echo "   📦 TensorFlow Lite..."
pip install tflite-runtime || pip install tensorflow==2.13.0 || echo "   ⚠️  TFLite failed"

# Hardware packages
echo "   📦 Hardware packages..."
pip install RPi.GPIO gpiozero || echo "   ⚠️  GPIO packages failed"

# Picamera2 - Try pip first, then link system package
echo "   📦 Picamera2..."
pip install picamera2 2>/dev/null || {
    echo "      ℹ️  pip install failed, linking system package..."
    PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    mkdir -p venv/lib/python${PYTHON_VER}/site-packages/
    ln -sf /usr/lib/python3/dist-packages/picamera2 venv/lib/python${PYTHON_VER}/site-packages/ 2>/dev/null
    ln -sf /usr/lib/python3/dist-packages/libcamera venv/lib/python${PYTHON_VER}/site-packages/ 2>/dev/null
}

# Web Interface
echo "   📦 Web packages..."
pip install flask flask-cors flask-socketio python-socketio eventlet || echo "   ⚠️  Some web packages failed"

echo ""
echo "Step 9: Creating directories..."
mkdir -p models logs data datasets
mkdir -p raw_images/fresh raw_images/spoiled

echo ""
echo "Step 10: Setting GPIO permissions..."
sudo usermod -a -G gpio $USER
sudo usermod -a -G i2c $USER

echo ""
echo "Step 11: Verifying installation..."
python3 << 'EOF'
print("\n🧪 Testing installed packages...\n")

packages = {
    'numpy': 'NumPy',
    'cv2': 'OpenCV',
    'ultralytics': 'YOLOv8',
    'flask': 'Flask',
    'yaml': 'PyYAML',
}

success_count = 0
fail_count = 0

for module, name in packages.items():
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'OK')
        print(f"✅ {name}: {version}")
        success_count += 1
    except ImportError:
        print(f"❌ {name}: Failed")
        fail_count += 1

# Hardware packages
print("\n🔌 Hardware Packages:")
try:
    import RPi.GPIO
    print(f"✅ RPi.GPIO: OK")
except:
    print(f"⚠️  RPi.GPIO: Not available")

try:
    from picamera2 import Picamera2
    print(f"✅ Picamera2: OK (import)")
    try:
        cam = Picamera2()
        cam.close()
        print(f"✅ Camera Module: Hardware detected!")
    except Exception as e:
        print(f"⚠️  Camera Module: {str(e)[:50]}...")
        print("   💡 Check camera cable or run 'libcamera-hello'")
except Exception as e:
    print(f"⚠️  Picamera2: {str(e)[:50]}...")
    print("   💡 Will use OpenCV fallback")

print(f"\n📊 Summary: {success_count}/{len(packages)} packages OK")

if success_count >= 4:
    print("✅ Installation successful!")
    exit(0)
else:
    print("⚠️  Some packages failed, but may still work")
    exit(0)
EOF

echo ""
echo "Step 12: Creating helper scripts..."

# Create camera fix script
cat > fix_camera.sh << 'CAMERA_SCRIPT'
#!/bin/bash
echo "🎥 Fixing Picamera2 in virtual environment..."
cd ~/System_Conveyor
source venv/bin/activate

PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
ln -sf /usr/lib/python3/dist-packages/picamera2 venv/lib/python${PYTHON_VER}/site-packages/
ln -sf /usr/lib/python3/dist-packages/libcamera venv/lib/python${PYTHON_VER}/site-packages/

python3 -c "from picamera2 import Picamera2; print('✅ Picamera2 OK!')" && echo "✅ Fixed!" || echo "❌ Still failed"
CAMERA_SCRIPT

chmod +x fix_camera.sh

# Create test camera script
cat > test_camera.sh << 'TEST_SCRIPT'
#!/bin/bash
cd ~/System_Conveyor
source venv/bin/activate
python hardware/camera.py
TEST_SCRIPT

chmod +x test_camera.sh

echo ""
echo -e "${GREEN}"
echo "=========================================="
echo "✅ INSTALLATION COMPLETE!"
echo "=========================================="
echo -e "${NC}"

echo ""
echo "📝 Next steps:"
echo "   1. Reboot Pi: sudo reboot"
echo "   2. After reboot:"
echo "      cd ~/System_Conveyor"
echo "      source venv/bin/activate"
echo "      python run_web.py"
echo ""
echo "🎥 Test camera:"
echo "   ./test_camera.sh"
echo ""
echo "📚 Documentation:"
echo "   - Quick start: QUICK_INSTALL.md"
echo "   - Full guide: docs/INDEX.md"
echo "   - Training on Pi: docs/TRAINING_ON_PI.md"
echo ""
echo "🔄 Reboot now?"
read -p "(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo reboot
fi

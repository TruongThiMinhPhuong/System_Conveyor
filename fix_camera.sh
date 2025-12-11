#!/bin/bash
echo "🎥 Fixing Picamera2 in virtual environment..."
cd ~/System_Conveyor
source venv/bin/activate

PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
ln -sf /usr/lib/python3/dist-packages/picamera2 venv/lib/python${PYTHON_VER}/site-packages/
ln -sf /usr/lib/python3/dist-packages/libcamera venv/lib/python${PYTHON_VER}/site-packages/

python3 -c "from picamera2 import Picamera2; print('✅ Picamera2 OK!')" && echo "✅ Fixed!" || echo "❌ Still failed"

# 🆘 Hướng Dẫn Khắc Phục Lỗi Cài Đặt

## ❌ Lỗi: "You need to install libcap development headers"

### Triệu chứng:
```
ERROR: Failed to build 'python-prctl' when getting requirements to build wheel
❌ OpenCV: No module named 'cv2'
❌ NumPy: No module named 'numpy'
```

### ✅ Giải pháp:

```bash
# 1. Cài dependencies còn thiếu
sudo apt install -y libcap-dev libffi-dev

# 2. Activate virtual environment
cd ~/System_Conveyor
source venv/bin/activate

# 3. Cài lại Python packages
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ❌ Lỗi: pip install timeout/failed

### Triệu chứng:
```
ERROR: Could not install packages...
Read timed out
```

### ✅ Giải pháp:

```bash
# Tăng timeout và cài từng nhóm
source venv/bin/activate

# Nhóm 1: Core
pip install --default-timeout=100 numpy pillow pyyaml loguru

# Nhóm 2: OpenCV
pip install --default-timeout=100 opencv-python

# Nhóm 3: AI
pip install --default-timeout=100 ultralytics

# Nhóm 4: TensorFlow Lite
pip install --default-timeout=100 tflite-runtime

# Nhóm 5: Hardware
pip install RPi.GPIO gpiozero picamera2

# Nhóm 6: Web
pip install flask flask-cors flask-socketio eventlet
```

---

## ❌ Lỗi: Out of Memory khi pip install

### Triệu chứng:
```
Killed
ERROR: Failed building wheel for...
```

### ✅ Giải pháp:

```bash
# Tăng swap trước
sudo dphys-swapfile swapoff
sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=4096/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Verify swap
free -h
# Swap phải hiển thị 4.0G

# Cài lại
cd ~/System_Conveyor
source venv/bin/activate
pip install -r requirements.txt
```

---

## ❌ Lỗi: Camera not found

### Triệu chứng:
```
FileNotFoundError: /dev/video0
RuntimeError: Camera not detected
```

### ✅ Giải pháp:

```bash
# 1. Enable camera
sudo raspi-config
# Interface Options → Camera → Yes

# 2. Reboot
sudo reboot

# 3. Test
libcamera-hello -t 5000

# 4. Verify trong Python
python3 -c "from picamera2 import Picamera2; print('✅ Camera OK')"
```

---

## ❌ Lỗi: GPIO Permission Denied

### Triệu chứng:
```
PermissionError: [Errno 13] Permission denied: '/dev/gpiomem'
```

### ✅ Giải pháp:

```bash
# Thêm user vào group
sudo usermod -a -G gpio $USER
sudo usermod -a -G i2c $USER

# Logout và login lại
# Hoặc:
su - $USER

# Test
python3 -c "import RPi.GPIO as GPIO; GPIO.setmode(GPIO.BCM); GPIO.cleanup(); print('✅ GPIO OK')"
```

---

## ❌ Lỗi: Virtual environment không hoạt động

### Triệu chứng:
```
bash: venv/bin/activate: No such file or directory
```

### ✅ Giải pháp:

```bash
# Tạo lại virtual environment
cd ~/System_Conveyor
rm -rf venv
python3 -m venv venv

# Activate
source venv/bin/activate

# Verify
which python
# Expected: ~/System_Conveyor/venv/bin/python
```

---

## ❌ Lỗi: Import cv2 failed sau khi cài

### Triệu chứng:
```
ImportError: libGL.so.1: cannot open shared object file
```

### ✅ Giải pháp:

```bash
# Cài thêm dependencies
sudo apt install -y libgl1-mesa-glx libglib2.0-0

# Test lại
python3 -c "import cv2; print('✅ OpenCV:', cv2.__version__)"
```

---

## ❌ Lỗi: ultralytics requires torch

### Triệu chứng:
```
ERROR: Could not find a version that satisfies the requirement torch
```

### ✅ Giải pháp:

```bash
source venv/bin/activate

# Cài PyTorch CPU version cho Pi
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Sau đó cài ultralytics
pip install ultralytics
```

---

## 🔍 Script Kiểm Tra Toàn Diện

```bash
cd ~/System_Conveyor
source venv/bin/activate

python3 << 'EOF'
import sys
print("=" * 60)
print("🔍 KIỂM TRA HỆ THỐNG TOÀN DIỆN")
print("=" * 60)

# Python version
print(f"\n📍 Python: {sys.version}")

# Check packages
print("\n📦 Thư viện Python:")
packages = {
    'numpy': 'NumPy',
    'cv2': 'OpenCV',
    'yaml': 'PyYAML',
    'PIL': 'Pillow',
    'ultralytics': 'YOLOv8',
    'flask': 'Flask',
    'loguru': 'Loguru',
}

for module, name in packages.items():
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'OK')
        print(f"  ✅ {name}: {version}")
    except ImportError as e:
        print(f"  ❌ {name}: {e}")

# Check hardware
print("\n🔌 Hardware Libraries:")
try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.cleanup()
    print("  ✅ RPi.GPIO")
except Exception as e:
    print(f"  ❌ RPi.GPIO: {e}")

try:
    from picamera2 import Picamera2
    print("  ✅ Picamera2")
except Exception as e:
    print(f"  ❌ Picamera2: {e}")

# Check directories
print("\n📁 Thư mục Project:")
import os
dirs = ['models', 'logs', 'data', 'datasets', 'venv']
for d in dirs:
    exists = os.path.isdir(d)
    print(f"  {'✅' if exists else '❌'} {d}/")

# Check swap
print("\n💾 Memory & Swap:")
import subprocess
result = subprocess.run(['free', '-h'], capture_output=True, text=True)
for line in result.stdout.split('\n'):
    if 'Mem:' in line or 'Swap:' in line:
        print(f"  {line}")

print("\n" + "=" * 60)
print("✅ KIỂM TRA HOÀN TẤT")
print("=" * 60)
EOF
```

---

## 🆘 Nếu Tất Cả Đều Thất Bại

### Option 1: Cài Lại Từ Đầu

```bash
# Xóa virtual environment cũ
cd ~/System_Conveyor
rm -rf venv

# Chạy lại install script
./install.sh
```

### Option 2: Manual Install

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Cài dependencies
sudo apt install -y \
    python3-pip python3-dev python3-venv \
    libopencv-dev python3-opencv libatlas-base-dev \
    libcamera-dev python3-picamera2 \
    libcap-dev libffi-dev \
    git cmake

# Tạo venv
cd ~/System_Conveyor
python3 -m venv venv
source venv/bin/activate

# Cài packages từng cái
pip install --upgrade pip
pip install numpy
pip install opencv-python
pip install ultralytics
pip install RPi.GPIO gpiozero picamera2
pip install flask flask-cors flask-socketio eventlet
```

---

## 📞 Liên Hệ Hỗ Trợ

Nếu vẫn gặp vấn đề:
1. Chụp ảnh toàn bộ lỗi
2. Chạy script kiểm tra ở trên
3. Cung cấp output

**Thường thì vấn đề do:**
- ❌ Chưa tăng swap (phải 4GB)
- ❌ Thiếu system dependencies (libcap-dev, ...)
- ❌ Chưa activate virtual environment
- ❌ Mạng chậm → timeout

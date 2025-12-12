

## 📋 Yêu Cầu Trước Khi Bắt Đầu

### Phần Cứng:
- ✅ Raspberry Pi 4 (8GB RAM)
- ✅ MicroSD Card 32GB+ (Class 10)
- ✅ Card reader (để flash OS)
- ✅ Màn hình + bàn phím/chuột (hoặc dùng SSH)
- ✅ Kết nối Internet (WiFi hoặc Ethernet)

### Phần Mềm Cần Có:
- Raspberry Pi Imager (download từ raspberrypi.com)
- PC/Laptop để flash SD card

---

## 📀 BƯỚC 1: Cài Đặt Raspberry Pi OS

### 1.1. Download Raspberry Pi Imager

```bash
# Trên Windows/Mac/Linux
https://www.raspberrypi.com/software/
```

### 1.2. Flash OS Lên SD Card

1. Mở **Raspberry Pi Imager**
2. Click **CHOOSE OS** → **Raspberry Pi OS (64-bit)** → **Full (recommended)**
3. Click **CHOOSE STORAGE** → Chọn SD card
4. Click **Settings** (biểu tượng ⚙️):
   ```
   ✓ Set hostname: fruit-sorter
   ✓ Enable SSH: ✓ Use password authentication
   ✓ Set username: pi
   ✓ Set password: [your-password]
   ✓ Configure WiFi: [tên wifi + password của bạn]
   ✓ Set locale: Asia/Ho_Chi_Minh, Keyboard: us
   ```
5. Click **SAVE** → **WRITE** → Chờ hoàn thành (~10-15 phút)

### 1.3. Khởi Động Raspberry Pi

1. Cắm SD card vào Pi
2. Kết nối:
   - HDMI → Màn hình
   - USB → Bàn phím + chuột
   - Ethernet hoặc WiFi
   - Nguồn USB-C 5V 3A
3. Bật nguồn
4. Đợi boot xong (~1-2 phút)

### 1.4. First Boot Setup

```bash
# Nếu dùng desktop GUI:
# - Chọn timezone, keyboard
# - Kết nối WiFi (nếu chưa)
# - Update software khi được hỏi

# Nếu dùng SSH từ PC:
ssh pi@fruit-sorter.local
# Hoặc: ssh pi@<ip-address>
```

---

## 🔧 BƯỚC 2: Cập Nhật Hệ Thống

```bash
# Update package list
sudo apt update

# Upgrade tất cả packages (mất ~10-20 phút)
sudo apt upgrade -y

# Reboot
sudo reboot
```

Chờ Pi reboot (~1 phút), sau đó SSH lại vào.

---

## 📦 BƯỚC 3: Cài Đặt System Dependencies

### 3.1. Python & Development Tools

```bash
# Cài Python 3.9+ (nếu chưa có)
sudo apt install -y python3 python3-pip python3-dev python3-venv

# Verify
python3 --version
# Expected: Python 3.9.x hoặc 3.11.x

pip3 --version
# Expected: pip 23.x
```

### 3.2. OpenCV Dependencies

```bash
sudo apt install -y \
    libopencv-dev \
    python3-opencv \
    libatlas-base-dev \
    libjasper-dev \
    libqt4-test \
    libqtgui4 \
    libhdf5-dev \
    libhdf5-103
```

### 3.3. Camera & GPIO Libraries

```bash
sudo apt install -y \
    libcamera-dev \
    python3-libcamera \
    python3-kms++ \
    python3-picamera2
```

### 3.4. Build Tools

```bash
sudo apt install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl
```

---

## 🎥 BƯỚC 4: Kích Hoạt Camera & GPIO

```bash
sudo raspi-config
```

Trong menu:
1. **Interface Options** → **Camera** → **Yes**
2. **Interface Options** → **I2C** → **Yes**
3. **Interface Options** → **SPI** → **Yes**
4. **Interface Options** → **SSH** → **Yes** (nếu chưa enable)
5. Chọn **Finish** → **Yes** (để reboot)

---

## 💾 BƯỚC 5: Tăng Swap Space

**QUAN TRỌNG** cho training AI models!

```bash
# Stop swap
sudo dphys-swapfile swapoff

# Edit config
sudo nano /etc/dphys-swapfile

# Tìm dòng:
# CONF_SWAPSIZE=100
# Sửa thành:
CONF_SWAPSIZE=4096

# Lưu: Ctrl+O, Enter, Ctrl+X

# Setup lại swap
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Verify
free -h
# Swap phải hiển thị 4.0G
```

---

## 📂 BƯỚC 6: Clone Project

### 6.1. Tạo Thư Mục

```bash
cd ~
mkdir -p Projects
cd Projects
```

### 6.2. Clone Repository

**Option A: Từ GitHub**
```bash
git clone https://github.com/your-username/System_Conveyor.git
cd System_Conveyor
```

**Option B: Copy từ USB/PC**
```bash
# Nếu copy từ USB
cp -r /media/pi/USB_DRIVE/System_Conveyor ~/Projects/

# Hoặc dùng scp từ PC
# Trên PC (Windows PowerShell):
scp -r D:\System_Conveyor pi@fruit-sorter.local:~/Projects/

# Vào thư mục
cd ~/Projects/System_Conveyor
```

---

## 🐍 BƯỚC 7: Tạo Virtual Environment

```bash
cd ~/Projects/System_Conveyor

# Tạo virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate

# Khi active, prompt sẽ có (venv) ở đầu:
# (venv) pi@fruit-sorter:~/Projects/System_Conveyor$
```

**Lưu ý**: Mỗi lần mở terminal mới phải chạy:
```bash
cd ~/Projects/System_Conveyor
source venv/bin/activate
```

---

## 📚 BƯỚC 8: Cài Python Dependencies

### 8.1. Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### 8.2. Cài Từ requirements.txt

```bash
pip install -r requirements.txt
```

**Thời gian**: ~15-30 phút (tùy tốc độ mạng)

**Nếu gặp lỗi**, cài từng nhóm:

#### A. Core Dependencies
```bash
pip install numpy pillow pyyaml python-dotenv loguru
```

#### B. Computer Vision
```bash
pip install opencv-python
```

#### C. AI Models
```bash
# YOLOv8
pip install ultralytics

# TensorFlow Lite
pip install tflite-runtime
# Nếu lỗi, dùng: pip install tensorflow
```

#### D. Hardware Control
```bash
pip install RPi.GPIO gpiozero picamera2
```

#### E. Web Interface
```bash
pip install flask flask-cors flask-socketio python-socketio eventlet
```

### 8.3. Verify Installation

```bash
python3 << 'EOF'
import sys
print("\n🧪 Kiểm tra các thư viện...")

packages = [
    ('numpy', 'NumPy'),
    ('cv2', 'OpenCV'),
    ('ultralytics', 'YOLOv8'),
    ('flask', 'Flask'),
    ('yaml', 'PyYAML'),
]

for module, name in packages:
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'OK')
        print(f"✅ {name}: {version}")
    except ImportError as e:
        print(f"❌ {name}: FAILED - {e}")

# Check hardware libs (chỉ chạy trên Pi)
try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.cleanup()
    print("✅ RPi.GPIO: OK")
except Exception as e:
    print(f"⚠️ RPi.GPIO: {e}")

try:
    from picamera2 import Picamera2
    print("✅ Picamera2: OK")
except Exception as e:
    print(f"⚠️ Picamera2: {e}")

print("\n✅ Kiểm tra hoàn tất!")
EOF
```

---

## 🔨 BƯỚC 9: Tạo Thư Mục Cần Thiết

```bash
cd ~/Projects/System_Conveyor

# Tạo thư mục
mkdir -p models
mkdir -p logs
mkdir -p data
mkdir -p datasets
mkdir -p raw_images/fresh
mkdir -p raw_images/spoiled

# Verify
ls -la
```

---

## ⚙️ BƯỚC 10: Cấu Hình GPIO Permissions

```bash
# Thêm user vào group gpio
sudo usermod -a -G gpio $USER
sudo usermod -a -G i2c $USER
sudo usermod -a -G spi $USER

# Logout và login lại để apply
# Hoặc:
su - $USER
```

---

## 🧪 BƯỚC 11: Test Hardware

### 11.1. Test Camera

```bash
cd ~/Projects/System_Conveyor
source venv/bin/activate

# Test với libcamera
libcamera-hello -t 5000
# Phải thấy camera preview 5 giây

# Test Python camera
python3 hardware/camera.py
```

### 11.2. Test GPIO (chưa nối hardware)

```bash
python3 << 'EOF'
import RPi.GPIO as GPIO
import time

# Setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)

# Blink test (nếu nối LED vào GPIO 18)
print("Testing GPIO 18...")
for i in range(5):
    GPIO.output(18, GPIO.HIGH)
    time.sleep(0.5)
    GPIO.output(18, GPIO.LOW)
    time.sleep(0.5)

GPIO.cleanup()
print("GPIO test OK!")
EOF
```

---

## 📦 BƯỚC 12: Cài Training Dependencies (Nếu Train Trên Pi)

Nếu muốn training trên Pi:

```bash
source venv/bin/activate

# PyTorch (CPU version) - MẤT ~10-15 phút
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# TensorFlow (full version)
pip3 install tensorflow

# Training tools
pip3 install matplotlib scikit-learn

# Annotation tool (nếu cần)
pip3 install labelImg
```

---

## ✅ BƯỚC 13: Verify Toàn Bộ Setup

```bash
cd ~/Projects/System_Conveyor
source venv/bin/activate

# Run check script
python3 << 'EOF'
print("="*60)
print("🍓 KIỂM TRA SETUP HỆ THỐNG")
print("="*60)

import sys
print(f"\n📍 Python: {sys.version}")

# Check imports
print("\n📦 Kiểm tra thư viện:")
libs = {
    'numpy': 'NumPy',
    'cv2': 'OpenCV', 
    'yaml': 'PyYAML',
    'flask': 'Flask',
    'ultralytics': 'YOLOv8',
}

for module, name in libs.items():
    try:
        __import__(module)
        print(f"  ✅ {name}")
    except:
        print(f"  ❌ {name}")

# Check GPIO
print("\n🔌 Kiểm tra GPIO:")
try:
    import RPi.GPIO
    print("  ✅ RPi.GPIO")
except:
    print("  ❌ RPi.GPIO")

# Check Camera
print("\n📸 Kiểm tra Camera:")
try:
    from picamera2 import Picamera2
    print("  ✅ Picamera2")
except:
    print("  ❌ Picamera2")

# Check directories
print("\n📁 Kiểm tra thư mục:")
import os
dirs = ['models', 'logs', 'data', 'datasets', 'raw_images']
for d in dirs:
    exists = os.path.isdir(d)
    status = "✅" if exists else "❌"
    print(f"  {status} {d}/")

print("\n" + "="*60)
print("✅ KIỂM TRA HOÀN TẤT!")
print("="*60)
EOF
```

---

## 🎯 BƯỚC 14: Chạy Test Đầu Tiên

```bash
# Test các module riêng lẻ
python3 hardware/camera.py      # Test camera
python3 hardware/servo_control.py   # Test servo (sau khi nối)
python3 hardware/motor_control.py   # Test motor (sau khi nối)
```

---

## 🌐 BƯỚC 15: Setup Web Interface (Optional)

```bash
# Chạy web server
python3 run_web.py

# Truy cập từ browser:
# http://fruit-sorter.local:5000
# Hoặc: http://<pi-ip>:5000
```

---

## 🔄 BƯỚC 16: Auto-Start (Optional)

Nếu muốn hệ thống tự động chạy khi boot:

```bash
# Tạo service file
sudo nano /etc/systemd/system/fruit-sorter.service
```

Nội dung:
```ini
[Unit]
Description=AI Fruit Sorting System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/Projects/System_Conveyor
ExecStart=/home/pi/Projects/System_Conveyor/venv/bin/python run_web.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable service:
```bash
sudo systemctl enable fruit-sorter
sudo systemctl start fruit-sorter
sudo systemctl status fruit-sorter
```

---

## 🆘 Troubleshooting

### Lỗi: `pip install` failed

```bash
# Nếu thiếu dependencies:
sudo apt install -y python3-dev libatlas-base-dev

# Nếu lỗi memory:
# Tăng swap (xem Bước 5)
```

### Lỗi: Camera not found

```bash
# Kiểm tra camera được enable
sudo raspi-config
# Interface Options → Camera → Enable

# Test camera
libcamera-hello

# Reboot
sudo reboot
```

### Lỗi: GPIO Permission denied

```bash
# Thêm vào group
sudo usermod -a -G gpio $USER

# Logout/login lại
```

### Lỗi: Import tflite_runtime failed

```bash
# Thử cài tensorflow đầy đủ
pip install tensorflow
```

---

## 📋 Checklist Hoàn Thành

- [ ] ✅ Raspberry Pi OS đã flash và boot OK
- [ ] ✅ System đã update (apt update & upgrade)
- [ ] ✅ Camera & GPIO đã enable trong raspi-config
- [ ] ✅ Swap đã tăng lên 4GB
- [ ] ✅ Project đã clone về Pi
- [ ] ✅ Virtual environment đã tạo
- [ ] ✅ Tất cả dependencies đã cài (requirements.txt)
- [ ] ✅ Thư mục models/, logs/, data/ đã tạo
- [ ] ✅ Camera test OK
- [ ] ✅ GPIO permissions OK
- [ ] ✅ Verification script chạy thành công

---

## 🎓 Commands Tóm Tắt

```bash
# Activate virtual environment (mỗi khi mở terminal mới)
cd ~/Projects/System_Conveyor
source venv/bin/activate

# Update code (nếu có thay đổi)
git pull

# Chạy hệ thống
python3 fruit_sorter.py

# Chạy web interface
python3 run_web.py

# Deactivate virtual environment
deactivate
```

---

## 📞 Next Steps

1. ✅ **Lắp ráp phần cứng** → Xem [POWER_SUPPLY_QUICK_GUIDE.md](POWER_SUPPLY_QUICK_GUIDE.md)
2. ✅ **Thu thập dữ liệu** → `python training/data_collection/collect_images.py`
3. ✅ **Training models** → Xem [TRAINING_ON_PI.md](TRAINING_ON_PI.md)
4. ✅ **Chạy hệ thống** → `python fruit_sorter.py` hoặc `python run_web.py`

---

**Chúc bạn cài đặt thành công! 🎉**



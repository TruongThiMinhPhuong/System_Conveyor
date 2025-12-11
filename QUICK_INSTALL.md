# � Hướng Dẫn Cài Đặt Nhanh

## ⚡ 3 BƯỚC ĐƠN GIẢN

### Bước 1: Copy Project Vào Pi

**Cách A: Từ USB**
```bash
# Cắm USB vào Pi
cd ~
cp -r /media/pi/USB_DRIVE/System_Conveyor .
```

**Cách B: Từ PC qua SSH**
```bash
# Trên PC (Windows PowerShell hoặc Linux/Mac Terminal):
scp -r D:\System_Conveyor pi@raspberrypi.local:~/

# Hoặc dùng IP:
scp -r D:\System_Conveyor pi@192.168.1.100:~/
```

**Cách C: Từ GitHub**
```bash
cd ~
git clone https://github.com/your-username/System_Conveyor.git
```

---

### Bước 2: Chạy Install Script

```bash
# SSH vào Pi:
ssh pi@raspberrypi.local

# Vào thư mục:
cd ~/System_Conveyor

# Cho phép execute:
chmod +x install.sh

# CHẠY INSTALL:
./install.sh
```

**Script sẽ tự động làm 12 bước:**

1. ✅ Tăng swap lên 4GB (cho training AI)
2. ✅ Update hệ thống
3. ✅ Cài system dependencies (libcap-dev, libffi-dev...)
4. ✅ Cài camera packages (libcamera, picamera2)
5. ✅ Enable camera & GPIO
6. ✅ Tạo virtual environment
7. ✅ Upgrade pip
8. ✅ Cài Python packages (OpenCV, YOLOv8, Flask...)
9. ✅ Tạo thư mục (models, logs, data...)
10. ✅ Setup GPIO permissions
11. ✅ Verify cài đặt
12. ✅ Tạo helper scripts (fix_camera.sh, test_camera.sh)

**⏱️ Thời gian:** ~30-45 phút (tự động)

---

### Bước 3: Reboot

```bash
# Script sẽ hỏi, nhấn 'y':
Reboot now? (y/n) y
```

---

## ✅ SAU KHI REBOOT

```bash
# SSH lại vào Pi
ssh pi@raspberrypi.local

# Vào project
cd ~/System_Conveyor

# Activate environment
source venv/bin/activate

# Test camera (nếu đã nối)
./test_camera.sh
# Hoặc:
python hardware/camera.py

# Chạy web interface
python run_web.py
```

**Truy cập:** http://raspberrypi.local:5000

---

## 📋 CHECKLIST ĐẦY ĐỦ

### Trước Khi Cài:
- [ ] Raspberry Pi 4 (8GB RAM) đã có Pi OS 64-bit
- [ ] Camera Module đã kết nối vào CSI port
- [ ] SD card ≥ 32GB
- [ ] Kết nối internet (WiFi hoặc Ethernet)
- [ ] Có keyboard + monitor HOẶC SSH

### Sau Khi Chạy install.sh:
- [ ] Script chạy xong không có lỗi nghiêm trọng
- [ ] Verification step hiển thị 4+ packages OK
- [ ] Swap = 4GB (`free -h`)
- [ ] Camera enabled (`vcgencmd get_camera` → detected=1)

### Sau Reboot:
- [ ] `source venv/bin/activate` hoạt động
- [ ] `python hardware/camera.py` chụp ảnh được
- [ ] `python run_web.py` khởi động OK
- [ ] Web interface truy cập được

---

## 🆘 KHẮC PHỤC LỖI THƯỜNG GẶP

### Lỗi 1: Camera không hoạt động

```bash
# Nếu Picamera2 lỗi, camera.py tự động dùng OpenCV
# Để fix Picamera2:
cd ~/System_Conveyor
./fix_camera.sh

# Hoặc enable V4L2 cho OpenCV:
sudo modprobe bcm2835-v4l2
echo "bcm2835-v4l2" | sudo tee -a /etc/modules
```

### Lỗi 2: Package import failed

```bash
cd ~/System_Conveyor
source venv/bin/activate

# Cài lại package bị lỗi:
pip install opencv-python ultralytics flask

# Test:
python3 -c "import cv2, ultralytics, flask; print('✅ OK!')"
```

### Lỗi 3: Out of memory

```bash
# Kiểm tra swap:
free -h

# Nếu swap < 4GB:
sudo dphys-swapfile swapoff
sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=4096/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### Lỗi 4: Permission denied

```bash
# GPIO permission:
sudo usermod -a -G gpio $USER
sudo usermod -a -G i2c $USER

# Logout và login lại
```

---

## 🎯 TÓM TẮT 3 LỆNH CHÍNH

```bash
cd ~/System_Conveyor
chmod +x install.sh
./install.sh
```

**VẬY LÀ XONG! 🎉**

---

## 📚 TÀI LIỆU BỔ SUNG

### Cho Người Mới:
- **[docs/INDEX.md](docs/INDEX.md)** - Lộ trình đầy đủ từ A-Z
- **[docs/INSTALLATION_GUIDE.md](docs/INSTALLATION_GUIDE.md)** - 16 bước chi tiết

### Khắc Phục Lỗi:
- **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** - Lỗi phổ biến
- **[docs/CAMERA_ALTERNATIVES.md](docs/CAMERA_ALTERNATIVES.md)** - Camera options

### Lắp Ráp Phần Cứng:
- **[docs/POWER_SUPPLY_QUICK_GUIDE.md](docs/POWER_SUPPLY_QUICK_GUIDE.md)** - Kết nối nguồn
- **[docs/detailed_wiring_diagram.md](docs/detailed_wiring_diagram.md)** - Sơ đồ chi tiết

### Training Models:
- **[docs/TRAINING_ON_PI.md](docs/TRAINING_ON_PI.md)** - Train trên Pi 4
- **[docs/training_guide.md](docs/training_guide.md)** - Train trên PC/GPU

### Hoạt Động:
- **[docs/user_manual.md](docs/user_manual.md)** - Vận hành hệ thống
- **[docs/web_interface_guide.md](docs/web_interface_guide.md)** - Giao diện web

---

## 🎓 NEXT STEPS SAU KHI CÀI XONG

### 1. Test Hardware
```bash
cd ~/System_Conveyor
source venv/bin/activate

# Test camera
python hardware/camera.py

# Test servo (sau khi nối phần cứng)
python hardware/servo_control.py

# Test motor
python hardware/motor_control.py
```

### 2. Thu Thập Dữ Liệu
```bash
# Chụp ảnh cho training
python training/data_collection/collect_images.py --mode classification --count 100
```

### 3. Training Models
```bash
# Xem hướng dẫn:
cat docs/TRAINING_ON_PI.md

# Train YOLOv8 (chạy qua đêm)
cd training/yolo
python train_yolo.py --epochs 50 --batch 4

# Train MobileNetV2
cd training/mobilenet
python train_mobilenet.py --epochs 30 --batch 8
```

### 4. Chạy Hệ Thống
```bash
# Web interface (khuyến nghị)
python run_web.py
# Truy cập: http://raspberrypi.local:5000

# Hoặc CLI
python fruit_sorter.py
```

---

## � TIPS

### Mỗi Lần SSH Vào Pi:
```bash
cd ~/System_Conveyor
source venv/bin/activate
```

### Auto-activate (Optional):
```bash
# Thêm vào ~/.bashrc:
echo "cd ~/System_Conveyor && source venv/bin/activate" >> ~/.bashrc
```

### Chạy Tự Động Khi Boot:
```bash
# Tạo systemd service (xem docs/user_manual.md)
sudo nano /etc/systemd/system/fruit-sorter.service
```

---

## � ĐẶC ĐIỂM NỔI BẬT

- ✅ **Tự động 100%** - Chỉ cần chạy 1 script
- ✅ **Error handling** - Bỏ qua packages lỗi, tiếp tục cài
- ✅ **Camera fallback** - Picamera2 → OpenCV tự động
- ✅ **Helper scripts** - fix_camera.sh, test_camera.sh
- ✅ **Verification** - Tự động test sau cài
- ✅ **Documentation** - Link đầy đủ mọi tài liệu

---

**🍓 Hệ thống sẵn sàng trong 1 giờ! Bắt đầu ngay! 🚀**

**Xem chi tiết từng bước:** [docs/INSTALLATION_GUIDE.md](docs/INSTALLATION_GUIDE.md)

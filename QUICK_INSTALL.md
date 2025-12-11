# 🚀 Hướng Dẫn Cài Đặt Nhanh

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

**Script sẽ tự động:**
- ✅ Tăng swap lên 4GB (cho training)
- ✅ Update hệ thống
- ✅ Cài tất cả system dependencies
- ✅ Cài OpenCV, camera, GPIO libraries
- ✅ Tạo virtual environment
- ✅ Cài Python packages từ requirements.txt
- ✅ Tạo các thư mục cần thiết
- ✅ Setup GPIO permissions
- ✅ Test cài đặt

**⏱️ Thời gian:** ~30-45 phút

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
python hardware/camera.py

# Test servo (nếu đã nối)
python hardware/servo_control.py

# Chạy web interface
python run_web.py
```

Truy cập: **http://raspberrypi.local:5000**

---

## 📋 CHECKLIST

- [ ] Project đã copy vào Pi (`~/System_Conveyor`)
- [ ] Đã chạy `chmod +x install.sh`
- [ ] Đã chạy `./install.sh` thành công
- [ ] Đã reboot
- [ ] Virtual environment hoạt động (`source venv/bin/activate`)
- [ ] Camera enabled (test với `libcamera-hello`)
- [ ] Swap = 4GB (`free -h`)

---

## 🆘 NẾU GẶP LỖI

### Lỗi: Permission denied khi chạy install.sh
```bash
chmod +x install.sh
./install.sh
```

### Lỗi: pip install failed
```bash
# Tăng swap trước:
sudo dphys-swapfile swapoff
sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=4096/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Chạy lại
./install.sh
```

### Lỗi: Camera not found
```bash
# Enable camera:
sudo raspi-config
# Interface Options → Camera → Yes → Reboot
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

## 📚 Tài Liệu Đầy Đủ

Xem chi tiết từng bước tại: **[INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)**

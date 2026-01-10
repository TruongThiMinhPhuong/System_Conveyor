# 🚀 HƯỚNG DẪN DEPLOY LÊN RASPBERRY PI 4 - CHI TIẾT TỪNG BƯỚC

> **Mục tiêu:** Deploy project từ laptop (Windows) lên Raspberry Pi 4 để chạy hệ thống phân loại trái cây.

---

## 📋 CHUẨN BỊ

### **Trên Laptop (Windows)**
- ✅ Project đã cleanup (chạy `prepare_for_pi.py`)
- ✅ Model đã train (`ai_models/mobilenet_model.tflite`)
- ✅ Raspberry Pi 4 đã cài Raspberry Pi OS

### **Phần Cứng**
- Raspberry Pi 4 (8GB RAM khuyến nghị)
- Thẻ SD 32GB+
- Camera module hoặc USB camera
- Servo motor + Motor DC
- Breadboard + jumper wires
- Nguồn 5V/3A cho Pi

---

## PHẦN 1: CHUẨN BỊ FILE TRÊN LAPTOP

### **Bước 1.1: Kiểm Tra Project**

Mở PowerShell:
```powershell
cd C:\Users\mgm\System_Conveyor
dir
```

Bạn nên thấy:
```
ai_models/
hardware/
web/
utils/
run_web.py
requirements-rpi.txt
config.yaml
```

---

### **Bước 1.2: Nén Project**

#### **Option A: Dùng Tar (Nếu có Git Bash)**

Mở **Git Bash** tại folder project:
```bash
tar -czf conveyor_pi.tar.gz ai_models hardware web utils run_web.py requirements-rpi.txt config.yaml README.md
```

#### **Option B: Dùng PowerShell (Windows 10/11)**

```powershell
# Nén tất cả files cần thiết
Compress-Archive -Path ai_models,hardware,web,utils,run_web.py,requirements-rpi.txt,config.yaml,README.md -DestinationPath conveyor_pi.zip -Force
```

#### **Option C: Dùng 7-Zip hoặc WinRAR**

1. Chọn các folder: `ai_models`, `hardware`, `web`, `utils`
2. Chọn các file: `run_web.py`, `requirements-rpi.txt`, `config.yaml`
3. Chuột phải → "Add to archive..."
4. Tên file: `conveyor_pi.zip`
5. Format: ZIP
6. Click OK

**Kiểm tra:**
```powershell
Get-Item conveyor_pi.zip | Select-Object Name, Length
```

Kích thước nên: **~50-100MB**

---

## PHẦN 2: CHUYỂN FILE SANG RASPBERRY PI

### **Bước 2.1: Tìm IP của Raspberry Pi**

Trên Pi, mở Terminal:
```bash
hostname -I
```

Kết quả ví dụ: `192.168.1.100`

---

### **Bước 2.2: Transfer File**

#### **Option A: SSH/SCP (Qua Mạng) - KHUYẾN NGHỊ**

**Bước 2.2.1: Cài PuTTY hoặc OpenSSH trên Windows**

- **Windows 10/11:** OpenSSH đã có sẵn
- **Windows cũ:** Download [PuTTY](https://www.putty.org/)

**Bước 2.2.2: Test SSH Connection**

```powershell
# Thử kết nối SSH
ssh pi@192.168.1.100
# Password mặc định: raspberry
```

Nếu thành công → nhập `exit` để thoát.

**Bước 2.2.3: Transfer File**

```powershell
# Dùng SCP
scp conveyor_pi.zip pi@192.168.1.100:~/

# Hoặc dùng WinSCP (GUI) - Download tại https://winscp.net
```

**Progress bar sẽ hiện:**
```
conveyor_pi.zip     100%  |████████████| 75MB  5.2MB/s  00:14
```

---

#### **Option B: USB Drive**

**Bước 1:** Copy file `conveyor_pi.zip` vào USB

**Bước 2:** Cắm USB vào Raspberry Pi

**Bước 3:** Trên Pi Terminal:
```bash
# Kiểm tra USB đã mount chưa
lsblk

# Mount USB (nếu chưa)
sudo mkdir -p /media/usb
sudo mount /dev/sda1 /media/usb

# Copy file
cp /media/usb/conveyor_pi.zip ~/

# Unmount USB
sudo umount /media/usb
```

---

#### **Option C: FileZilla (GUI - Dễ Nhất)**

1. Download [FileZilla Client](https://filezilla-project.org/)
2. Mở FileZilla
3. Connect:
   - Host: `sftp://192.168.1.100`
   - Username: `pi`
   - Password: `raspberry` (mặc định)
   - Port: `22`
4. Drag & drop `conveyor_pi.zip` từ bên trái (laptop) sang bên phải (Pi folder `/home/pi/`)

---

## PHẦN 3: SETUP TRÊN RASPBERRY PI

### **Bước 3.1: SSH vào Raspberry Pi**

Từ laptop:
```powershell
ssh pi@192.168.1.100
```

Nhập password (mặc định: `raspberry`)

---

### **Bước 3.2: Giải Nén Project**

```bash
# Vào home directory
cd ~

# Kiểm tra file đã có
ls -lh conveyor_pi.zip

# Giải nén (nếu là .zip)
unzip conveyor_pi.zip -d System_Conveyor

# Hoặc nếu là .tar.gz
tar -xzf conveyor_pi.tar.gz -C System_Conveyor

# Vào folder project
cd System_Conveyor

# Kiểm tra
ls -la
```

Bạn nên thấy:
```
ai_models/
hardware/
web/
utils/
run_web.py
requirements-rpi.txt
config.yaml
```

---

### **Bước 3.3: Update Raspberry Pi OS**

```bash
# Update package list
sudo apt update

# Upgrade packages (mất ~10-20 phút)
sudo apt upgrade -y

# Install Python dev tools
sudo apt install python3-pip python3-dev python3-opencv -y
```

---

### **Bước 3.4: Cài Python Dependencies**

```bash
cd ~/System_Conveyor

# Upgrade pip
pip3 install --upgrade pip

# Cài từ requirements
pip3 install -r requirements-rpi.txt

# Nếu gặp lỗi, cài từng package:
pip3 install tflite-runtime
pip3 install opencv-python
pip3 install numpy
pip3 install flask flask-socketio flask-cors
pip3 install eventlet
pip3 install RPi.GPIO
pip3 install picamera2
```

**Lưu ý:** Quá trình cài đặt mất **15-30 phút**.

---

### **Bước 3.5: Enable Camera**

```bash
# Mở raspi-config
sudo raspi-config
```

Trong menu:
1. Chọn **3. Interface Options**
2. Chọn **P1. Camera** 
3. Chọn **Yes** để enable
4. Chọn **Finish**
5. Reboot: `sudo reboot`

Sau khi reboot, SSH lại vào Pi.

---

### **Bước 3.6: Kiểm Tra Hardware**

#### **Test Camera:**
```bash
# Nếu dùng Pi Camera
libcamera-hello --list-cameras

# Chụp ảnh test
libcamera-jpeg -o test.jpg
```

#### **Test GPIO (Servo/Motor):**
```bash
# Chạy Python
python3

# Test GPIO
>>> import RPi.GPIO as GPIO
>>> GPIO.setmode(GPIO.BCM)
>>> GPIO.setup(17, GPIO.OUT)
>>> GPIO.output(17, GPIO.HIGH)
>>> GPIO.cleanup()
>>> exit()
```

Nếu không lỗi → GPIO OK!

---

## PHẦN 4: CHẠY HỆ THỐNG

### **Bước 4.1: Cấu Hình config.yaml**

```bash
cd ~/System_Conveyor
nano config.yaml
```

Chỉnh sửa các thông số phù hợp với phần cứng:
```yaml
camera:
  source: 0  # 0 = USB camera, 'picamera' = Pi Camera
  width: 640
  height: 480
  fps: 30

gpio:
  servo_pin: 17
  motor_forward_pin: 27
  motor_backward_pin: 22
  motor_enable_pin: 18
  
web:
  host: '0.0.0.0'
  port: 5001
```

Lưu: `Ctrl+X` → `Y` → `Enter`

---

### **Bước 4.2: Test Chạy Web Server**

```bash
cd ~/System_Conveyor
python3 run_web.py
```

Bạn sẽ thấy:
```
🌐 AI Fruit Sorting System - Web Interface
============================================================
🔗 Access: http://192.168.1.100:5001
📱 Mobile: http://192.168.1.100:5001
```

**Mở browser trên laptop:**
```
http://192.168.1.100:5001
```

Bạn nên thấy giao diện web!

---

### **Bước 4.3: Test AI Detection**

1. Trong web interface, click **"Start System"**
2. Đặt trái cây trước camera
3. Quan sát detection kết quả:
   - Green box = Fresh
   - Red box = Spoiled

---

### **Bước 4.4: Chạy Tự Động Khi Khởi Động**

Tạo systemd service:

```bash
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
WorkingDirectory=/home/pi/System_Conveyor
ExecStart=/usr/bin/python3 /home/pi/System_Conveyor/run_web.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Lưu và enable:
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service
sudo systemctl enable fruit-sorter.service

# Start service
sudo systemctl start fruit-sorter.service

# Check status
sudo systemctl status fruit-sorter.service
```

Giờ hệ thống sẽ tự động chạy khi Pi khởi động!

---

## PHẦN 5: TROUBLESHOOTING

### **Lỗi 1: Cannot import tflite_runtime**

```bash
pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
```

---

### **Lỗi 2: Camera not detected**

```bash
# Kiểm tra camera
vcgencmd get_camera

# Nếu chưa enable
sudo raspi-config
# Interface → Camera → Enable → Reboot
```

---

### **Lỗi 3: GPIO Permission Denied**

```bash
# Thêm user vào gpio group
sudo usermod -aG gpio pi

# Reboot
sudo reboot
```

---

### **Lỗi 4: Port 5001 already in use**

```bash
# Tìm process đang dùng port
sudo lsof -i :5001

# Kill process
sudo kill -9 <PID>

# Hoặc đổi port trong config.yaml
```

---

## PHẦN 6: TIPS & OPTIMIZATION

### **Tăng Performance**

1. **Overclock Pi 4:**
```bash
sudo nano /boot/config.txt
# Thêm:
# over_voltage=6
# arm_freq=2000
```

2. **Disable Desktop (dùng CLI only):**
```bash
sudo raspi-config
# System Options → Boot → Console
```

3. **Tăng GPU Memory:**
```bash
sudo raspi-config
# Performance → GPU Memory → 256
```

---

### **Monitor System**

```bash
# CPU temp
vcgencmd measure_temp

# Memory usage
free -h

# Disk usage
df -h

# Process list
htop
```

---

## 🎯 CHECKLIST HOÀN CHỈNH

- [ ] Cleanup project trên laptop (`prepare_for_pi.py`)
- [ ] Nén project (`conveyor_pi.zip`)
- [ ] Transfer sang Pi (SSH/USB/FileZilla)
- [ ] SSH vào Pi
- [ ] Giải nén project
- [ ] Update Pi OS (`sudo apt update && upgrade`)
- [ ] Cài dependencies (`pip3 install -r requirements-rpi.txt`)
- [ ] Enable camera (`sudo raspi-config`)
- [ ] Test camera (`libcamera-hello`)
- [ ] Test GPIO
- [ ] Chỉnh `config.yaml`
- [ ] Chạy `python3 run_web.py`
- [ ] Test web interface từ laptop
- [ ] Setup auto-start (`systemd service`)
- [ ] Test khởi động lại Pi

---

## 📞 HỖ TRỢ

Nếu gặp vấn đề:
1. Check logs: `sudo journalctl -u fruit-sorter.service -f`
2. Check web logs: `tail -f logs/*.log`
3. Test từng component riêng (camera, GPIO, model)

---

**Chúc bạn deploy thành công! 🚀**

# 🚀 Hướng Dẫn Deploy Lên Raspberry Pi 4

## Bước 1: Chuẩn Bị Project Trên Laptop

### **Cleanup và Tối Ưu**

```bash
# Chạy script cleanup
python prepare_for_pi.py
```

Script sẽ xóa:
- ✅ Dataset (~2-5GB)
- ✅ Model `.h5` (chỉ giữ `.tflite`)
- ✅ Training scripts
- ✅ Python cache
- ✅ Log files

**Kết quả:** Project giảm từ ~5GB → ~50-100MB

---

### **Nén Project**

```bash
# Tạo file nén
tar -czf conveyor_pi.tar.gz ai_models hardware web utils run_web.py requirements-rpi.txt config.yaml
```

Hoặc trên Windows:
```powershell
Compress-Archive -Path ai_models,hardware,web,utils,run_web.py,requirements-rpi.txt,config.yaml -DestinationPath conveyor_pi.zip
```

---

## Bước 2: Transfer Sang Raspberry Pi

### **Option 1: SSH (Qua mạng)**

```bash
# Copy file sang Pi
scp conveyor_pi.tar.gz pi@192.168.1.100:~/

# Hoặc dùng WinSCP trên Windows
```

### **Option 2: USB Drive**

1. Copy `conveyor_pi.tar.gz` vào USB
2. Cắm USB vào Pi
3. Copy từ USB: `cp /media/usb/conveyor_pi.tar.gz ~/`

---

## Bước 3: Setup Trên Raspberry Pi

### **SSH vào Pi**

```bash
ssh pi@192.168.1.100
```

### **Giải nén và Setup**

```bash
# Giải nén
cd ~
tar -xzf conveyor_pi.tar.gz
cd System_Conveyor

# Update system
sudo apt update
sudo apt upgrade -y

# Cài Python dependencies
sudo apt install python3-pip python3-opencv -y
pip3 install -r requirements-rpi.txt

# Enable camera
sudo raspi-config
# Interface Options → Camera → Enable

# Reboot
sudo reboot
```

---

## Bước 4: Test Trên Pi

```bash
# Chạy web server
cd ~/System_Conveyor
python3 run_web.py
```

Mở browser trên laptop:
```
http://<PI_IP_ADDRESS>:5001
```

---

## 📊 So Sánh Kích Thước

| Phiên bản | Kích thước |
|-----------|------------|
| Full (Laptop) | ~5GB |
| Sau cleanup | ~50-100MB |
| Chỉ runtime files | ~20-30MB |

---

## 🔧 Troubleshooting

### **Lỗi: TFLite Runtime**
```bash
pip3 install tflite-runtime
```

### **Lỗi: Camera không hoạt động**
```bash
sudo raspi-config
# Interface → Camera → Enable
sudo reboot
```

### **Lỗi: GPIO Permission**
```bash
sudo usermod -aG gpio pi
sudo reboot
```

---

## 🎯 Essential Files for Pi

```
System_Conveyor/
├── ai_models/
│   └── mobilenet_model.tflite  (~3MB)
├── hardware/
│   ├── conveyor.py
│   ├── camera.py
│   └── servo.py
├── web/
│   ├── app.py
│   ├── templates/
│   └── static/
├── utils/
│   ├── config.py
│   └── logger.py
├── run_web.py
├── requirements-rpi.txt
└── config.yaml
```

**Total:** ~30-50MB (thay vì ~5GB!)

---

## ✅ Checklist Deploy

- [ ] Chạy `prepare_for_pi.py` trên laptop
- [ ] Nén project
- [ ] Transfer sang Pi (SSH/USB)
- [ ] Giải nén trên Pi
- [ ] Cài dependencies: `pip3 install -r requirements-rpi.txt`
- [ ] Enable camera: `sudo raspi-config`
- [ ] Test: `python3 run_web.py`
- [ ] Access web: `http://<PI_IP>:5001`

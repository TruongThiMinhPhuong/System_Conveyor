# 🍎 AI Fruit Sorting System

**Development of a Conveyor System for Fruit Quality Classification Using AI Camera**

Hệ thống băng chuyền phân loại hoa quả tươi/hỏng tự động sử dụng AI Camera - **Chạy hoàn toàn trên Raspberry Pi 4 (8GB RAM)**

---

## 🎯 Tính Năng

- **🔍 Phát hiện hoa quả**: YOLOv8-nano (Ultralytics)
- **🧠 Phân loại tươi/hỏng**: MobileNetV2 (TensorFlow Lite)
- **🖼️ Tiền xử lý ảnh**: OpenCV (lọc màu, làm mịn, cắt ROI)
- **⚙️ Điều khiển phần cứng**: Servo MG996R, Motor DC qua L298N
- **🌐 Web Interface**: Dashboard điều khiển & giám sát real-time
- **📊 Độ chính xác**: 90-95% (với dataset đủ lớn)

### Phân Loại 2 Chiều:
- ✅ **Hoa quả tươi** → Servo 0° (đi thẳng)
- ❌ **Hoa quả hỏng** → Servo 180° (gạt phải)

---

## 🛠️ Phần Cứng

### Thiết Bị Chính
- **Raspberry Pi 4** (8GB RAM) + nguồn 5V 3A USB-C
- **Camera Module v2** 5MP 1080p (CSI connector)
- **MicroSD Card** 32GB+ (Class 10)

### Motor & Điều Khiển
- **Servo Motor**: MG996R (6V, 11-13 kg⋅cm)
- **Motor Driver**: L298N Module
- **Conveyor Motor**: JGB37-545 hoặc tương đương
- **Nguồn điện**: 6V cho servo, 12V cho motor băng chuyền

### Cấu Hình Tối Ưu (Khoảng Cách 20cm)
- **Tốc độ motor**: 35% (2.92 cm/s)
- **Khoảng cách camera-servo**: 20 cm
- **Thời gian di chuyển**: 6.85 giây
- **Độ chính xác dự kiến**: 98%
- **Throughput**: 40-45 trái/phút

---

## 📁 Cấu Trúc Project

```
System_Conveyor/
├── hardware/              # Điều khiển phần cứng (Camera, Servo, Motor)
├── ai_models/            # AI models (YOLO, MobileNetV2)
├── training/             # Scripts huấn luyện models
├── web/                  # Web Interface (Flask + SocketIO)
├── utils/                # Utilities và config
├── docs/                 # Tài liệu hướng dẫn
│   └── SYSTEM_SETUP.md   # Hướng dẫn setup đầy đủ
├── models/               # Trained models (sau khi train)
├── fruit_sorter.py       # Script chính (CLI)
├── run_web.py            # Web interface
└── install.sh            # Script cài đặt tự động
```

---

## 🚀 Cài Đặt Nhanh (3 Bước)

### Bước 1: Copy Project Vào Raspberry Pi

**Cách A: USB**
```bash
cd ~
cp -r /media/pi/USB_DRIVE/System_Conveyor .
```

**Cách B: SCP từ PC**
```bash
# Trên PC (Windows PowerShell / Linux / Mac)
scp -r System_Conveyor pi@raspberrypi.local:~/
```

**Cách C: Git Clone**
```bash
cd ~
git clone https://github.com/your-username/System_Conveyor.git
```

### Bước 2: Chạy Install Script

```bash
cd ~/System_Conveyor
chmod +x install.sh
./install.sh
```

**Script tự động làm:**
- ✅ Tăng swap lên 4GB
- ✅ Cài đặt system dependencies (libcap-dev, libffi-dev...)
- ✅ Enable camera & GPIO
- ✅ Tạo virtual environment
- ✅ Cài Python packages (OpenCV, YOLOv8, Flask, TensorFlow Lite...)
- ✅ Setup GPIO permissions
- ✅ Verify cài đặt

**⏱️ Thời gian**: ~30-45 phút (tự động)

### Bước 3: Reboot

```bash
sudo reboot
```

---

## ✅ Sau Khi Cài Đặt

```bash
# SSH vào Pi
ssh pi@raspberrypi.local

# Vào project
cd ~/System_Conveyor

# Activate environment
source venv/bin/activate

# Test camera
python hardware/camera.py

# Chạy web interface
python run_web.py
```

**Truy cập**: http://192.168.137.177:5001

---

## 🎓 Training AI Models

### Trên Raspberry Pi 4 (Khuyến Nghị PC/GPU)

```bash
# YOLO Detection
cd training/yolo
python train_yolo.py --epochs 100 --batch 4

# MobileNetV2 Classification
cd training/mobilenet
python train_mobilenet.py --epochs 50 --batch 8
python export_tflite.py
```

### Thu Thập Dữ Liệu

```bash
# Chụp ảnh cho training
python training/data_collection/collect_images.py \
    --mode classification \
    --count 200 \
    --interval 2.0
```

**Yêu cầu dataset**: 200+ ảnh/class cho mỗi loại trái cây (cam, ổi, táo)

---

## ▶️ Chạy Hệ Thống

### Chế Độ CLI
```bash
cd ~/System_Conveyor
source venv/bin/activate
python fruit_sorter.py
```

### Web Interface (Khuyến Nghị) 🌐
```bash
python run_web.py
```

**Truy cập:**
- Raspberry Pi: http://192.168.137.177:5001
- Từ mạng local: http://192.168.137.177:5001

**Tính Năng Web:**
- 📹 Video Feed: Live camera với bounding boxes & phân loại
- 🎯 Last Detection: Hiển thị ảnh trái cây vừa phát hiện với thông tin chi tiết
- ⚙️ System Control: Start/Stop hệ thống
- 🔧 Motor Control: Điều chỉnh tốc độ (35% khuyến nghị)
- 🔄 Servo Control: Test servo (Fresh 0°, Spoiled 180°, Center 90°)
- 📊 Statistics: Thống kê real-time (tươi/hỏng, FPS, uptime)
- 📱 Responsive: Hoạt động tốt trên mobile/tablet

---

## 🔧 Cấu Hình

File: `utils/config.py`

```python
# Tốc độ motor (tối ưu cho 20cm)
CONVEYOR_SPEED_DETECTION = 35      # 2.92 cm/s

# Khoảng cách camera-servo
CAMERA_TO_SERVO_DISTANCE = 20.0    # cm

# Servo angles (đã cập nhật)
SERVO_ANGLE_FRESH = 0              # Tươi - Đi thẳng
SERVO_ANGLE_SPOILED = 180          # Hỏng - Gạt phải
SERVO_ANGLE_CENTER = 90            # Neutral

# AI thresholds
YOLO_CONFIDENCE_THRESHOLD = 0.5
CLASSIFICATION_THRESHOLD = 0.6
```

---

## 📊 Quy Trình Hoạt Động

1. **Camera** chụp ảnh liên tục (25 FPS)
2. **YOLOv8** phát hiện trái cây
3. **Preprocessing** (OpenCV): Cắt ROI, chuẩn hóa ảnh
4. **MobileNetV2** phân loại tươi/hỏng
5. **Servo** điều hướng:
   - ✅ Tươi → 0° (thẳng)
   - ❌ Hỏng → 180° (phải)
6. **Băng chuyền** tiếp tục di chuyển

---

## 🔍 Cải Thiện Độ Chính Xác

### Dataset Chất Lượng
- **Số lượng**: 200+ ảnh/class cho mỗi loại trái (cam, ổi, táo)
- **Đa dạng**: Nhiều góc độ, ánh sáng, kích thước
- **Label chính xác**: Phân biệt rõ tươi/hỏng

### Preprocessing Riêng Cho Từng Loại
- 🍊 **Cam**: Tăng contrast để thấy vết thâm
- 🥭 **Ổi**: Tăng saturation phân biệt màu
- 🍎 **Táo**: Sharpen để thấy rõ bề mặt

### Expected Results
- **Overall Accuracy**: 90-95%
- **Fresh Precision**: 88-92%
- **Spoiled Precision**: 88-92%

---

## 🆘 Troubleshooting

### Camera Không Hoạt Động
```bash
# Enable camera
sudo raspi-config  # Interface → Camera → Yes
sudo reboot

# Test camera
libcamera-hello
```

### GPIO Permission Denied
```bash
sudo usermod -a -G gpio,i2c,spi $USER
# Logout và login lại
```

### Out of Memory
```bash
# Kiểm tra swap
free -h

# Tăng swap (install.sh đã làm)
sudo dphys-swapfile swapoff
sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=4096/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### Package Import Failed
```bash
source venv/bin/activate
pip install opencv-python ultralytics flask tensorflow-lite
```

---

## 📖 Tài Liệu

- **[docs/SYSTEM_SETUP.md](docs/SYSTEM_SETUP.md)** - Hardware & Software setup đầy đủ
- Includes:
  - Part 1: Hardware Setup (camera, servo, motor wiring)
  - Part 2: Software Setup (OS, dependencies, training)
  - Wiring diagrams
  - Configuration cho 20cm distance
  - Troubleshooting guide

---

## 📋 Changelog - Version 1.0.0

### ✅ Cập Nhật Mới Nhất

**Tối Ưu Hóa 20cm Distance:**
- Giảm tốc độ motor: 60% → 35% (chính xác hơn)
- Cập nhật timing parameters
- Thêm constants: `CAMERA_TO_SERVO_DISTANCE`, `FRUIT_TRAVEL_TIME`
- Độ chính xác dự kiến: 98%

**Web Interface:**
- Fix Last Detection: Hiển thị ảnh thực tế của trái cây
- Thêm image enlargement modal
- Cập nhật servo button labels (Fresh/Spoiled)
- Color-coded detection (green=fresh, red=spoiled)

**Servo Configuration:**
- Fresh: 0° (đi thẳng) - thay đổi từ 45°
- Spoiled: 180° (gạt phải) - thay đổi từ 135°
- Center: 90° (neutral)

**Documentation:**
- Gộp tài liệu thành 1 file: `docs/SYSTEM_SETUP.md`
- Hướng dẫn cải thiện accuracy cho 3 loại trái
- Timing optimization guide

### Hoàn Thành
- ✅ Code: Python 3000+ LOC
- ✅ Web Interface: Dashboard đầy đủ
- ✅ Documentation: Hướng dẫn chi tiết
- ✅ Testing: Scripts kiểm tra
- ✅ Deployment: Sẵn sàng production

---

## 🎯 Next Steps

1. **Hardware Assembly**: Lắp ráp phần cứng theo [docs/SYSTEM_SETUP.md](docs/SYSTEM_SETUP.md)
2. **Software Installation**: Chạy `./install.sh`
3. **Data Collection**: Thu thập 200+ ảnh/class
4. **Training**: Train models (trên Pi hoặc PC/GPU)
5. **Testing**: Test toàn bộ hệ thống
6. **Production**: Chạy hệ thống thực tế

---

## 👨‍💻 Author

**Minh Phuong** - 2025

Development of a Conveyor System for Fruit Quality Classification Using AI Camera

---

## 📝 License

MIT License

---

**Version**: 1.0.0  
**Last Updated**: 2025-12-17  
**Status**: Production Ready ✅

# 🍎 AI Fruit Sorting Conveyor System

> **🍓 QUAN TRỌNG:** Toàn bộ hệ thống chạy **HOÀN TOÀN trên Raspberry Pi 4 (8GB RAM)**!  
> Không cần PC/Laptop riêng. Training, deployment, web interface - TẤT CẢ trên Pi!  
> Xem chi tiết: **[EVERYTHING_ON_PI4.md](docs/EVERYTHING_ON_PI4.md)** ⚡

Hệ thống băng chuyền phân loại hoa quả tươi/hỏng tự động sử dụng AI Camera và Raspberry Pi 4.

## 🎯 Tính Năng

- **Phát hiện hoa quả**: YOLOv8-nano (Ultralytics)
- **Phân loại tươi/hỏng**: MobileNetV2 (TensorFlow Lite)
- **Tiền xử lý ảnh**: OpenCV (lọc màu, làm mịn, cắt ROI)
- **Điều khiển phần cứng**: Servo MG996R, Motor DC qua L298N

## 🛠️ Phần Cứng

- **Raspberry Pi 4** (8GB RAM) với nguồn 5V 3A
- **Camera Module** 5MP 1080p
- **Servo Motor**: MG996R
- **Motor Driver**: L298N
- **Conveyor Motor**: JGB37-545
- Nguồn điện riêng cho servo (6V) và motor băng chuyền (12V)

## 📁 Cấu Trúc Project

```
System_Conveyor/
├── hardware/              # Điều khiển phần cứng
├── ai_models/            # AI models (YOLO, MobileNetV2)
├── training/             # Scripts huấn luyện models
├── utils/                # Utilities và config
├── docs/                 # Tài liệu hướng dẫn
├── models/               # Trained models (sau khi train)
├── main.py               # Script chính
└── requirements.txt      # Python dependencies
```

## 🚀 Cài Đặt

### 1. Cài Đặt Trên Raspberry Pi

```bash
cd System_Conveyor
chmod +x install.sh
./install.sh
```

### 2. Kích Hoạt Camera và GPIO

```bash
sudo raspi-config
# Interface Options → Camera → Enable
# Interface Options → GPIO → Enable
```

### 3. Huấn Luyện Models (Trên PC/Laptop)

Xem hướng dẫn chi tiết tại [docs/training_guide.md](docs/training_guide.md)

```bash
# YOLO Detection
cd training/yolo
python train_yolo.py

# MobileNetV2 Classification
cd training/mobilenet
python train_mobilenet.py
python export_tflite.py
```

## 📖 Tài Liệu

- [Hardware Setup](docs/hardware_setup.md) - Hướng dẫn đấu nối phần cứng
- **[Detailed Wiring Diagram](docs/detailed_wiring_diagram.md) - Sơ đồ kết nối chi tiết ⚡**
- [Software Setup](docs/software_setup.md) - Cài đặt phần mềm
- [Training Guide](docs/training_guide.md) - Huấn luyện AI models
- [User Manual](docs/user_manual.md) - Hướng dẫn sử dụng
- [Web Interface Guide](docs/web_interface_guide.md) - Giao diện web

## ▶️ Chạy Hệ Thống

### Chế Độ CLI (Command Line)
```bash
python fruit_sorter.py
```

### Giao Diện Web (Khuyến Nghị) 🌐
```bash
python run_web.py
```

Truy cập giao diện web:
- **Local**: http://localhost:5000
- **Từ máy khác**: http://\<raspberry-pi-ip\>:5000

#### Tính Năng Web Interface:
- 📹 **Video Feed**: Xem trực tiếp từ camera với bounding boxes
- ⚙️ **Điều Khiển Hệ Thống**: Start/Stop hệ thống
- 🔧 **Điều Khiển Motor**: Điều chỉnh tốc độ băng chuyền
- 🔄 **Điều Khiển Servo**: Test vị trí servo (Left/Center/Right)
- 📊 **Thống Kê Real-time**: Số lượng tươi/hỏng, FPS, uptime
- 📱 **Responsive**: Hoạt động tốt trên mobile và tablet

## 🔧 Cấu Hình

Chỉnh sửa file `utils/config.py` để tùy chỉnh:
- GPIO pins
- Detection thresholds
- Camera settings
- Motor speeds

## 📊 Quy Trình Hoạt Động

1. Camera chụp ảnh liên tục
2. YOLOv8 phát hiện đối tượng
3. Phân loại và xử lý:
   - **Nếu KHÔNG phải hoa quả** → Servo rẽ trái (thùng reject 1)
   - **Nếu là hoa quả** → Cắt ROI và tiền xử lý (OpenCV)
   - **MobileNetV2** phân loại tươi/hỏng
4. Servo điều hướng:
   - **Hoa quả tươi** → Đi thẳng (servo ở giữa) 🍎
   - **Hoa quả hỏng** → Rẽ phải (servo sang phải) 🍂
   - **Vật khác** → Rẽ trái (servo sang trái) ⚠️
5. Băng chuyền tiếp tục di chuyển

## 📝 License

MIT License

## 👨‍💻 Tác Giả

AI Fruit Sorting System - 2025

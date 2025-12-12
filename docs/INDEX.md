
## 📚 Index - Danh Mục Tài Liệu

### 🍓 TẤT CẢ TRÊN PI 4 (8GB RAM)
**[EVERYTHING_ON_PI4.md](EVERYTHING_ON_PI4.md)** ⭐ - Xác nhận: Chỉ cần duy nhất Pi 4!

### 🎯 BẮT ĐẦU NHANH (Quick Start)
1. **[README.md](../README.md)** - Tổng quan dự án
2. **[POWER_SUPPLY_QUICK_GUIDE.md](POWER_SUPPLY_QUICK_GUIDE.md)** ⚡ - Hướng dẫn nối nguồn nhanh

### 🔧 LẮP ĐẶT PHẦN CỨNG
3. **[hardware_setup.md](hardware_setup.md)** - Hướng dẫn lắp ráp đầy đủ
4. **[detailed_wiring_diagram.md](detailed_wiring_diagram.md)** 🔌 - Sơ đồ kết nối chi tiết

### 💻 CÀI ĐẶT PHẦN MỀM  
5. **[software_setup.md](software_setup.md)** - Cài đặt Raspberry Pi OS & dependencies
6. **[web_interface_guide.md](web_interface_guide.md)** 🌐 - Giao diện web dashboard

### 🤖 HUẤN LUYỆN AI
7. **[training_guide.md](training_guide.md)** - Training YOLOv8 & MobileNetV2 (PC/GPU)
8. **[TRAINING_ON_PI.md](TRAINING_ON_PI.md)** 🍓 - Training trực tiếp trên Raspberry Pi 4

### 📖 VẬN HÀNH
9. **[user_manual.md](user_manual.md)** - Hướng dẫn sử dụng hệ thống

---

## 🚀 Quy Trình Hoàn Chỉnh

### Phase 1: MUA SẮM & CHUẨN BỊ (1-2 ngày)

#### Phần Cứng Cơ Bản:
- [x] Raspberry Pi 4 (8GB) + case + tản nhiệt
- [x] Camera Module v2 (5MP 1080p)
- [x] Thẻ nhớ microSD 32GB+ (Class 10)
- [x] Servo MG996R
- [x] Motor Driver L298N  
- [x] Motor băng chuyền JGB37-545

#### Nguồn Điện:
- [x] Adapter 5V 3A USB-C cho Pi
- [x] Adapter 12V 5A DC cho motor + servo
- [x] Buck Converter LM2596 (12V→6V)

#### Phụ Kiện:
- [x] Breadboard hoặc PCB
- [x] Jumper wires (M-M, M-F)
- [x] Dây nguồn DC 18-20 AWG
- [x] Terminal blocks
- [x] Cấu trúc băng chuyền (tự làm/mua)

📄 **Tham khảo:** [POWER_SUPPLY_QUICK_GUIDE.md](POWER_SUPPLY_QUICK_GUIDE.md)

---

### Phase 2: LẮP RÁP PHẦN CỨNG (2-3 giờ)

#### Bước 1: Kết Nối Nguồn Điện
```
1. Điều chỉnh Buck Converter về 6V
2. Kết nối nguồn 12V → Buck + L298N
3. Tạo Common Ground
4. Kết nối servo với 6V từ Buck
```

#### Bước 2: Kết Nối GPIO
```
GPIO 18 → Servo Signal
GPIO 22 → L298N ENA
GPIO 23 → L298N IN1
GPIO 24 → L298N IN2
GND     → Common GND
```

#### Bước 3: Lắp Camera
```
1. Kết nối ribbon cable vào CSI port
2. Cố định camera ở vị trí quan sát băng chuyền
```

📄 **Tham khảo:** 
- [hardware_setup.md](hardware_setup.md) - Chi tiết đầy đủ
- [detailed_wiring_diagram.md](detailed_wiring_diagram.md) - Sơ đồ

---

### Phase 3: CÀI ĐẶT PHẦN MỀM (1-2 giờ)

#### Trên Raspberry Pi:
```bash
# 1. Flash Raspberry Pi OS
# 2. Cài đặt hệ thống
cd System_Conveyor
chmod +x install.sh
./install.sh

# 3. Kích hoạt camera
sudo raspi-config
# Interface → Camera → Enable
```

#### Trên PC/Laptop:
```bash
# Clone project
git clone <repo-url>

# Cài môi trường training
conda create -n fruit_training python=3.9
conda activate fruit_training
pip install torch torchvision tensorflow ultralytics
```

📄 **Tham khảo:** [software_setup.md](software_setup.md)

---

### Phase 4: THU THẬP DỮ LIỆU (1-2 ngày)

#### Thu Thập Ảnh Trái Cây:
```bash
# Trên Raspberry Pi
python training/data_collection/collect_images.py \
    --mode classification \
    --count 200 \
    --interval 2
```

#### Yêu Cầu Dataset:
- **Phát hiện (YOLO):** 200-500 ảnh có trái cây
- **Phân loại (MobileNetV2):**
  - 100-150 ảnh tươi
  - 100-150 ảnh hỏng

📄 **Tham khảo:** [training_guide.md](training_guide.md) - Phần Data Collection

---

### Phase 5: GÁN NHÃN DỮ LIỆU (2-4 giờ)

#### Cho YOLO (Object Detection):
```bash
# Trên PC
labelImg
# Chọn format YOLO
# Vẽ bounding box cho mỗi trái cây
```

#### Cho MobileNetV2 (Classification):
```bash
python training/mobilenet/prepare_data.py \
    --source raw_images \
    --train-split 0.7 \
    --val-split 0.15
```

📄 **Tham khảo:** [training_guide.md](training_guide.md) - Annotation Section

---

### Phase 6: HUẤN LUYỆN MODELS (2-6 giờ)

#### Train YOLOv8 (1-3 giờ trên GPU):
```bash
cd training/yolo
python train_yolo.py \
    --epochs 100 \
    --batch 16 \
    --imgsz 640
```

#### Train MobileNetV2 (30-60 phút):
```bash
cd training/mobilenet
python train_mobilenet.py \
    --epochs 50 \
    --batch 32

# Export to TFLite
python export_tflite.py \
    --model ./mobilenet_training/mobilenet_fruit_classifier_best.keras \
    --quantize
```

📄 **Tham khảo:** [training_guide.md](training_guide.md)

---

### Phase 7: TRIỂN KHAI LÊN RASPBERRY PI (30 phút)

```bash
# 1. Transfer models từ PC sang Pi
scp models/yolov8n_fruit.pt pi@raspberrypi:~/System_Conveyor/models/
scp models/mobilenet_classifier.tflite pi@raspberrypi:~/System_Conveyor/models/

# 2. Verify models
cd ~/System_Conveyor
python -c "from ai_models import YOLODetector; print('YOLO OK')"
python -c "from ai_models import MobileNetClassifier; print('MobileNet OK')"
```

📄 **Tham khảo:** [software_setup.md](software_setup.md) - Model Deployment

---

### Phase 8: TEST & CALIBRATE (1-2 giờ)

#### Test Từng Module:
```bash
# Test camera
python hardware/camera.py

# Test servo
python hardware/servo_control.py

# Test motor
python hardware/motor_control.py

# Test AI models
python ai_models/yolo_detector.py
python ai_models/mobilenet_classifier.py
```

#### Calibrate:
```bash
# Chỉnh góc servo trong utils/config.py
SERVO_ANGLE_LEFT = 45   # Fresh
SERVO_ANGLE_RIGHT = 135 # Spoiled

# Chỉnh tốc độ motor
CONVEYOR_SPEED_DEFAULT = 60
```

📄 **Tham khảo:** [user_manual.md](user_manual.md) - Calibration

---

### Phase 9: VẬN HÀNH (Sẵn sàng!)

#### Chế độ CLI:
```bash
python fruit_sorter.py
```

#### Chế độ Web Interface (Khuyến nghị):
```bash
python run_web.py
# Truy cập: http://raspberrypi-ip:5000
```

📄 **Tham khảo:** 
- [user_manual.md](user_manual.md)
- [web_interface_guide.md](web_interface_guide.md)

---

## 🎯 Các Tài Liệu Theo Mục Đích

### Nếu Bạn Muốn...

#### ❓ Hiểu tổng quan dự án
→ Đọc [README.md](../README.md)

#### 🔌 Kết nối nguồn điện
→ Đọc [POWER_SUPPLY_QUICK_GUIDE.md](POWER_SUPPLY_QUICK_GUIDE.md) ⭐ (NHANH)
→ Hoặc [detailed_wiring_diagram.md](detailed_wiring_diagram.md) (CHI TIẾT)

#### 🔧 Lắp ráp phần cứng đầy đủ
→ Đọc [hardware_setup.md](hardware_setup.md)

#### ⚙️ Cài đặt phần mềm
→ Đọc [software_setup.md](software_setup.md)

#### 🤖 Train AI models
→ Đọc [training_guide.md](training_guide.md)

#### 🌐 Dùng web interface
→ Đọc [web_interface_guide.md](web_interface_guide.md)

#### 🚀 Vận hành hệ thống
→ Đọc [user_manual.md](user_manual.md)

---

## ⏱️ Timeline Ước Tính

| Phase | Thời Gian | Ghi Chú |
|-------|-----------|---------|
| Mua sắm linh kiện | 1-2 ngày | Tùy thời gian ship |
| Lắp ráp phần cứng | 2-3 giờ | Nếu đã có kinh nghiệm |
| Cài đặt phần mềm | 1-2 giờ | Bao gồm Pi + PC |
| Thu thập dữ liệu | 1-2 ngày | 300-400 ảnh |
| Gán nhãn dữ liệu | 2-4 giờ | YOLO + Classification |
| Train models (GPU) | 2-6 giờ | Có thể chạy qua đêm |
| Deploy & Test | 1-2 giờ | Calibration |
| **TỔNG** | **~3-5 ngày** | Làm part-time |

---

## 🆘 Troubleshooting - Tra Cứu Nhanh

### Lỗi Phần Cứng

| Triệu Chứng | Nguyên Nhân | Giải Pháp | Tài Liệu |
|-------------|-------------|-----------|----------|
| Servo không chạy | Thiếu nguồn 6V | Kiểm tra Buck converter | [POWER_QUICK](POWER_SUPPLY_QUICK_GUIDE.md) |
| Motor không quay | Thiếu 12V | Kiểm tra L298N | [Hardware Setup](hardware_setup.md) |
| GPIO không hoạt động | Chưa nối Common GND | Nối tất cả GND chung | [Wiring Diagram](detailed_wiring_diagram.md) |
| Camera không nhận | Ribbon cable lỏng | Kiểm tra kết nối CSI | [Hardware Setup](hardware_setup.md) |

### Lỗi Phần Mềm

| Triệu Chứng | Nguyên Nhân | Giải Pháp | Tài Liệu |
|-------------|-------------|-----------|----------|
| Model not found | Chưa train/copy model | Copy model vào `/models` | [Software Setup](software_setup.md) |
| Import error | Thiếu dependency | Chạy lại `install.sh` | [Software Setup](software_setup.md) |
| Low accuracy | Dataset kém | Thu thập thêm dữ liệu | [Training Guide](training_guide.md) |
| Slow FPS | Resolution cao | Giảm resolution | [User Manual](user_manual.md) |

---

## 📞 Hỗ Trợ

### Liên Hệ
- GitHub Issues: [Create Issue]
- Email: support@example.com

### Tài Nguyên Bổ Sung
- YOLOv8 Docs: https://docs.ultralytics.com
- TensorFlow Lite: https://www.tensorflow.org/lite
- Raspberry Pi: https://www.raspberrypi.com/documentation

---

## ✅ Checklist Tổng Thể

### Trước Khi Bắt Đầu:
- [ ] Đã đọc README.md
- [ ] Đã mua đủ linh kiện
- [ ] Có PC/Laptop với GPU (cho training)

### Lắp Ráp:
- [ ] Hoàn thành kết nối nguồn điện
- [ ] Hoàn thành kết nối GPIO
- [ ] Camera đã test OK
- [ ] Servo đã test OK
- [ ] Motor đã test OK

### Phần Mềm:
- [ ] Đã cài Raspberry Pi OS
- [ ] Đã chạy install.sh
- [ ] Camera được enable
- [ ] Test code chạy được

### AI Models:
- [ ] Đã thu thập đủ dữ liệu
- [ ] Đã gán nhãn dataset
- [ ] Đã train YOLO
- [ ] Đã train MobileNetV2
- [ ] Đã copy models lên Pi

### Vận Hành:
- [ ] Hệ thống chạy được
- [ ] Độ chính xác chấp nhận được
- [ ] Web interface hoạt động
- [ ] Đã calibrate servo/motor

---

**Chúc bạn thành công với dự án! 🍎🤖✨**



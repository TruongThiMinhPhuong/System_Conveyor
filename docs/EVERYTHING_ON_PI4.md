

### Raspberry Pi 4 (8GB RAM) CÓ THỂ LÀM TẤT CẢ:

✅ **Thu thập dữ liệu** → Camera trên Pi  
✅ **Gán nhãn dữ liệu** → LabelImg trên Pi (hoặc VNC)  
✅ **Training YOLOv8** → Trực tiếp trên Pi (8-10 giờ)  
✅ **Training MobileNetV2** → Trực tiếp trên Pi (2-3 giờ)  
✅ **Deploy & Inference** → Chạy hệ thống trên Pi  
✅ **Web Interface** → Giao diện web chạy trên Pi  
✅ **Điều khiển phần cứng** → GPIO/Servo/Motor trên Pi  



**CHỈ CẦN:**
- ✅ 1x Raspberry Pi 4 (8GB RAM)
- ✅ Camera Module
- ✅ Servo + Motor + Driver
- ✅ Nguồn điện

---

## 🚀 QUY TRÌNH ĐẦY ĐỦ TRÊN PI 4 (8GB)

### PHASE 1: Thiết Lập Ban Đầu (1 giờ)

```bash
# 1. Flash Raspberry Pi OS (64-bit, Desktop)
# Dùng Raspberry Pi Imager

# 2. First boot - Update
sudo apt update && sudo apt upgrade -y

# 3. Enable Camera & GPIO
sudo raspi-config
# Interface Options → Camera → Enable
# Interface Options → I2C → Enable
# Interface Options → SPI → Enable

# 4. Tăng Swap (QUAN TRỌNG cho training!)
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Sửa: CONF_SWAPSIZE=4096
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
free -h  # Verify swap = 4GB

# 5. Clone project
git clone <your-repo> ~/System_Conveyor
cd ~/System_Conveyor

# 6. Chạy install script
chmod +x install.sh
./install.sh
# Cài TẤT CẢ dependencies (AI + Hardware + Web)

# 7. Reboot
sudo reboot
```

---

### PHASE 2: Lắp Ráp Phần Cứng (2-3 giờ)

Theo hướng dẫn:
- **[POWER_SUPPLY_QUICK_GUIDE.md](POWER_SUPPLY_QUICK_GUIDE.md)** - Kết nối nguồn
- **[detailed_wiring_diagram.md](detailed_wiring_diagram.md)** - Đấu nối chi tiết

**Test từng module:**
```bash
cd ~/System_Conveyor
source venv/bin/activate

python hardware/camera.py      # ✓ Camera OK
python hardware/servo_control.py   # ✓ Servo OK  
python hardware/motor_control.py   # ✓ Motor OK
```

---

### PHASE 3: Thu Thập Dữ Liệu (1-2 ngày)

**TRỰC TIẾP trên Pi 4:**

```bash
cd ~/System_Conveyor
source venv/bin/activate

# Thu thập ảnh cho classification
python training/data_collection/collect_images.py \
    --mode classification \
    --count 150 \
    --interval 2.0

# Kết quả:
# raw_images/fresh/   → 75 ảnh
# raw_images/spoiled/ → 75 ảnh
```

**Yêu cầu tối thiểu cho Pi:**
- YOLO: 100-150 ảnh (có trái cây)
- MobileNetV2: 50-75 ảnh mỗi class (fresh/spoiled)

---

### PHASE 4: Gán Nhãn (2-3 giờ)

**Cách 1: Trực tiếp trên Pi Desktop**
```bash
# Cài LabelImg
pip3 install labelImg

# Chạy
labelImg
# Format: YOLO
# Vẽ bounding box cho từng trái cây
```

**Cách 2: Qua VNC (từ PC/laptop)**
```bash
# Bật VNC trên Pi
sudo raspi-config
# Interface → VNC → Enable

# Từ PC: Dùng VNC Viewer kết nối
# Chạy labelImg như bình thường
```

**Cách 3: Copy ảnh → Gán nhãn offline → Copy lại**
```bash
# Từ Pi
scp -r raw_images your-pc:~/

# Trên PC: Gán nhãn
# Copy ngược lại Pi
scp -r labeled_images pi@raspberrypi:~/System_Conveyor/datasets/
```

**Nhưng tốt nhất: Làm trực tiếp trên Pi qua VNC!**

---

### PHASE 5: Training AI Models TRÊN PI (6-15 giờ total)

#### 5.1. Chuẩn Bị Dataset

```bash
cd ~/System_Conveyor

# Tổ chức dataset YOLO
mkdir -p datasets/fruit_detection/{images,labels}/{train,val}
# Copy ảnh và labels đã annotate vào

# Tổ chức dataset MobileNetV2
python training/mobilenet/prepare_data.py \
    --source raw_images \
    --train-split 0.7
```

#### 5.2. Train YOLOv8 (8-10 giờ)

```bash
cd ~/System_Conveyor/training/yolo

# Chạy trong screen để không bị ngắt
screen -S yolo_training

# Training tối ưu cho Pi 4 8GB
python train_yolo.py \
    --data dataset.yaml \
    --epochs 50 \
    --batch 4 \
    --imgsz 416 \
    --device cpu \
    --workers 2 \
    --cache False

# Detach: Ctrl+A, D
# Monitor: screen -r yolo_training
```

**Theo dõi nhiệt độ trong terminal khác:**
```bash
watch -n 5 'vcgencmd measure_temp && free -h'
```

#### 5.3. Train MobileNetV2 (2-3 giờ)

```bash
cd ~/System_Conveyor/training/mobilenet

# Chạy trong screen
screen -S mobilenet_training

# Training
python train_mobilenet.py \
    --dataset ./datasets/fruit_classification \
    --epochs 30 \
    --batch 8 \
    --image-size 160

# Export TFLite
python export_tflite.py \
    --model ./mobilenet_training/mobilenet_fruit_classifier_best.keras \
    --output ../../models/mobilenet_classifier.tflite \
    --quantize

# Detach: Ctrl+A, D
```

**Models được lưu tại:**
- `models/yolov8n_fruit.pt` ✓
- `models/mobilenet_classifier.tflite` ✓

---

### PHASE 6: Deploy & Test (30 phút)

```bash
cd ~/System_Conveyor
source venv/bin/activate

# Models đã ở đúng chỗ (không cần copy)

# Test AI models
python ai_models/yolo_detector.py       # ✓
python ai_models/mobilenet_classifier.py # ✓

# Test toàn bộ hệ thống
python hardware/conveyor.py
```

---

### PHASE 7: Chạy Hệ Thống (Sẵn sàng!)

#### Cách 1: Command Line
```bash
python fruit_sorter.py
```

#### Cách 2: Web Interface (Khuyến nghị)
```bash
python run_web.py
# Truy cập: http://raspberrypi.local:5000
# Hoặc: http://<pi-ip>:5000
```

**Từ điện thoại/tablet/PC khác:** Chỉ cần vào địa chỉ web!

---

## 📊 Thông Số Kỹ Thuật

### Raspberry Pi 4 (8GB RAM) - Spec

| Thông Số | Giá Trị | Ghi Chú |
|----------|---------|---------|
| **CPU** | Quad-core Cortex-A72 (1.5GHz) | ARM v8 64-bit |
| **RAM** | 8GB LPDDR4 | Đủ cho training + inference |
| **GPU** | VideoCore VI | Tăng tốc video decode |
| **Storage** | microSD 32GB+ | Class 10 khuyến nghị |
| **USB** | 2x USB 3.0, 2x USB 2.0 | Cho webcam/storage |
| **GPIO** | 40 pins | Điều khiển servo/motor |
| **Camera** | CSI interface | Camera Module v2 |
| **Ethernet** | Gigabit | Tốt hơn WiFi cho training |

### Hiệu Năng Thực Tế

| Task | Pi 4 8GB (CPU) | PC GPU (RTX 3060) |
|------|----------------|-------------------|
| **Data Collection** | ✓ Native | Remote SSH |
| **YOLO Training (50 epochs)** | 8-10 giờ | 45-60 phút |
| **MobileNetV2 (30 epochs)** | 2-3 giờ | 20-30 phút |
| **Inference YOLO** | 5-10 FPS | 60+ FPS |
| **Inference MobileNetV2** | 20-30 FPS | 100+ FPS |
| **Web Interface** | 10-15 FPS | 30+ FPS |
| **Power Consumption** | 15W | 200W+ |

**Kết luận:** Pi 4 8GB **ĐỦ MẠNH** cho toàn bộ quy trình!

---

## 💾 Yêu Cầu Lưu Trữ

### Trên microSD Card (32GB khuyến nghị)

| Mục | Dung Lượng |
|-----|------------|
| Raspberry Pi OS | ~8GB |
| System_Conveyor code | ~500MB |
| Dependencies (Python) | ~2GB |
| Raw images (300 ảnh) | ~300MB |
| Labeled dataset | ~500MB |
| Trained models | ~50MB |
| Logs & temp files | ~500MB |
| **TỔNG** | **~12GB** |

**Còn dư ~20GB** cho mở rộng sau!

---

## 🌡️ Quản Lý Nhiệt Độ

### QUAN TRỌNG cho training dài hạn!

```bash
# Monitor liên tục
watch -n 2 vcgencmd measure_temp

# Script cảnh báo
cat > ~/check_temp.sh << 'EOF'
#!/bin/bash
while true; do
    temp=$(vcgencmd measure_temp | awk -F= '{print $2}' | awk -F\' '{print $1}')
    echo "$(date '+%H:%M:%S'): $temp°C"
    if (( $(echo "$temp > 80" | bc -l) )); then
        echo "⚠️ HIGH TEMP! Consider pausing..."
    fi
    sleep 60
done
EOF

chmod +x ~/check_temp.sh
./check_temp.sh &
```

### Nhiệt Độ An Toàn

- ✅ **< 70°C**: Tốt, tiếp tục
- ⚠️ **70-80°C**: Bình thường khi training
- 🔥 **> 80°C**: Cần quạt tản nhiệt
- 🛑 **> 85°C**: DỪNG, thêm cooling

### Solution Tản Nhiệt

1. **Fan 5V** (gắn vào GPIO hoặc USB)
2. **Heatsink** nhôm/đồng
3. **Case có quạt** tích hợp
4. **Để nơi thoáng mát**

---

## 🔋 Nguồn Điện Đầy Đủ

### Cho Raspberry Pi 4 (8GB)
- **Adapter chính hãng**: 5V 3A USB-C
- **Hoặc**: Adapter 5V 3.5A-4A (an toàn hơn khi training)

### Cho Servo + Motor
- **12V 5A DC Adapter** (đã đủ cho tất cả)
- **Buck Converter** LM2596 (hạ 12V→6V cho servo)

**Tổng: 2 nguồn điện** (Pi + Hardware)

---

## ✅ Checklist Hoàn Chỉnh

### Phần Cứng
- [ ] Raspberry Pi 4 (8GB RAM) ✓
- [ ] Camera Module v2 (5MP)
- [ ] Case + Quạt + Heatsink
- [ ] microSD 32GB+ (Class 10)
- [ ] Servo MG996R
- [ ] L298N Motor Driver
- [ ] Motor JGB37-545
- [ ] Nguồn 5V 3A (Pi)
- [ ] Nguồn 12V 5A (Motor/Servo)
- [ ] Buck Converter (12V→6V)
- [ ] Breadboard + Jumpers
- [ ] Cấu trúc băng chuyền

### Phần Mềm (Trên Pi)
- [ ] Raspberry Pi OS 64-bit Desktop
- [ ] Swap tăng lên 4GB
- [ ] Camera & GPIO enabled
- [ ] Project đã clone
- [ ] Đã chạy install.sh
- [ ] Dependencies đã cài đủ

### Dataset
- [ ] 100-150 ảnh cho YOLO
- [ ] 50-75 ảnh/class cho classification
- [ ] Đã gán nhãn (LabelImg)
- [ ] Dataset organized

### Training
- [ ] YOLOv8 trained (models/yolov8n_fruit.pt)
- [ ] MobileNetV2 trained (models/mobilenet_classifier.tflite)
- [ ] Models validated

### Deployment
- [ ] Hardware đã test OK
- [ ] AI inference chạy OK
- [ ] Web interface hoạt động
- [ ] Hệ thống phân loại chính xác

---

## 🆘 Troubleshooting Trên Pi

### 1. Out of Memory khi training
```bash
# Kiểm tra swap
free -h

# Nếu swap < 4GB
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile  # CONF_SWAPSIZE=4096
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Giảm batch size
# YOLO: --batch 2
# MobileNetV2: --batch 4
```

### 2. Training quá chậm
```bash
# Giảm epochs
# YOLO: 30-40 epochs thay vì 50
# MobileNetV2: 20 epochs thay vì 30

# Giảm image size
# YOLO: --imgsz 320
```

### 3. Pi bị treo khi training
```bash
# Kiểm tra nhiệt độ
vcgencmd measure_temp

# Nếu > 80°C: Thêm quạt tản nhiệt!

# Giảm overclock (nếu có)
sudo nano /boot/config.txt
# Comment out over_voltage & arm_freq
```

### 4. Web interface lag
```bash
# Giảm FPS
nano utils/config.py
# MAX_FPS = 5  # Thay vì 10

# Giảm resolution
# CAMERA_RESOLUTION = (1280, 720)  # Thay vì 1920x1080
```

---

## 📚 Tài Liệu Chi Tiết

1. **[INDEX.md](INDEX.md)** - Lộ trình đầy đủ từ A-Z
2. **[TRAINING_ON_PI.md](TRAINING_ON_PI.md)** - Chi tiết training trên Pi
3. **[POWER_SUPPLY_QUICK_GUIDE.md](POWER_SUPPLY_QUICK_GUIDE.md)** - Kết nối nguồn
4. **[detailed_wiring_diagram.md](detailed_wiring_diagram.md)** - Sơ đồ đấu nối
5. **[web_interface_guide.md](web_interface_guide.md)** - Giao diện web

---

## 🎯 Kết Luận

### ✅ Raspberry Pi 4 (8GB RAM) HOÀN TOÀN ĐỦ cho:

1. ✓ Thu thập & gán nhãn dữ liệu
2. ✓ Training YOLOv8-nano
3. ✓ Training MobileNetV2
4. ✓ Inference real-time
5. ✓ Web interface dashboard
6. ✓ Điều khiển phần cứng (servo, motor)
7. ✓ Phân loại trái cây tự động

### 📊 So Sánh Chi Phí

| Phương Án | Chi Phí | Thời Gian |
|-----------|---------|-----------|
| **Pi 4 8GB** | ~2-3 triệu VNĐ | 10-15 giờ training |
| **PC GPU** | ~15-20 triệu VNĐ | 2-3 giờ training |

**Tiết kiệm: ~15 triệu VNĐ** 💰

### ⏰ Timeline Thực Tế

| Giai Đoạn | Thời Gian |
|-----------|-----------|
| Setup Pi | 1-2 giờ |
| Lắp phần cứng | 2-3 giờ |
| Thu thập dữ liệu | 1-2 ngày |
| Gán nhãn | 2-3 giờ |
| Training (Pi chạy qua đêm) | 10-15 giờ |
| Deploy & Test | 1 giờ |
| **TỔNG** | **~3-4 ngày** |

---

**🍓 RASPBERRY PI 4 (8GB) = HỆ THỐNG HOÀN CHỈNH! 🤖**



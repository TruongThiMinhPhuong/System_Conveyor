# Training AI Models Trực Tiếp Trên Raspberry Pi 4

## ⚠️ LƯU Ý QUAN TRỌNG

**Raspberry Pi 4 (8GB RAM) CÓ THỂ train cả YOLO và MobileNetV2**, nhưng sẽ:
- ⏰ **Chậm hơn nhiều** so với PC GPU (10-50 lần)
- 🔥 **Nóng máy** - CẦN tản nhiệt tốt
- ⚡ **Tốn điện** - Training có thể chạy cả ngày

**Thời gian ước tính:**
- YOLOv8-nano: 6-12 giờ (so với 1-2 giờ trên GPU)
- MobileNetV2: 2-4 giờ (so với 30-60 phút trên GPU)

---

## 🔧 Chuẩn Bị Raspberry Pi 4 Để Training

### 1. Tăng Swap Space (Bắt Buộc!)

```bash
# Tắt swap hiện tại
sudo dphys-swapfile swapoff

# Chỉnh swap size lên 4GB
sudo nano /etc/dphys-swapfile
# Sửa dòng: CONF_SWAPSIZE=4096

# Khởi động lại swap
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Kiểm tra
free -h
# Swap phải hiển thị 4GB
```

### 2. Overclock (Tùy Chọn - Cẩn Thận!)

```bash
sudo nano /boot/config.txt
# Thêm vào cuối file:
over_voltage=6
arm_freq=2000
gpu_freq=600

# Lưu và reboot
sudo reboot
```

⚠️ **Chú ý:** Overclock CẦN tản nhiệt tốt (quạt + heatsink)!

### 3. Cài Dependencies Đầy Đủ

```bash
cd ~/System_Conveyor
source venv/bin/activate

# Cài thêm PyTorch cho Pi (CPU version)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Cài TensorFlow
pip3 install tensorflow

# Các thư viện training
pip3 install matplotlib scikit-learn
pip3 install labelImg  # Annotation tool
```

---

## 📊 Training YOLOv8 Trên Raspberry Pi 4

### Script Training Tối Ưu Cho Pi

Tạo file `training/yolo/train_yolo_pi.py`:

```python
"""
YOLOv8 Training Optimized for Raspberry Pi 4
"""
from ultralytics import YOLO
import torch

def train_yolo_on_pi():
    print("=" * 60)
    print("YOLOv8 Training on Raspberry Pi 4")
    print("=" * 60)
    
    # Kiểm tra RAM
    import psutil
    ram_gb = psutil.virtual_memory().total / (1024**3)
    print(f"Total RAM: {ram_gb:.1f} GB")
    
    if ram_gb < 7:
        print("⚠️ WARNING: RAM < 8GB, training có thể bị lỗi!")
    
    # Load model
    model = YOLO('yolov8n.pt')  # Nano - nhẹ nhất
    
    # Training với config tối ưu cho Pi
    results = model.train(
        data='dataset.yaml',
        
        # Giảm batch size cho Pi
        batch=4,  # Thay vì 16
        
        # Giảm epochs (có thể tăng nếu muốn accuracy cao hơn)
        epochs=50,  # Thay vì 100
        
        # Giảm image size
        imgsz=416,  # Thay vì 640
        
        # Giảm workers
        workers=2,  # Pi 4 có 4 cores
        
        # Bật mixed precision (nếu hỗ trợ)
        amp=False,  # Pi CPU không hỗ trợ AMP
        
        # Cache images in RAM (nếu đủ RAM)
        cache=False,  # Để False nếu RAM < 8GB
        
        # Project name
        project='fruit_detection',
        name='yolov8n_pi',
        
        # Device
        device='cpu',
        
        # Patience
        patience=20,
        
        # Save period
        save_period=10
    )
    
    print("\n✅ Training completed!")
    print(f"Best model: fruit_detection/yolov8n_pi/weights/best.pt")
    
    return results

if __name__ == '__main__':
    train_yolo_on_pi()
```

### Chạy Training:

```bash
cd ~/System_Conveyor/training/yolo

# Chạy trong screen để không bị ngắt khi SSH disconnect
screen -S yolo_training

# Start training
python train_yolo_pi.py

# Detach: Ctrl+A, D
# Reattach: screen -r yolo_training
```

### Theo Dõi Tiến Độ:

```bash
# Monitor temperature
watch -n 2 vcgencmd measure_temp

# Monitor memory
watch -n 5 free -h

# Monitor CPU
htop
```

---

## 🧠 Training MobileNetV2 Trên Raspberry Pi 4

### Script Tối Ưu Cho Pi

Sửa file `training/mobilenet/train_mobilenet.py`:

```python
# Thay đổi các tham số sau:

# Giảm batch size
BATCH_SIZE = 8  # Thay vì 32

# Giảm epochs
EPOCHS = 30  # Thay vì 50

# Image size nhỏ hơn
IMG_SIZE = 160  # Thay vì 224

# Giảm learning rate
LEARNING_RATE = 0.0005  # Thay vì 0.001
```

### Chạy Training:

```bash
cd ~/System_Conveyor/training/mobilenet

# Trong screen
screen -S mobilenet_training

python train_mobilenet.py \
    --dataset ./datasets/fruit_classification \
    --epochs 30 \
    --batch 8 \
    --image-size 160

# Detach: Ctrl+A, D
```

---

## 💾 Giảm Yêu Cầu Dataset

Do Pi train chậm, bạn có thể bắt đầu với dataset nhỏ hơn:

### Dataset Tối Thiểu:

- **YOLO Detection:** 100-150 ảnh (thay vì 300+)
- **MobileNetV2:** 50-70 ảnh mỗi class (thay vì 100+)

### Thu Thập Nhanh:

```bash
# Thu thập ít ảnh hơn
python training/data_collection/collect_images.py \
    --mode classification \
    --count 100 \
    --interval 1.5
```

---

## ⚡ Tối Ưu Hiệu Suất

### 1. Đóng Các Tiến Trình Không Cần

```bash
# Tắt GUI (nếu đang dùng Desktop)
sudo systemctl stop lightdm

# Tắt Bluetooth
sudo systemctl stop bluetooth

# Tắt WiFi (nếu dùng Ethernet)
sudo rfkill block wifi
```

### 2. Giới Hạn RAM Cho Training

```bash
# Giới hạn TensorFlow chỉ dùng 6GB RAM
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

### 3. Chạy Qua Đêm

```bash
# Bật No sleep
sudo systemctl mask sleep.target suspend.target

# Hoặc dùng caffeinate nếu có
```

---

## 📈 Kỳ Vọng Về Accuracy

Training trên Pi với dataset nhỏ hơn:

| Model | Accuracy Kỳ Vọng | Có Thể Cải Thiện |
|-------|-------------------|------------------|
| YOLOv8 | 60-75% mAP | Train lâu hơn, thêm data |
| MobileNetV2 | 80-90% | Tăng epochs lên 50 |

**Đây vẫn đủ tốt cho prototype/testing!**

---

## 🔥 Giám Sát Nhiệt Độ

```bash
# Script check nhiệt độ liên tục
cat > check_temp.sh << 'EOF'
#!/bin/bash
while true; do
    temp=$(vcgencmd measure_temp | cut -d= -f2 | cut -d\' -f1)
    echo "$(date): Temperature: $temp°C"
    
    # Cảnh báo nếu quá 80°C
    if (( $(echo "$temp > 80" | bc -l) )); then
        echo "⚠️ WARNING: High temperature!"
    fi
    
    sleep 60
done
EOF

chmod +x check_temp.sh
./check_temp.sh &
```

**Nhiệt độ an toàn:** < 80°C
**Cần dừng nếu:** > 85°C

---

## 🛡️ Backup & Recovery

### Backup Model Định Kỳ:

```bash
# Tạo cron job backup
crontab -e

# Thêm dòng (backup mỗi 2 giờ):
0 */2 * * * cp -r ~/System_Conveyor/training/*/weights ~/backup_models/
```

### Nếu Training Bị Gián Đoạn:

```bash
# YOLOv8 tự động resume từ last checkpoint
python train_yolo_pi.py --resume

# MobileNetV2: Load best checkpoint và tiếp tục
# (Đã tích hợp ModelCheckpoint trong code)
```

---

## 📊 So Sánh: Pi vs PC với GPU

| Tiêu Chí | Raspberry Pi 4 | PC với RTX 3060 |
|----------|----------------|-----------------|
| **YOLOv8 (50 epochs)** | 8-10 giờ | 45-60 phút |
| **MobileNetV2 (30 epochs)** | 2-3 giờ | 20-30 phút |
| **Chi phí** | $0 (đã có Pi) | $300+ (GPU) |
| **Điện năng** | 15W | 200W+ |
| **Nhiệt độ** | 70-80°C | 60-70°C |
| **Độ linh hoạt** | Train + Deploy = 1 thiết bị | 2 thiết bị |

**Ưu điểm train trên Pi:**
- ✅ Không cần PC riêng
- ✅ Tiết kiệm điện
- ✅ Tất cả trong 1 thiết bị

**Nhược điểm:**
- ⏰ Chậm hơn nhiều
- 🔥 Dễ nóng
- 📉 Có thể accuracy thấp hơn (do batch size nhỏ)

---

## 🚀 Quy Trình Training Trên Pi (Tóm Tắt)

```bash
# 1. Chuẩn bị Pi
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile  # CONF_SWAPSIZE=4096
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# 2. Thu thập data
cd ~/System_Conveyor
python training/data_collection/collect_images.py --count 100

# 3. Train YOLO (trong screen)
screen -S yolo
cd training/yolo
python train_yolo_pi.py
# Ctrl+A, D để detach

# 4. Train MobileNetV2 (sau khi YOLO xong)
screen -S mobilenet
cd ../mobilenet
python train_mobilenet.py --batch 8 --epochs 30
# Ctrl+A, D để detach

# 5. Monitor
watch -n 5 vcgencmd measure_temp
screen -r yolo  # Kiểm tra tiến độ

# 6. Export models
python export_tflite.py
```

---

## ✅ Checklist Training Trên Pi

- [ ] Đã tăng swap lên 4GB
- [ ] Có quạt tản nhiệt hoặc heatsink
- [ ] Đã cài đầy đủ dependencies (torch, tensorflow)
- [ ] Dataset đã chuẩn bị (tối thiểu 100 ảnh)
- [ ] Chạy trong `screen` để tránh mất session
- [ ] Monitor nhiệt độ (< 80°C)
- [ ] Đã backup models định kỳ

---

## 🆘 Troubleshooting

### Lỗi: Out of Memory
```bash
# Giảm batch size xuống 2
# Tắt cache=True
# Kiểm tra swap: free -h
```

### Lỗi: Training quá chậm
```bash
# Giảm epochs
# Giảm image size (320 cho YOLO)
# Đóng các app khác
```

### Lỗi: Pi bị treo
```bash
# Kiểm tra nhiệt độ
# Thêm quạt tản nhiệt
# Giảm overclock
```

---

**Kết luận:** Training trên Pi 4 hoàn toàn khả thi, chỉ cần kiên nhẫn và theo dõi nhiệt độ! 🍓🤖

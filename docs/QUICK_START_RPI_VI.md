# Hướng Dẫn Chạy Nhanh Trên Raspberry Pi

## 🚀 Chạy Ngay Trên Raspberry Pi

### Bước 1: Copy File Lên Raspberry Pi

**Trên Windows PC**, mở PowerShell:

```powershell
# Copy quick fix script
scp d:\System_Conveyor\quick_fix_rpi.sh pi@192.168.137.177:~/
scp d:\System_Conveyor\fruit_sorter.py pi@192.168.137.177:~/System_Conveyor/
```

### Bước 2: Chạy Script Trên Raspberry Pi

**SSH vào Raspberry Pi** (nếu chưa SSH):

```bash
ssh pi@192.168.137.177
```

**Chạy quick fix script**:

```bash
cd ~
chmod +x quick_fix_rpi.sh
./quick_fix_rpi.sh
```

Script sẽ:
- ✅ Tải YOLO model (phát hiện trái cây)
- ✅ Cấu hình hệ thống
- ✅ Sẵn sàng chạy

### Bước 3: Chạy Hệ Thống

```bash
cd ~/System_Conveyor
python3 fruit_sorter.py
```

---

## ⚠️ Lưu Ý Quan Trọng

### Chức Năng Hiện Tại

✅ **Có thể làm**:
- Phát hiện trái cây (YOLO)
- Di chuyển băng tải
- Điều khiển servo

❌ **Chưa thể làm**:
- Phân loại Fresh/Spoiled (cần train MobileNet trên PC)

### Để Có Đầy Đủ Chức Năng

Bạn cần **train model MobileNet trên Windows PC**:

1. **Trên Windows PC**:
   ```powershell
   cd d:\System_Conveyor
   .\setup_pc.ps1
   python quick_train.py
   ```

2. **Copy model sang Raspberry Pi**:
   ```powershell
   scp models\mobilenet_classifier.tflite pi@192.168.137.177:~/System_Conveyor/models/
   ```

3. **Khởi động lại hệ thống** trên Raspberry Pi

---

## 🔧 Các Lệnh Hữu Ích

### Chạy hệ thống
```bash
cd ~/System_Conveyor
python3 fruit_sorter.py
```

### Chạy web interface
```bash
cd ~/System_Conveyor  
python3 run_web.py
# Truy cập: http://192.168.137.177:5000
```

### Kiểm tra models
```bash
ls -lh ~/System_Conveyor/models/
```

### Xem logs
```bash
tail -f ~/System_Conveyor/logs/system.log
```

---

## 🐛 Xử Lý Lỗi

### Lỗi: "YOLO model not found"

```bash
cd ~/System_Conveyor
python3 << EOF
from ultralytics import YOLO
YOLO('yolov8n.pt').save('models/yolov8n_fruit.pt')  
EOF
```

### Lỗi: "Camera not detected"

```bash
# Bật camera
sudo raspi-config
# Interface Options > Camera > Enable
sudo reboot
```

### Lỗi: "GPIO permission denied"

```bash
sudo usermod -a -G gpio pi
sudo reboot
```

### Hệ thống chậm (FPS thấp)

Chỉnh trong `utils/config.py`:
```python
CAMERA_RESOLUTION = (320, 320)  # Giảm resolution
YOLO_INPUT_SIZE = 320
FAST_PREPROCESSING = True
```

---

## 📊 Kết Quả Mong Đợi

Sau khi chạy thành công:

```
============================================================
🍎 Conveyor System for Fruit Classification
============================================================
✅ YOLO model loaded
⚠️ MobileNet: Demo mode (train on PC for full features)
✅ System initialized successfully!
🚀 Starting main system loop...

⚡ FPS: 12.3
⏱️ YOLO: 45ms | Total: 82ms
```

---

## ✅ Checklist

- [ ] Quick fix script chạy thành công
- [ ] YOLO model đã tải
- [ ] Hệ thống khởi động không lỗi
- [ ] Camera hoạt động
- [ ] Servo di chuyển
- [ ] Băng tải chạy được

### Để Có Đầy Đủ Tính Năng
- [ ] Train MobileNet trên Windows PC
- [ ] Copy model lên Raspberry Pi
- [ ] Accuracy > 90%

---

**🎉 Xong! Hệ thống đã sẵn sàng chạy trên Raspberry Pi!**

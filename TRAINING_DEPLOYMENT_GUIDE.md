# 🚀 Hướng Dẫn Train Model Trên Laptop và Triển Khai Lên Raspberry Pi 4

**Training on Laptop (GPU) → Optimize → Deploy to Raspberry Pi 4 (8GB RAM)**

## 📋 Tổng Quan Quy Trình / Workflow Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         LAPTOP (GPU)                                 │
├──────────────────────────────────────────────────────────────────────┤
│  1. Thu thập data (Data Collection)                                 │
│     └─→ dataset/raw_images/                                         │
│                                                                      │
│  2. Gán nhãn (Labeling)                                             │
│     └─→ dataset/train/, dataset/valid/, dataset/test/              │
│                                                                      │
│  3. Train model (Training)                                          │
│     ├─→ YOLOv8: ai_models/yolo_best.pt (object detection)          │
│     └─→ MobileNetV2: ai_models/mobilenet_model.h5 (classification) │
│                                                                      │
│  4. Tối ưu model (Optimization)                                     │
│     ├─→ Resize ảnh (224x224 → 96x96 hoặc 128x128)                 │
│     ├─→ TFLite INT8 Quantization                                   │
│     └─→ ai_models/mobilenet_model_int8.tflite                      │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
                        COPY MODEL
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│                    RASPBERRY PI 4 (8GB RAM)                          │
├──────────────────────────────────────────────────────────────────────┤
│  5. Chạy suy luận real-time (Inference on CPU)                      │
│     ├─→ YOLOv8 (ncnn format hoặc .pt nhẹ)                          │
│     └─→ MobileNetV2 INT8 TFLite                                     │
│                                                                      │
│  6. Kết quả (Results)                                               │
│     ├─→ Độ chính xác: >85%                                          │
│     ├─→ FPS: 15-25 (tùy độ phức tạp)                               │
│     └─→ Hệ thống phân loại trái cây real-time                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 📦 BƯỚC 1: Chuẩn Bị Môi Trường (Setup Environment)

### Trên Laptop (Windows/Linux với GPU)

```powershell
# 1. Clone repository
git clone https://github.com/TruongThiMinhPhuong/System_Conveyor.git
cd System_Conveyor

# 2. Tạo môi trường ảo (Virtual Environment)
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Cài đặt dependencies cho training
pip install -r requirements-pc.txt

# requirements-pc.txt bao gồm:
# - tensorflow>=2.10.0 (hỗ trợ GPU)
# - opencv-python>=4.8.0
# - ultralytics>=8.0.0 (YOLOv8)
# - numpy, matplotlib, scikit-learn
```

### Kiểm Tra GPU

```python
# test_gpu.py
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))
print("CUDA:", tf.test.is_built_with_cuda())
```

---

## 🗂️ BƯỚC 2: Thu Thập và Gán Nhãn Dữ Liệu (Data Collection & Labeling)

### 2.1 Thu Thập Ảnh

**Sử dụng script có sẵn (với webcam/camera):**

```bash
python data_collection_script.py
```

Script này sẽ:
- Mở camera và hiển thị preview
- Nhấn `SPACE` để chụp ảnh
- Ảnh được lưu vào `raw_images/`
- Nhấn `q` để thoát

**Hoặc thu thập ảnh thủ công:**
- Chụp ít nhất **500-1000 ảnh** cho mỗi loại trái cây
- Đa dạng góc độ, ánh sáng, nền
- Đặt trong thư mục `raw_images/apple/`, `raw_images/orange/`, v.v.

### 2.2 Gán Nhãn (Labeling)

#### Cho YOLOv8 (Object Detection):

1. **Sử dụng LabelImg hoặc Roboflow:**
   ```bash
   pip install labelImg
   labelImg
   ```

2. **Format YOLO:**
   - Mỗi ảnh có 1 file `.txt` tương ứng
   - Format: `class_id center_x center_y width height` (normalized 0-1)
   - Ví dụ: `0 0.5 0.5 0.8 0.8`

3. **Cấu trúc thư mục:**
   ```
   dataset/
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── valid/
   │   ├── images/
   │   └── labels/
   └── data.yaml
   ```

4. **File data.yaml:**
   ```yaml
   train: dataset/train/images
   val: dataset/valid/images
   nc: 2
   names: ['apple', 'orange']
   ```

#### Cho MobileNetV2 (Classification):

1. **Cấu trúc thư mục đơn giản hơn:**
   ```
   dataset/
   ├── train/
   │   ├── fresh/
   │   │   ├── img1.jpg
   │   │   └── img2.jpg
   │   └── spoiled/
   │       ├── img1.jpg
   │       └── img2.jpg
   ├── valid/
   │   ├── fresh/
   │   └── spoiled/
   └── test/
       ├── fresh/
       └── spoiled/
   ```

2. **Kiểm tra chất lượng dataset:**
   ```bash
   python dataset_quality_checker.py
   ```

---

## 🏋️ BƯỚC 3: Train Model Trên Laptop (Training)

### 3.1 Train YOLOv8 (Object Detection)

```bash
# Quick training script
python quick_train.py
```

**Hoặc train thủ công:**

```python
# train_yolo.py
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')  # nano version (nhẹ nhất)

# Train
results = model.train(
    data='dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,  # GPU
    project='training',
    name='yolo_fruit_detection',
    patience=20,
    save=True
)

# Model saved to: training/yolo_fruit_detection/weights/best.pt
```

**Đánh giá model:**
```python
# Validate
metrics = model.val()
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
```

### 3.2 Train MobileNetV2 (Classification)

**Script có sẵn:**
```bash
python retrain_model.py
```

**Hoặc sử dụng Google Colab (khuyến nghị):**

1. Upload `Train_MobileNet_Colab.ipynb` lên Google Colab
2. Nén folder dataset: `dataset.zip`
3. Upload lên Colab hoặc Google Drive
4. Chạy notebook (sử dụng GPU T4 miễn phí)
5. Download model đã train: `mobilenet_model.h5`

**Hoặc train thủ công:**

```python
# train_mobilenet.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Hyperparameters
IMG_SIZE = 224  # MobileNetV2 standard
BATCH_SIZE = 32
EPOCHS = 50

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1./255)

# Load data
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    'dataset/valid',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Build model
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base layers
base_model.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(2, activation='softmax')(x)  # 2 classes: fresh, spoiled

model = Model(inputs=base_model.input, outputs=predictions)

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=valid_generator,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
    ]
)

# Save model
model.save('ai_models/mobilenet_model.h5')
print("✅ Model saved to ai_models/mobilenet_model.h5")
```

**Đánh giá:**
```python
# Evaluate on test set
test_generator = valid_datagen.flow_from_directory(
    'dataset/test',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

loss, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy*100:.2f}%")
```

---

## ⚡ BƯỚC 4: Tối Ưu Model (Optimization)

### 4.1 Convert sang TensorFlow Lite INT8

**Sử dụng script có sẵn:**
```bash
python convert_to_tflite.py
```

**Script chi tiết:**

```python
# convert_to_tflite.py
import tensorflow as tf
import numpy as np
from pathlib import Path

# Load trained model
model = tf.keras.models.load_model('ai_models/mobilenet_model.h5')

# Representative dataset for quantization
def representative_dataset():
    """Generate sample data for INT8 calibration"""
    import cv2
    
    image_dir = Path('dataset/valid/fresh')
    images = list(image_dir.glob('*.jpg'))[:100]  # Use 100 samples
    
    for img_path in images:
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        yield [img]

# Convert to TFLite with INT8 quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# Convert
tflite_model = converter.convert()

# Save
output_path = 'ai_models/mobilenet_model_int8.tflite'
with open(output_path, 'wb') as f:
    f.write(tflite_model)

# Check file size
import os
h5_size = os.path.getsize('ai_models/mobilenet_model.h5') / (1024*1024)
tflite_size = os.path.getsize(output_path) / (1024*1024)

print(f"✅ Conversion complete!")
print(f"📦 Original model (H5): {h5_size:.2f} MB")
print(f"📦 Optimized model (TFLite INT8): {tflite_size:.2f} MB")
print(f"🎯 Size reduction: {(1 - tflite_size/h5_size)*100:.1f}%")
```

**Kết quả mong đợi:**
- Model gốc (H5): ~10-15 MB
- Model tối ưu (TFLite INT8): ~2-4 MB
- Giảm kích thước: ~70-80%
- Độ chính xác: giảm <2% (vẫn >85%)

### 4.2 Tối Ưu YOLOv8

YOLOv8 đã nhẹ rồi, nhưng có thể export sang ONNX hoặc NCNN:

```python
from ultralytics import YOLO

model = YOLO('training/yolo_fruit_detection/weights/best.pt')

# Export to ONNX (faster inference)
model.export(format='onnx')

# Hoặc export to NCNN (tốt hơn cho Pi)
model.export(format='ncnn')
```

---

## 📤 BƯỚC 5: Copy Model Sang Raspberry Pi

### 5.1 Chuẩn Bị Files

**Trên Laptop, tạo folder để copy:**
```powershell
# Tạo folder models_to_deploy
mkdir models_to_deploy
cd models_to_deploy

# Copy models
copy ..\ai_models\mobilenet_model_int8.tflite .
copy ..\ai_models\yolo_best.pt .

# Copy code
xcopy ..\*.py . /s /e
```

### 5.2 Transfer Files

**Phương án 1: USB Drive**
```powershell
# Copy toàn bộ folder vào USB
# Trên Pi, mount USB và copy vào home directory
```

**Phương án 2: SCP (qua mạng)**
```powershell
# Trên laptop (nếu có SSH)
scp -r models_to_deploy pi@raspberrypi.local:~/System_Conveyor
```

**Phương án 3: Git (khuyến nghị)**
```bash
# Commit và push lên GitHub
git add .
git commit -m "Add trained models"
git push origin main

# Trên Pi, pull code
cd ~/System_Conveyor
git pull origin main
```

---

## 🍓 BƯỚC 6: Setup Raspberry Pi 4

### 6.1 Cài Đặt Hệ Điều Hành

1. Download **Raspberry Pi OS (64-bit)** - Bullseye hoặc Bookworm
2. Flash vào SD card bằng Raspberry Pi Imager
3. Enable SSH và WiFi trong imager settings
4. Boot Pi và SSH vào: `ssh pi@raspberrypi.local`

### 6.2 Cài Đặt Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python packages
sudo apt install -y python3-pip python3-venv
sudo apt install -y python3-opencv python3-numpy
sudo apt install -y python3-picamera2 python3-libcamera

# Install hardware control
sudo apt install -y python3-rpi.gpio python3-gpiozero

# Create virtual environment
cd ~/System_Conveyor
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements-rpi.txt

# Install TFLite Runtime (lightweight)
pip install tflite-runtime
```

### 6.3 Test Models

```bash
# Test MobileNet TFLite
python -c "
import tflite_runtime.interpreter as tflite
interpreter = tflite.Interpreter(model_path='ai_models/mobilenet_model_int8.tflite')
print('✅ TFLite model loaded successfully!')
"

# Test YOLO
python -c "
from ultralytics import YOLO
model = YOLO('ai_models/yolo_best.pt')
print('✅ YOLO model loaded successfully!')
"
```

---

## 🚀 BƯỚC 7: Chạy Hệ Thống Trên Raspberry Pi

### 7.1 Test Hardware

```bash
# Test camera
python hardware/camera.py

# Test servo
python hardware/servo_control.py

# Test motor
python hardware/motor_control.py
```

### 7.2 Chạy Web Interface

```bash
# Start web server
python run_web.py
```

Console sẽ hiển thị:
```
🌐 AI Fruit Sorting System - Web Interface
🔗 Access at: http://raspberrypi.local:5001
```

### 7.3 Truy Cập từ Laptop

1. Mở browser trên laptop
2. Vào `http://raspberrypi.local:5001`
3. Nhấn "▶️ Start System"
4. Xem camera feed và phân loại real-time!

---

## 📊 BƯỚC 8: Đánh Giá Hiệu Suất (Performance Evaluation)

### 8.1 Chạy Script Đánh Giá

```bash
python evaluate_system.py
```

Script sẽ đo:
- **FPS** (frames per second)
- **Độ chính xác** (accuracy)
- **Inference time** cho mỗi model
- **CPU/RAM usage**

### 8.2 Kết Quả Mong Đợi

| Metric | YOLOv8 | MobileNet INT8 | Combined |
|--------|--------|----------------|----------|
| FPS | 15-20 | 25-30 | 15-25 |
| Accuracy | 90%+ | 85%+ | 85%+ |
| Inference Time | 50-70ms | 30-40ms | 80-110ms |
| RAM Usage | ~800MB | ~200MB | ~1GB |

---

## 🔧 Troubleshooting

### Lỗi Thường Gặp

**1. TensorFlow GPU không hoạt động trên laptop:**
```bash
# Reinstall TensorFlow with GPU support
pip uninstall tensorflow
pip install tensorflow[and-cuda]
# Hoặc install CUDA Toolkit + cuDNN manually
```

**2. Camera không hoạt động trên Pi:**
```bash
# Enable camera
sudo raspi-config
# Interface Options → Camera → Enable

# Test camera
libcamera-hello
```

**3. Model không load được:**
```python
# Check file exists
import os
print(os.path.exists('ai_models/mobilenet_model_int8.tflite'))

# Check file permissions
ls -l ai_models/
```

**4. FPS quá thấp trên Pi:**
- Giảm resolution ảnh (640x480 → 320x240)
- Tăng `DETECTION_INTERVAL` trong config
- Sử dụng model nhẹ hơn (YOLOv8n thay vì YOLOv8s)

---

## 📚 Tài Liệu Tham Khảo

- **YOLOv8 Documentation**: https://docs.ultralytics.com
- **TensorFlow Lite**: https://www.tensorflow.org/lite
- **Raspberry Pi Camera**: https://www.raspberrypi.com/documentation/computers/camera_software.html
- **Project GitHub**: https://github.com/TruongThiMinhPhuong/System_Conveyor

---

## ✅ Checklist Hoàn Thành

- [ ] Chuẩn bị môi trường laptop (GPU)
- [ ] Thu thập dataset (>500 ảnh/class)
- [ ] Gán nhãn dữ liệu
- [ ] Train YOLOv8 (mAP50 >0.8)
- [ ] Train MobileNetV2 (accuracy >85%)
- [ ] Convert sang TFLite INT8
- [ ] Test models trên laptop
- [ ] Setup Raspberry Pi 4
- [ ] Copy models sang Pi
- [ ] Test hardware (camera, servo, motor)
- [ ] Chạy web interface
- [ ] Test hệ thống hoàn chỉnh
- [ ] Đánh giá FPS và accuracy

---

**🎉 Chúc bạn thành công với dự án phân loại trái cây tự động!**

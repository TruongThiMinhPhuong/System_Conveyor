# Giải Pháp Tối Ưu FPS và Accuracy - YOLOv8-nano + MobileNetV2

## 📊 Phân Tích Vấn Đề Hiện Tại

### Kết Quả Đánh Giá (evaluation_20251219_113305.json):
- **FPS hiện tại: 0.8 FPS** ❌ (Mục tiêu: >8-10 FPS)
- **Accuracy: 87.5%** ⚠️ (Vượt 85% nhưng có vấn đề)
- **YOLO Detection: 1061ms** ❌ (bottleneck chính!)
- **Preprocessing: 95ms** ⚠️
- **Classification: 86ms** ✅

### Vấn Đề Nghiêm Trọng:
1. **Spoiled Class: 0% precision/recall** - Model không phát hiện được trái hỏng!
2. **4/20 ảnh (20%) failed detection** - YOLO không detect được
3. **Classification confidence thấp: 67.4%** - Model không tự tin
4. **YOLO quá chậm** - Chiếm 85% thời gian xử lý
5. **Dataset test chỉ có fresh** - Không có spoiled để test đầy đủ

---

## 🎯 Giải Pháp Toàn Diện

### 1. TỐI ƯU YOLO DETECTION (Giảm từ 1061ms → <200ms)

#### A. Giảm Input Resolution
```python
# Config hiện tại: (640, 480) → Đổi sang (320, 320)
CAMERA_RESOLUTION = (320, 320)  # Giảm 4x pixels
```

#### B. Enable YOLO Optimization Flags
```python
# Trong yolo_detector.py - thêm optimization
def detect(self, image, verbose=False):
    results = self.model(
        image,
        conf=self.confidence_threshold,
        iou=self.iou_threshold,
        verbose=verbose,
        half=True,  # FP16 inference (nhanh gấp 2x)
        device='cpu',  # Explicit CPU
        imgsz=320  # Resize input to 320x320
    )
```

#### C. Batch Processing & Async
```python
# Xử lý nhiều frames cùng lúc (nếu có queue)
results = self.model(
    [image1, image2, image3],  # Batch inference
    stream=True  # Streaming mode
)
```

### 2. TỐI ƯU PREPROCESSING (Giảm từ 95ms → <30ms)

#### A. FAST_MODE Configuration
```python
# Trong preprocessing.py
def __init__(self, fast_mode=True):
    if fast_mode:
        self.clahe_tile_size = (2, 2)  # Giảm từ (4,4) → (2,2)
        self.clahe_clip_limit = 1.5     # Giảm từ 2.0 → 1.5
        self.apply_blur = False         # Tắt blur
        self.enhance_contrast = False    # Tắt CLAHE nếu không cần
```

#### B. Sử dụng cv2.resize tối ưu
```python
def resize_image(self, image, size=None):
    # Dùng INTER_NEAREST cho resize nhanh nhất
    return cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
```

### 3. TỐI ƯU MOBILENET CLASSIFICATION

#### A. Đảm bảo TFLite Delegate
```python
# Cài đặt XNNPACK cho ARM optimization
sudo apt-get install -y libxnnpack-dev

# Trong mobilenet_classifier.py - force XNNPACK
try:
    self.interpreter = tflite.Interpreter(
        model_path=self.model_path,
        num_threads=4  # Dùng 4 cores
    )
except:
    # Fallback to single thread
    self.interpreter = tflite.Interpreter(model_path=self.model_path)
```

#### B. Quantize Model (INT8)
```python
# Chuyển model từ FP32 → INT8 (nhanh 4x)
# File: convert_to_int8.py
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]

# Representative dataset
def representative_dataset():
    for i in range(100):
        yield [np.random.rand(1, 224, 224, 3).astype(np.float32)]

converter.representative_dataset = representative_dataset
tflite_model = converter.convert()
```

### 4. CẢI THIỆN DATA QUALITY (Tăng Accuracy lên >95%)

#### A. Thu Thập Dữ Liệu Đúng Cách
```bash
# CẦN:
# - 300+ ảnh fresh (đa dạng loại trái, góc độ, ánh sáng)
# - 300+ ảnh spoiled (thực sự hỏng, thối, dập nát)
# - 100+ ảnh test fresh
# - 100+ ảnh test spoiled
```

#### B. Data Augmentation Mạnh Hơn
```python
# Trong training script
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)
```

#### C. Class Balancing
```python
# Đảm bảo số lượng ảnh fresh = spoiled
# Nếu thiếu, dùng augmentation để tăng
from imblearn.over_sampling import RandomOverSampler
```

### 5. TRAINING PARAMETERS TỐI ƯU

```python
# Training MobileNetV2
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',  # hoặc 'categorical_crossentropy' nếu >2 class
    metrics=['accuracy', 'AUC', 'Precision', 'Recall']
)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
    ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy')
]

history = model.fit(
    train_generator,
    epochs=50,  # Tăng từ 20 → 50
    validation_data=val_generator,
    callbacks=callbacks,
    class_weight={0: 1.0, 1: 1.5}  # Weight cao hơn cho spoiled class
)
```

### 6. YOLO TRAINING (Nếu cần train lại)

```bash
# Train YOLOv8n với dataset riêng
yolo train model=yolov8n.pt data=fruit_data.yaml epochs=100 imgsz=320 \
    batch=32 device=0 patience=20 optimizer=Adam lr0=0.001

# fruit_data.yaml
path: /path/to/dataset
train: images/train
val: images/val
nc: 3  # number of classes
names: ['apple', 'orange', 'guava']
```

---

## 📋 Implementation Plan

### Bước 1: Tối Ưu Ngay (Quick Wins)
```python
# utils/config.py - ĐỔI NGAY
CAMERA_RESOLUTION = (320, 320)  # Từ (640, 480)
FAST_PREPROCESSING = True
YOLO_CONFIDENCE_THRESHOLD = 0.35  # Giảm từ 0.5 để detect nhiều hơn
BLUR_KERNEL_SIZE = 3  # Giảm từ 5
```

### Bước 2: Tối Ưu Code
- Implement YOLO half precision (FP16)
- Tắt CLAHE trong fast mode
- Sử dụng INTER_NEAREST cho resize
- Enable multi-threading cho TFLite

### Bước 3: Cải Thiện Data
- Thu thập 200+ ảnh spoiled thực tế
- Augmentation mạnh
- Balance dataset
- Tạo test set đầy đủ

### Bước 4: Retrain Models
- Train MobileNetV2 với data mới (50 epochs)
- Quantize sang INT8
- Validate accuracy >95%

### Bước 5: Final Testing
- Test với full dataset
- Đảm bảo FPS >8
- Đảm bảo Accuracy >90%
- Đảm bảo Spoiled class có F1 >85%

---

## 🎯 KẾT QUẢ KỲ VỌNG

### Trước Tối Ưu:
- FPS: 0.8 ❌
- YOLO: 1061ms ❌
- Preprocessing: 95ms ⚠️
- Classification: 86ms ✅
- Accuracy: 87.5% (spoiled = 0%) ❌

### Sau Tối Ưu:
- FPS: **>8-10** ✅
- YOLO: **<200ms** ✅ (giảm 5x)
- Preprocessing: **<30ms** ✅ (giảm 3x)
- Classification: **<50ms** ✅
- Accuracy: **>95%** ✅
- Spoiled F1: **>85%** ✅

---

## 🚀 Triển Khai Ngay

Tôi sẽ tạo các file tối ưu sau:
1. `utils/config_optimized.py` - Configuration tối ưu
2. `ai_models/yolo_detector_optimized.py` - YOLO tối ưu
3. `ai_models/preprocessing_fast.py` - Preprocessing nhanh
4. `training/mobilenet/train_optimized.py` - Training script tối ưu
5. `data_augmentation_pipeline.py` - Pipeline augment data
6. `evaluate_system_fast.py` - Evaluation nhanh

Bạn muốn tôi implement ngay không?

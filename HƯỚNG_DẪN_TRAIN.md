# 🚀 HƯỚNG DẪN TRAIN MODEL - ĐẦY ĐỦ VÀ DỄ HIỂU

> **Mục đích**: Train model MobileNet để phân loại trái cây Fresh/Spoiled  
> **Thời gian**: 30-40 phút  
> **Chi phí**: MIỄN PHÍ 100%

---

## 📋 MỤC LỤC

1. [Tổng Quan](#tổng-quan)
2. [Phương Án 1: Google Colab (Khuyên Dùng)](#phương-án-1-google-colab)
3. [Phương Án 2: Train Trên PC](#phương-án-2-train-trên-pc)
4. [Chuẩn Bị Dataset](#chuẩn-bị-dataset)
5. [Deploy Model](#deploy-model)
6. [Xử Lý Lỗi](#xử-lý-lỗi)
7. [Tips & Tricks](#tips--tricks)

---

## 🎯 TỔNG QUAN

### Tại Sao Cần Train?

Hệ thống cần 2 models:
- ✅ **YOLO** - Phát hiện trái cây (có thể dùng pretrained)
- ✅ **MobileNet** - Phân loại Fresh/Spoiled (CẦN TRAIN với ảnh của bạn)

### So Sánh Các Phương Án

| Phương Án | Thời Gian | GPU | Chi Phí | Khuyên Dùng |
|-----------|-----------|-----|---------|-------------|
| **Google Colab** | 15-20 phút | ✅ Free GPU | Miễn Phí | ⭐⭐⭐⭐⭐ |
| **PC Windows** | 30-60 phút | Tùy máy | Miễn Phí | ⭐⭐⭐⭐ |
| **Raspberry Pi** | 10-20 giờ | ❌ | Miễn Phí | ⭐ (Không khuyên) |

### Yêu Cầu Dataset

- **Tối thiểu**: 50 ảnh fresh + 50 ảnh spoiled
- **Khuyên dùng**: 200+ ảnh mỗi loại
- **Format**: JPG, PNG
- **Chất lượng**: Rõ nét, đa dạng góc độ

---

# PHƯƠNG ÁN 1: GOOGLE COLAB

## ⭐ Tại Sao Chọn Colab?

- ✅ **100% Miễn Phí**
- ✅ **GPU Mạnh** (nhanh hơn PC 10-20 lần)
- ✅ **Không Cần Cài Đặt**
- ✅ **Chạy Được Trên Raspberry Pi Browser**
- ✅ **Đơn Giản Nhất**

---

## 📦 BƯỚC 1: Chuẩn Bị Dataset

### 1.1. Tạo Folder Ảnh

Trên Raspberry Pi hoặc PC:

```bash
# Raspberry Pi
cd ~
mkdir -p my_fruits/train/fresh
mkdir -p my_fruits/train/spoiled
mkdir -p my_fruits/val/fresh
mkdir -p my_fruits/val/spoiled
mkdir -p my_fruits/test/fresh
mkdir -p my_fruits/test/spoiled

# Windows PC
cd d:\
mkdir my_fruits\train\fresh
mkdir my_fruits\train\spoiled
mkdir my_fruits\val\fresh
mkdir my_fruits\val\spoiled
```

### 1.2. Sắp Xếp Ảnh

**Cấu trúc mong đợi**:
```
my_fruits/
├── train/              (70% ảnh)
│   ├── fresh/          (50+ ảnh)
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── spoiled/        (50+ ảnh)
│       ├── img001.jpg
│       └── ...
├── val/                (15% ảnh)
│   ├── fresh/          (10-20 ảnh)
│   └── spoiled/        (10-20 ảnh)
└── test/               (15% ảnh)
    ├── fresh/          (10-20 ảnh)
    └── spoiled/        (10-20 ảnh)
```

### 1.3. Nén Thành ZIP

**Raspberry Pi**:
```bash
cd ~
zip -r dataset.zip my_fruits/
```

**Windows**:
```powershell
# Chuột phải folder my_fruits → Send to → Compressed (zipped) folder
# Đổi tên thành dataset.zip
```

✅ **Xong! File dataset.zip đã sẵn sàng**

---

## 🌐 BƯỚC 2: Mở Google Colab

### 2.1. Trên Raspberry Pi

```bash
cd ~/System_Conveyor
chmod +x start_colab_training.sh
./start_colab_training.sh
```

Script sẽ tự động mở browser!

### 2.2. Hoặc Mở Thủ Công

1. Mở browser (Chromium/Chrome/Firefox)
2. Truy cập: https://colab.research.google.com
3. Đăng nhập Gmail

---

## 📤 BƯỚC 3: Upload Notebook

### Trong Google Colab:

1. Click: **File** → **Upload notebook**
2. Chọn tab **Upload**
3. Click **Browse**
4. Chọn file: `Train_MobileNet_Colab.ipynb`
   - **Raspberry Pi**: `/home/pi/System_Conveyor/Train_MobileNet_Colab.ipynb`
   - **Windows**: `d:\System_Conveyor\Train_MobileNet_Colab.ipynb`
5. Đợi upload xong

✅ **Notebook đã mở!**

---

## ⚡ BƯỚC 4: Chọn GPU (QUAN TRỌNG!)

### Kích Hoạt GPU Miễn Phí:

1. Click: **Runtime** → **Change runtime type**
2. **Hardware accelerator**: Chọn **T4 GPU**
3. Click **Save**

**Kiểm tra**: Trong notebook sẽ hiện "GPU available: YES"

✅ **GPU đã sẵn sàng! Train sẽ nhanh hơn 10-20 lần!**

---

## ▶️ BƯỚC 5: Chạy Training

### 5.1. Chạy Cell Setup

**Cell 1: Setup Environment**

- Click vào cell
- Nhấn `Shift + Enter`
- Đợi ~1-2 phút

Output:
```
✅ TensorFlow version: 2.x.x
✅ GPU available: YES
✅ Setup complete!
```

### 5.2. Upload Dataset

**Cell 2: Upload Dataset**

- Chạy cell (Shift + Enter)
- Sẽ xuất hiện nút **"Choose Files"**
- Click và chọn file `dataset.zip`
- Đợi upload (phụ thuộc tốc độ mạng)

Output:
```
📊 Dataset Summary:
   Train: 150 fresh, 145 spoiled
   Val: 25 fresh, 23 spoiled
   Total: 343 images
✅ Dataset OK!
```

### 5.3. Chạy Các Cell Còn Lại

Chạy lần lượt từng cell (Shift + Enter):

| Cell | Tên | Thời Gian |
|------|-----|-----------|
| 3 | Data Augmentation | 10s |
| 4 | Create Model | 30s |
| 5 | Prepare Data | 20s |
| **6** | **Train Model** | **15-20 phút** ⏱️ |
| 7 | Evaluate | 1 phút |
| 8 | Convert to TFLite | 30s |
| 9 | Download | Auto |

### 5.4. Theo Dõi Training (Cell 6)

**Output mẫu**:
```
🚀 Starting training...
   Epochs: 50
   Batch size: 32
   Using GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

Epoch 1/50
32/32 [==============================] - 15s
loss: 0.6234 - accuracy: 0.7123 - val_loss: 0.5123 - val_accuracy: 0.7856

...

Epoch 35/50
32/32 [==============================] - 12s
loss: 0.1234 - accuracy: 0.9456 - val_loss: 0.1523 - val_accuracy: 0.9234

✅ Training complete!
```

**Kết quả tốt**: `val_accuracy > 0.90` (90%+)

---

## 📥 BƯỚC 6: Download Model

**Cell 9: Download**

Sẽ tự động download 3 files:
1. **mobilenet_classifier.tflite** ← **Quan trọng nhất!**
2. best_model.keras (backup)
3. training_history.png (biểu đồ)

Files sẽ xuất hiện trong folder **Downloads**.

---

## 📋 BƯỚC 7: Copy Model Về Raspberry Pi

### Nếu Train Trên Raspberry Pi:

```bash
# File đã tải về ~/Downloads/
cp ~/Downloads/mobilenet_classifier.tflite ~/System_Conveyor/models/

# Kiểm tra
ls -lh ~/System_Conveyor/models/mobilenet_classifier.tflite
```

### Nếu Train Trên PC (Windows):

```powershell
# Copy qua SSH
scp C:\Users\YourName\Downloads\mobilenet_classifier.tflite pi@192.168.137.177:~/System_Conveyor/models/

# Hoặc dùng WinSCP/FileZilla
```

✅ **Model đã deploy!**

---

## 🚀 BƯỚC 8: Chạy Hệ Thống

```bash
# Trên Raspberry Pi
cd ~/System_Conveyor
python3 fruit_sorter.py
```

### Kết Quả Mong Đợi:

```
============================================================
🍎 Conveyor System for Fruit Classification
============================================================

🤖 Loading YOLO model...
✅ YOLO model loaded successfully

🤖 Loading MobileNetV2 model...
   Attempting XNNPACK delegate (ARM optimization)...
   ✅ Using XNNPACK delegate
✅ MobileNetV2 model loaded successfully

✅ System initialized successfully!
🚀 Starting main system loop...

⚡ FPS: 12.3
⏱️ Processing Times (avg):
   YOLO: 45.2ms
   MobileNet: 28.5ms  
   Preprocessing: 9.8ms
   Total: 83.5ms

🎯 Detected: apple (confidence: 0.87)
📊 Classified: Fresh (confidence: 0.923)
➡️ Sorting: CENTER (Fresh)
```

🎉 **Thành công!**

---

# PHƯƠNG ÁN 2: TRAIN TRÊN PC

## 🖥️ Khi Nào Dùng?

- Có PC Windows mạnh
- Muốn kiểm soát hoàn toàn
- Không có internet ổn định
- Muốn train offline

---

## 📦 BƯỚC 1: Setup PC

### 1.1. Chạy Script Tự Động

```powershell
cd d:\System_Conveyor
.\setup_pc.ps1
```

Script sẽ:
- ✅ Kiểm tra Python
- ✅ Cài TensorFlow
- ✅ Cài các packages cần thiết
- ✅ Tạo folders
- ✅ Kiểm tra dataset

### 1.2. Kiểm Tra TensorFlow

```powershell
python -c "import tensorflow as tf; print(tf.__version__)"
```

Output: `2.x.x` ← OK!

---

## 📊 BƯỚC 2: Chuẩn Bị Dataset

### Option A: Tự Tổ Chức

```powershell
# Tạo folders
cd d:\System_Conveyor\training\mobilenet
mkdir datasets\fruit_classification\train\fresh
mkdir datasets\fruit_classification\train\spoiled
mkdir datasets\fruit_classification\val\fresh
mkdir datasets\fruit_classification\val\spoiled

# Copy ảnh vào
# fresh → datasets\fruit_classification\train\fresh\
# spoiled → datasets\fruit_classification\train\spoiled\
```

### Option B: Dùng Script

```powershell
cd training\mobilenet
python prepare_data.py --source "D:\your_images" --output ./datasets/fruit_classification --verify
```

---

## 🚀 BƯỚC 3: Train Model

### Option A: Quick Train (Khuyên Dùng)

```powershell
cd d:\System_Conveyor
python quick_train.py
```

Script sẽ tự động:
1. Train model
2. Evaluate
3. Convert to TFLite
4. Hỏi có deploy lên Pi không

### Option B: Train Thủ Công

```powershell
cd training\mobilenet

# Train
python train_mobilenet.py --dataset ./datasets/fruit_classification --epochs 50 --batch 32

# Evaluate
python evaluate_model.py --model ./mobilenet_training/mobilenet_fruit_classifier_best.keras

# Convert
python export_tflite.py --model ./mobilenet_training/mobilenet_fruit_classifier_best.keras --output ../../models/mobilenet_classifier.tflite
```

---

## 📤 BƯỚC 4: Deploy

```powershell
# Copy to Raspberry Pi
scp models\mobilenet_classifier.tflite pi@192.168.137.177:~/System_Conveyor/models/

# Test trên Pi
ssh pi@192.168.137.177
cd ~/System_Conveyor
python3 fruit_sorter.py
```

---

# 📂 CHUẨN BỊ DATASET

## 🎨 Nguyên Tắc Thu Thập Ảnh

### 1. Số Lượng

- **Tối thiểu**: 50 ảnh/loại (100 tổng)
- **Khuyên dùng**: 200+ ảnh/loại (400+ tổng)
- **Nhiều = Tốt**: Càng nhiều càng chính xác

### 2. Đa Dạng

**Góc độ**:
- Top view (nhìn từ trên)
- Side view (nhìn từ bên)
- 45° angle
- Xoay 360°

**Ánh sáng**:
- Sáng
- Tối
- Trung bình
- Backlight (ngược sáng)

**Background**:
- Băng tải
- Bàn trắng
- Bàn đen
- Tự nhiên

### 3. Chất Lượng

✅ **TỐT**:
- Rõ nét (không mờ)
- Đủ sáng
- Toàn bộ trái cây trong khung
- Kích thước phù hợp (640x480+)

❌ **TRÁNH**:
- Mờ, nhoè
- Quá tối hoặc quá sáng
- Bị cắt xén
- Quá nhỏ (<200x200)

### 4. Cân Bằng

- Fresh ≈ Spoiled
- Nếu lệch quá nhiều, model sẽ bias

---

## 🗂️ Tổ Chức Dataset

### Cấu Trúc Chuẩn

```
fruit_classification/
├── train/                    70% data
│   ├── fresh/               
│   │   ├── fresh_001.jpg
│   │   ├── fresh_002.jpg
│   │   └── ... (150+ ảnh)
│   └── spoiled/
│       ├── spoiled_001.jpg
│       └── ... (150+ ảnh)
│
├── val/                      15% data
│   ├── fresh/               (20-30 ảnh)
│   └── spoiled/             (20-30 ảnh)
│
└── test/                     15% data
    ├── fresh/               (20-30 ảnh)
    └── spoiled/             (20-30 ảnh)
```

### Script Tự Động Chia Dataset

```python
# Nếu bạn có tất cả ảnh trong 2 folder: all_fresh/, all_spoiled/
cd training/mobilenet
python prepare_data.py \
    --source /path/to/all_images \
    --output ./datasets/fruit_classification \
    --split 0.7 0.15 0.15 \
    --verify
```

---

# 🚀 DEPLOY MODEL

## 📋 Checklist Trước Khi Deploy

- [ ] Model accuracy > 90%
- [ ] File .tflite đã tạo
- [ ] Kích thước ~3-5 MB
- [ ] Đã test trên validation set

---

## 📤 Copy Model Lên Raspberry Pi

### Method 1: SCP (Khuyên Dùng)

```bash
# Từ PC hoặc máy khác
scp /path/to/mobilenet_classifier.tflite pi@192.168.137.177:~/System_Conveyor/models/
```

### Method 2: USB

```bash
# Copy to USB trên PC
# Cắm USB vào Pi
sudo mount /dev/sda1 /mnt/usb
cp /mnt/usb/mobilenet_classifier.tflite ~/System_Conveyor/models/
sudo umount /mnt/usb
```

### Method 3: Git

```bash
# Trên PC: Add to git
git add models/mobilenet_classifier.tflite
git commit -m "Add trained model"
git push

# Trên Pi: Pull
cd ~/System_Conveyor
git pull
```

---

## ✅ Verify Deployment

```bash
# Check file tồn tại
ls -lh ~/System_Conveyor/models/mobilenet_classifier.tflite

# Kích thước nên ~3-5 MB
# Output: -rw-r--r-- 1 pi pi 3.8M ...

# Test load model
python3 -c "
from ai_models import MobileNetClassifier
m = MobileNetClassifier()
if m.load_model():
    print('✅ Model loaded successfully!')
else:
    print('❌ Failed to load model')
"
```

---

## 🎮 Chạy Hệ Thống

```bash
cd ~/System_Conveyor
python3 fruit_sorter.py
```

Hoặc với web interface:

```bash
python3 run_web.py
# Truy cập: http://192.168.137.177:5000
```

---

# 🐛 XỬ LÝ LỖI

## ❌ Lỗi Colab

### "No GPU available"

**Nguyên nhân**: Quên chọn GPU hoặc hết quota

**Giải pháp**:
```
1. Runtime → Change runtime type → T4 GPU → Save
2. Runtime → Restart runtime
3. Chạy lại từ cell 1
```

### "Runtime disconnected"

**Nguyên nhân**: Colab free timeout (90 phút)

**Giải pháp**:
```
- Model đã save checkpoint
- Chạy lại cell Training
- Hoặc giảm EPOCHS xuống 30
```

### Upload dataset bị lỗi

**Giải pháp**:
```bash
# Kiểm tra cấu trúc ZIP
unzip -l dataset.zip

# Phải thấy:
dataset/train/fresh/...
dataset/train/spoiled/...
dataset/val/...

# Nếu sai, nén lại đúng cấu trúc
```

---

## ❌ Lỗi Training

### Out of Memory

**Giải pháp**:
```python
# Trong notebook, giảm batch size
BATCH_SIZE = 16  # Thay vì 32
```

### Accuracy thấp (<85%)

**Nguyên nhân**:
- Dataset quá ít
- Ảnh không đa dạng
- Ảnh chất lượng kém

**Giải pháp**:
1. Thu thêm ảnh (200+ mỗi loại)
2. Đảm bảo đa dạng góc độ, ánh sáng
3. Loại ảnh mờ, kém chất lượng
4. Train lại với epochs cao hơn

### Overfitting (train acc >> val acc)

```
Train accuracy: 98%
Val accuracy: 75%  ← Overfitting!
```

**Giải pháp**:
```python
# Tăng dropout
dropout_rate = 0.6  # Thay vì 0.5

# Thêm augmentation
# Đã có sẵn trong notebook

# Giảm epochs
EPOCHS = 30
```

---

## ❌ Lỗi Deployment

### Model không load được

```bash
# Test load
python3 << EOF
from ai_models import MobileNetClassifier
m = MobileNetClassifier(model_path='models/mobilenet_classifier.tflite')
print(m.load_model())
EOF
```

Nếu lỗi:
```
# Kiểm tra file
ls -lh models/mobilenet_classifier.tflite

# Re-copy
scp mobilenet_classifier.tflite pi@192.168.137.177:~/System_Conveyor/models/
```

### XNNPACK không hoạt động

```
⚠️ Using CPU inference (no hardware acceleration)
```

**Giải pháp**:
```bash
# Cài TFLite với XNNPACK
pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
```

### FPS quá thấp (<8)

**Giải pháp**:
```python
# Edit utils/config.py
CAMERA_RESOLUTION = (320, 320)  # Giảm từ 416
YOLO_INPUT_SIZE = 320
FAST_PREPROCESSING = True
APPLY_BLUR = False
```

---

# 💡 TIPS & TRICKS

## 🚀 Training Nhanh Hơn

### 1. Dùng GPU (Colab)
- Luôn chọn T4 GPU
- Nhanh hơn CPU 10-20 lần

### 2. Giảm Epochs Khi Test
```python
EPOCHS = 20  # Thay vì 50, để test nhanh
```

### 3. Tăng Batch Size (Nếu Có RAM)
```python
BATCH_SIZE = 64  # Nếu GPU cho phép
```

---

## 📈 Accuracy Cao Hơn

### 1. Nhiều Dữ Liệu
- 200+ ảnh/loại tốt nhất
- Càng nhiều càng tốt

### 2. Đa Dạng
- Nhiều góc độ
- Nhiều điều kiện ánh sáng
- Nhiều background

### 3. Quality Control
```python
# Loại ảnh:
- Quá mờ
- Quá tối/sáng
- Bị cắt
- Sai label
```

### 4. Fine-tuning
```python
# Unfreeze base model layers (advanced)
base_model.trainable = True
# Train lại with learning rate nhỏ
learning_rate = 0.0001
```

---

## 🔧 Debugging

### Test Model Riêng

```python
import cv2
import numpy as np
from ai_models import MobileNetClassifier

# Load model
classifier = MobileNetClassifier()
classifier.load_model()

# Test 1 ảnh
img = cv2.imread("test.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img = img.astype(np.float32) / 127.5 - 1.0

result = classifier.classify_with_details(img)
print(f"Class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Check Processing Time

```python
import time

start = time.time()
result = classifier.classify_with_details(img)
print(f"Time: {(time.time() - start)*1000:.1f}ms")

# Mục tiêu: < 30ms
```

---

## 📊 Monitor Performance

```bash
# Xem real-time stats
cd ~/System_Conveyor
python3 fruit_sorter.py

# Sẽ in mỗi 10 giây:
⚡ FPS: 12.3
⏱️ YOLO: 45ms | MobileNet: 28ms
```

---

# ✅ CHECKLIST HOÀN CHỈNH

## 📋 Before Training

- [ ] Có ít nhất 50 ảnh fresh
- [ ] Có ít nhất 50 ảnh spoiled
- [ ] Ảnh rõ nét, đa dạng
- [ ] Fresh ≈ Spoiled (cân bằng)
- [ ] Đã tổ chức theo structure chuẩn
- [ ] (Colab) Đã nén thành dataset.zip
- [ ] (PC) Đã cài TensorFlow

## 🚀 During Training

- [ ] (Colab) Đã chọn T4 GPU
- [ ] Training chạy thành công
- [ ] Không có error
- [ ] val_accuracy tăng dần
- [ ] Chờ đủ epochs (hoặc early stopping)

## 📊 After Training

- [ ] val_accuracy > 90%
- [ ] val_loss giảm
- [ ] Không overfitting (train acc ≈ val acc)
- [ ] Confusion matrix tốt
- [ ] Đã download file .tflite
- [ ] File size ~3-5 MB

## 🚀 Deployment

- [ ] Đã copy model to Pi
- [ ] File tồn tại: `~/System_Conveyor/models/mobilenet_classifier.tflite`
- [ ] Model load thành công
- [ ] XNNPACK delegate hoạt động
- [ ] Hệ thống chạy không lỗi
- [ ] FPS > 10
- [ ] Classification chính xác

## 🎯 Real-World Testing

- [ ] Test với ảnh thật
- [ ] Accuracy thực tế > 85%
- [ ] Confidence > 80%
- [ ] Không miss fruits
- [ ] Servo phản ứng đúng
- [ ] Băng tải hoạt động

---

# 🎯 QUICK REFERENCE

## Colab Training (30 phút)

```
1. Chuẩn bị dataset → ZIP
2. Mở Colab → Upload notebook
3. Chọn GPU (T4)
4. Upload dataset.zip
5. Run all cells
6. Đợi 15-20 phút
7. Download .tflite
8. Copy to Pi → Deploy
```

## PC Training (60 phút)

```powershell
1. .\setup_pc.ps1
2. Organize dataset
3. python quick_train.py
4. Đợi training
5. scp model to Pi
6. Deploy
```

## Deploy to Pi

```bash
scp mobilenet_classifier.tflite pi@192.168.137.177:~/System_Conveyor/models/
ssh pi@192.168.137.177
cd ~/System_Conveyor
python3 fruit_sorter.py
```

---

# 🆘 HELP & SUPPORT

## 📚 Tài Liệu Thêm

- Google Colab Docs: https://colab.research.google.com/notebooks/intro.ipynb
- TensorFlow Docs: https://tensorflow.org/tutorials
- MobileNetV2: https://keras.io/api/applications/mobilenet/

## 🎥 Video Tutorials

- Google Colab Basics: https://youtube.com/watch?v=inN8seMm7UI
- Upload Files to Colab: https://youtube.com/watch?v=V2Mq_8D60rg
- Transfer Learning: https://youtube.com/watch?v=i_LwzRVP7bg

## 💬 Common Questions

**Q: Colab có giới hạn gì không?**  
A: Free tier có 90 phút timeout và quota GPU hàng ngày. Đủ để train 2-3 lần/ngày.

**Q: Train mất bao lâu?**  
A: Colab (GPU): 15-20 phút. PC (CPU): 30-60 phút. Pi: 10-20 giờ (không khuyên).

**Q: Cần bao nhiêu ảnh?**  
A: Tối thiểu 50/loại. Khuyên dùng 200+/loại.

**Q: Accuracy thấp phải làm sao?**  
A: Thêm ảnh, đa dạng hơn, train lâu hơn, check quality.

---

# 🎉 KẾT LUẬN

## 🏆 Best Practice

✅ **Dùng Google Colab**:
- Miễn phí 100%
- Có GPU mạnh
- Đơn giản nhất
- Thời gian nhanh nhất

✅ **Dataset tốt**:
- 200+ ảnh/loại  
- Đa dạng đầy đủ
- Chất lượng tốt

✅ **Monitor kỹ**:
- Check metrics
- Test real-world
- Fine-tune nếu cần

## 📊 Kết Quả Mong Đợi

- ⏱️ **Training time**: 15-20 phút (Colab)
- 🎯 **Accuracy**: > 90%
- 💾 **Model size**: ~3-5 MB
- ⚡ **Inference time**: < 30ms
- 📈 **FPS**: > 10

## 🚀 Next Steps

1. Train model theo hướng dẫn
2. Deploy lên Pi
3. Test thực tế
4. Fine-tune nếu cần
5. Enjoy! 🎉

---

**💪 Chúc bạn thành công!**

*Mọi thắc mắc xem lại phần [Xử Lý Lỗi](#xử-lý-lỗi) phía trên.*

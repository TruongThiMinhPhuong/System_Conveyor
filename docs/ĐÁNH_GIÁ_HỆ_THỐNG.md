# 📊 ĐÁNH GIÁ ĐỘ CHÍNH XÁC HỆ THỐNG

## 🎯 Mục Đích

Đánh giá toàn diện độ chính xác của hệ thống khi:
1. **Train với hoa quả thực tế**
2. **Chạy trên Raspberry Pi**  
3. **Xử lý trong điều kiện thực tế**

---

## 📋 MỤC LỤC

1. [Chuẩn Bị](#chuẩn-bị)
2. [Thu Thập Ảnh Test](#thu-thập-ảnh-test)
3. [Chạy Đánh Giá](#chạy-đánh-giá)
4. [Phân Tích Kết Quả](#phân-tích-kết-quả)
5. [Cải Thiện](#cải-thiện)

---

## 🎓 CHUẨN BỊ

### 1. Yêu Cầu

- ✅ Đã train model với **hoa quả thực tế**
- ✅ Model đã deploy lên **Raspberry Pi**
- ✅ Hệ thống chạy được (test OK)
- ✅ Có dataset test riêng (KHÔNG dùng data training)

### 2. Cấu Trúc Dataset Test

```
test_dataset/
├── fresh/              (20-50 ảnh)
│   ├── fresh_001.jpg
│   ├── fresh_002.jpg
│   └── ...
└── spoiled/            (20-50 ảnh)
    ├── spoiled_001.jpg
    ├── spoiled_002.jpg
    └── ...
```

**Quan trọng**: 
- ❌ KHÔNG dùng ảnh từ training set
- ✅ Chụp fresh ảnh mới trong điều kiện thực tế
- ✅ Đa dạng: nhiều góc độ, ánh sáng, loại quả

---

## 📸 THU THẬP ẢNH TEST

### Option 1: Chụp Trực Tiếp Từ Hệ Thống

```bash
# Trên Raspberry Pi
cd ~/System_Conveyor

# Tạo script chụp ảnh
python3 << 'EOF'
from picamera2 import Picamera2
import cv2
from pathlib import Path

# Initialize camera
camera = Picamera2()
config = camera.create_still_configuration()
camera.configure(config)
camera.start()

# Create directories
Path("test_dataset/fresh").mkdir(parents=True, exist_ok=True)
Path("test_dataset/spoiled").mkdir(parents=True, exist_ok=True)

print("📸 Camera ready! Press SPACE to capture, Q to quit")

count = 0
while True:
    frame = camera.capture_array()
    cv2.imshow("Capture", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # Spacebar
        label = input("Label (f=fresh, s=spoiled): ")
        folder = "fresh" if label == 'f' else "spoiled"
        filename = f"test_dataset/{folder}/img_{count:03d}.jpg"
        cv2.imwrite(filename, frame)
        print(f"✅ Saved: {filename}")
        count += 1
    elif key == ord('q'):
        break

camera.stop()
cv2.destroyAllWindows()
print(f"✅ Captured {count} images")
EOF
```

### Option 2: Chụp Bằng Điện Thoại/Camera Khác

1. Chụp ảnh hoa quả thực tế
2. Phân loại thủ công (fresh/spoiled)
3. Copy vào Pi:

```bash
# Từ PC
scp -r test_dataset/ pi@192.168.137.177:~/System_Conveyor/
```

### Tips Thu Thập Ảnh Test

✅ **Làm**:
- Chụp trong điều kiện giống production (ánh sáng, góc độ)
- Bao gồm các trường hợp khó (ảnh mờ nhẹ, ánh sáng yếu)
- Cân bằng fresh vs spoiled
- 20-50 ảnh mỗi loại là đủ

❌ **Tránh**:
- Dùng ảnh từ dataset training
- Ảnh quá dễ (perfect conditions)
- Ảnh không đại diện cho real-world

---

## 🚀 CHẠY ĐÁNH GIÁ

### Trên Raspberry Pi

```bash
cd ~/System_Conveyor

# Đánh giá với dataset test
python3 evaluate_system.py --test_dir test_dataset --output evaluation_results
```

### Output

Script sẽ:
1. ✅ Load models (YOLO + MobileNet)
2. ✅ Process từng ảnh test
3. ✅ Tính toán metrics
4. ✅ In kết quả lên màn hình
5. ✅ Lưu báo cáo chi tiết

**Thời gian**: ~2-5 phút cho 50 ảnh

---

## 📊 PHÂN TÍCH KẾT QUẢ

### 1. Metrics Quan Trọng

#### **Accuracy (Độ Chính Xác)**
```
Accuracy = (Số ảnh đúng) / (Tổng số ảnh)
```

**Đánh giá**:
- ✅ **≥95%**: Xuất sắc
- ✅ **≥90%**: Tốt  
- ⚠️ **≥85%**: Khá, cần cải thiện
- ❌ **<85%**: Thấp, cần train lại

#### **Precision (Độ Chính Xác Dự Đoán)**
```
Precision_Fresh = (Fresh đúng) / (Tất cả dự đoán Fresh)
```

**Ý nghĩa**: 
- Khi model nói "Fresh", % thực sự fresh
- Quan trọng nếu không muốn fresh bị loại nhầm

#### **Recall (Độ Phủ)**
```
Recall_Fresh = (Fresh đúng) / (Tất cả Fresh thật)
```

**Ý nghĩa**:
- % fresh thực tế được nhận diện
- Quan trọng nếu không muốn bỏ sót fresh

#### **F1 Score (Cân Bằng)**
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**Đánh giá**:
- ✅ **≥90%**: Tốt
- ⚠️ **≥85%**: Chấp nhận được
- ❌ **<85%**: Cần cải thiện

### 2. Confusion Matrix

```
                 Predicted Fresh  |  Predicted Spoiled
Actual Fresh:         40          |         5
Actual Spoiled:       3           |        42
```

**Phân tích**:
- **40**: Fresh đúng (True Positive)
- **42**: Spoiled đúng (True Negative)
- **5**: Fresh nhầm thành Spoiled (False Negative) ← **Lãng phí!**
- **3**: Spoiled nhầm thành Fresh (False Positive) ← **Nguy hiểm!**

**Mong muốn**: False Positive (Spoiled→Fresh) gần 0

### 3. Performance Metrics

#### **Processing Time**
```
Avg Total Time: 85ms
  - YOLO: 45ms
  - Preprocessing: 10ms
  - Classification: 30ms
```

**Đánh giá**:
- ✅ **≤100ms**: Tốt (real-time)
- ⚠️ **100-150ms**: Chấp nhận được
- ❌ **>150ms**: Quá chậm

#### **FPS (Frames Per Second)**
```
Estimated FPS = 1000 / Avg_Total_Time
```

**Yêu cầu**:
- ✅ **≥10 FPS**: Đủ nhanh cho conveyor
- ⚠️ **8-10 FPS**: Có thể dùng, giảm tốc độ belt
- ❌ **<8 FPS**: Quá chậm

### 4. Confidence Scores

```
Avg Detection Confidence: 87%
Avg Classification Confidence: 92%
```

**Đánh giá**:
- ✅ **≥85%**: Tốt, model tin tưởng
- ⚠️ **70-85%**: Chấp nhận được
- ❌ **<70%**: Model không chắc chắn

---

## 📋 MẪU KẾT QUẢ

### Kết Quả Tốt ✅

```
============================================================
📊 KẾT QUẢ ĐÁNH GIÁ HỆ THỐNG
============================================================

📈 Tổng Quan:
   Tổng số ảnh: 90
   Phát hiện thành công: 88
   Phân loại chính xác: 84
   Phân loại sai: 4

🎯 Độ Chính Xác:
   Overall Accuracy: 95.45%

🍏 Fresh Class:
   Precision: 93.33%
   Recall: 96.67%
   F1 Score: 95.00%

🍎 Spoiled Class:
   Precision: 97.67%
   Recall: 94.44%
   F1 Score: 96.03%

⚡ Hiệu Năng (Raspberry Pi):
   YOLO Detection: 42.3ms
   Preprocessing: 9.2ms
   Classification: 28.5ms
   Total: 80.0ms
   Estimated FPS: 12.5

🔍 Độ Tin Cậy:
   Avg Detection Confidence: 89.2%
   Avg Classification Confidence: 93.5%

🎓 ĐÁNH GIÁ:
   ✅ Accuracy: XUẤT SẮC (≥95%)
   ✅ F1 Score: TỐT (≥90%)
   ✅ Performance: ĐỦ NHANH (≥10 FPS)
```

### Kết Quả Cần Cải Thiện ⚠️

```
============================================================
📊 KẾT QUẢ ĐÁNH GIÁ HỆ THỐNG
============================================================

📈 Tổng Quan:
   Tổng số ảnh: 80
   Phát hiện thành công: 76
   Phân loại chính xác: 68
   Phân loại sai: 8

🎯 Độ Chính Xác:
   Overall Accuracy: 89.47%  ← Dưới 90%!

🍏 Fresh Class:
   Precision: 87.50%
   Recall: 91.67%
   F1 Score: 89.54%  ← Dưới 90%!

🍎 Spoiled Class:
   Precision: 92.11%
   Recall: 87.50%
   F1 Score: 89.74%  ← Dưới 90%!

⚡ Hiệu Năng:
   Total: 125.3ms
   Estimated FPS: 7.98  ← Dưới 10 FPS!

🎓 ĐÁNH GIÁ:
   ⚠️  Accuracy: KHÁ (<90%, cần cải thiện)
   ⚠️  F1 Score: KHÁ (<90%)
   ❌ Performance: QUÁ CHẬM (<8 FPS)

💡 KHUYẾN NGHỊ CẢI THIỆN:
   - Thu thập thêm dữ liệu (200+ ảnh/loại)
   - Đảm bảo ảnh đa dạng (góc độ, ánh sáng)
   - Train lại với epochs cao hơn
   
💡 KHUYẾN NGHỊ TỐI ƯU:
   - Giảm CAMERA_RESOLUTION xuống 320x320
   - Set FAST_PREPROCESSING = True
   - Kiểm tra XNNPACK delegate
```

---

## 🔧 CẢI THIỆN HỆ THỐNG

### Nếu Accuracy Thấp (<90%)

#### 1. Cải Thiện Dataset

```bash
# Thu thêm dữ liệu
- Minimum: 100 ảnh/loại
- Recommended: 200-300 ảnh/loại
- Đa dạng:
  ✅ Nhiều góc độ
  ✅ Nhiều điều kiện ánh sáng
  ✅ Nhiều background
  ✅ Nhiều loại quả
```

#### 2. Train Lại Model

```powershell
# Trên PC hoặc Colab
cd training/mobilenet

# Train với epochs cao hơn
python train_mobilenet.py --dataset ./datasets/fruit_classification --epochs 70

# Hoặc dùng Colab với GPU
```

#### 3. Kiểm Tra Data Quality

```python
# Loại ảnh kém chất lượng
- Ảnh quá mờ
- Ảnh quá tối/sáng
- Ảnh sai label
- Ảnh trùng lặp
```

#### 4. Điều Chỉnh Threshold

```python
# utils/config.py
CLASSIFICATION_THRESHOLD = 0.7  # Tăng từ 0.6 nếu muốn chắc chắn hơn
YOLO_CONFIDENCE_THRESHOLD = 0.5  # Giảm nếu miss detection
```

### Nếu FPS Thấp (<10)

#### 1. Giảm Resolution

```python
# utils/config.py
CAMERA_RESOLUTION = (320, 320)  # Từ 416
YOLO_INPUT_SIZE = 320  # Từ 416
```

#### 2. Bật Fast Mode

```python
# utils/config.py
FAST_PREPROCESSING = True
APPLY_BLUR = False  # Tắt blur nếu không cần
```

#### 3. Kiểm Tra Hardware Acceleration

```bash
# Test XNNPACK
python3 << EOF
from ai_models import MobileNetClassifier
m = MobileNetClassifier()
m.load_model()
# Should see: "✅ Using XNNPACK delegate"
EOF
```

#### 4. Nâng Cấp Hardware

- Raspberry Pi 5 (nhanh hơn 2x)
- Google Coral USB Accelerator (TPU)
- Overclock Pi 4 (caution!)

### Nếu False Positive Cao (Spoiled→Fresh)

**Nguy hiểm**: Hoa quả hỏng đi vào kênh fresh!

#### Giải pháp:

```python
# Tăng threshold
CLASSIFICATION_THRESHOLD = 0.75  # Từ 0.6

# Hoặc bias về spoiled
# Trong code, default to spoiled khi low confidence
if confidence < 0.8:
    default_to_spoiled = True
```

### Nếu False Negative Cao (Fresh→Spoiled)

**Lãng phí**: Hoa quả tốt bị loại bỏ!

#### Giải pháp:

```python
# Giảm threshold
CLASSIFICATION_THRESHOLD = 0.55  # Từ 0.6

# Thêm augmentation
# Train lại với data augmentation mạnh hơn
```

---

## 📈 CONTINUOUS IMPROVEMENT

### Quy Trình Cải Thiện Liên Tục

```
1. Chạy đánh giá
   ↓
2. Phân tích kết quả
   ↓
3. Thu thập thêm data (nếu cần)
   ↓
4. Train lại model
   ↓
5. Deploy & test
   ↓
6. Lặp lại từ bước 1
```

### Tracking Progress

```bash
# Lưu kết quả mỗi lần đánh giá
ls evaluation_results/
evaluation_20251218_120000.json
evaluation_20251219_150000.json
evaluation_20251220_140000.json

# So sánh accuracy qua thời gian
# V1: 85% → V2: 91% → V3: 94% ✅
```

---

## 🎯 TARGET METRICS

### Mục Tiêu Tối Thiểu

- ✅ Accuracy: **≥90%**
- ✅ F1 Score: **≥88%**
- ✅ FPS: **≥10**
- ✅ False Positive (Spoiled→Fresh): **<3%**

### Mục Tiêu Lý Tưởng

- 🎯 Accuracy: **≥95%**
- 🎯 F1 Score: **≥92%**
- 🎯 FPS: **≥12**
- 🎯 False Positive: **<1%**

---

## 📝 CHECKLIST ĐÁNH GIÁ

### Trước Khi Đánh Giá
- [ ] Model đã train xong
- [ ] Model đã deploy lên Pi
- [ ] Có dataset test (20+ ảnh/loại)
- [ ] Test data KHÔNG trùng training data
- [ ] Hệ thống chạy được

### Trong Quá Trình
- [ ] Chụp/thu thập ảnh test đa dạng
- [ ] Label đúng (fresh/spoiled)
- [ ] Chạy script đánh giá
- [ ] Ghi chú kết quả

### Sau Đánh Giá
- [ ] Phân tích metrics
- [ ] Xác định vấn đề (nếu có)
- [ ] Lập kế hoạch cải thiện
- [ ] Document kết quả
- [ ] Train lại (nếu cần)

---

## 💡 TIPS

### Best Practices

1. **Đánh giá thường xuyên**
   - Sau mỗi lần train
   - Khi thay đổi config
   - Khi có data mới

2. **Sử dụng real-world data**
   - Ảnh trong điều kiện thực tế
   - Bao gồm edge cases
   - Đa dạng điều kiện

3. **Track metrics theo thời gian**
   - Save mỗi lần đánh giá
   - So sánh versions
   - Monitor trends

4. **A/B Testing**
   - Test nhiều configs
   - So sánh kết quả
   - Chọn best performing

---

## 📄 FILES QUAN TRỌNG

| File | Mục Đích |
|------|----------|
| `evaluate_system.py` | Script đánh giá chính |
| `evaluation_results/*.json` | Kết quả chi tiết |
| `evaluation_results/*.txt` | Báo cáo tóm tắt |

---

## 🆘 TROUBLESHOOTING

### Lỗi "Model not found"

```bash
# Kiểm tra models
ls -lh ~/System_Conveyor/models/
# Phải có:
# - yolov8n_fruit.pt
# - mobilenet_classifier.tflite
```

### Lỗi "Test directory not found"

```bash
# Kiểm tra structure
ls -R test_dataset/
# Phải có:
# test_dataset/fresh/
# test_dataset/spoiled/
```

### FPS quá thấp trong test

```bash
# Đóng các process khác
sudo systemctl stop bluetooth
# Overclock (optional)
# Kiểm tra temperature
vcgencmd measure_temp
```

---

## ✅ TÓM LẠI

**Để đánh giá độ chính xác hệ thống**:

1. ✅ Thu thập test data (real-world)
2. ✅ Chạy `evaluate_system.py`
3. ✅ Phân tích kết quả
4. ✅ Cải thiện nếu cần
5. ✅ Lặp lại cho đến khi đạt target

**Target**: Accuracy ≥90%, FPS ≥10, F1 ≥88%

🎉 **Thành công khi hệ thống hoạt động tốt trong điều kiện thực tế!**

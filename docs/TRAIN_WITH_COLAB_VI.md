# 🚀 Hướng Dẫn Train Model Với Google Colab (Miễn Phí)

## ⚡ Tại Sao Dùng Google Colab?

- ✅ **MIỄN PHÍ** - Không tốn tiền
- ✅ **GPU MIỄN PHÍ** - Train nhanh hơn 10-20 lần
- ✅ **Không cần cài đặt** - Chạy trên trình duyệt
- ✅ **Dùng được trên Raspberry Pi** - Mở browser là được

---

## 📋 Bước 1: Chuẩn Bị Dataset

### Tạo Folder Ảnh

Tổ chức ảnh của bạn như sau:

```
my_fruits/
├── train/
│   ├── fresh/
│   │   ├── fresh_1.jpg
│   │   ├── fresh_2.jpg
│   │   └── ... (ít nhất 50 ảnh)
│   └── spoiled/
│       ├── spoiled_1.jpg
│       ├── spoiled_2.jpg
│       └── ... (ít nhất 50 ảnh)
├── val/
│   ├── fresh/
│   │   └── ... (10-20 ảnh)
│   └── spoiled/
│       └── ... (10-20 ảnh)
└── test/
    ├── fresh/
    │   └── ... (10-20 ảnh)
    └── spoiled/
        └── ... (10-20 ảnh)
```

### Nén Thành ZIP

**Windows/Linux**: Click chuột phải → Send to → Compressed folder  
**Tên file**: `dataset.zip`

---

## 🌐 Bước 2: Mở Google Colab

### Trên Raspberry Pi:
1. Mở **Chromium browser**
2. Truy cập: https://colab.research.google.com
3. Đăng nhập Gmail

### Upload Notebook:
1. **File** → **Upload notebook**
2. Chọn file `Train_MobileNet_Colab.ipynb` (trong thư mục System_Conveyor)

---

## 🚀 Bước 3: Chạy Training

### 3.1. Chọn GPU (Miễn Phí!)

1. **Runtime** → **Change runtime type**
2. **Hardware accelerator** → Chọn **T4 GPU**
3. Click **Save**

### 3.2. Chạy Từng Cell

**Cách chạy**: Click vào cell → Nhấn `Shift + Enter`

#### Cell 1: Setup Environment
```python
# Install dependencies
!pip install -q tensorflow opencv-python ...
```
⏱️ Chờ ~1-2 phút

#### Cell 2: Upload Dataset
```python
# Upload ZIP file
from google.colab import files
uploaded = files.upload()
```
📁 Chọn file `dataset.zip` của bạn  
⏱️ Chờ upload (phụ thuộc tốc độ mạng)

#### Cell 3-5: Chuẩn bị dữ liệu
Chạy lần lượt, mỗi cell ~10-30 giây

#### Cell 6: Train Model 🎯
```python
history = model.fit(...)
```
⏱️ **Quan trọng**: Đây là bước lâu nhất (~15-20 phút)

**Theo dõi progress**:
```
Epoch 1/50
32/32 [==============================] - 15s
...
val_accuracy: 0.9234
```

✅ **Kết quả tốt**: val_accuracy > 0.90 (90%)

#### Cell 7: Evaluate
Xem kết quả đánh giá và biểu đồ

#### Cell 8: Convert to TFLite
Chuyển sang định dạng Raspberry Pi

#### Cell 9: Download 📥
```python
files.download('output/mobilenet_classifier.tflite')
```

File sẽ tải về máy bạn!

---

## 📥 Bước 4: Copy Model Về Raspberry Pi

### Từ Máy Bạn (có file đã download):

**Nếu trên Raspberry Pi** (đã download trực tiếp):
```bash
# File ở ~/Downloads/
cd ~/Downloads
cp mobilenet_classifier.tflite ~/System_Conveyor/models/
```

**Nếu trên Windows PC** (cần copy sang Pi):
```powershell
# Copy qua mạng
scp ~/Downloads/mobilenet_classifier.tflite pi@192.168.137.177:~/System_Conveyor/models/
```

---

## 🎮 Bước 5: Chạy Hệ Thống

```bash
# Trên Raspberry Pi
cd ~/System_Conveyor
python3 fruit_sorter.py
```

**Kết quả mong đợi**:
```
✅ YOLO model loaded
✅ MobileNetV2 model loaded
   Using XNNPACK delegate
✅ System initialized successfully!
🚀 Starting main system loop...

⚡ FPS: 12.3
⏱️ YOLO: 45ms | MobileNet: 28ms
📊 Classified: Fresh (92.5%)
```

---

## 📊 Kết Quả Mong Đợi

### Metrics Tốt:
- ✅ Accuracy: > 90%
- ✅ Precision: > 88%
- ✅ Recall: > 88%
- ✅ F1 Score: > 90%

### Nếu Kết Quả Thấp:
- 📸 Thu thêm ảnh (tối thiểu 100-200 mỗi loại)
- 🔄 Train lại với epochs cao hơn (70-100)
- 🎨 Đảm bảo ảnh đa dạng (nhiều góc độ, ánh sáng)

---

## 🐛 Xử Lý Lỗi

### ❌ "Runtime disconnected"
**Nguyên nhân**: Colab timeout sau 90 phút free  
**Giải pháp**: Chạy lại từ cell training (model đã save checkpoint)

### ❌ "Out of memory"
**Nguyên nhân**: Dataset quá lớn  
**Giải pháp**: 
```python
BATCH_SIZE = 16  # Giảm từ 32
```

### ❌ "No GPU available"
**Nguyên nhân**: Quên chọn GPU  
**Giải pháp**: Runtime → Change runtime type → T4 GPU

### ❌ Upload dataset lỗi
**Nguyên nhân**: File ZIP sai cấu trúc  
**Giải pháp**: Kiểm tra lại folder structure (xem Bước 1)

---

## 💡 Tips & Tricks

### Train Nhanh Hơn
1. Dùng ảnh nhỏ hơn (nếu quá nhiều)
2. Giảm EPOCHS nếu test: `EPOCHS = 20`
3. Dùng GPU (QUAN TRỌNG!)

### Cải Thiện Accuracy
1. **Thêm ảnh**: 200+ mỗi loại là tốt nhất
2. **Đa dạng**: Nhiều góc độ, ánh sáng, background
3. **Chất lượng**: Ảnh rõ nét, không mờ
4. **Cân bằng**: Số lượng Fresh ≈ Spoiled

### Kiểm Tra Model
```python
# Test 1 ảnh
import cv2
test_img = cv2.imread('test.jpg')
# ... preprocessing ...
prediction = model.predict(test_img)
print(f"Fresh: {prediction[0][0]:.2%}")
print(f"Spoiled: {prediction[0][1]:.2%}")
```

---

## 📱 Quy Trình Hoàn Chỉnh

### Lần Đầu Setup:
1. ✅ Chuẩn bị dataset (50+ ảnh/loại)
2. ✅ Nén thành ZIP
3. ✅ Mở Colab notebook
4. ✅ Chọn GPU
5. ✅ Upload dataset
6. ✅ Train (15-20 phút)
7. ✅ Download model
8. ✅ Copy to Raspberry Pi
9. ✅ Chạy hệ thống

### Train Lại (Khi Cần):
1. ✅ Thêm ảnh mới vào dataset
2. ✅ Nén lại ZIP
3. ✅ Mở lại notebook cũ
4. ✅ Upload dataset mới
5. ✅ Runtime → Restart and run all
6. ✅ Download model mới
7. ✅ Copy đè lên model cũ

---

## 🎯 Checklist Hoàn Thành

### Chuẩn Bị:
- [ ] Dataset có ít nhất 50 ảnh fresh
- [ ] Dataset có ít nhất 50 ảnh spoiled
- [ ] Đã nén thành dataset.zip
- [ ] Có tài khoản Gmail

### Training:
- [ ] Đã upload notebook lên Colab
- [ ] Đã chọn GPU (T4)
- [ ] Đã upload dataset
- [ ] Training chạy thành công
- [ ] Accuracy > 90%
- [ ] Đã download file .tflite

### Deployment:
- [ ] File .tflite đã copy to Raspberry Pi
- [ ] Hệ thống chạy không lỗi
- [ ] Classification hoạt động
- [ ] FPS > 10

---

## 🆘 Hỗ Trợ

### Link Hữu Ích:
- 📚 Google Colab: https://colab.research.google.com
- 📖 TensorFlow Docs: https://tensorflow.org
- 💬 Colab FAQ: https://research.google.com/colaboratory/faq.html

### Video Tham Khảo:
- How to use Google Colab: https://youtube.com/watch?v=inN8seMm7UI
- Upload files to Colab: https://youtube.com/watch?v=V2Mq_8D60rg

---

**🎉 Chúc bạn train model thành công!**

*Lưu ý: GPU miễn phí của Colab có giới hạn. Nếu hết quota, đợi 12-24h hoặc dùng Colab Pro ($9.99/tháng)*

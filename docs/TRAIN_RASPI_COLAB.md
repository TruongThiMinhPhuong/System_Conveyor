# 🎯 HƯỚNG DẪN TRAIN TRÊN RASPBERRY PI VỚI GOOGLE COLAB

## 📱 Siêu Đơn Giản - Chỉ 7 Bước!

---

## 🔰 Bước 1: Chuẩn Bị Dataset

### Tạo Folder Ảnh

```bash
# Trên Raspberry Pi
cd ~
mkdir -p my_fruits/train/fresh
mkdir -p my_fruits/train/spoiled
mkdir -p my_fruits/val/fresh
mkdir -p my_fruits/val/spoiled
```

### Copy Ảnh Vào

```bash
# Copy ảnh fresh
cp /path/to/your/fresh_images/*.jpg ~/my_fruits/train/fresh/

# Copy ảnh spoiled
cp /path/to/your/spoiled_images/*.jpg ~/my_fruits/train/spoiled/

# Tương tự cho val (10-20 ảnh mỗi loại)
```

### Nén Thành ZIP

```bash
cd ~
zip -r dataset.zip my_fruits/
```

✅ **Xong bước 1!** File `dataset.zip` đã sẵn sàng

---

## 🌐 Bước 2: Chạy Script Tự Động

```bash
cd ~/System_Conveyor
chmod +x start_colab_training.sh
./start_colab_training.sh
```

Script sẽ:
- ✅ Kiểm tra browser
- ✅ Mở Google Colab tự động
- ✅ Hiện hướng dẫn chi tiết

---

## 🚀 Bước 3: Upload Notebook (Trong Browser)

### Trên Google Colab:

1. **Đăng nhập Gmail** (nếu chưa)

2. **Upload Notebook**:
   ```
   File → Upload notebook
   → Browse
   → Chọn: /home/pi/System_Conveyor/Train_MobileNet_Colab.ipynb
   ```

✅ **Notebook đã sẵn sàng!**

---

## ⚡ Bước 4: Chọn GPU Miễn Phí

### Trong Colab:

```
Runtime → Change runtime type
→ Hardware accelerator: T4 GPU
→ Save
```

**Quan trọng**: Phải chọn GPU để train nhanh!

✅ **GPU đã kích hoạt!**

---

## 📦 Bước 5: Upload Dataset

### Chạy Cell Đầu Tiên:

1. Click vào cell đầu tiên (Setup Environment)
2. Nhấn `Shift + Enter`
3. Đợi cài đặt xong (~1-2 phút)

### Chạy Cell Upload:

1. Cell thứ 2: Upload Dataset
2. Click nút **"Choose Files"**
3. Chọn file `dataset.zip` (ở ~/dataset.zip)
4. Đợi upload (phụ thuộc tốc độ mạng)

✅ **Dataset đã upload!**

---

## 🎯 Bước 6: Train Model

### Chạy Lần Lượt Các Cell:

Click vào từng cell và nhấn `Shift + Enter`:

```
Cell 3: Data Augmentation   → 10 giây
Cell 4: Create Model        → 30 giây
Cell 5: Prepare Data        → 20 giây
Cell 6: Train Model         → 15-20 PHÚT ⏱️
Cell 7: Evaluate            → 1 phút
Cell 8: Convert to TFLite   → 30 giây
Cell 9: Download            → Tự động
```

### Theo Dõi Training (Cell 6):

```
Epoch 1/50
32/32 [==============================] - 15s
loss: 0.4523 - accuracy: 0.8234 - val_loss: 0.3421 - val_accuracy: 0.8756
```

**Chờ đến khi**:
```
Epoch 50/50
val_accuracy: 0.9234  ← Kết quả tốt (> 0.90)
```

✅ **Training xong!**

---

## 📥 Bước 7: Download & Deploy

### Download Model:

Cell cuối cùng sẽ tự động download:
- `mobilenet_classifier.tflite` → File này là model đã train!

File sẽ xuất hiện trong folder **Downloads**.

### Copy to System:

```bash
# Quay lại terminal Raspberry Pi

# Copy model vào project
cp ~/Downloads/mobilenet_classifier.tflite ~/System_Conveyor/models/

# Kiểm tra
ls -lh ~/System_Conveyor/models/mobilenet_classifier.tflite
```

✅ **Model đã deploy!**

---

## ✅ Bước 8: Chạy Hệ Thống

```bash
cd ~/System_Conveyor
python3 fruit_sorter.py
```

### Kết Quả Mong Đợi:

```
============================================================
🍎 Conveyor System for Fruit Classification
============================================================

🤖 Loading MobileNetV2 model...
   Using XNNPACK delegate
✅ MobileNetV2 model loaded successfully

🚀 Starting main system loop...

⚡ FPS: 12.3
⏱️ YOLO: 45ms | MobileNet: 28ms | Preprocessing: 10ms

🎯 Detected: apple (confidence: 0.87)
📊 Classified: Fresh (confidence: 0.923)
➡️ Sorting: CENTER (Fresh)
```

🎉 **Thành công!** Hệ thống đã hoạt động với model mới!

---

## 🐛 Xử Lý Lỗi Thường Gặp

### ❌ "No GPU available" trong Colab

**Giải pháp**:
```
Runtime → Change runtime type → T4 GPU → Save
Sau đó: Runtime → Restart runtime
```

### ❌ Upload dataset bị lỗi

**Giải pháp**:
- Kiểm tra file ZIP có đúng cấu trúc không
- Thử nén lại: `zip -r dataset.zip my_fruits/`
- File size < 100MB tốt nhất

### ❌ "Runtime disconnected" giữa chừng

**Nguyên nhân**: Colab free timeout sau 90 phút

**Giải pháp**:
- Chạy lại cell Training (model đã save checkpoint)
- Hoặc giảm EPOCHS xuống còn 30

### ❌ Accuracy thấp (< 85%)

**Giải pháp**:
- Thu thêm ảnh (tối thiểu 100-200/loại)
- Đảm bảo ảnh đa dạng (nhiều góc độ, ánh sáng)
- Train lại với EPOCHS cao hơn (70-100)

### ❌ Model download không tự động

**Giải pháp**:
```python
# Chạy lại cell cuối
from google.colab import files
files.download('output/mobilenet_classifier.tflite')
```

---

## 💡 Tips Hay

### Training Nhanh Hơn:
1. **Luôn chọn GPU** (Runtime → T4 GPU)
2. Giảm BATCH_SIZE nếu out of memory: `BATCH_SIZE = 16`
3. Giảm EPOCHS để test: `EPOCHS = 20`

### Accuracy Cao Hơn:
1. **Nhiều ảnh hơn**: 200+ ảnh/loại
2. **Đa dạng**: Nhiều góc độ, ánh sáng, background
3. **Chất lượng**: Ảnh rõ, không mờ
4. **Cân bằng**: Fresh ≈ Spoiled

### Tiết Kiệm Thời Gian:
1. Chuẩn bị dataset trước khi mở Colab
2. Zip dataset trước (không zip trong Colab)
3. Có thể pause và resume training (checkpoint)

---

## 📊 Checklist Hoàn Chỉnh

### Chuẩn Bị:
- [ ] Có ít nhất 50 ảnh fresh
- [ ] Có ít nhất 50 ảnh spoiled
- [ ] Đã nén thành dataset.zip
- [ ] File < 100MB (nếu lớn hơn, xóa ảnh thừa)

### Training:
- [ ] Đã mở Colab
- [ ] Đã upload notebook
- [ ] Đã chọn T4 GPU
- [ ] Đã upload dataset.zip
- [ ] Training chạy thành công (15-20 phút)
- [ ] val_accuracy > 0.90
- [ ] Đã download file .tflite

### Deploy:
- [ ] Copy model to ~/System_Conveyor/models/
- [ ] Chạy fruit_sorter.py thành công
- [ ] YOLO detect được
- [ ] MobileNet classify được
- [ ] FPS > 10
- [ ] Accuracy trong thực tế > 85%

---

## 🎬 Video Hướng Dẫn (Tham Khảo)

Nếu chưa rõ, xem các video này:

1. **Google Colab cơ bản**:
   - https://www.youtube.com/watch?v=inN8seMm7UI

2. **Upload files to Colab**:
   - https://www.youtube.com/watch?v=V2Mq_8D60rg

3. **Train model with Colab**:
   - https://www.youtube.com/watch?v=i_LwzRVP7bg

---

## 🆘 Cần Trợ Giúp?

### Check Log Errors:
```bash
# Xem log hệ thống
tail -f ~/System_Conveyor/logs/system.log

# Test model riêng
python3 -c "from ai_models import MobileNetClassifier; m = MobileNetClassifier(); m.load_model()"
```

### Verify Model:
```bash
# Kiểm tra model file
ls -lh ~/System_Conveyor/models/mobilenet_classifier.tflite

# Kích thước nên ~3-5 MB
```

---

## 🎯 Tóm Tắt Quy Trình

```
📁 Chuẩn bị dataset
    ↓
📦 Nén thành ZIP
    ↓
🌐 Mở Google Colab
    ↓
⬆️ Upload notebook + dataset
    ↓
⚡ Chọn GPU
    ↓
▶️ Chạy training (15-20 phút)
    ↓
📥 Download model
    ↓
📋 Copy to Raspberry Pi
    ↓
🚀 Chạy hệ thống
    ↓
🎉 Thành công!
```

---

**⏱️ Tổng thời gian**: ~30-40 phút (bao gồm upload + training)

**💰 Chi phí**: 100% MIỄN PHÍ

**🎯 Kết quả**: Model accuracy > 90%

---

**🎉 Chúc bạn train model thành công!**

*Lưu ý: Nếu hết quota GPU miễn phí, đợi 12-24h hoặc dùng Colab Pro ($9.99/tháng)*

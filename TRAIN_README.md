# 🚀 QUICK START - Train Model Miễn Phí

## 🎯 Chọn Phương Án Train

### ✅ KHUYÊN DÙNG: Google Colab (Miễn Phí + GPU)

**Ưu điểm**:
- ✅ Miễn phí 100%
- ✅ Có GPU (nhanh hơn 10-20x)
- ✅ Không cần cài gì
- ✅ Chạy được trên Raspberry Pi browser

**Làm thế nào?**

1. **Mở notebook**:
   - Upload file `Train_MobileNet_Colab.ipynb` lên https://colab.research.google.com
   - Hoặc click: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TruongThiMinhPhuong/System_Conveyor/blob/main/Train_MobileNet_Colab.ipynb)

2. **Đọc hướng dẫn chi tiết**:
   - Xem file: [`docs/TRAIN_WITH_COLAB_VI.md`](docs/TRAIN_WITH_COLAB_VI.md)

3. **Tóm tắt**:
   ```
   1. Chọn GPU (Runtime → Change runtime → T4 GPU)
   2. Upload dataset.zip
   3. Run All cells
   4. Đợi 15-20 phút
   5. Download mobilenet_classifier.tflite
   6. Copy to Raspberry Pi
   ```

---

### Option 2: Train Trên PC Windows

**Khi nào dùng?**
- Có PC mạnh
- Cần kiểm soát hoàn toàn
- Không có internet ổn định

**Làm thế nào?**

```powershell
# 1. Setup
cd d:\System_Conveyor
.\setup_pc.ps1

# 2. Chuẩn bị dataset
python training\mobilenet\prepare_data.py --source YOUR_IMAGES --output training/mobilenet/datasets/fruit_classification

# 3. Train nhanh
python quick_train.py

# Hoặc train thủ công
python training\mobilenet\train_mobilenet.py --dataset training/mobilenet/datasets/fruit_classification --epochs 50
```

**Chi tiết**: [`docs/COMPLETE_SETUP.md`](docs/COMPLETE_SETUP.md)

---

### ❌ KHÔNG Nên: Train Trên Raspberry Pi

**Tại sao?**
- Quá chậm (10-20 giờ)
- Dễ crash
- Thiếu RAM
- Cần cài TensorFlow nặng (2GB+)

---

## 📊 Dataset Yêu Cầu

### Tối Thiểu:
- 50 ảnh fresh
- 50 ảnh spoiled

### Khuyến Nghị:
- 200+ ảnh fresh
- 200+ ảnh spoiled
- Đa dạng góc độ, ánh sáng

### Cấu Trúc:
```
dataset/
├── train/
│   ├── fresh/
│   └── spoiled/
├── val/
│   ├── fresh/
│   └── spoiled/
└── test/
    ├── fresh/
    └── spoiled/
```

---

## 🎯 Sau Khi Train Xong

### 1. Copy Model to Raspberry Pi

```bash
scp mobilenet_classifier.tflite pi@192.168.137.177:~/System_Conveyor/models/
```

### 2. Chạy Hệ Thống

```bash
# Trên Raspberry Pi
cd ~/System_Conveyor
python3 fruit_sorter.py
```

### 3. Kiểm Tra Kết Quả

Mong đợi:
```
✅ MobileNetV2 model loaded
   Using XNNPACK delegate
⚡ FPS: 12.3
📊 Classified: Fresh (92.5%)
```

---

## 📚 Tài Liệu

| Tài liệu | Mô tả |
|----------|-------|
| [`TRAIN_WITH_COLAB_VI.md`](docs/TRAIN_WITH_COLAB_VI.md) | **Hướng dẫn chi tiết Google Colab** |
| [`COMPLETE_SETUP.md`](docs/COMPLETE_SETUP.md) | Setup PC & Raspberry Pi |
| [`QUICK_START_RPI_VI.md`](docs/QUICK_START_RPI_VI.md) | Quick start cho Pi |
| `Train_MobileNet_Colab.ipynb` | **Colab notebook sẵn sàng dùng** |

---

## ✅ Checklist

- [ ] Có dataset (50+ ảnh/loại)
- [ ] Đã chọn phương án train (Colab/PC)
- [ ] Model train xong (accuracy > 90%)
- [ ] File .tflite đã copy to Pi
- [ ] Hệ thống chạy thành công

---

**🎉 Good luck!**

# 📸 Hướng Dẫn Đưa Ảnh Vào Laptop Để Training

## Bước 1: Chọn Phương Án Thu Thập Ảnh

### ✅ **Phương Án 1: Chụp Bằng Webcam** (Nhanh nhất)

```bash
cd c:\Users\mgm\System_Conveyor
python data_collection_script.py
```

- Nhấn **SPACE** để chụp ảnh
- Nhấn **Q** để thoát
- Ảnh lưu tự động vào `raw_images/`

---

### 📱 **Phương Án 2: Từ Điện Thoại**

#### A. Qua USB Cable:
1. Kết nối điện thoại với laptop (cáp USB)
2. Chọn chế độ **"Transfer files"** trên điện thoại
3. Mở **File Explorer** → **This PC** → Tìm điện thoại
4. Copy ảnh từ `DCIM/Camera/` sang:
   ```
   c:\Users\mgm\System_Conveyor\raw_images\
   ```

#### B. Qua Google Drive:
1. Trên điện thoại: Upload ảnh lên Google Drive
2. Trên laptop: Download từ drive về `raw_images/`

---

### 🌐 **Phương Án 3: Download Dataset Có Sẵn**

```bash
# Từ Kaggle
pip install kaggle
kaggle datasets download -d moltean/fruits
unzip fruits.zip -d raw_images/
```

---

## Bước 2: Sắp Xếp Ảnh

Sau khi có ảnh trong `raw_images/`, chạy script:

```bash
python organize_images.py
```

**Chọn 1 trong 2 mode:**

### Mode 1: Automatic (Đã phân loại thủ công)
Nếu bạn đã tổ chức như sau:
```
raw_images/
  ├── fresh/
  │   ├── img1.jpg
  │   └── img2.jpg
  └── spoiled/
      ├── img1.jpg
      └── img2.jpg
```

→ Chọn **1** và script tự động chia thành train/valid/test

### Mode 2: Interactive (Phân loại từng ảnh)
Script hiển thị từng ảnh và bạn nhấn:
- **F** = Fresh (tươi)
- **S** = Spoiled (hỏng)
- **Q** = Quit

→ Script tự động lưu vào folder tương ứng

---

## Bước 3: Kiểm Tra Dataset

```bash
python dataset_quality_checker.py
```

Kiểm tra:
- ✅ Số lượng ảnh mỗi class
- ✅ Kích thước ảnh
- ✅ Format ảnh
- ✅ Ảnh bị lỗi

---

## Cấu Trúc Thư Mục Cuối Cùng

```
System_Conveyor/
├── raw_images/              # Ảnh gốc (backup)
│   ├── img1.jpg
│   └── img2.jpg
│
└── dataset/                 # Dataset để training
    ├── train/ (70%)
    │   ├── fresh/
    │   └── spoiled/
    ├── valid/ (20%)
    │   ├── fresh/
    │   └── spoiled/
    └── test/ (10%)
        ├── fresh/
        └── spoiled/
```

---

## Số Lượng Ảnh Khuyến Nghị

| Loại | Tối Thiểu | Khuyến Nghị | Tối Ưu |
|------|-----------|-------------|--------|
| Fresh | 200 | 500 | 1000+ |
| Spoiled | 200 | 500 | 1000+ |
| **Total** | **400** | **1000** | **2000+** |

---

## Lưu Ý Khi Chụp/Thu Thập Ảnh

### ✅ Nên:
- **Đa dạng góc độ**: trên, dưới, nghiêng, cận cảnh, xa
- **Đa dạng ánh sáng**: sáng, tối, đèn vàng, đèn trắng
- **Đa dạng nền**: trắng, đen, gỗ, vải
- **Đa dạng trạng thái**: vừa hỏng, hỏng nhiều, tươi mới, hơi héo

### ❌ Tránh:
- Ảnh mờ, nhòe
- Ảnh quá tối hoặc quá sáng
- Ảnh có nhiều vật thể khác
- Ảnh trùng lặp (copy paste)

---

## Troubleshooting

### Lỗi: "ModuleNotFoundError: No module named 'cv2'"
```bash
pip install opencv-python
```

### Lỗi: Script không mở camera
- Kiểm tra camera có kết nối không
- Đóng các app khác đang dùng camera (Zoom, Skype...)
- Thử camera khác

### Ảnh không được copy
- Kiểm tra đường dẫn folder có đúng không
- Kiểm tra quyền ghi file (Run as Administrator)

---

## Bước Tiếp Theo

Sau khi có dataset:

```bash
# 1. Kiểm tra dataset
python dataset_quality_checker.py

# 2. Train model
python quick_train.py

# Hoặc xem hướng dẫn đầy đủ
# Đọc file: TRAINING_DEPLOYMENT_GUIDE.md
```

---

✅ **Hoàn thành! Bây giờ bạn đã có dataset để train model**

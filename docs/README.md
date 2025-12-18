# 📚 TÀI LIỆU HƯỚNG DẪN - AI FRUIT SORTING SYSTEM

## 🎯 BẮT ĐẦU TỪ ĐÂU?

### 🚀 Quick Start - Chạy Ngay

| Mục đích | File cần đọc |
|----------|--------------|
| **Train model (Colab)** | [`HƯỚNG_DẪN_TRAIN.md`](../HƯỚNG_DẪN_TRAIN.md) ⭐ |
| **Chạy trên Raspberry Pi** | [`QUICK_START_RPI_VI.md`](QUICK_START_RPI_VI.md) |
| **Setup PC training** | [`COMPLETE_SETUP.md`](COMPLETE_SETUP.md) |

---

## 📖 CẤU TRÚC TÀI LIỆU

### 🌟 File Chính (Đọc Trước)

1. **[`HƯỚNG_DẪN_TRAIN.md`](../HƯỚNG_DẪN_TRAIN.md)** ⭐⭐⭐
   - **Mô tả**: Hướng dẫn HOÀN CHỈNH train model
   - **Nội dung**: Colab + PC training, dataset, deploy, troubleshooting
   - **Độ dài**: 1000+ dòng
   - **Cho ai**: Ai cũng nên đọc file này!

2. **[`QUICK_START_RPI_VI.md`](QUICK_START_RPI_VI.md)** ⭐⭐
   - **Mô tả**: Quick start cho Raspberry Pi
   - **Nội dung**: Copy files, run system
   - **Độ dài**: ~200 dòng
   - **Cho ai**: Deploy lên Pi

3. **[`TRAIN_README.md`](../TRAIN_README.md)** ⭐
   - **Mô tả**: Tổng quan các phương án training
   - **Nội dung**: So sánh Colab vs PC vs Pi
   - **Độ dài**: ~150 dòng
   - **Cho ai**: Chưa quyết định train thế nào

---

### 📚 Files Chi Tiết (Đọc Khi Cần)

4. **[`TRAIN_WITH_COLAB_VI.md`](TRAIN_WITH_COLAB_VI.md)**
   - **Mô tả**: Chi tiết Google Colab training
   - **Nội dung**: Step-by-step Colab, screenshots guide
   - **Độ dài**: ~500 dòng
   - **Cho ai**: Muốn biết chi tiết về Colab

5. **[`TRAIN_RASPI_COLAB.md`](TRAIN_RASPI_COLAB.md)**
   - **Mô tả**: Train Colab từ Raspberry Pi browser
   - **Nội dung**: 7 bước siêu cụ thể
   - **Độ dài**: ~400 dòng
   - **Cho ai**: Train trực tiếp trên Pi qua Colab

6. **[`COMPLETE_SETUP.md`](COMPLETE_SETUP.md)**
   - **Mô tả**: Setup toàn bộ hệ thống
   - **Nội dung**: PC setup, Pi setup, deployment
   - **Độ dài**: ~600 dòng
   - **Cho ai**: First-time setup hoàn chỉnh

7. **[`SYSTEM_SETUP.md`](SYSTEM_SETUP.md)**
   - **Mô tả**: System architecture và setup
   - **Nội dung**: Hardware requirements, installation
   - **Độ dài**: ~300 dòng
   - **Cho ai**: Hiểu kiến trúc hệ thống

8. **[`FRESH_SPOILED_FIX.md`](../docs/FRESH_SPOILED_FIX.md)**
   - **Mô tả**: Performance fixes summary
   - **Nội dung**: All optimizations applied
   - **Độ dài**: ~100 dòng
   - **Cho ai**: Developers

---

### 🛠️ Files Kỹ Thuật

9. **[`../setup_pc.ps1`](../setup_pc.ps1)**
   - **Mô tả**: PowerShell script setup PC
   - **Cho ai**: Windows users

10. **[`../setup_rpi.sh`](../setup_rpi.sh)**
    - **Mô tả**: Bash script setup Raspberry Pi
    - **Cho ai**: Pi users

11. **[`../start_colab_training.sh`](../start_colab_training.sh)**
    - **Mô tả**: Script tự động mở Colab
    - **Cho ai**: Pi users train Colab

12. **[`../quick_train.py`](../quick_train.py)**
    - **Mô tả**: Python script train nhanh
    - **Cho ai**: PC users

---

## 🎯 ROADMAP ĐỌC TÀI LIỆU

### 🌱 Người Mới Bắt Đầu

```
1. Đọc HƯỚNG_DẪN_TRAIN.md (overview)
   ↓
2. Chọn phương án:
   - Colab → TRAIN_WITH_COLAB_VI.md
   - PC → COMPLETE_SETUP.md (phần PC)
   ↓
3. Train xong → QUICK_START_RPI_VI.md
   ↓
4. Deploy & Run!
```

### 🚀 Người Có Kinh Nghiệm

```
1. TRAIN_README.md (chọn phương án)
   ↓
2. Chạy scripts:
   - PC: setup_pc.ps1 → quick_train.py
   - Pi: start_colab_training.sh
   ↓
3. Deploy & Done!
```

### 🔧 Developer

```
1. COMPLETE_SETUP.md (architecture)
   ↓
2. SYSTEM_SETUP.md (technical details)
   ↓
3. FRESH_SPOILED_FIX.md (optimizations)
   ↓
4. Code modifications
```

---

## 📝 CHEAT SHEET

### Train Model (Fastest)

```bash
# Colab (Pi hoặc PC)
./start_colab_training.sh
# Follow browser instructions

# PC
.\setup_pc.ps1
python quick_train.py
```

### Deploy to Pi

```bash
scp mobilenet_classifier.tflite pi@192.168.137.177:~/System_Conveyor/models/
```

### Run System

```bash
cd ~/System_Conveyor
python3 fruit_sorter.py
```

---

## 🆘 TÌM GÌ Ở ĐÂU?

| Câu hỏi | File |
|---------|------|
| Làm sao train model? | `HƯỚNG_DẪN_TRAIN.md` |
| Train trên Pi được không? | `HƯỚNG_DẪN_TRAIN.md` → Phương Án 1 (Colab) |
| Cần bao nhiêu ảnh? | `HƯỚNG_DẪN_TRAIN.md` → Chuẩn Bị Dataset |
| Lỗi Colab "No GPU"? | `HƯỚNG_DẪN_TRAIN.md` → Xử Lý Lỗi |
| Deploy model thế nào? | `HƯỚNG_DẪN_TRAIN.md` → Deploy Model |
| Chạy hệ thống ra sao? | `QUICK_START_RPI_VI.md` |
| Setup lần đầu? | `COMPLETE_SETUP.md` |
| Accuracy thấp? | `HƯỚNG_DẪN_TRAIN.md` → Tips & Tricks |
| FPS thấp? | `FRESH_SPOILED_FIX.md` |
| Hardware setup? | `SYSTEM_SETUP.md` |

---

## 📂 CẤU TRÚC THƯ MỤC

```
System_Conveyor/
├── docs/                           ← BẠN ĐANG Ở ĐÂY
│   ├── README.md                   ← File này
│   ├── TRAIN_WITH_COLAB_VI.md      Chi tiết Colab
│   ├── TRAIN_RASPI_COLAB.md        Train từ Pi
│   ├── COMPLETE_SETUP.md           Setup đầy đủ
│   ├── QUICK_START_RPI_VI.md       Quick start Pi
│   ├── SYSTEM_SETUP.md             System architecture
│   └── FRESH_SPOILED_FIX.md        Performance fixes
│
├── HƯỚNG_DẪN_TRAIN.md             ⭐ MAIN GUIDE
├── TRAIN_README.md                 Training overview
├── Train_MobileNet_Colab.ipynb     Colab notebook
│
├── setup_pc.ps1                    PC setup script
├── setup_rpi.sh                    Pi setup script
├── start_colab_training.sh         Open Colab script
├── quick_train.py                  Quick train script
│
└── ...                             (code files)
```

---

## 💡 RECOMMENDATIONS

### Đọc File Nào?

**Nếu bạn**:

- 🆕 **Mới bắt đầu**: Đọc [`HƯỚNG_DẪN_TRAIN.md`](../HƯỚNG_DẪN_TRAIN.md) từ đầu đến cuối
- ⚡ **Muốn nhanh**: Đọc [`TRAIN_README.md`](../TRAIN_README.md) → Chọn phương án
- 🥧 **Dùng Pi**: Đọc [`TRAIN_RASPI_COLAB.md`](TRAIN_RASPI_COLAB.md)
- 🖥️ **Dùng PC**: Đọc [`COMPLETE_SETUP.md`](COMPLETE_SETUP.md) phần PC
- 🔧 **Đã train xong**: Đọc [`QUICK_START_RPI_VI.md`](QUICK_START_RPI_VI.md)

### Files Nào Quan Trọng Nhất?

1. ⭐⭐⭐ `HƯỚNG_DẪN_TRAIN.md` - **MUST READ**
2. ⭐⭐ `QUICK_START_RPI_VI.md` - Deploy
3. ⭐ `TRAIN_README.md` - Quick overview

Các files khác đọc khi cần chi tiết!

---

## 🎉 TÓM LẠI

- ✅ **Tất cả files đều quan trọng** - Mỗi file phục vụ 1 mục đích
- ✅ **Bắt đầu từ HƯỚNG_DẪN_TRAIN.md** - Complete guide
- ✅ **Files khác là chi tiết** - Đọc khi cần thêm info
- ✅ **README này giúp điều hướng** - Biết đọc file nào

**Happy training! 🚀**

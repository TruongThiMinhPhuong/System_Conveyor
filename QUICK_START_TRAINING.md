# Quick Start: Train MobileNetV2

## Trước khi Train

### 1. Chuẩn Bị Dataset

**Cấu trúc cần thiết:**
```
System_Conveyor/
└── data/
    └── fruits/
        ├── train/
        │   ├── fresh/      # Ít nhất 100 ảnh trái cây tươi
        │   └── spoiled/    # Ít nhất 100 ảnh trái cây hỏng
        └── validation/
            ├── fresh/      # Ít nhất 20 ảnh
            └── spoiled/    # Ít nhất 20 ảnh
```

**Lấy dataset:**
- Thu thập: Xem [`HUONG_DAN_NHAP_ANH.md`](file:///c:/Users/mgm/System_Conveyor/HUONG_DAN_NHAP_ANH.md)
- Download mẫu: Kaggle Fruit dataset
- Sử dụng dataset có sẵn

### 2. Cài Đặt Dependencies

```bash
pip install tensorflow numpy matplotlib pillow
```

---

## Bắt Đầu Training

### Option 1: Automated Script (KHUYẾN NGHỊ)

```bash
cd C:\Users\mgm\System_Conveyor
python start_training.py
```

**Script tự động:**
- ✅ Check dataset structure
- ✅ Check dependencies
- ✅ Train model
- ✅ Export to TFLite
- ✅ Verify models
- ✅ Show next steps

### Option 2: Manual Training

```bash
cd training/mobilenet

python train_mobilenet.py \
  --dataset ../../data/fruits \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --dropout 0.5
```

---

## Training Configuration

| Parameter | Value | Note |
|-----------|-------|------|
| **Epochs** | 50 | Tăng nếu chưa converge |
| **Batch Size** | 32 | Giảm nếu out of memory |
| **Learning Rate** | 0.001 | Adam optimizer |
| **Image Size** | 224x224 | MobileNetV2 standard |
| **Dropout** | 0.5 | Prevent overfitting |
| **Base Model** | MobileNetV2 | Pretrained, frozen |

---

## Evaluation Criteria

### ✅ Good Model
- **Validation Accuracy > 85%**
- **Train acc - Val acc < 5%** (no overfitting)
- **Loss decreasing smoothly**

### ⚠️ Overfitting Signs
- Train acc >> Val acc (gap > 5%)
- Training loss decreases, validation loss increases

**Solutions:**
1. Increase dropout: 0.3 → 0.5
2. Add more data augmentation
3. Collect more training data
4. Early stopping (already enabled)

### ❌ Poor Model
- **Validation Accuracy < 80%**
- **Loss not decreasing**

**Solutions:**
1. Check dataset quality
2. Increase epochs
3. Adjust learning rate
4. Verify data labeling

---

## After Training

### 1. Review Results

**Check training plots:**
```
training/mobilenet/mobilenet_training/training_history.png
```

**Verify files:**
```
training/mobilenet/mobilenet_training/
├── mobilenet_fruit_classifier.h5     # Keras model (~14 MB)
└── training_history.png              # Training curves

models/
└── mobilenet_classifier.tflite       # TFLite model (~4 MB)
```

### 2. Export to TFLite (if not done automatically)

```bash
cd training/mobilenet
python export_tflite.py
```

**Benefits:**
- Size: 14 MB → 4 MB (3.5x smaller)
- Speed: 2-3x faster on Pi
- Accuracy loss: < 1%

### 3. Test Model Locally

```bash
cd training/mobilenet
python evaluate_model.py --test-data ../../data/fruits/validation
```

---

## Transfer to Raspberry Pi

### Method 1: Deployment Package (Recommended)

```bash
# Create package
python prepare_for_pi.py

# Copy to Pi
scp conveyor_pi_deploy.zip pi@192.168.137.177:~/

# On Pi
cd ~
unzip conveyor_pi_deploy.zip
cd System_Conveyor
python3 fruit_sorter.py
```

### Method 2: Copy Model Only

```bash
# Copy just the TFLite model
scp models/mobilenet_classifier.tflite pi@192.168.137.177:~/System_Conveyor/models/
```

---

## Test on Raspberry Pi

### CLI Mode

```bash
cd ~/System_Conveyor
python3 fruit_sorter.py
```

### Web Interface

```bash
python3 run_web.py
# Access: http://<pi_ip>:5001
```

---

## Troubleshooting

### Error: Out of Memory

**Solution:** Reduce batch size
```bash
python train_mobilenet.py --batch-size 16
```

### Error: Dataset not found

**Solution:** Check paths
```bash
ls -la data/fruits/train/fresh/
ls -la data/fruits/train/spoiled/
```

### Error: TensorFlow not installed

**Solution:**
```bash
pip install tensorflow
# Or for specific version
pip install tensorflow==2.13.0
```

### Training too slow

**Solutions:**
1. Use GPU if available
2. Reduce epochs: `--epochs 30`
3. Use smaller batch: `--batch-size 16`
4. Train on Google Colab (faster)

### Model accuracy too low

**Solutions:**
1. Check data quality
2. Increase training data
3. Adjust data augmentation
4. Unfreeze some layers
5. Increase epochs

---

## Expected Results

### Training Time
- **CPU only:** 2-4 hours (50 epochs)
- **GPU:** 20-40 minutes (50 epochs)
- **Google Colab (free GPU):** 15-30 minutes

### Model Performance
- **Accuracy:** 85-95%
- **Inference time (Pi):** 50-100 ms
- **Model size:** ~4 MB (TFLite)

---

## Next Steps

After successful training:

1. ✅ **Verify model**: Check accuracy > 85%
2. ✅ **Export TFLite**: Optimize for Pi
3. ✅ **Transfer to Pi**: Use deployment package
4. ✅ **Integrate**: Test full system
5. ✅ **Calibrate**: Fine-tune thresholds
6. ✅ **Deploy**: Production ready!

---

## Resources

- **Training Guide:** [`HƯỚNG_DẪN_TRAIN.md`](file:///c:/Users/mgm/System_Conveyor/HƯỚNG_DẪN_TRAIN.md)
- **Data Collection:** [`HUONG_DAN_NHAP_ANH.md`](file:///c:/Users/mgm/System_Conveyor/HUONG_DAN_NHAP_ANH.md)
- **Deployment:** [`HUONG_DAN_DEPLOY_PI.md`](file:///c:/Users/mgm/System_Conveyor/HUONG_DAN_DEPLOY_PI.md)
- **Colab Notebook:** [`Train_MobileNet_Colab.ipynb`](file:///c:/Users/mgm/System_Conveyor/Train_MobileNet_Colab.ipynb)

---

## Quick Command Reference

```bash
# Start training (automated)
python start_training.py

# Manual training
cd training/mobilenet
python train_mobilenet.py --dataset ../../data/fruits

# Export to TFLite
python export_tflite.py

# Evaluate model
python evaluate_model.py --test-data ../../data/fruits/validation

# Create deployment package
python prepare_for_pi.py

# Transfer to Pi
scp conveyor_pi_deploy.zip pi@<ip>:~/
```

Good luck with training! 🚀

# Training Guide

Complete guide for training YOLOv8 and MobileNetV2 models for the fruit sorting system.

> **💡 LƯU Ý:** Hướng dẫn này dành cho training trên **PC/Laptop có GPU**.  
> Nếu muốn train **TRỰC TIẾP trên Raspberry Pi 4**, xem [TRAINING_ON_PI.md](TRAINING_ON_PI.md) ⚡

## Overview

The system uses two AI models:
1. **YOLOv8-nano**: Detects fruits on conveyor belt
2. **MobileNetV2**: Classifies fruits as fresh or spoiled

Both models must be trained on a **PC/Laptop with GPU** (not on Raspberry Pi).

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with 4GB+ VRAM (recommended)
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ free space

### Software Requirements
```bash
# Python 3.8+
python --version

# CUDA Toolkit (for GPU training)
nvidia-smi
```

## Step 1: Setup Training Environment

### Create Conda Environment
```bash
conda create -n fruit_training python=3.9
conda activate fruit_training
```

### Install Dependencies
```bash
# PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# TensorFlow
pip install tensorflow

# Training tools
pip install ultralytics
pip install matplotlib scikit-learn
pip install pillow opencv-python

# Annotation tool
pip install labelImg
```

## Step 2: Data Collection

### Collect Images on Raspberry Pi

```bash
cd ~/System_Conveyor
source venv/bin/activate

python training/data_collection/collect_images.py \
    --mode classification \
    --count 200 \
    --interval 2.0 \
    --output ./raw_images
```

**Tips for Good Dataset**:
- **Lighting**: Vary lighting conditions (bright, dim, natural light)
- **Angles**: Capture from different angles
- **Backgrounds**: Mix backgrounds (belt, table, hand)
- **Variety**: Different fruit sizes, colors, ripeness stages
- **Damage types**: Bruises, mold, discoloration, soft spots

### Transfer to PC
```bash
# On PC
scp -r pi@fruit-sorter.local:~/System_Conveyor/raw_images ./System_Conveyor/
```

## Step 3: Train YOLO Detection Model

### Prepare YOLO Dataset

1. **Organize Images**:
```
datasets/fruit_detection/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

2. **Annotate with LabelImg**:
```bash
labelImg
```

**Annotation Steps**:
- Open Dir → Select `images/train`
- Change Save Dir → Select `labels/train`
- Select **YOLO** format (important!)
- Draw box around each fruit
- Assign class (apple, orange, banana, etc.)
- Save (creates .txt file)
- Repeat for all images

**YOLO Label Format**:
```
<class_id> <x_center> <y_center> <width> <height>
```
All values normalized to [0, 1]

3. **Update dataset.yaml**:
```yaml
path: ./datasets/fruit_detection
train: images/train
val: images/val

nc: 3  # Number of classes

names:
  0: apple
  1: orange
  2: banana
```

### Train YOLO
```bash
cd training/yolo

python train_yolo.py \
    --data ./dataset.yaml \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --device 0
```

**Training Parameters**:
- `--epochs`: 100-200 recommended
- `--batch`: Adjust based on GPU memory (16, 32)
- `--imgsz`: 640 default (use 416 for faster inference)
- `--device`: 0 for GPU, cpu for CPU

**Expected Time**: 1-3 hours on GPU

### Evaluate YOLO
```bash
python -c "
from ultralytics import YOLO
model = YOLO('fruit_detection/yolov8n_fruit/weights/best.pt')
metrics = model.val()
print(f'mAP50: {metrics.box.map50:.3f}')
print(f'mAP50-95: {metrics.box.map:.3f}')
"
```

**Target Metrics**:
- mAP50 > 0.70
- Precision > 0.75
- Recall > 0.70

## Step 4: Train MobileNetV2 Classification

### Prepare Classification Dataset

```bash
cd training/mobilenet

python prepare_data.py \
    --source ../../raw_images \
    --output ./datasets/fruit_classification \
    --train-split 0.7 \
    --val-split 0.15 \
    --test-split 0.15
```

Creates:
```
datasets/fruit_classification/
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

### Train MobileNetV2
```bash
python train_mobilenet.py \
    --dataset ./datasets/fruit_classification \
    --epochs 50 \
    --batch 32 \
    --lr 0.001 \
    --image-size 224 \
    --output ./mobilenet_training
```

**Training Parameters**:
- `--epochs`: 50-100 recommended
- `--batch`: 32 default (adjust for GPU)
- `--lr`: 0.001 (Adam optimizer)
- `--image-size`: 224 (MobileNetV2 default)

**Expected Time**: 30-60 minutes on GPU

### Monitor Training
Training produces:
- `mobilenet_training/mobilenet_fruit_classifier_best.keras` - Best model
- `mobilenet_training/training_history.png` - Training curves

Check for:
- **No overfitting**: Val accuracy follows train accuracy
- **Converged**: Loss plateaus
- **Good accuracy**: Val accuracy > 90%

### Export to TensorFlow Lite
```bash
python export_tflite.py \
    --model ./mobilenet_training/mobilenet_fruit_classifier_best.keras \
    --output ../../models/mobilenet_classifier.tflite \
    --quantize
```

**Quantization Benefits**:
- Smaller model size (5-10MB → 2-3MB)
- Faster inference on Raspberry Pi
- Minimal accuracy loss (<1%)

## Step 5: Test Models Locally

### Test YOLO
```python
from ultralytics import YOLO
import cv2

model = YOLO('../../models/yolov8n_fruit.pt')
img = cv2.imread('test_image.jpg')

results = model(img)
results[0].show()  # Display detections
```

### Test MobileNetV2
```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load TFLite model
interpreter = tf.lite.Interpreter('../../models/mobilenet_classifier.tflite')
interpreter.allocate_tensors()

# Test image
img = Image.open('test_fruit.jpg').resize((224, 224))
img_array = np.array(img, dtype=np.float32) / 255.0
img_array = np.expand_dims(img_array, 0)

# Inference
interpreter.set_tensor(interpreter.get_input_details()[0]['index'], img_array)
interpreter.invoke()
output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

print(f"Fresh: {output[0][0]:.2%}, Spoiled: {output[0][1]:.2%}")
```

## Step 6: Deploy to Raspberry Pi

### Transfer Models
```bash
# Copy to Raspberry Pi
scp models/yolov8n_fruit.pt pi@fruit-sorter.local:~/System_Conveyor/models/
scp models/mobilenet_classifier.tflite pi@fruit-sorter.local:~/System_Conveyor/models/
```

### Verify on Raspberry Pi
```bash
# SSH to Raspberry Pi
ssh pi@fruit-sorter.local

cd ~/System_Conveyor
source venv/bin/activate

# Test YOLO
python ai_models/yolo_detector.py

# Test MobileNetV2
python ai_models/mobilenet_classifier.py
```

## Improving Model Accuracy

### If Low YOLO mAP (<0.5)
1. **More data**: Collect 500+ images
2. **Better annotations**: Review and fix labels
3. **Longer training**: Increase epochs
4. **Data augmentation**: Already enabled in script
5. **Smaller input size**: Try 416 instead of 640

### If Low Classification Accuracy (<85%)
1. **More balanced data**: Equal fresh/spoiled samples
2. **Better image quality**: Ensure good lighting
3. **More variety**: Different damage types
4. **Data augmentation**: Check augmentation.py settings
5. **Longer training**: Increase epochs to 100

### Fine-Tuning Tips
- **Learning rate**: Try 0.0001 if overfitting
- **Batch size**: Reduce if GPU memory errors
- **Dropout**: Increase to 0.4-0.5 if overfitting
- **Freeze fewer layers**: Unfreeze last few layers of base model

## Dataset Guidelines

### Minimum Dataset Sizes
- **YOLO Detection**: 200+ images, 500-1000 ideal
- **Classification**: 100+ per class, 300+ ideal

### Quality over Quantity
- ✅ Well-lit, clear images
- ✅ Varied conditions
- ✅ Accurate labels
- ❌ Blurry or dark images
- ❌ Incorrect labels

### Class Balance
Ensure roughly equal samples per class:
- Fresh apples: 150
- Spoiled apples: 145
- ✅ Balanced!

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python train_yolo.py --batch 8
python train_mobilenet.py --batch 16
```

### Model Not Learning
- Check data loading (print sample batch)
- Verify labels are correct
- Try lower learning rate
- Increase epochs

### Poor Performance on Raspberry Pi
- Use smaller YOLO input (416)
- Apply more quantization
- Reduce inference frequency

## Next Steps

1. ✅ Deploy models to Raspberry Pi
2. ✅ Test on real conveyor system
3. ✅ Fine-tune thresholds in config.py
4. ✅ Monitor performance and retrain if needed

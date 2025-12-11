# YOLOv8 Training Guide

## Overview
Train YOLOv8-nano model to detect fruits on the conveyor belt.

## Prerequisites
- Python 3.8+
- GPU with CUDA support (recommended)
- At least 100-200 images of fruits

## Installation (Training PC)

```bash
pip install torch torchvision
pip install ultralytics
pip install labelImg  # For annotation
```

## Dataset Preparation

### 1. Collect Images
- Capture images of fruits using Raspberry Pi camera or regular camera
- Vary lighting conditions, angles, backgrounds
- Include different fruit types you want to detect

### 2. Annotation with LabelImg

Install and run LabelImg:
```bash
pip install labelImg
labelImg
```

**Annotation Steps:**
1. Open LabelImg
2. Select "Open Dir" → Choose your images directory
3. Select "Change Save Dir" → Choose labels directory
4. Select "YOLO" format (not PascalVOC!)
5. For each image:
   - Draw bounding box around fruit
   - Enter class name (e.g., "apple", "orange")
   - Save (creates .txt file)

### 3. Organize Dataset

Create directory structure:
```
datasets/fruit_detection/
├── images/
│   ├── train/      # 80% of images
│   ├── val/        # 15% of images
│   └── test/       # 5% of images
└── labels/
    ├── train/      # Corresponding labels
    ├── val/
    └── test/
```

### 4. Update dataset.yaml

Edit `dataset.yaml`:
```yaml
path: ./datasets/fruit_detection
train: images/train
val: images/val

nc: 3  # Number of classes

names:
  0: apple
  1: orange
  2: banana  # Add your fruit classes
```

## Training

### Basic Training
```bash
cd training/yolo
python train_yolo.py --data ./dataset.yaml --epochs 100
```

### Advanced Training Options
```bash
python train_yolo.py \
    --data ./dataset.yaml \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --device 0 \
    --project fruit_detection \
    --name yolov8n_v1
```

**Parameters:**
- `--epochs`: Training iterations (100-200 recommended)
- `--batch`: Batch size (adjust based on GPU memory)
- `--imgsz`: Image size (640 default, 416 for faster inference)
- `--device`: GPU ID (0) or 'cpu'

## Monitoring Training

Training outputs will be saved in `fruit_detection/yolov8n_fruit/`:
- `weights/best.pt` - Best model weights
- `weights/last.pt` - Last epoch weights
- `results.png` - Training curves
- `confusion_matrix.png` - Confusion matrix

## Evaluation

The script automatically validates and shows:
- **mAP50**: Mean Average Precision at IoU 0.5
- **mAP50-95**: mAP across IoU thresholds
- **Precision**: Correct detections / total detections
- **Recall**: Detected objects / total ground truth

**Target Metrics:**
- mAP50 > 0.7 (Good)
- Precision > 0.75
- Recall > 0.70

## Model Export

After training, the best model is automatically copied to:
```
models/yolov8n_fruit.pt
```

## Deploy to Raspberry Pi

1. Copy model file:
```bash
scp models/yolov8n_fruit.pt pi@<raspberry-pi-ip>:~/System_Conveyor/models/
```

2. Test on Raspberry Pi:
```bash
python -c "from ai_models.yolo_detector import YOLODetector; detector = YOLODetector(); detector.load_model(); detector.test()"
```

## Troubleshooting

**Low mAP (<0.5):**
- Add more training images
- Improve annotation quality
- Increase training epochs
- Try data augmentation

**Overfitting:**
- Add more diverse images
- Use data augmentation
- Reduce model complexity

**Slow inference on Raspberry Pi:**
- Reduce `imgsz` to 416 or 320
- Use confidence threshold 0.6+
- Optimize image preprocessing

## Tips

1. **Quality over quantity**: Better to have 200 well-annotated images than 1000 poor ones
2. **Balanced dataset**: Equal number of each fruit class
3. **Varied conditions**: Different lighting, angles, backgrounds
4. **Regular testing**: Test on Raspberry Pi frequently

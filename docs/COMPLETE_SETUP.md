# Complete Setup and Deployment Guide

## 🎯 Overview

This guide covers the **complete setup** for the AI Fruit Sorting System:
- **PC**: Training environment (Windows)
- **Raspberry Pi**: Deployment environment (inference only)

---

## 🖥️ Part 1: PC Setup (Training Environment)

### Prerequisites
- Windows 10/11
- Python 3.8+
- 8GB+ RAM recommended
- Internet connection

### Quick Setup

```powershell
# Run the automated setup script
cd d:\System_Conveyor
.\setup_pc.ps1
```

This will:
- ✅ Install TensorFlow and training dependencies
- ✅ Create necessary directories
- ✅ Verify installation
- ✅ Check for dataset

### Manual Setup (if script fails)

```powershell
# 1. Install Python packages
pip install tensorflow opencv-python numpy matplotlib scikit-learn seaborn ultralytics Pillow

# 2. Verify TensorFlow
python -c "import tensorflow as tf; print(tf.__version__)"

# 3. Create directories
mkdir models, logs, data
mkdir training\mobilenet\datasets
mkdir training\mobilenet\mobilenet_training
```

---

## 📊 Part 2: Prepare Dataset

### Option A: Use Existing Images

If you have images organized in folders:

```powershell
cd training\mobilenet

# Prepare dataset
python prepare_data.py --source <path_to_your_images> --output ./datasets/fruit_classification --verify
```

**Expected source structure**:
```
your_images/
├── fresh/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── spoiled/
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

### Option B: Collect New Images

```powershell
# Use data collection script
cd training\data_collection
python collect_images.py
```

### Dataset Requirements

- **Minimum**: 50 images per class (Fresh/Spoiled)
- **Recommended**: 200+ images per class
- **Format**: JPG, PNG
- **Quality**: Clear, well-lit images

---

## 🚀 Part 3: Train the Model

### Quick Training (Automated)

```powershell
# Use the quick training script
cd d:\System_Conveyor
python quick_train.py
```

This automates:
1. Training
2. Evaluation
3. TFLite conversion
4. Deployment to Raspberry Pi (optional)

### Manual Training

```powershell
cd training\mobilenet

# 1. Train
python train_mobilenet.py --dataset ./datasets/fruit_classification --epochs 50

# 2. Evaluate
python evaluate_model.py --model ./mobilenet_training/mobilenet_fruit_classifier_best.keras

# 3. Convert to TFLite
python export_tflite.py --model ./mobilenet_training/mobilenet_fruit_classifier_best.keras --output ../../models/mobilenet_classifier.tflite
```

### Expected Results

After training, you should see:
- ✅ Accuracy: 90-95%
- ✅ F1 Score: > 90%
- ✅ Confusion matrix showing good separation
- ✅ TFLite model: `models/mobilenet_classifier.tflite`

---

## 🥧 Part 4: Raspberry Pi Setup

### 1. Transfer Files

From your PC:

```powershell
# Copy the entire project
scp -r d:\System_Conveyor pi@192.168.137.177:~/

# Or just copy the setup script first
scp d:\System_Conveyor\setup_rpi.sh pi@192.168.137.177:~/
```

### 2. Run Setup on Raspberry Pi

```bash
# SSH to Raspberry Pi
ssh pi@192.168.137.177

# Run setup
cd ~/System_Conveyor
chmod +x setup_rpi.sh
./setup_rpi.sh
```

This will:
- ✅ Install TFLite Runtime (lightweight, no TensorFlow)
- ✅ Install OpenCV, NumPy
- ✅ Install hardware libraries (GPIO, PiCamera)
- ✅ Create directories
- ✅ Verify hardware

### 3. Copy Trained Models

```powershell
# From PC, copy the TFLite model
scp models\mobilenet_classifier.tflite pi@192.168.137.177:~/System_Conveyor/models/

# Copy YOLO model (if you have it)
scp models\yolov8n_fruit.pt pi@192.168.137.177:~/System_Conveyor/models/
```

---

## ✅ Part 5: Verification

### On PC

```powershell
# Test preprocessing
python training\mobilenet\debug_preprocessing.py --image <test_image.jpg>

# Test evaluation
python training\mobilenet\evaluate_model.py --model training\mobilenet\mobilenet_training\mobilenet_fruit_classifier_best.keras
```

### On Raspberry Pi

```bash
cd ~/System_Conveyor

# Test AI models
python3 -c "from ai_models import MobileNetClassifier; m = MobileNetClassifier(); m.load_model(); print('MobileNet OK')"

# Test hardware
python3 -c "from hardware import ConveyorSystem; print('Hardware imports OK')"

# Run the system
python3 fruit_sorter.py
```

---

## 🎮 Part 6: Running the System

### Start Main System

```bash
# On Raspberry Pi
cd ~/System_Conveyor
python3 fruit_sorter.py
```

**Expected output**:
```
🍎 Development of a Conveyor System for Fruit Quality Classification Using AI Camera
============================================================
✅ MobileNetV2 model loaded successfully
   Using XNNPACK delegate  # Hardware acceleration
✅ System initialized successfully!
🚀 Starting main system loop...

⚡ FPS: 12.3
⏱️ YOLO: 45ms | MobileNet: 28ms | Preprocessing: 10ms
```

### Start Web Interface

```bash
# In a separate terminal
cd ~/System_Conveyor
python3 run_web.py
```

Access at: `http://192.168.137.177:5000`

---

## 🔧 Troubleshooting

### PC Issues

#### TensorFlow Installation Failed
```powershell
# Try installing specific version
pip install tensorflow==2.13.0

# Or use conda
conda install tensorflow
```

#### Out of Memory During Training
```powershell
# Reduce batch size
python train_mobilenet.py --batch 16  # Instead of 32
```

### Raspberry Pi Issues

#### TFLite Runtime Not Found
```bash
# Install from Google Coral repo
pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
```

#### XNNPACK Acceleration Not Working
```bash
# Check if ARM NEON is supported
cat /proc/cpuinfo | grep -i neon

# If not available, system will fall back to CPU (slower but works)
```

#### Camera Not Detected
```bash
# Enable camera in raspi-config
sudo raspi-config
# Navigate to: Interface Options > Camera > Enable

# Test camera
libcamera-hello
```

#### GPIO Permission Denied
```bash
# Add user to gpio group
sudo usermod -a -G gpio pi
sudo reboot
```

#### Low FPS (<8 FPS)
```python
# Edit utils/config.py
CAMERA_RESOLUTION = (320, 320)  # Reduce resolution
YOLO_INPUT_SIZE = 320
APPLY_BLUR = False  # Disable blur
```

---

## 📋 Quick Reference

### PC Commands (Windows)

| Task | Command |
|------|---------|
| Setup environment | `.\setup_pc.ps1` |
| Quick train | `python quick_train.py` |
| Manual train | `python training\mobilenet\train_mobilenet.py --dataset <path> --epochs 50` |
| Evaluate model | `python training\mobilenet\evaluate_model.py --model <path>` |
| Convert to TFLite | `python training\mobilenet\export_tflite.py --model <path>` |
| Deploy model | `scp models\*.tflite pi@192.168.137.177:~/System_Conveyor/models/` |

### Raspberry Pi Commands

| Task | Command |
|------|---------|
| Setup environment | `./setup_rpi.sh` |
| Run system | `python3 fruit_sorter.py` |
| Run web interface | `python3 run_web.py` |
| Test models | `python3 -c "from ai_models import *"` |
| Check performance | Monitor logs in console |

---

## 🎯 Complete Workflow Summary

### First Time Setup

1. **PC**:
   ```powershell
   cd d:\System_Conveyor
   .\setup_pc.ps1
   python quick_train.py
   ```

2. **Raspberry Pi**:
   ```bash
   ssh pi@192.168.137.177
   cd ~/System_Conveyor
   ./setup_rpi.sh
   ```

3. **Deploy**:
   ```powershell
   scp models\mobilenet_classifier.tflite pi@192.168.137.177:~/System_Conveyor/models/
   ```

4. **Run**:
   ```bash
   ssh pi@192.168.137.177
   cd ~/System_Conveyor
   python3 fruit_sorter.py
   ```

### Retraining Workflow

When you need to retrain:

1. **PC**: Collect more data
2. **PC**: `python quick_train.py`
3. **PC**: Deploy new model
4. **Raspberry Pi**: Restart system

---

## ✅ Checklist

### PC Setup
- [ ] Python 3.8+ installed
- [ ] TensorFlow installed and working
- [ ] Dataset prepared (50+ images per class)
- [ ] Model trained successfully
- [ ] TFLite model generated
- [ ] Evaluation shows >90% accuracy

### Raspberry Pi Setup
- [ ] TFLite Runtime installed
- [ ] All dependencies installed
- [ ] Camera enabled and working
- [ ] GPIO accessible
- [ ] Models copied to Raspberry Pi
- [ ] System runs without errors
- [ ] FPS > 10

### System Verification
- [ ] Fruits detected correctly
- [ ] Classification accuracy >85%
- [ ] Servo responds to classifications
- [ ] Web interface accessible
- [ ] Performance monitoring working

---

**🎉 Everything should now be working!**

For issues or questions, check the troubleshooting section above.

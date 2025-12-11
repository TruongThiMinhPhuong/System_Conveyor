# Software Setup Guide

Complete software installation and configuration for Raspberry Pi 4.

## Prerequisites

- ✅ Raspberry Pi 4 (8GB RAM recommended)
- ✅ MicroSD Card (32GB+, Class 10)
- ✅ Hardware assembled per [Hardware Setup Guide](hardware_setup.md)
- ✅ PC/Laptop for model training (with GPU recommended)

## Step 1: Install Raspberry Pi OS

### Download Raspberry Pi Imager
- Download from: https://www.raspberrypi.com/software/
- Install on your PC

### Flash OS
1. Open Raspberry Pi Imager
2. Choose **Raspberry Pi OS (64-bit)** - Full version
3. Select your microSD card
4. Click **Settings** (gear icon):
   - Set hostname: `fruit-sorter`
   - Enable SSH
   - Set username/password
   - Configure WiFi (optional)
5. Write OS to SD card

### First Boot
1. Insert SD card into Raspberry Pi
2. Connect monitor, keyboard, mouse
3. Power on and complete setup wizard
4. Update system:
```bash
sudo apt update
sudo apt upgrade -y
```

## Step 2: Enable Camera and GPIO

```bash
sudo raspi-config
```

Navigate and enable:
- **Interface Options** → **Camera** → **Enable**
- **Interface Options** → **SSH** → **Enable** (if not done)
- **Interface Options** → **I2C** → **Enable**
- **Interface Options** → **SPI** → **Enable**

Reboot:
```bash
sudo reboot
```

## Step 3: Transfer Project Files

### Option A: From GitHub (if uploaded)
```bash
cd ~
git clone https://github.com/your-username/System_Conveyor.git
cd System_Conveyor
```

### Option B: Via  SCP (from PC)
```bash
# On your PC (Windows PowerShell or Linux/Mac terminal)
scp -r System_Conveyor pi@fruit-sorter.local:~/
```

### Option C: USB Drive
1. Copy project files to USB drive
2. Insert USB into Raspberry Pi
3. Copy files:
```bash
cp -r /media/pi/USB_DRIVE/System_Conveyor ~/
cd ~/System_Conveyor
```

## Step 4: Run Installation Script

```bash
cd ~/System_Conveyor
chmod +x install.sh
./install.sh
```

This script will:
- Update system packages
- Install Python dependencies
- Enable camera and GPIO
- Create virtual environment
- Install OpenCV, YOLOv8, TensorFlow Lite
- Test all installations

Installation takes **15-30 minutes**.

## Step 5: Verify Installation

### Activate Virtual Environment
```bash
cd ~/System_Conveyor
source venv/bin/activate
```

### Test Camera
```bash
python hardware/camera.py
```
Expected: Camera captures test image

### Test Hardware
```bash
# Test servo
python hardware/servo_control.py

# Test motor
python hardware/motor_control.py

# Test complete conveyor
python hardware/conveyor.py
```

### Check Configuration
```bash
python utils/config.py
```

## Step 6: Train AI Models (on PC with GPU)

> **IMPORTANT**: Model training should be done on a PC/laptop with GPU, NOT on Raspberry Pi

### Setup Training Environment (PC)

```bash
# Create conda environment
conda create -n fruit_training python=3.9
conda activate fruit_training

# Install training dependencies
pip install torch torchvision
pip install tensorflow
pip install ultralytics
pip install matplotlib scikit-learn
pip install labelImg
```

### Collect Training Data

**On Raspberry Pi**:
```bash
cd ~/System_Conveyor
source venv/bin/activate

# Collect images for classification
python training/data_collection/collect_images.py \
    --mode classification \
    --count 200 \
    --interval 2.0
```

This creates:
- `raw_images/fresh/` - Fresh fruit images
- `raw_images/spoiled/` - Spoiled fruit images

Transfer images to PC for training:
```bash
# On PC
scp -r pi@fruit-sorter.local:~/System_Conveyor/raw_images ./
```

### Train YOLO Model (Detection)

See detailed guide: [training/yolo/README.md](../training/yolo/README.md)

```bash
cd training/yolo

# Annotate images with LabelImg
labelImg

# Train YOLO
python train_yolo.py --epochs 100 --batch 16
```

Output: `models/yolov8n_fruit.pt`

### Train MobileNetV2 (Classification)

```bash
cd training/mobilenet

# Prepare dataset
python prepare_data.py --source ../../raw_images

# Train model
python train_mobilenet.py --epochs 50 --batch 32

# Export to TFLite
python export_tflite.py --model ./mobilenet_training/mobilenet_fruit_classifier_best.keras
```

Output: `models/mobilenet_classifier.tflite`

## Step 7: Deploy Models to Raspberry Pi

### Transfer Models
```bash
# On PC
cd System_Conveyor
scp models/yolov8n_fruit.pt pi@fruit-sorter.local:~/System_Conveyor/models/
scp models/mobilenet_classifier.tflite pi@fruit-sorter.local:~/System_Conveyor/models/
```

### Verify Models on Raspberry Pi
```bash
# On Raspberry Pi
cd ~/System_Conveyor
source venv/bin/activate

# Test YOLO
python -c "from ai_models import YOLODetector; d = YOLODetector(); d.load_model(); print('YOLO OK')"

# Test MobileNetV2
python -c "from ai_models import MobileNetClassifier; c = MobileNetClassifier(); c.load_model(); print('MobileNet OK')"
```

## Step 8: Configure System

Edit configuration file:
```bash
nano utils/config.py
```

Key settings to adjust:
- `CAMERA_RESOLUTION` - Match your camera
- `YOLO_CONFIDENCE_THRESHOLD` - Detection sensitivity
- `CLASSIFICATION_THRESHOLD` - Classification confidence
- `CONVEYOR_SPEED_DEFAULT` - Belt speed
- `SERVO_ANGLE_LEFT/RIGHT` - Sorting angles

## Step 9: Run System

### Test Run
```bash
cd ~/System_Conveyor
source venv/bin/activate
python fruit_sorter.py
```

### Run at Startup (Optional)

Create systemd service:
```bash
sudo nano /etc/systemd/system/fruit-sorter.service
```

Content:
```ini
[Unit]
Description=AI Fruit Sorting System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/System_Conveyor
ExecStart=/home/pi/System_Conveyor/venv/bin/python fruit_sorter.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable fruit-sorter
sudo systemctl start fruit-sorter
sudo systemctl status fruit-sorter
```

## Troubleshooting

### Camera Issues
```bash
# Check camera detection
libcamera-hello

# If failed, check connections and reboot
sudo reboot
```

### GPIO Permission Errors
```bash
sudo usermod -a -G gpio,i2c,spi $USER
# Logout and login
```

### "Model not found" Errors
- Ensure models are in `System_Conveyor/models/` directory
- Check file names match config
- Verify files transferred correctly

### Slow Inference
- Reduce YOLO input size in config (640 → 416)
- Reduce image resolution
- Increase detection interval
- Disable visual debug mode

### ImportError for tflite_runtime
```bash
pip install tflite-runtime
# Or
pip install tensorflow
```

## Performance Optimization

### For Faster Inference
1. **Overclock Raspberry Pi** (use cooling):
```bash
sudo nano /boot/config.txt
# Add:
over_voltage=6
arm_freq=2000
```

2. **Reduce Resolution**:
- Camera: 1280x720 instead of 1920x1080
- YOLO input: 416 instead of 640

3. **Optimize Code**:
- Skip frames: `SKIP_FRAMES = 2`
- Lower MAX_FPS: `MAX_FPS = 5`

### Monitor System
```bash
# CPU temperature
vcgencmd measure_temp

# CPU usage
htop

# Log file
tail -f logs/fruitsorter_*.log
```

## Next Steps

1. ✅ Read [User Manual](user_manual.md) for operation
2. ✅ Fine-tune sorting parameters
3. ✅ Calibrate servo angles
4. ✅ Test with real fruits
5. ✅ Monitor and improve accuracy

# AI Fruit Sorting System - Complete Setup Guide

Development of a Conveyor System for Fruit Quality Classification Using AI Camera  
Raspberry Pi 4 - YOLOv8 + MobileNetV2

---

## Table of Contents
- [Part 1: Hardware Setup](#part-1-hardware-setup)
- [Part 2: Software Setup](#part-2-software-setup)

---

# Part 1: Hardware Setup

Complete guide for assembling and wiring the AI Fruit Sorting Conveyor System.

## Safety First ⚠️

- **Power off** all devices before wiring
- Check voltage ratings before connecting
- Use appropriate power supplies
- Never connect motors directly to Raspberry Pi GPIO
- Double-check polarity (+/-)

## Components Needed

### Main Components
- ✅ Raspberry Pi 4 (8GB RAM)
- ✅ Camera Module v2 (5MP 1080p)
- ✅ MicroSD Card (32GB+, Class 10)
- ✅ Power Supply for RPi (5V 3A USB-C)

### Motors & Control
- ✅ Servo Motor: MG996R
- ✅ Motor Driver: L298N Module
- ✅ Conveyor Motor: JGB37-545 with encoder
- ✅ Power Supply for Motors (6-12V, 2A+)

### Miscellaneous
- ✅ Breadboard or PCB
- ✅ Jumper wires (M-M, M-F)
- ✅ Mounting hardware
- ✅ Conveyor belt structure

## Raspberry Pi 4 GPIO Pinout

```
      3.3V  1 ◉ ◉  2   5V
     GPIO2  3 ◉ ◉  4   5V
     GPIO3  5 ◉ ◉  6   GND
     GPIO4  7 ◉ ◉  8   GPIO14
       GND  9 ◉ ◉ 10   GPIO15
    GPIO17 11 ◉ ◉ 12   GPIO18 (PWM0) ← SERVO
    GPIO27 13 ◉ ◉ 14   GND
    GPIO22 15 ◉ ◉ 16   GPIO23 ← MOTOR IN1
      3.3V 17 ◉ ◉ 18   GPIO24 ← MOTOR IN2
    GPIO10 19 ◉ ◉ 20   GND
     GPIO9 21 ◉ ◉ 22   GPIO25
    GPIO11 23 ◉ ◉ 24   GPIO8
       GND 25 ◉ ◉ 26   GPIO7
```

## Camera Module Installation

1. **Power off** Raspberry Pi
2. Locate the **CSI camera connector** (between HDMI and audio jack)
3. Gently pull up the plastic clip
4. Insert **ribbon cable** with blue side facing audio jack
5. Push down the clip to secure

Test camera:
```bash
libcamera-hello
```

## Servo Motor (MG996R) Wiring

### Specifications
- Voltage: 4.8-7.2V
- Torque: 11-13 kg⋅cm
- Control: PWM (50Hz)

### Connections
```
MG996R Wire → Connection
├─ Brown  → GND
├─ Red    → 6V External PSU
└─ Orange → GPIO 18 (Pin 12)
```

**⚠️ IMPORTANT**: Use external 6V power, NOT from Raspberry Pi!

## L298N Motor Driver Wiring

### Pin Connections

```
Raspberry Pi          L298N
├─ GPIO 22 (Pin 15) → ENA (PWM Speed)
├─ GPIO 23 (Pin 16) → IN1 (Direction)
├─ GPIO 24 (Pin 18) → IN2 (Direction)
└─ GND (Pin 20)     → GND

L298N               Conveyor Motor
├─ OUT1         →   Motor Wire 1
└─ OUT2         →   Motor Wire 2

Power Supply
├─ +12V         →   L298N +12V
└─ GND          →   Common GND
```

## Complete Wiring Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Raspberry Pi 4 (8GB)                     │
│                                                             │
│  [Camera CSI] ← Camera Module v2                           │
│                                                             │
│  GPIO 18 (Pin 12) ──────────┐                             │
│  GPIO 22-24 (Motor) ─────┐  │                             │
│  GND ─────────────────┐  │  │                             │
└───────────────────────┼──┼──┼─────────────────────────────┘
                        │  │  │
                        │  │  └──► Servo Signal (Orange)
                        │  └─────► L298N Control (ENA,IN1,IN2)
                        └────────► Common GND

┌──────────┐          ┌─────────────┐         ┌──────────────┐
│ Servo    │          │   L298N     │         │ Conveyor PSU │
│ PSU 6V   │          │   Driver    │         │   (12V 2A)   │
└────┬─────┘          │             │         └──────┬───────┘
     │                │  +12V ◄─────┼────────────────┘
     ├──► Servo VCC   │  GND  ◄─────┼──► Common GND
     └──► Common GND  │             │
                      │  OUT1 ──────┼──► Motor Wire 1
                      │  OUT2 ──────┼──► Motor Wire 2
                      └─────────────┘
```

## Power Supply Requirements

1. **Raspberry Pi**: 5V 3A USB-C
2. **Servo Motor**: 6V 2A
3. **Conveyor Motor**: 12V 2A+

**⚠️ CRITICAL**: All power supplies must share **common ground**!

```
RPi GND ←───┬───→ Servo GND ←───┬───→ L298N GND ←──→ Motor PSU GND
            └─────────All Connected Together─────────┘
```

## Camera Positioning

- **Distance to servo**: 20 cm (optimal)
- **Height**: 20-30 cm above conveyor
- **Angle**: Perpendicular to belt
- **Lighting**: Even LED lighting recommended

## Component Testing

```bash
# Test camera
python hardware/camera.py

# Test servo
python hardware/servo_control.py

# Test motor
python hardware/motor_control.py

# Test complete system
python hardware/conveyor.py
```

## Hardware Safety Checklist ✅

- [ ] All connections double-checked
- [ ] No bare wires touching
- [ ] Correct voltage for each component
- [ ] Common ground established
- [ ] Motors secured
- [ ] Camera ribbon properly inserted

---

# Part 2: Software Setup

Complete software installation and configuration for Raspberry Pi 4.

## Prerequisites

- ✅ Raspberry Pi 4 with hardware assembled
- ✅ MicroSD Card (32GB+)
- ✅ PC/Laptop for model training (GPU recommended)

## Step 1: Install Raspberry Pi OS

1. Download **Raspberry Pi Imager**: https://www.raspberrypi.com/software/
2. Flash **Raspberry Pi OS (64-bit)** - Full version
3. Configure settings (hostname: `fruit-sorter`, enable SSH)
4. Boot and update:

```bash
sudo apt update && sudo apt upgrade -y
```

## Step 2: Enable Camera and GPIO

```bash
sudo raspi-config
```

Enable:
- Interface Options → Camera
- Interface Options → SSH
- Interface Options → I2C
- Interface Options → SPI

Reboot:
```bash
sudo reboot
```

## Step 3: Transfer Project Files

Choose one method:

### Option A: Git Clone
```bash
cd ~
git clone https://github.com/your-username/System_Conveyor.git
cd System_Conveyor
```

### Option B: SCP from PC
```bash
scp -r System_Conveyor pi@fruit-sorter.local:~/
```

### Option C: USB Drive
```bash
cp -r /media/pi/USB_DRIVE/System_Conveyor ~/
```

## Step 4: Run Installation Script

```bash
cd ~/System_Conveyor
chmod +x install.sh
./install.sh
```

**Installation time**: 15-30 minutes

This installs:
- Python dependencies
- OpenCV, YOLOv8, TensorFlow Lite
- Camera and GPIO libraries
- Virtual environment

## Step 5: Verify Installation

```bash
cd ~/System_Conveyor
source venv/bin/activate

# Test camera
python hardware/camera.py

# Test hardware
python hardware/servo_control.py
python hardware/motor_control.py

# Check configuration
python utils/config.py
```

## Step 6: Train AI Models (on PC/Laptop)

> **⚠️ IMPORTANT**: Train on PC with GPU, NOT on Raspberry Pi!

### Setup Training Environment (PC)

```bash
conda create -n fruit_training python=3.9
conda activate fruit_training

pip install torch torchvision tensorflow
pip install ultralytics matplotlib scikit-learn
```

### Collect Training Data (on Raspberry Pi)

```bash
cd ~/System_Conveyor
source venv/bin/activate

# Collect images
python training/data_collection/collect_images.py \
    --mode classification \
    --count 200 \
    --interval 2.0
```

Transfer to PC:
```bash
scp -r pi@fruit-sorter.local:~/System_Conveyor/raw_images ./
```

### Train YOLO (Detection)

```bash
cd training/yolo
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
python export_tflite.py
```

Output: `models/mobilenet_classifier.tflite`

## Step 7: Deploy Models to Raspberry Pi

```bash
# On PC
scp models/yolov8n_fruit.pt pi@fruit-sorter.local:~/System_Conveyor/models/
scp models/mobilenet_classifier.tflite pi@fruit-sorter.local:~/System_Conveyor/models/
```

Verify on Pi:
```bash
cd ~/System_Conveyor
source venv/bin/activate

python -c "from ai_models import YOLODetector; d = YOLODetector(); d.load_model(); print('YOLO OK')"
python -c "from ai_models import MobileNetClassifier; c = MobileNetClassifier(); c.load_model(); print('MobileNet OK')"
```

## Step 8: Configure System

Edit configuration:
```bash
nano utils/config.py
```

Key settings:
- `CONVEYOR_SPEED_DETECTION = 35` (for 20cm distance)
- `CAMERA_TO_SERVO_DISTANCE = 20.0`
- `SERVO_ANGLE_FRESH = 0` (straight)
- `SERVO_ANGLE_SPOILED = 180` (right)

## Step 9: Run System

### Test Run
```bash
cd ~/System_Conveyor
source venv/bin/activate
python fruit_sorter.py
```

### Web Interface
```bash
python run_web.py
```

Access: http://fruit-sorter.local:5000

### Run at Startup (Optional)

Create service:
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

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl enable fruit-sorter
sudo systemctl start fruit-sorter
```

## Troubleshooting

### Camera Not Detected
```bash
libcamera-hello
sudo raspi-config  # Enable camera
sudo reboot
```

### GPIO Permission Errors
```bash
sudo usermod -a -G gpio,i2c,spi $USER
# Logout and login
```

### Model Not Found
- Check files in `models/` directory
- Verify file names match config
- Re-transfer models if needed

### Slow Performance
- Reduce YOLO input size (640 → 416)
- Lower camera resolution
- Increase detection interval
- Optimize: `SKIP_FRAMES = 2`, `MAX_FPS = 5`

## Performance Optimization

### Overclock (with cooling)
```bash
sudo nano /boot/config.txt
# Add:
over_voltage=6
arm_freq=2000
```

### Monitor System
```bash
vcgencmd measure_temp  # Temperature
htop                   # CPU usage
tail -f logs/*.log     # System logs
```

---

## System Configuration Summary

**Optimal Settings for 20cm Distance:**
- Motor Speed: 35% (2.92 cm/s)
- Travel Time: 6.85 seconds
- Processing Timeout: 1.5s
- Servo Delay: 0.6s
- Expected Accuracy: 98%
- Throughput: 40-45 fruits/min

---

## Quick Reference Commands

```bash
# Activate environment
source venv/bin/activate

# Run main system
python fruit_sorter.py

# Run web interface
python run_web.py

# Test hardware
python hardware/conveyor.py

# Check system status
sudo systemctl status fruit-sorter

# View logs
tail -f logs/*.log
```

---

## Support & Next Steps

1. Fine-tune sorting parameters
2. Calibrate servo angles
3. Test with real fruits (cam, ổi, táo)
4. Monitor and improve accuracy
5. Optimize for production use

**Target Accuracy**: 90-95%  
**Recommended Dataset**: 200+ images per class per fruit type

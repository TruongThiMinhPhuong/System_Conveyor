# User Manual

Operation guide for the AI Fruit Sorting Conveyor System.

## Quick Start

### Power On System
1. ✅ **Hardware Check**:
   - Camera connected
   - Servo and motors wired correctly
   - Power supplies connected
   - Common ground established

2. ✅ **Start Raspberry Pi**:
```bash
# SSH to Raspberry Pi
ssh pi@fruit-sorter.local

# Navigate to project
cd ~/System_Conveyor

# Activate virtual environment
source venv/bin/activate
```

3. ✅ **Run System**:
```bash
python fruit_sorter.py
```

System will:
- Initialize camera
- Load AI models
- Start hardware (servo, motors)
- Begin processing

## System Operation

### Normal Operation Flow

```
1. Conveyor belt moves continuously
2. Camera captures frames
3. YOLO detects fruit → Bounding box
4. ROI extracted and preprocessed
5. MobileNetV2 classifies → Fresh/Spoiled
6. Servo sorts fruit:
   ├─ Fresh → CENTER (straight path)
   └─ Spoiled → RIGHT (reject bin)
7. Conveyor continues
```

### Starting the System

```bash
cd ~/System_Conveyor
source venv/bin/activate
python fruit_sorter.py
```

Expected output:
```
======================================
🍎 AI Fruit Sorting Conveyor System
======================================

📁 Paths:
   YOLO model exists: ✅
   MobileNet model exists: ✅

🚀 Initializing Fruit Sorting System...
⚙️ Initializing hardware...
🎥 Initializing camera...
✅ Camera initialized
🔧 Initializing servo...
✅ Servo initialized
🔧 Initializing motor controller...
✅ Motor controller initialized

🤖 Loading AI models...
✅ YOLO model loaded
✅ MobileNetV2 model loaded
✅ System initialized successfully!
▶️ Conveyor system started
🚀 Starting main system loop...
```

### During Operation

System logs show:
```
[10:30:15] [INFO] - 🎯 Detected: apple (confidence: 92.5%)
[10:30:15] [INFO] - 🍎 Classified: Fresh (confidence: 87.3%)
[10:30:15] [INFO] - ↔️ Sorting: LEFT (Fresh)
```

### Stopping the System

Press `Ctrl+C`:
```
^C
⚙️ Interrupt received
⚙️ Stopping system...
🛑 Motor stopped
🛑 Servo stopped
📴 Camera closed

====================================
System Statistics
====================================
Total detections: 47
Fresh sorted: 32 (68.1%)
Spoiled sorted: 15 (31.9%)
Errors: 0
====================================

✅ System stopped
```

## Configuration

### Adjusting System Parameters

Edit `utils/config.py`:

```python
# Detection sensitivity
YOLO_CONFIDENCE_THRESHOLD = 0.5  # Lower = more detections
CLASSIFICATION_THRESHOLD = 0.6   # Higher = more confident sorting

# Conveyor speeds
CONVEYOR_SPEED_DEFAULT = 60      # 0-100%
CONVEYOR_SPEED_DETECTION = 40    # Slower for better detection

# Servo angles
SERVO_ANGLE_LEFT = 45       # Fresh fruit angle
SERVO_ANGLE_RIGHT = 135     # Spoiled fruit angle
```

### Camera Settings

```python
# Resolution (lower = faster processing)
CAMERA_RESOLUTION = (1920, 1080)  # Try (1280, 720) for speed

# Adjustments
CAMERA_BRIGHTNESS = 0.0    # -1.0 to 1.0
CAMERA_CONTRAST = 1.0      # 0.0 to 2.0
```

## Calibration

### Servo Calibration

1. **Test servo movement**:
```bash
python hardware/servo_control.py
```

2. **Adjust angles** in `utils/config.py`:
```python
SERVO_ANGLE_LEFT = 45    # Adjust until fruits go left
SERVO_ANGLE_CENTER = 90
SERVO_ANGLE_RIGHT = 135  # Adjust until fruits go right
```

3. **Test with fruit**:
- Place fruit on belt
- Observe servo movement
- Adjust angles as needed

### Motor Speed Calibration

```bash
python hardware/motor_control.py
```

Test different speeds in `utils/config.py`:
```python
CONVEYOR_SPEED_DETECTION = 40  # Slow enough for detection
CONVEYOR_SPEED_DEFAULT = 60    # Normal operation
```

## Monitoring

### View Logs

```bash
# Real-time logs
tail -f logs/fruitsorter_*.log

# All logs
cat logs/fruitsorter_*.log
```

### System Statistics

Statistics printed every 10 seconds (configurable):
```python
STATS_UPDATE_INTERVAL = 10  # seconds
```

### Performance Metrics

Check processing FPS:
```
📊 FPS: 8.5, Processing: 120.5ms
```

**Target Performance**:
- FPS: 5-10 (adequate for sorting)
- Processing time: <200ms per fruit

## Troubleshooting

### No Detections

**Symptoms**: No fruits detected, empty logs

**Solutions**:
1. Check camera:
```bash
python hardware/camera.py
```

2. Lower YOLO threshold:
```python
YOLO_CONFIDENCE_THRESHOLD = 0.3  # More sensitive
```

3. Improve lighting
4. Check fruit is in camera view

### Wrong Classification

**Symptoms**: Fresh classified as spoiled (or vice versa)

**Solutions**:
1. Check classification threshold:
```python
CLASSIFICATION_THRESHOLD = 0.5  # Less strict
```

2. Retrain MobileNetV2 with more data
3. Improve image quality (lighting, focus)

### Servo Not Sorting

**Symptoms**: Detection works, but servo doesn't move

**Solutions**:
1. Check servo power (6V external)
2. Test servo:
```bash
python hardware/servo_control.py
```

3. Check wiring (GPIO 18)
4. Verify common ground

### Conveyor Not Moving

**Symptoms**: Motor doesn't run

**Solutions**:
1. Check L298N power (12V)
2. Test motor:
```bash
python hardware/motor_control.py
```

3. Check wiring
4. Try different speed:
```bash
python -c "from hardware import MotorControl; m = MotorControl(); m.initialize(); m.start_forward(80)"
```

### Camera Errors

**Symptoms**: "Failed to capture frame"

**Solutions**:
1. Check camera connection
2. Enable camera:
```bash
sudo raspi-config
# Interface Options → Camera → Enable
```

3. Reboot:
```bash
sudo reboot
```

### Slow Performance

**Symptoms**: FPS < 3, system laggy

**Solutions**:
1. Reduce camera resolution
2. Reduce YOLO input size to 416
3. Increase detection interval:
```python
DETECTION_INTERVAL = 1.0  # seconds
```

4. Skip frames:
```python
SKIP_FRAMES = 1  # Process every other frame
```

## Maintenance

### Daily Checks
- ✅ Camera lens clean
- ✅ Belt running smoothly
- ✅ Servo moving freely
- ✅ No loose wires

### Weekly Checks
- ✅ Review logs for errors
- ✅ Check statistics (accuracy)
- ✅ Clean camera module
- ✅ Verify sorting accuracy

### Monthly Maintenance
- ✅ Retrain models if accuracy drops
- ✅ Update system packages
- ✅ Backup configuration
- ✅ Check motor temperatures

## Advanced Features

### Auto-Start on Boot

Enable systemd service:
```bash
sudo systemctl enable fruit-sorter
sudo systemctl start fruit-sorter
```

Check status:
```bash
sudo systemctl status fruit-sorter
```

### Remote Monitoring

Access logs remotely:
```bash
ssh pi@fruit-sorter.local 'tail -f ~/System_Conveyor/logs/fruitsorter_*.log'
```

### Data Collection for Retraining

Collect more training data:
```bash
python training/data_collection/collect_images.py \
    --mode classification \
    --count 100
```

## Safety

### Emergency Stop

Press `Ctrl+C` to stop immediately.

Emergency stop in code:
```python
def emergency_stop(self):
    """Stop all operations immediately"""
    self.motor.brake()
    self.servo.move_to_center()
```

### Power Safety
- Never hot-swap motor connections
- Always power off before wiring changes
- Use appropriate power supplies
- Ensure common ground

## Performance Tuning

### For Maximum Accuracy
```python
YOLO_CONFIDENCE_THRESHOLD = 0.6
CLASSIFICATION_THRESHOLD = 0.7
CONVEYOR_SPEED_DETECTION = 30  # Slower
CAMERA_RESOLUTION = (1920, 1080)  # Higher
```

### For Maximum Speed
```python
YOLO_CONFIDENCE_THRESHOLD = 0.4
CLASSIFICATION_THRESHOLD = 0.5
CONVEYOR_SPEED_DETECTION = 60  # Faster
CAMERA_RESOLUTION = (1280, 720)  # Lower
SKIP_FRAMES = 1  # Process every other frame
```

## Getting Help

### Check Logs
```bash
grep ERROR logs/fruitsorter_*.log
```

### Test Components Individually
```bash
# Test camera
python hardware/camera.py

# Test servo
python hardware/servo_control.py

# Test motor
python hardware/motor_control.py

# Test YOLO
python ai_models/yolo_detector.py

# Test MobileNetV2
python ai_models/mobilenet_classifier.py
```

### Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| "Camera not initialized" | Camera not detected | Check connection, enable in raspi-config |
| "Model not found" | Missing AI model | Copy models to `models/` directory |
| "GPIO not available" | Permission issue | `sudo usermod -a -G gpio $USER` |
| "Low confidence" | Poor lighting/image | Improve lighting, retrain model |

## Tips for Best Results

1. **Lighting**: Use consistent, bright LED lighting
2. **Belt Speed**: Balance speed vs accuracy
3. **Camera Position**: 20-30cm above belt, perpendicular
4. **Calibration**: Regularly test and adjust servo angles
5. **Data Quality**: Retrain with high-quality images
6. **Monitoring**: Check logs regularly for issues
7. **Maintenance**: Keep system clean and wiring secure

## Support

For issues:
1. Check documentation
2. Review logs
3. Test components individually
4. Check hardware connections
5. Verify model files exist

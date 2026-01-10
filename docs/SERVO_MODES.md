# Servo Operating Modes Guide

This guide explains the different operating modes for the fruit sorting system and how to use the servo independently of AI models.

## Operating Modes

The system now supports 3 operating modes that can be configured in `utils/config.py`:

### 1. NORMAL Mode (Default)

**Requirements:** Both YOLO and MobileNet models must be available

**Behavior:**
- Full AI pipeline: Detection → Classification → Sorting
- Uses YOLO for fruit detection
- Uses MobileNet for freshness classification
- Servo controlled based on AI classification results

**Use case:** Production operation with trained models

**Configuration:**
```python
OPERATING_MODE = 'NORMAL'
```

---

### 2. DETECTION_ONLY Mode

**Requirements:** Only YOLO model required (MobileNet optional)

**Behavior:**
- Uses YOLO for fruit detection only
- Classification uses fallback rules (configured via `DEFAULT_CLASSIFICATION`)
- Servo operates based on fallback classification
- Useful when MobileNet model is not available or being retrained

**Use case:** Testing detection accuracy, MobileNet model unavailable

**Configuration:**
```python
OPERATING_MODE = 'DETECTION_ONLY'
DEFAULT_CLASSIFICATION = 'FRESH'  # or 'SPOILED'
```

---

### 3. MANUAL Mode

**Requirements:** No AI models required

**Behavior:**
- No AI processing
- Servo controlled by timer or manual triggers
- Uses `MANUAL_DEFAULT_FRESH` for classification
- Excellent for hardware testing

**Use case:** Hardware testing, calibration, demonstrations

**Configuration:**
```python
OPERATING_MODE = 'MANUAL'
MANUAL_SORT_INTERVAL = 5.0  # seconds between sorts
MANUAL_DEFAULT_FRESH = True  # True=fresh, False=spoiled
```

## Fallback Classification

When AI models fail or produce low-confidence results, the system uses fallback classification:

```python
# Configuration in utils/config.py
DEFAULT_CLASSIFICATION = 'FRESH'  # Options: 'FRESH', 'SPOILED', 'SKIP'
REQUIRE_CLASSIFICATION = True  # Set False to always use fallback
```

**Options:**
- `FRESH`: Default to fresh (servo at 0°, straight through)
- `SPOILED`: Default to spoiled (servo at 180°, push right)
- `SKIP`: Skip sorting, keep servo centered

## Testing Servo Without AI Models

### Option 1: Standalone Test Script

Run the dedicated servo test script (no AI models needed):

```bash
# Windows
python test_servo_only.py

# Raspberry Pi
python3 test_servo_only.py
```

**Test modes:**
1. **Basic test**: 3 predefined tests (fresh, spoiled, alternating)
2. **Continuous**: Continuous sorting with 5-second intervals

### Option 2: Manual Mode in Main System

1. Edit `utils/config.py`:
   ```python
   OPERATING_MODE = 'MANUAL'
   MANUAL_SORT_INTERVAL = 5.0
   MANUAL_DEFAULT_FRESH = True
   ```

2. Run the main system:
   ```bash
   python fruit_sorter.py
   ```

The system will sort fruit every 5 seconds using the manual default classification.

### Option 3: Hardware Test Cycle

Use the built-in hardware test:

```python
from hardware import ConveyorSystem

with ConveyorSystem() as conveyor:
    conveyor.run_test_cycle()
```

## Troubleshooting

### Servo not working in NORMAL mode

**Problem:** AI models not loading correctly

**Solution:** Switch to DETECTION_ONLY or MANUAL mode:
```python
# In utils/config.py
OPERATING_MODE = 'DETECTION_ONLY'  # or 'MANUAL'
```

### Low confidence classifications

**Problem:** Model produces low-confidence results

**Solution:** Configure fallback behavior:
```python
DEFAULT_CLASSIFICATION = 'FRESH'  # Safe default
CLASSIFICATION_THRESHOLD = 0.6    # Adjust threshold
```

### Testing on PC without Raspberry Pi

**Problem:** GPIO not available on development PC

**Solution:** The system automatically runs in simulation mode when GPIO is not available. All servo commands will be logged but not executed.

## Examples

### Example 1: Test servo on Raspberry Pi (no models)

```bash
# Set to MANUAL mode
nano utils/config.py
# Change: OPERATING_MODE = 'MANUAL'

# Run main system
python3 fruit_sorter.py
```

### Example 2: Run with detection only

```bash
# Requires YOLO model only
# Set to DETECTION_ONLY mode
nano utils/config.py
# Change: OPERATING_MODE = 'DETECTION_ONLY'
# Change: DEFAULT_CLASSIFICATION = 'FRESH'

python3 fruit_sorter.py
```

### Example 3: Quick hardware test

```bash
# No configuration changes needed
python3 test_servo_only.py
```

## Configuration Reference

All servo-related settings in `utils/config.py`:

```python
# Operating Modes
OPERATING_MODE = 'NORMAL'  # NORMAL, DETECTION_ONLY, MANUAL

# Classification fallback
REQUIRE_CLASSIFICATION = True
DEFAULT_CLASSIFICATION = 'FRESH'  # FRESH, SPOILED, SKIP

# Manual mode settings
MANUAL_SORT_INTERVAL = 5.0
MANUAL_DEFAULT_FRESH = True

# Servo angles (hardware level)
SERVO_ANGLE_FRESH = 0      # Fresh: straight (0°)
SERVO_ANGLE_SPOILED = 180  # Spoiled: push right (180°)
SERVO_ANGLE_CENTER = 90    # Neutral position

# Timing
SERVO_MOVE_DELAY = 0.6          # Servo movement time
CONVEYOR_STOP_DELAY = 0.4       # Pause for sorting
```

## Summary

The system is now **flexible and robust**:

✅ **Works without AI models** (MANUAL mode)  
✅ **Works with partial AI** (DETECTION_ONLY mode)  
✅ **Full AI when available** (NORMAL mode)  
✅ **Graceful fallback** when models fail  
✅ **Easy hardware testing** via test scripts

This ensures the servo mechanism can always be tested and operated, regardless of AI model availability.

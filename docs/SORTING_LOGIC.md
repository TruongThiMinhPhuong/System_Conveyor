# Logic Phân Loại 3 Chiều

## 🎯 Tổng Quan

Hệ thống sử dụng **phân loại 3 chiều** để xử lý tất cả các đối tượng trên băng chuyền:

```
┌─────────────────────────────────────────────┐
│         BĂNG CHUYỀN DI CHUYỂN →             │
│                                             │
│     [Camera + AI phát hiện]                 │
│              │                              │
│              ▼                              │
│      ┌───────────────┐                      │
│      │ YOLOv8 Check  │                      │
│      └───────┬───────┘                      │
│              │                              │
│    ┌─────────┴─────────┐                   │
│    │                   │                   │
│    ▼                   ▼                   │
│ [Hoa quả?]         [Vật khác]              │
│    │                   │                   │
│    ▼                   │                   │
│ [MobileNetV2]          │                   │
│    │                   │                   │
│ ┌──┴──┐                │                   │
│ │     │                │                   │
│ ▼     ▼                ▼                   │
│ Tươi  Hỏng          Vật khác               │
│ │     │                │                   │
│ ▼     ▼                ▼                   │
│ CENTER RIGHT          LEFT                 │
│ (90°) (135°)          (45°)                │
└─────────────────────────────────────────────┘
```

## 🔄 3 Trường Hợp Phân Loại

### 1. 🍎 Hoa Quả Tươi → GIỮA (Đi Thẳng)
- **YOLO phát hiện**: Là hoa quả (apple, orange, banana...)
- **MobileNetV2 phân loại**: Fresh (confidence > threshold)
- **Servo**: 90° (CENTER)
- **Kết quả**: Rơi thẳng vào thùng tươi

### 2. 🍂 Hoa Quả Hỏng → PHẢI (Reject)
- **YOLO phát hiện**: Là hoa quả
- **MobileNetV2 phân loại**: Spoiled
- **Servo**: 135° (RIGHT)
- **Kết quả**: Rẽ phải vào thùng reject 2

### 3. ⚠️ Không Phải Hoa Quả → TRÁI (Reject)
- **YOLO phát hiện**: Không phải hoa quả (hoặc không detect)
- **MobileNetV2**: Không chạy
- **Servo**: 45° (LEFT)
- **Kết quả**: Rẽ trái vào thùng reject 1

## 🗂️ Bố Trí Thùng Chứa

```
        [Camera]
           │
    ═══════▼════════  ← Băng chuyền
         [Servo]
       /    |    \
      /     |     \
     /      |      \
    ↙       ↓       ↘
[Reject 1] [Tươi] [Reject 2]
   LEFT    CENTER   RIGHT
 (Vật khác) (Fresh) (Spoiled)
    45°      90°      135°
```

## 💻 Code Implementation

### Trong `hardware/conveyor.py`:

```python
def sort_fruit(self, is_fresh=None, is_fruit=True, pause_conveyor=True):
    """
    3-way sorting logic
    
    Args:
        is_fresh: True/False for fruit, None if not applicable
        is_fruit: Whether object is a fruit (from YOLO)
        pause_conveyor: Pause belt during sorting
    """
    if not is_fruit:
        # Non-fruit → LEFT
        self.servo.move_to_left()   # 45°
    elif is_fresh:
        # Fresh fruit → CENTER
        self.servo.move_to_center() # 90°
    else:
        # Spoiled fruit → RIGHT
        self.servo.move_to_right()  # 135°
```

### Trong `fruit_sorter.py`:

```python
# Kiểm tra YOLO có detect hoa quả không
detections = self.detector.detect(frame)

if not detections or not is_fruit_class(detection):
    # Không phải hoa quả
    conveyor.sort_fruit(is_fruit=False)
else:
    # Là hoa quả → Classify
    classification = self.classifier.classify(roi)
    is_fresh = classification['is_fresh']
    
    conveyor.sort_fruit(is_fresh=is_fresh, is_fruit=True)
```

## 🎨 Training Dataset

### YOLOv8 Classes:
Cần train YOLO với các class hoa quả cụ thể:
```yaml
# dataset.yaml
names:
  0: apple
  1: orange  
  2: banana
  # ... các loại hoa quả khác
```

**Quan trọng**: YOLOv8 phải học **CHỈ** detect các loại hoa quả bạn muốn.
- Nếu detect được → is_fruit = True
- Nếu không detect hoặc class khác → is_fruit = False

### MobileNetV2 Classes:
```python
# 2 classes
FRESHNESS_CLASSES = ['Fresh', 'Spoiled']
```

## 🧪 Test Cases

### Test 1: Hoa Quả Tươi
```bash
# Input: Táo tươi
# Expected: 
#   - YOLO detect: "apple"
#   - MobileNetV2: "Fresh" (90%+)
#   - Servo: 90° (CENTER)
#   - Kết quả: Rơi vào thùng tươi ✓
```

### Test 2: Hoa Quả Hỏng
```bash
# Input: Cam hỏng
# Expected:
#   - YOLO detect: "orange"
#   - MobileNetV2: "Spoiled" (85%+)
#   - Servo: 135° (RIGHT)
#   - Kết quả: Rẽ phải vào reject bin 2 ✓
```

### Test 3: Vật Khác
```bash
# Input: Chai nhựa / Giấy / Đá
# Expected:
#   - YOLO: No detection (hoặc class không phải fruit)
#   - MobileNetV2: Không chạy
#   - Servo: 45° (LEFT)
#   - Kết quả: Rẽ trái vào reject bin 1 ✓
```

## ⚙️ Configuration

Trong `utils/config.py`:

```python
# YOLO fruit classes (chỉ detect những class này)
FRUIT_CLASSES = ['apple', 'orange', 'banana', 'mango']

# Detection confidence
YOLO_CONFIDENCE_THRESHOLD = 0.5  # Chỉ accept nếu > 50%

# Classification confidence  
CLASSIFICATION_THRESHOLD = 0.6   # Tươi/hỏng confidence

# Servo timing
SORT_DURATION = 1.0  # Thời gian servo mở
PAUSE_BEFORE_SORT = 0.3
PAUSE_AFTER_SORT = 0.5
```

## 📊 Expected Accuracy

| Case | Detection | Classification | Total |
|------|-----------|----------------|-------|
| Fresh Fruit | >90% | >85% | ~77% |
| Spoiled Fruit | >90% | >85% | ~77% |
| Non-Fruit | 100% (reject all) | N/A | 100% |

**Note**: Vật khác sẽ luôn được loại trừ (LEFT) nên không ảnh hưởng đến chất lượng sản phẩm cuối.

## 🔧 Calibration

### Bước 1: Test YOLO Detection
```bash
python ai_models/yolo_detector.py
# Kiểm tra: Chỉ detect hoa quả, không detect vật khác
```

### Bước 2: Test MobileNetV2
```bash
python ai_models/mobilenet_classifier.py
# Kiểm tra: Fresh vs Spoiled accuracy
```

### Bước 3: Test Servo Positions
```bash
python hardware/servo_control.py
# Test: LEFT (45°), CENTER (90°), RIGHT (135°)
```

### Bước 4: Test Full System
```bash
python fruit_sorter.py
# Hoặc dùng web interface
python run_web.py
```

Đặt từng loại lên băng chuyền:
1. ✓ Hoa quả tươi → CENTER
2. ✓ Hoa quả hỏng → RIGHT
3. ✓ Vật khác (chai, giấy) → LEFT

## ✅ Advantages

1. **Sạch hơn**: Vật lạ không lẫn vào sản phẩm
2. **An toàn hơn**: Phát hiện dị vật
3. **Linh hoạt hơn**: Dễ mở rộng thêm class
4. **Đơn giản hóa cơ khí**: Hoa quả tươi rơi tự nhiên

## 🎯 Summary

```
┌─────────────────────────────────────┐
│   INPUT: Bất kỳ vật gì trên băng    │
└────────────┬────────────────────────┘
             │
             ▼
    ┌────────────────┐
    │ YOLO Detection │
    └────────┬───────┘
             │
    ┌────────┴────────┐
    │ Is Fruit?       │
    └─┬───────────┬───┘
      │           │
   NO │           │ YES
      │           │
      ▼           ▼
   [LEFT]    ┌──────────┐
   45°       │MobileNetV2│
   Reject 1  └─┬──────┬─┘
               │      │
            FRESH  SPOILED
               │      │
               ▼      ▼
           [CENTER] [RIGHT]
            90°     135°
           Good    Reject 2
└─────────────────────────────────────┘
```

**TL;DR:**
- Hoa quả tươi → Thẳng ✓
- Hoa quả hỏng → Phải ✗
- Vật khác → Trái ✗✗

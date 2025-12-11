# Hướng Dẫn Hiệu Chỉnh Servo Cho Logic Mới

## 🎯 Logic Phân Loại Mới

- **Trái Cây Tươi** → Đi thẳng (Servo ở giữa - 90°)
- **Trái Cây Hỏng** → Rẽ phải (Servo sang phải - 135°)

## 🔧 Cấu Hình Servo

### Trong `hardware/gpio_config.py`:

```python
# Servo angles (degrees)
SERVO_ANGLE_LEFT = 45       # Không dùng (test only)
SERVO_ANGLE_CENTER = 90     # Tươi - đi thẳng ✓
SERVO_ANGLE_RIGHT = 135     # Hỏng - rẽ phải ✓
```

### Nghĩa Là:
- **90°** (Center) = Cổng thẳng → Trái cây tươi đi thẳng
- **135°** (Right) = Cổng nghiêng phải → Trái cây hỏng rơi vào thùng reject

## 🛠️ Hiệu Chỉnh Vật Lý

### Bước 1: Chuẩn Bị Cơ Khí

**Đặt 2 thùng chứa:**
```
                 [Thùng Tươi]
                      ║
    ════════════════╬════════════════  (Băng chuyền)
    →  →  →  →  →  ║  →  →  →  →
                    ║
              [Servo ở giữa]
                    ║
                    ╚═══════▶ [Thùng Hỏng]
                              (Bên phải)
```

### Bước 2: Test Servo

```bash
cd ~/System_Conveyor
source venv/bin/activate
python hardware/servo_control.py
```

**Kiểm tra:**
- Khi servo ở **90°** (center) → Cổng THẲNG, trái cây rơi vào thùng tươi
- Khi servo ở **135°** (right) → Cổng NGHIÊNG, trái cây rơi vào thùng hỏng

### Bước 3: Điều Chỉnh Góc (Nếu Cần)

Nếu góc không chuẩn, sửa trong `utils/config.py`:

```python
# Ví dụ: Nếu cần nghiêng nhiều hơn
SERVO_ANGLE_CENTER = 85    # Giảm để nghiêng trái chút
SERVO_ANGLE_RIGHT = 140    # Tăng để nghiêng phải nhiều hơn
```

## 🧪 Test Thực Tế

### Test 1: Servo Manual
```bash
python hardware/servo_control.py
# Quan sát cổng mở/đóng
```

### Test 2: Với Trái Cây Thật
```bash
python fruit_sorter.py
# Hoặc dùng web interface
python run_web.py
```

**Đặt trái cây lên băng chuyền:**
1. Trái cây tươi → Phải rơi thẳng vào thùng tươi
2. Trái cây hỏng → Phải rẽ phải vào thùng reject

## 📐 Bố Trí Thùng Chứa

### Cấu Hình Khuyến Nghị:

```
        [Camera nhìn xuống]
               │
    ═══════════▼══════════════  (Băng chuyền di chuyển →)
         Servo MG996R
         (ở giữa băng)
               │
        ┌──────┴──────┐
        │             │
    [Thùng Tươi] [Thùng Hỏng]
    (Dưới băng)  (Bên phải)
```

**Vị trí:**
- **Thùng Tươi**: Đặt ngay dưới băng chuyền (trái cây rơi thẳng)
- **Thùng Hỏng**: Đặt bên phải, servo đẩy trái cây qua đó

## ⚙️ Code Flow

```python
# Trong conveyor.py:

if is_fresh:
    # Tươi → Servo ở giữa (đường thẳng)
    self.servo.move_to_center()  # 90°
    # Trái cây rơi thẳng vào thùng tươi
else:
    # Hỏng → Servo sang phải
    self.servo.move_to_right()   # 135°
    # Trái cây bị đẩy sang phải, rơi vào thùng hỏng
```

## 🔍 Kiểm Tra Logs

```bash
tail -f logs/fruitsorter_*.log

# Expected output:
# [INFO] - 🍎 Fresh fruit detected → Going straight
# [INFO] - 🍂 Spoiled fruit detected → Turning right
```

## 📊 Fine-Tuning

Nếu trái cây không rơi đúng chỗ:

### Điều chỉnh góc:
```python
# utils/config.py

# Nếu tươi cần nghiêng chút:
SERVO_ANGLE_CENTER = 88  # Nghiêng trái 2°

# Nếu hỏng cần nghiêng nhiều hơn:
SERVO_ANGLE_RIGHT = 140  # Nghiêng phải nhiều hơn
```

### Điều chỉnh thời gian:
```python
# utils/config.py

SORT_DURATION = 1.5        # Tăng nếu cần thêm thời gian
PAUSE_BEFORE_SORT = 0.3    # Dừng băng trước khi sort
PAUSE_AFTER_SORT = 0.5     # Chờ servo về center
```

## ✅ Checklist

- [ ] Servo test OK (90° thẳng, 135° phải)
- [ ] 2 thùng đã đặt đúng vị trí
- [ ] Test với trái cây thật
- [ ] Tươi rơi vào thùng tươi ✓
- [ ] Hỏng rơi vào thùng hỏng ✓
- [ ] Logs hiển thị đúng
- [ ] Web interface hiển thị đúng

## 🎯 Kết Luận

**Logic mới đơn giản hơn:**
- Không cần servo "đẩy" trái cây tươi sang trái
- Trái cây tươi tự nhiên rơi thẳng (tiết kiệm cơ khí)
- Chỉ cần servo đẩy trái cây hỏng sang phải

**Ưu điểm:**
- ✅ Đơn giản hóa cơ khí
- ✅ Giảm wear-and-tear servo
- ✅ Độ tin cậy cao hơn

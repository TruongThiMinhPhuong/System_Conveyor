# Hướng dẫn cập nhật code trên Raspberry Pi

## Vấn đề hiện tại

Bạn đã `git reset --hard 879011f` nên mất code mới. Cần pull code mới nhất từ GitHub.

## Giải pháp

Chạy các lệnh sau trên Raspberry Pi:

```bash
# 1. Pull code mới nhất (sẽ fast-forward về commit mới)
cd ~/System_Conveyor
git pull origin main

# 2. Kiểm tra bạn đã ở commit mới nhất
git log --oneline -5
# Bạn sẽ thấy:
# 451f9ee Fix web app logger error - add warning method to SystemLogger
# 0c8bdb4 Add servo independence from AI models - support NORMAL/DETECTION_ONLY/MANUAL modes
# 879011f fix up
```

## Các tính năng mới sau khi pull

### 1. Servo Independence (0c8bdb4)
- Servo hoạt động độc lập với AI models
- 3 chế độ: NORMAL, DETECTION_ONLY, MANUAL
- Script test riêng: `test_servo_only.py`

### 2. Web App Logger Fix (451f9ee)
- Fix lỗi `'SystemLogger' object has no attribute 'warning'`
- Web app giờ xử lý lỗi khi thiếu AI modules một cách ổn định

## Test sau khi pull

### Test 1: Web Server
```bash
python3 run_web.py
# Không còn lỗi AttributeError
```

### Test 2: Servo Only
```bash
python3 test_servo_only.py
# Test servo không cần AI models
```

### Test 3: Main System (nếu có models)
```bash
python3 fruit_sorter.py
# Hoặc với chế độ MANUAL nếu không có models:
# Sửa utils/config.py: OPERATING_MODE = 'MANUAL'
```

## Nếu git pull gặp lỗi authentication

Nếu bạn gặp lỗi authentication khi pull, làm theo hướng dẫn này:

### Option 1: Pull mà không cần credentials (read-only)
```bash
# Đổi remote URL sang HTTPS read-only
git remote set-url origin https://github.com/TruongThiMinhPhuong/System_Conveyor.git

# Pull code
git pull origin main
```

### Option 2: Sử dụng Personal Access Token (nếu cần push)
```bash
# 1. Tạo Personal Access Token trên GitHub:
#    - Vào Settings > Developer settings > Personal access tokens
#    - Generate new token (classic)
#    - Chọn repo scope
#    - Copy token

# 2. Sử dụng token khi pull/push
git pull https://YOUR_TOKEN@github.com/TruongThiMinhPhuong/System_Conveyor.git main
```

## Kiểm tra code đã cập nhật

```bash
# Kiểm tra các file mới/thay đổi
ls -la test_servo_only.py   # File mới
ls -la docs/SERVO_MODES.md  # File mới

# Kiểm tra git status
git status
# Nên thấy: Your branch is up to date with 'origin/main'.
```

## Tóm tắt

1. **Pull code mới**: `git pull origin main`
2. **Test ngay**: `python3 test_servo_only.py`
3. **Chạy web app**: `python3 run_web.py` (không còn lỗi)

Nếu có vấn đề gì, hãy chạy `git log` và `git status` để kiểm tra.

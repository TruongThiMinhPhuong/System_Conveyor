# Fix lỗi picamera2 numpy incompatibility

## Vấn đề

```
⚠️ picamera2 not available: numpy.dtype size changed, may indicate binary incompatibility
Expected 96 from C header, got 88 from PyObject
```

**Nguyên nhân:** Numpy version không tương thích với picamera2 đã được compile.

## Giải pháp nhanh

### Option 1: Chạy fix script (Khuyến nghị)

```bash
cd ~/System_Conveyor
chmod +x fix_picamera2_numpy.sh
./fix_picamera2_numpy.sh
```

### Option 2: Manual fix

```bash
# Activate venv nếu có
source venv/bin/activate

# Uninstall cả hai
pip3 uninstall -y numpy picamera2

# Install numpy version tương thích
pip3 install "numpy<1.24"

# Reinstall picamera2
pip3 install picamera2

# Verify
python3 -c "import picamera2; print('OK')"
```

## Verify fix

```bash
# Test camera
python3 -c "from hardware import Camera; c = Camera(); c.initialize()"

# Hoặc run test script
python3 test_accuracy_improvements.py
```

## Nếu vẫn lỗi

### Option A: Rebuild picamera2

```bash
sudo apt update
sudo apt install -y python3-picamera2
```

### Option B: Sử dụng system picamera2

```bash
# Thay vì cài trong venv, dùng system package
sudo apt install -y python3-picamera2 python3-numpy

# Link vào venv (nếu dùng venv)
cd ~/System_Conveyor/venv/lib/python3.11/site-packages/
ln -s /usr/lib/python3/dist-packages/picamera2 .
ln -s /usr/lib/python3/dist-packages/libcamera .
```

## Root cause

Picamera2 được compile với một numpy version cụ thể. Khi update numpy sang version mới hơn, binary interface thay đổi → incompatibility.

**Solution:** Downgrade numpy về version tương thích (<1.24) hoặc rebuild picamera2.

## Tránh vấn đề trong tương lai

Trong `requirements-rpi.txt`, pin numpy version:

```txt
numpy<1.24,>=1.21.0
picamera2>=0.3.0
```

Sau đó:
```bash
pip3 install -r requirements-rpi.txt --upgrade
```

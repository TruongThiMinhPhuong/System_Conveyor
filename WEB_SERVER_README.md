# 🌐 Web Server Quick Start Guide

## Starting the Web Interface

### On Laptop (Development)

```bash
# Navigate to project directory
cd c:\Users\mgm\System_Conveyor

# Activate virtual environment (if using one)
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Start web server
python run_web.py
```

**Console Output:**
```
============================================================
🌐 AI Fruit Sorting System - Web Interface
============================================================

🔗 Access the interface at:
   🖥️  Localhost:    http://localhost:5001
   🌐 Local IP:     http://127.0.0.1:5001

💡 For network access, use your computer's IP address
📝 Press Ctrl+C to stop the server
```

### On Raspberry Pi

```bash
cd ~/System_Conveyor
source venv/bin/activate
python run_web.py
```

Access from any device on the network:
- From Pi: `http://localhost:5001`
- From laptop: `http://raspberrypi.local:5001`
- From mobile: `http://<pi-ip-address>:5001`

---

## Web Interface Features

### 📹 Camera Feed
- Real-time video stream from Pi Camera or USB webcam
- Shows bounding boxes for detected fruits
- Color-coded: Green (fresh), Red (spoiled)

### ⚙️ System Control
- **Start System**: Begins fruit sorting (conveyor + AI detection)
- **Stop System**: Stops all operations
- **System Status**: Shows if system is running/stopped
- **Uptime**: How long the system has been running

### 🔧 Motor Control
- **Start/Stop Motor**: Control conveyor belt
- **Speed Slider**: Adjust motor speed (0-100%)

### 🔄 Servo Control
- **Fresh (0°)**: Move to fresh fruit position
- **Center (90°)**: Neutral position
- **Spoiled (180°)**: Move to spoiled fruit position

### 📊 Statistics
- **Total Detected**: Count of all fruits detected
- **Fresh Sorted**: Number of fresh fruits sorted
- **Spoiled Sorted**: Number of spoiled fruits sorted
- **Errors**: System errors count
- **FPS**: Current processing speed
- **Reset Statistics**: Clear all counters

### 🎯 Last Detection
- Shows thumbnail of most recently detected fruit
- Classification result (fresh/spoiled)
- Confidence percentage
- Timestamp

---

## Features Available on PC vs Raspberry Pi

| Feature | PC (No Hardware) | Raspberry Pi |
|---------|------------------|--------------|
| Web UI | ✅ Full | ✅ Full |
| Camera Feed | ⚠️ Webcam only | ✅ Pi Camera |
| Motor Control | ❌ No GPIO | ✅ Full |
| Servo Control | ❌ No GPIO | ✅ Full |
| AI Detection | ✅ If models exist | ✅ Full |
| Statistics | ✅ Full | ✅ Full |

**Note:** On PC without hardware, GPIO-related features (motor, servo) will show error messages but won't crash the web interface.

---

## Port Configuration

Default port: **5001**

### Change Port (if 5001 is in use):

Edit `run_web.py` line 22:
```python
socketio.run(
    app,
    host='0.0.0.0',
    port=5001,  # Change this number
    debug=False,
    allow_unsafe_werkzeug=True
)
```

### Find Process Using Port 5001:

**Windows:**
```powershell
netstat -ano | findstr :5001
taskkill /PID <process_id> /F
```

**Linux/Pi:**
```bash
lsof -i :5001
kill -9 <process_id>
```

---

## Troubleshooting

### Web Server Won't Start

**Issue:** `Address already in use`
- **Solution:** Port 5001 is occupied. Change port or kill existing process.

**Issue:** `ModuleNotFoundError: No module named 'flask'`
- **Solution:** Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Web Page Loads But No Styles

**Issue:** CSS/JS files not loading (blank page)
- **Solution:** Check that `web/static/` folder exists with `css/` and `js/` subdirectories
- Verify file paths in browser console (F12)

### Camera Feed Shows "Waiting for feed..."

**On PC:**
- No camera connected, or camera in use by another app
- Close other camera apps (Zoom, Skype, etc.)

**On Pi:**
- Camera not enabled: `sudo raspi-config` → Interface → Camera → Enable
- Wrong camera module: Check `hardware/camera.py` configuration

### SocketIO Connection Failed

**Issue:** Red "Offline" indicator
- **Solution:** 
  - Check browser console for errors
  - Verify SocketIO library loaded: View page source, check for `socket.io.min.js`
  - Clear browser cache and reload

### System Starts But No Detection

**Issue:** Stats stay at 0, no fruits detected
- **Solution:**
  - Check that AI models exist in `ai_models/` folder
  - Models required:
    - `yolo_best.pt` (or `yolov8n.pt`)
    - `mobilenet_model_int8.tflite` (or `mobilenet_model.h5`)
  - If missing, train models using `TRAINING_DEPLOYMENT_GUIDE.md`

---

## Browser Compatibility

| Browser | Windows | Linux | macOS | Mobile |
|---------|---------|-------|-------|--------|
| Chrome | ✅ | ✅ | ✅ | ✅ |
| Firefox | ✅ | ✅ | ✅ | ✅ |
| Edge | ✅ | ✅ | ✅ | - |
| Safari | - | - | ✅ | ✅ |

**Recommended:** Chrome or Firefox for best compatibility.

---

## Security Notes

⚠️ **Important:**
- Web server runs on `0.0.0.0` (all network interfaces)
- **No authentication** - anyone on your network can access
- For production, add authentication or use firewall rules
- Secret key is hardcoded - change in production:
  ```python
  app.config['SECRET_KEY'] = 'your-secure-random-key'
  ```

---

## API Endpoints (Advanced)

For custom integrations or testing:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/api/status` | GET | Get system status JSON |
| `/api/start` | POST | Start sorting system |
| `/api/stop` | POST | Stop sorting system |
| `/api/motor/<action>` | POST | Control motor (start/stop/speed) |
| `/api/servo/<action>` | POST | Control servo (fresh/center/spoiled) |
| `/api/stats/reset` | POST | Reset statistics |
| `/video_feed` | GET | MJPEG video stream |

**Example:**
```bash
# Get system status
curl http://localhost:5001/api/status

# Start system
curl -X POST http://localhost:5001/api/start

# Move servo to fresh position
curl -X POST http://localhost:5001/api/servo/fresh
```

---

## Need Help?

- **Training Guide:** See `TRAINING_DEPLOYMENT_GUIDE.md`
- **Hardware Setup:** See `docs/` folder
- **Configuration:** Check `utils/config.py`
- **GitHub Issues:** https://github.com/TruongThiMinhPhuong/System_Conveyor/issues

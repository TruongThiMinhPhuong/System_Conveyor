# Web Interface Guide

Complete guide for using the web interface to control and monitor the AI Fruit Sorting System.

## Overview

The web interface provides a user-friendly dashboard to:
- Monitor live camera feed with AI detections
- Control system start/stop
- Adjust conveyor belt speed
- Test servo positions
- View real-time statistics
- Track sorting performance

## Starting the Web Server

### On Raspberry Pi

```bash
cd ~/System_Conveyor
source venv/bin/activate
python run_web.py
```

Expected output:
```
============================================================
🌐 AI Fruit Sorting System - Web Interface
============================================================

🔗 Access the interface at:
   Local: http://localhost:5000
   Network: http://<raspberry-pi-ip>:5000

📝 Press Ctrl+C to stop the server
```

### Find Your Raspberry Pi IP Address

```bash
hostname -I
# Output: 192.168.1.100 (example)
```

## Accessing the Interface

### From Raspberry Pi
Open browser and go to: `http://localhost:5000`

### From Another Computer/Phone
Open browser and go to: `http://192.168.1.100:5000`
(Replace with your Raspberry Pi IP)

## Interface Layout

### Header
- **System Title**: AI Fruit Sorting System
- **Connection Status**: Shows if connected to server (green = online, red = offline)

### Left Panel: Camera Feed
- **Live Video**: Real-time camera feed from Raspberry Pi
- **AI Overlays**: Bounding boxes and classifications drawn on detected fruits
- **FPS Counter**: Current processing speed
- **Last Detection**: Shows most recent fruit detected with confidence

### Right Panel: Controls & Statistics

#### 1. System Control
- **▶️ Start System**: Initialize hardware and start processing
- **⏹️ Stop System**: Stop processing and hardware
- **System Status**: Current state (Running/Stopped)
- **Uptime**: Time since system started

#### 2. Motor Control
- **Start Motor**: Start conveyor belt
- **Stop Motor**: Stop conveyor belt
- **Speed Slider**: Adjust belt speed (0-100%)

#### 3. Servo Control
- **⬅️ Left**: Move servo to fresh fruit position
- **↔️ Center**: Move servo to neutral position
- **➡️ Right**: Move servo to spoiled fruit position

#### 4. Statistics
- **Total Detected**: Number of fruits detected
- **Fresh Sorted**: Number classified as fresh
- **Spoiled Sorted**: Number classified as spoiled
- **Errors**: System errors encountered
- **Reset Statistics**: Clear all counts

## Using the Interface

### Basic Workflow

1. **Open Interface**
   - Navigate to the web address
   - Verify "Connected" status (green dot)

2. **Start System**
   - Click "▶️ Start System"
   - Wait for initialization (hardware + AI models)
   - Video feed should appear

3. **Monitor Operations**
   - Watch live video feed
   - Check statistics updating in real-time
   - Observe FPS and detection info

4. **Manual Controls** (optional)
   - Test servo positions before starting
   - Adjust motor speed during operation
   - Pause/resume conveyor as needed

5. **Stop System**
   - Click "⏹️ Stop System"
   - Hardware will be safely stopped
   - Video feed pauses

### Testing Components

#### Test Camera
1. Open interface
2. Video feed should show (even without starting system)
3. Check image quality and focus

#### Test Servo
1. Click servo buttons (Left/Center/Right)
2. Physically observe servo movement
3. Adjust angles in `utils/config.py` if needed

#### Test Motor
1. Click "Start Motor"
2. Observe conveyor belt motion
3. Adjust speed slider
4. Click "Stop Motor"

## Features Explained

### Real-Time Video Streaming
- MJPEG stream from Raspberry Pi camera
- Low latency (~100-200ms)
- AI detections overlaid on video
- Bounding boxes show detected fruits
- Color coding: Green = Fresh, Red = Spoiled

### SocketIO Real-Time Updates
- Statistics update every second
- No page refresh needed
- Connection status automatically monitored
- Instant notification on disconnect

### Interactive Controls
- All controls work via REST API
- Immediate feedback with toast notifications
- Error handling with user-friendly messages

### Statistics Dashboard
- Live counters for all metrics
- Color-coded cards for easy reading
- Uptime tracker
- Reset functionality for new sessions

## Keyboard Shortcuts

(Future enhancement - currently use mouse/touch)

## Mobile Access

The interface is fully responsive:
- **Smartphone**: Vertical layout, one panel at a time
- **Tablet**: Optimized grid layout
- **Desktop**: Full side-by-side layout

Test on mobile:
1. Ensure device on same WiFi as Raspberry Pi
2. Navigate to `http://<raspberry-pi-ip>:5000`
3. Use touch controls for buttons and slider

## Troubleshooting

### Cannot Access Interface

**Problem**: Browser cannot load the page

**Solutions**:
1. Check server is running:
```bash
ps aux | grep run_web.py
```

2. Verify port 5000 is not blocked:
```bash
sudo ufw allow 5000
```

3. Check Raspberry Pi IP:
```bash
hostname -I
```

4. Ping Raspberry Pi from computer:
```bash
ping <raspberry-pi-ip>
```

### Video Feed Not Showing

**Problem**: Black screen or "Waiting for feed..."

**Solutions**:
1. Check camera connection
2. Start the system (video only streams when system running)
3. Check browser console for errors (F12)
4. Try different browser (Chrome/Firefox recommended)

### "Disconnected" Status

**Problem**: Red dot, cannot control system

**Solutions**:
1. Refresh page
2. Check server logs for errors
3. Restart web server
4. Check network connection

### Slow Performance

**Problem**: Low FPS, laggy video

**Solutions**:
1. Reduce camera resolution in `utils/config.py`
2. Lower MAX_FPS setting
3. Use wired Ethernet instead of WiFi
4. Close other browsers/applications
5. Reduce YOLO input size

### Controls Not Responding

**Problem**: Buttons don't work

**Solutions**:
1. Check console for JavaScript errors (F12)
2. Verify system is initialized (click Start System)
3. Check SocketIO connection status
4. Refresh page

## Advanced Configuration

### Change Port

Edit `run_web.py`:
```python
socketio.run(app, host='0.0.0.0', port=8080)  # Change from 5000
```

### Enable HTTPS

For production with SSL:
```python
socketio.run(app, host='0.0.0.0', port=443, 
             keyfile='key.pem', certfile='cert.pem')
```

### Add Authentication

Edit `web/app.py` to add Flask-Login or basic auth.

### Customize Interface

- **HTML**: Edit `web/templates/index.html`
- **CSS**: Edit `web/static/css/style.css`
- **JavaScript**: Edit `web/static/js/app.js`

## API Endpoints

For custom integrations:

### GET /api/status
Returns system status
```json
{
  "initialized": true,
  "running": true,
  "stats": {...}
}
```

### POST /api/start
Start system
```json
{"success": true, "message": "System started"}
```

### POST /api/stop
Stop system

### POST /api/motor/start
Start motor with optional speed
```json
{"speed": 60}
```

### POST /api/servo/left
Move servo (left/center/right)

### POST /api/stats/reset
Reset statistics

## Auto-Start Web Server

Create systemd service:

```bash
sudo nano /etc/systemd/system/fruit-sorter-web.service
```

Content:
```ini
[Unit]
Description=AI Fruit Sorting Web Interface
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/System_Conveyor
ExecStart=/home/pi/System_Conveyor/venv/bin/python run_web.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl enable fruit-sorter-web
sudo systemctl start fruit-sorter-web
```

## Security Considerations

### For Production Use:
1. Change SECRET_KEY in `web/app.py`
2. Enable authentication
3. Use HTTPS
4. Limit CORS origins
5. Add rate limiting
6. Use reverse proxy (nginx)

### Basic Firewall:
```bash
sudo ufw allow 5000/tcp
sudo ufw enable
```

## Next Steps

1. ✅ Test interface on local network
2. ✅ Configure for your network
3. ✅ Add to startup (optional)
4. ✅ Customize appearance (optional)
5. ✅ Integrate with other systems via API

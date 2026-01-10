# Troubleshooting Web Interface Not Loading

## Vấn đề

Website không load nội dung - màn hình trống hoặc spinner loading mãi.

## Common Causes & Solutions

### 1. Check Web Server Status

```bash
# Trên Raspberry Pi, check xem server có đang chạy không
ps aux | grep python | grep run_web
```

**Nếu không thấy process:**
```bash
cd ~/System_Conveyor
python3 run_web.py
```

---

### 2. Check Port Conflicts

```bash
# Check port 5001 có bị chiếm không
sudo netstat -tulpn | grep 5001

# Hoặc
sudo lsof -i :5001
```

**Solution:** Kill process trên port 5001 hoặc đổi port trong `run_web.py`

---

### 3. Check Browser Console

**Mở Developer Tools (F12) trong browser:**
1. Tab **Console** - check JavaScript errors
2. Tab **Network** - check failed requests (CSS, JS, API calls)

**Common errors:**
- `404 Not Found` → File không tồn tại
- `500 Internal Server Error` → Server error (check terminal logs)
- `ERR_CONNECTION_REFUSED` → Server không chạy

---

### 4. Access from Correct URL

**Local access (trên Pi):**
```
http://localhost:5001
http://127.0.0.1:5001
```

**Remote access (từ máy khác):**
```
http://192.168.137.177:5001
http://<raspberry_pi_ip>:5001
```

**Check Pi IP:**
```bash
hostname -I
```

---

### 5. Check Static Files Exist

```bash
cd ~/System_Conveyor/web
ls -la static/css/
ls -la static/js/
ls -la templates/
```

**Should see:**
- `static/css/style.css`
- `static/js/app.js`
- `templates/index.html`

**If missing:** Pull latest code:
```bash
git pull origin main
```

---

### 6. Check Firewall

```bash
# Allow port 5001
sudo ufw allow 5001

# Or disable firewall temporarily
sudo ufw disable
```

---

### 7. Check Server Logs

**Look for errors in terminal:**
```
# Common errors:
[ERROR] - ❌ Initialization failed
ModuleNotFoundError: No module named 'flask'
```

**Solution for missing modules:**
```bash
source venv/bin/activate  # If using venv
pip3 install -r requirements.txt
```

---

### 8. Browser Cache Issue

**Clear browser cache:**
- Chrome/Edge: `Ctrl + Shift + Delete`
- Or try incognito/private mode: `Ctrl + Shift + N`

---

### 9. Socket.IO Connection Issues

**If you see "Offline" in top right:**

This is normal if system not started yet. Click "▶️ Start System" button.

**If stays offline after starting:**
1. Check browser console for WebSocket errors
2. Verify Socket.IO CDN loads: `https://cdn.socket.io/4.5.4/socket.io.min.js`
3. Check if port 5001 WebSocket accessible

---

## Quick Diagnosis

### Test 1: Can you access the server?

```bash
# From Raspberry Pi
curl http://localhost:5001

# Should return HTML
```

### Test 2: Check static files

```bash
curl http://localhost:5001/static/css/style.css
curl http://localhost:5001/static/js/app.js
```

### Test 3: Check API

```bash
curl http://localhost:5001/api/status
# Should return JSON:
# {"initialized":false,"running":false,"stats":{...}}
```

---

## Step-by-Step Debugging

### 1. Kill existing processes

```bash
pkill -f "python.*run_web"
```

### 2. Start server with verbose output

```bash
cd ~/System_Conveyor
python3 run_web.py
```

### 3. Check terminal output

Look for:
```
✅ Using tflite-runtime
🌐 AI Fruit Sorting System - Web Interface
🔗 Access the interface at:
   http://localhost:5001
```

### 4. Open browser

```
http://<raspberry_pi_ip>:5001
```

### 5. Open Developer Tools (F12)

Check:
- **Console tab**: Any red errors?
- **Network tab**: Any failed requests (red)?

---

## Common Fixes

### Fix 1: Module Import Errors

```bash
source venv/bin/activate
pip3 install flask flask-socketio flask-cors opencv-python
```

### Fix 2: Permission Errors

```bash
chmod +x run_web.py
chmod -R 755 web/
```

### Fix 3: Port Already in Use

```bash
# Find and kill process on port 5001
sudo lsof -ti:5001 | xargs sudo kill -9

# Or change port in run_web.py to 5002
```

### Fix 4: Static Files Not Found

```bash
# Ensure files exist
cd ~/System_Conveyor/web
ls static/css/style.css
ls static/js/app.js

# If missing, pull from git
git pull origin main
```

---

## Emergency Reset

If nothing works:

```bash
# 1. Stop everything
pkill -f python

# 2. Pull fresh code
cd ~/System_Conveyor
git stash
git pull origin main

# 3. Reinstall dependencies
source venv/bin/activate
pip3 install -r requirements.txt

# 4. Start clean
python3 run_web.py
```

---

## Still Not Working?

**Provide these details:**

1. **Error in terminal** (if any)
2. **Browser console errors** (F12 → Console tab)
3. **Output of:**
   ```bash
   cd ~/System_Conveyor
   python3 run_web.py
   ```

4. **Can you access:**
   - `http://localhost:5001` from Pi?
   - `http://<pi_ip>:5001` from another computer?

5. **Network tab** showing which requests fail?

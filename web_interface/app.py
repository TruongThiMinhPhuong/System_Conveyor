"""
Giao diện web Flask để điều khiển và giám sát hệ thống
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import threading
import time
import json
import logging
from typing import Optional
import numpy as np

from core.ai_processor import AIProcessor

app = Flask(__name__)

# Global processor instance
processor: Optional[AIProcessor] = None
latest_frame: Optional[bytes] = None
frame_lock = threading.Lock()
system_status = {}
status_lock = threading.Lock()

def frame_processed_callback(frame, status):
    """Callback khi frame được xử lý"""
    global latest_frame, system_status
    
    # Encode frame thành JPEG
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    frame_bytes = buffer.tobytes()
    
    with frame_lock:
        latest_frame = frame_bytes
    
    with status_lock:
        system_status = status

def fruit_classified_callback(track):
    """Callback khi trái cây được phân loại"""
    logging.info(f"Fruit classified in UI: {track.class_name}")

def init_processor():
    """Khởi tạo AI processor"""
    global processor
    
    if processor is not None:
        return processor
    
    processor = AIProcessor()
    
    # Set callbacks
    processor.on_frame_processed = frame_processed_callback
    processor.on_fruit_classified = fruit_classified_callback
    
    # Initialize
    if processor.initialize():
        processor.start_processing()
        return processor
    else:
        return None

def generate_frames():
    """Generator để stream video"""
    while True:
        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame
            else:
                frame = None
        
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            # Send blank frame
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS

@app.route('/')
def index():
    """Trang chủ"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Stream video"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    """API lấy trạng thái hệ thống"""
    with status_lock:
        if processor:
            status = processor.get_status()
        else:
            status = system_status
    
    return jsonify(status)

@app.route('/api/control/motor', methods=['POST'])
def control_motor():
    """API điều khiển động cơ"""
    if not processor:
        return jsonify({"error": "Processor not initialized"}), 400
    
    data = request.json
    action = data.get('action')
    
    if action == 'start':
        processor.motor_controller.start_conveyor()
    elif action == 'stop':
        processor.motor_controller.stop_conveyor()
    elif action == 'speed':
        speed = data.get('speed', 70)
        processor.set_motor_speed(speed)
    else:
        return jsonify({"error": "Invalid action"}), 400
    
    return jsonify({"success": True})

@app.route('/api/control/system', methods=['POST'])
def control_system():
    """API điều khiển hệ thống"""
    global processor
    
    data = request.json
    action = data.get('action')
    
    if action == 'start':
        if processor is None:
            processor = init_processor()
        else:
            processor.start_processing()
    
    elif action == 'stop':
        if processor:
            processor.stop()
            processor = None
    
    elif action == 'restart':
        if processor:
            processor.stop()
            processor = None
        processor = init_processor()
    
    else:
        return jsonify({"error": "Invalid action"}), 400
    
    return jsonify({"success": True})

@app.route('/api/test/servo', methods=['POST'])
def test_servo():
    """API test servo"""
    if not processor:
        return jsonify({"error": "Processor not initialized"}), 400
    
    data = request.json
    angle = data.get('angle', 90)
    
    # Điều khiển servo trực tiếp (chỉ để test)
    try:
        processor.motor_controller.servo_queue.put(angle)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    """API quản lý cấu hình"""
    if request.method == 'GET':
        # Trả về cấu hình hiện tại
        config = {
            "motor_speed": processor.motor_controller.current_motor_speed if processor else 70,
            "confidence_threshold": 0.65,
            "camera_fps": 30
        }
        return jsonify(config)
    
    else:  # POST
        data = request.json
        # TODO: Cập nhật cấu hình
        return jsonify({"success": True})

if __name__ == '__main__':
    # Khởi tạo processor
    processor = init_processor()
    
    # Chạy Flask server
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
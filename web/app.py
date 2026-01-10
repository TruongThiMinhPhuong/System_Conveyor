"""
Flask Web Application for AI Fruit Sorting System
Provides web interface for monitoring and control
"""

from flask import Flask, render_template, jsonify, request, Response
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import time
import threading
import cv2
import numpy as np
from pathlib import Path
import sys
import base64

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from hardware import ConveyorSystem
from ai_models import YOLODetector, MobileNetClassifier, ImagePreprocessor
from utils import Config, SystemLogger

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'fruit-sorter-secret-key-2025'
CORS(app)

# Use threading mode - more compatible with Pi environment
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global system instance
system_instance = None
system_running = False
system_lock = threading.Lock()

# Statistics
stats = {
    'total_detections': 0,
    'fresh_sorted': 0,
    'spoiled_sorted': 0,
    'errors': 0,
    'uptime': 0,
    'fps': 0,
    'last_detection': None,
    'last_detection_image': None  # NEW: Store last detection image
}


class WebFruitSorter:
    """Web-enabled fruit sorting system"""
    
    def __init__(self):
        """Initialize system"""
        self.logger = SystemLogger(name="WebFruitSorter")
        self.conveyor = None
        self.detector = None
        self.classifier = None
        self.preprocessor = None
        self.is_running = False
        self.is_initialized = False
        self.latest_frame = None
        self.start_time = None
        
    def initialize(self):
        """Initialize hardware and AI models"""
        try:
            Config.create_directories()
            
            # Initialize hardware
            self.conveyor = ConveyorSystem()
            if not self.conveyor.initialize():
                return False
            
            # Load AI models
            self.detector = YOLODetector(
                model_path=Config.YOLO_MODEL_PATH,
                confidence_threshold=Config.YOLO_CONFIDENCE_THRESHOLD
            )
            
            self.classifier = MobileNetClassifier(
                model_path=Config.MOBILENET_MODEL_PATH,
                input_size=Config.MOBILENET_INPUT_SIZE
            )
            
            self.preprocessor = ImagePreprocessor(
                target_size=(Config.MOBILENET_INPUT_SIZE, Config.MOBILENET_INPUT_SIZE)
            )
            
            # Try to load models (OK if they don't exist yet)
            self.detector.load_model()
            self.classifier.load_model()
            
            self.is_initialized = True
            self.logger.system_event("✅ Web system initialized")
            return True
            
        except Exception as e:
            self.logger.error("Initialization failed", e)
            return False
    
    def process_frame(self, frame):
        """Process frame and update statistics"""
        global stats
        
        try:
            # Check if detector exists and is loaded
            if not self.detector or not self.detector.is_loaded:
                return frame, None
            
            # Run detection
            detections = self.detector.detect(frame, verbose=False)
            
            if not detections:
                return frame, None
            
            detection = max(detections, key=lambda x: x['confidence'])
            stats['total_detections'] += 1
            stats['last_detection'] = {
                'class': detection['class_name'],
                'confidence': detection['confidence'],
                'time': time.strftime('%H:%M:%S')
            }
            
            # Classify if classifier exists and is loaded
            if self.classifier and self.classifier.is_loaded:
                bbox = detection['bbox']
                preprocessed = self.preprocessor.preprocess_complete_pipeline(frame, bbox)
                
                if preprocessed is not None:
                    classification = self.classifier.classify_with_details(preprocessed)
                    is_fresh = classification['is_fresh']
                    
                    if is_fresh:
                        stats['fresh_sorted'] += 1
                    else:
                        stats['spoiled_sorted'] += 1
                    
                    # Update last detection
                    stats['last_detection']['classification'] = classification['predicted_class']
                    stats['last_detection']['class_confidence'] = classification['confidence']
                    
                    # Capture detection image (ROI)
                    x1, y1, x2, y2 = bbox
                    roi = frame[y1:y2, x1:x2].copy()
                    # Encode to base64 for web transfer
                    ret, jpeg = cv2.imencode('.jpg', roi, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        stats['last_detection_image'] = base64.b64encode(jpeg).decode('utf-8')
                    
                    # Sort fruit
                    self.conveyor.sort_fruit(is_fresh, pause_conveyor=False)
                    
                    # Draw on frame
                    color = (0, 255, 0) if is_fresh else (255, 0, 0)
                    label = f"{detection['class_name']}: {classification['predicted_class']} ({classification['confidence']:.0%})"
                else:
                    color = (255, 255, 0)
                    label = f"{detection['class_name']} ({detection['confidence']:.0%})"
            else:
                color = (255, 255, 0)
                label = f"{detection['class_name']} ({detection['confidence']:.0%})"
            
            # Draw bounding box
            x1, y1, x2, y2 = detection['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            return frame, detection
            
        except Exception as e:
            stats['errors'] += 1
            self.logger.error("Processing error", e)
            return frame, None
    
    def run(self):
        """Main processing loop - optimized for speed"""
        self.is_running = True
        self.start_time = time.time()
        self.conveyor.start_conveyor(Config.CONVEYOR_SPEED_DEFAULT)
        
        frame_count = 0
        fps_start = time.time()
        last_process_time = 0
        
        while self.is_running:
            try:
                loop_start = time.time()
                
                # Capture frame
                frame = self.conveyor.capture_image()
                
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Always update latest frame for streaming
                self.latest_frame = frame
                
                # Process frame only at detection interval
                if time.time() - last_process_time >= Config.DETECTION_INTERVAL:
                    processed_frame, detection = self.process_frame(frame)
                    if processed_frame is not None:
                        self.latest_frame = processed_frame
                    last_process_time = time.time()
                
                # Calculate FPS
                frame_count += 1
                if time.time() - fps_start >= 1.0:
                    stats['fps'] = frame_count
                    stats['uptime'] = int(time.time() - self.start_time)
                    frame_count = 0
                    fps_start = time.time()
                    
                    # Emit statistics via SocketIO
                    socketio.emit('stats_update', stats)
                
                # Control FPS - adaptive sleep
                elapsed = time.time() - loop_start
                target_time = 1.0 / Config.MAX_FPS
                if elapsed < target_time:
                    time.sleep(target_time - elapsed)
                
            except Exception as e:
                stats['errors'] += 1
                self.logger.error("Loop error", e)
                time.sleep(0.1)
    
    def stop(self):
        """Stop system"""
        self.is_running = False
        if self.conveyor:
            self.conveyor.stop_conveyor()
        self.logger.system_event("System stopped")
    
    def get_frame_jpeg(self):
        """Get latest frame as JPEG with optimized compression"""
        if self.latest_frame is None:
            # Return blank frame
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "No camera feed", (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, jpeg = cv2.imencode('.jpg', blank, [cv2.IMWRITE_JPEG_QUALITY, 70])
        else:
            # Resize if needed for faster streaming
            frame = self.latest_frame
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640 / w
                frame = cv2.resize(frame, (640, int(h * scale)))
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        
        return jpeg.tobytes() if ret else b''


# Routes
@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')


@app.route('/api/status')
def get_status():
    """Get system status"""
    global system_instance, system_running
    
    return jsonify({
        'initialized': system_instance.is_initialized if system_instance else False,
        'running': system_running,
        'stats': stats
    })


@app.route('/api/start', methods=['POST'])
def start_system():
    """Start sorting system"""
    global system_instance, system_running
    
    with system_lock:
        if system_running:
            return jsonify({'success': False, 'message': 'Already running'})
        
        if not system_instance:
            system_instance = WebFruitSorter()
            if not system_instance.initialize():
                return jsonify({'success': False, 'message': 'Initialization failed'})
        
        # Start in background thread
        system_running = True
        thread = threading.Thread(target=system_instance.run, daemon=True)
        thread.start()
        
        return jsonify({'success': True, 'message': 'System started'})


@app.route('/api/stop', methods=['POST'])
def stop_system():
    """Stop sorting system"""
    global system_instance, system_running
    
    with system_lock:
        if not system_running:
            return jsonify({'success': False, 'message': 'Not running'})
        
        system_running = False
        if system_instance:
            system_instance.stop()
        
        return jsonify({'success': True, 'message': 'System stopped'})


@app.route('/api/motor/<action>', methods=['POST'])
def control_motor(action):
    """Control motor"""
    global system_instance
    
    if not system_instance or not system_instance.is_initialized:
        return jsonify({'success': False, 'message': 'System not initialized'})
    
    try:
        speed = request.json.get('speed', Config.CONVEYOR_SPEED_DEFAULT) if request.json else Config.CONVEYOR_SPEED_DEFAULT
        
        if action == 'start':
            system_instance.conveyor.start_conveyor(speed)
        elif action == 'stop':
            system_instance.conveyor.stop_conveyor()
        elif action == 'speed':
            system_instance.conveyor.motor.set_speed(speed)
        else:
            return jsonify({'success': False, 'message': 'Invalid action'})
        
        return jsonify({'success': True, 'message': f'Motor {action} executed'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/servo/<action>', methods=['POST'])
def control_servo(action):
    """Control servo"""
    global system_instance
    
    if not system_instance or not system_instance.is_initialized:
        return jsonify({'success': False, 'message': 'System not initialized'})
    
    try:
        if action == 'fresh':
            system_instance.conveyor.servo.move_to_fresh()
        elif action == 'spoiled':
            system_instance.conveyor.servo.move_to_spoiled()
        elif action == 'center':
            system_instance.conveyor.servo.move_to_center()
        else:
            return jsonify({'success': False, 'message': 'Invalid action'})
        
        return jsonify({'success': True, 'message': f'Servo moved to {action}'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/stats/reset', methods=['POST'])
def reset_stats():
    """Reset statistics"""
    global stats
    
    stats = {
        'total_detections': 0,
        'fresh_sorted': 0,
        'spoiled_sorted': 0,
        'errors': 0,
        'uptime': 0,
        'fps': 0,
        'last_detection': None,
        'last_detection_image': None
    }
    
    return jsonify({'success': True, 'message': 'Statistics reset'})


@app.route('/video_feed')
def video_feed():
    """Video streaming route - optimized for low latency"""
    def generate():
        global system_instance
        last_frame_time = 0
        min_frame_interval = 1.0 / 30  # Max 30 FPS for streaming
        
        while True:
            current_time = time.time()
            
            # Rate limiting
            if current_time - last_frame_time < min_frame_interval:
                time.sleep(0.01)
                continue
            
            if system_instance and system_instance.latest_frame is not None:
                frame_jpeg = system_instance.get_frame_jpeg()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_jpeg + b'\r\n')
                last_frame_time = current_time
            else:
                # Send placeholder when no frame
                time.sleep(0.05)
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# SocketIO events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('connected', {'message': 'Connected to fruit sorter'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')


@socketio.on('request_stats')
def handle_stats_request():
    """Send current statistics"""
    emit('stats_update', stats)


def create_app():
    """Application factory"""
    return app


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)

"""
System Configuration
Centralized configuration for AI Fruit Sorting System
"""

from pathlib import Path
from typing import Tuple


class Config:
    """System configuration class"""
    
    # ========== Paths ==========
    BASE_DIR = Path(__file__).parent.parent
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    DATA_DIR = BASE_DIR / "data"
    
    # Model paths
    YOLO_MODEL_PATH = str(MODELS_DIR / "yolov8n_fruit.pt")
    MOBILENET_MODEL_PATH = str(MODELS_DIR / "mobilenet_classifier.tflite")
    
    # ======== Camera Settings ========
    CAMERA_RESOLUTION = (640, 480)  # Lower resolution for faster processing
    CAMERA_FRAMERATE = 30
    CAMERA_ROTATION = 0  # 0, 90, 180, or 270
    
    # Web streaming settings
    WEB_STREAM_RESOLUTION = (640, 480)  # Resolution for web streaming
    JPEG_QUALITY = 70  # JPEG compression quality (1-100)
    
    # Camera adjustments
    CAMERA_BRIGHTNESS = 0.0  # -1.0 to 1.0
    CAMERA_CONTRAST = 1.0    # 0.0 to 2.0
    CAMERA_SATURATION = 1.0  # 0.0 to 2.0
    
    # ======== AI Model Settings ========
    
    # YOLO Detection
    YOLO_CONFIDENCE_THRESHOLD = 0.5  # Minimum detection confidence
    YOLO_IOU_THRESHOLD = 0.45        # IoU threshold for NMS
    YOLO_INPUT_SIZE = 640            # Input image size for YOLO
    
    # MobileNetV2 Classification
    MOBILENET_INPUT_SIZE = 224       # Input size (224x224)
    CLASSIFICATION_THRESHOLD = 0.6   # Minimum confidence for classification
    
    # Class names
    FRUIT_CLASSES = ['apple', 'orange', 'banana']  # Update based on your dataset
    FRESHNESS_CLASSES = ('Fresh', 'Spoiled')
    
    # ======== Hardware Settings ========
    
    # Conveyor motor speeds (0-100%) - Optimized for 20cm distance
    CONVEYOR_SPEED_DEFAULT = 35       # Slower for 20cm distance precision
    CONVEYOR_SPEED_FAST = 60          # Fast mode (no detection)
    CONVEYOR_SPEED_SLOW = 20          # Very slow for calibration
    CONVEYOR_SPEED_DETECTION = 35     # Speed during detection (2.92 cm/s)
    
    # Servo angles (0-180 degrees)
    SERVO_ANGLE_FRESH = 0      # Fresh fruit - Go straight (0°)
    SERVO_ANGLE_SPOILED = 180  # Spoiled fruit - Push right (180°)
    SERVO_ANGLE_CENTER = 90    # Neutral/default position
    
    # ======== Timing Settings ========
    
    # Camera to Servo Distance (measured in cm)
    CAMERA_TO_SERVO_DISTANCE = 20.0  # Distance from camera to servo gate
    
    # Detection and processing
    DETECTION_INTERVAL = 0.1         # Seconds between detections
    DETECTION_ZONE_DELAY = 0.5       # Time in detection zone
    PROCESSING_TIMEOUT = 1.5         # Max time for AI processing (increased)
    
    # Motor control timing - Optimized for 20cm distance
    SERVO_MOVE_DELAY = 0.6          # Time for servo to move (increased)
    CONVEYOR_STOP_DELAY = 0.4       # Pause time for sorting (increased)
    CONVEYOR_RESUME_DELAY = 0.3     # Delay before resume (increased)
    
    # Calculated travel time for 20cm at 35% speed (2.92 cm/s)
    # Travel time = 20cm / 2.92cm/s = 6.85 seconds
    FRUIT_TRAVEL_TIME = 6.85        # Time for fruit to travel from camera to servo
    
    # ======== Image Preprocessing ========
    
    # ROI extraction
    ROI_PADDING = 10  # Pixels to pad around detection
    
    # Preprocessing options
    APPLY_BLUR = True
    BLUR_KERNEL_SIZE = 5
    ENHANCE_CONTRAST = True
    
    # HSV color filtering (optional, for specific scenarios)
    USE_HSV_FILTER = False
    HSV_LOWER_BOUND = (0, 50, 50)
    HSV_UPPER_BOUND = (180, 255, 255)
    
    # ======== Logging Settings ========
    
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
    LOG_TO_FILE = True
    LOG_TO_CONSOLE = True
    LOG_MAX_SIZE = 10 * 1024 * 1024  # 10 MB
    LOG_BACKUP_COUNT = 5
    
    # ======== System Behavior ========
    
    # Operation modes
    AUTO_START = False  # Auto-start conveyor on system startup
    CONTINUOUS_MODE = True  # Continuous operation vs single-shot
    VISUAL_DEBUG = False  # Save annotated images for debugging
    
    # Performance
    MAX_FPS = 25  # Maximum processing FPS (increased)
    SKIP_FRAMES = 0  # Number of frames to skip between processing
    
    # Safety
    ENABLE_EMERGENCY_STOP = True
    MAX_CONSECUTIVE_ERRORS = 5  # Stop after N errors
    
    # ======== Statistics ========
    
    SAVE_STATISTICS = True
    STATS_UPDATE_INTERVAL = 10  # Seconds
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_models(cls) -> Tuple[bool, bool]:
        """
        Check if required models exist
        
        Returns:
            Tuple of (yolo_exists, mobilenet_exists)
        """
        yolo_exists = Path(cls.YOLO_MODEL_PATH).exists()
        mobilenet_exists = Path(cls.MOBILENET_MODEL_PATH).exists()
        
        return yolo_exists, mobilenet_exists
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 60)
        print("System Configuration")
        print("=" * 60)
        
        print(f"\n📁 Paths:")
        print(f"   Base directory: {cls.BASE_DIR}")
        print(f"   YOLO model: {cls.YOLO_MODEL_PATH}")
        print(f"   MobileNet model: {cls.MOBILENET_MODEL_PATH}")
        
        yolo_ok, mobilenet_ok = cls.validate_models()
        print(f"   YOLO model exists: {'✅' if yolo_ok else '❌'}")
        print(f"   MobileNet model exists: {'✅' if mobilenet_ok else '❌'}")
        
        print(f"\n📷 Camera:")
        print(f"   Resolution: {cls.CAMERA_RESOLUTION[0]}x{cls.CAMERA_RESOLUTION[1]}")
        print(f"   Framerate: {cls.CAMERA_FRAMERATE} FPS")
        
        print(f"\n🤖 AI Models:")
        print(f"   YOLO confidence: {cls.YOLO_CONFIDENCE_THRESHOLD}")
        print(f"   Classification threshold: {cls.CLASSIFICATION_THRESHOLD}")
        
        print(f"\n⚙️ Hardware:")
        print(f"   Conveyor speed: {cls.CONVEYOR_SPEED_DEFAULT}%")
        print(f"   Servo angles: Fresh={cls.SERVO_ANGLE_FRESH}° / Center={cls.SERVO_ANGLE_CENTER}° / Spoiled={cls.SERVO_ANGLE_SPOILED}°")
        
        print(f"\n📊 Operation:")
        print(f"   Auto start: {cls.AUTO_START}")
        print(f"   Continuous mode: {cls.CONTINUOUS_MODE}")
        print(f"   Max FPS: {cls.MAX_FPS}")


if __name__ == '__main__':
    Config.create_directories()
    Config.print_config()

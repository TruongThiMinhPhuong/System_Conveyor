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
    CAMERA_RESOLUTION = (416, 416)  # Balanced for accuracy/speed (increased from 320x320)
    CAMERA_FRAMERATE = 30
    CAMERA_ROTATION = 0  # 0, 90, 180, or 270
    
    # Web streaming settings
    WEB_STREAM_RESOLUTION = (640, 480)  # Resolution for web streaming
    JPEG_QUALITY = 70  # JPEG compression quality (1-100)
    
    # Camera adjustments for better image quality
    CAMERA_BRIGHTNESS = 0.05  # Slight increase for better visibility
    CAMERA_CONTRAST = 1.1    # Slight increase for better edges
    CAMERA_SATURATION = 1.05  # Slight increase for color distinction
    
    # ======== AI Model Settings ========
    
    # YOLO Detection
    YOLO_CONFIDENCE_THRESHOLD = 0.32  # Slightly lower for better detection (from 0.35)
    YOLO_IOU_THRESHOLD = 0.45         # IoU threshold for NMS
    YOLO_INPUT_SIZE = 416             # Increased for better accuracy (from 320)
    
    # MobileNetV2 Classification
    MOBILENET_INPUT_SIZE = 224       # Input size (224x224)
    CLASSIFICATION_THRESHOLD = 0.55   # Slightly lower threshold (from 0.6)
    
    # Class names - Fresh/Spoiled classification
    FRUIT_CLASSES = ['fresh', 'spoiled']  # tươi, hỏng
    FRESHNESS_CLASSES = ('Fresh', 'Spoiled')  # Tươi, Hỏng
    
    # ======== Hardware Settings ========
    
    # Conveyor motor speeds (0-100%) - Optimized for 20cm distance
    CONVEYOR_SPEED_DEFAULT = 70       # Default speed set to 70%
    CONVEYOR_SPEED_FAST = 60          # Fast mode (no detection)
    CONVEYOR_SPEED_SLOW = 20          # Very slow for calibration
    CONVEYOR_SPEED_DETECTION = 70     # Speed during detection
    
    # Servo angles (0-180 degrees)
    SERVO_ANGLE_FRESH = 0      # Fresh fruit - Go straight (0°)
    SERVO_ANGLE_SPOILED = 180  # Spoiled fruit - Push right (180°)
    SERVO_ANGLE_CENTER = 90    # Neutral/default position
    
    # ======== Timing Settings ========
    
    # Camera to Servo Distance (measured in cm)
    CAMERA_TO_SERVO_DISTANCE = 20.0  # Distance from camera to servo gate
    
    # Detection and processing (optimized for real-time)
    DETECTION_INTERVAL = 0.05        # Faster detection checks (was 0.1)
    DETECTION_ZONE_DELAY = 0.3       # Reduced for faster throughput
    PROCESSING_TIMEOUT = 0.5         # Tighter timeout for real-time (was 1.5)
    
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
    
    # Preprocessing options (balanced for accuracy/speed)
    FAST_PREPROCESSING = True  # Keep fast mode but improved settings
    APPLY_BLUR = True          # Enable blur for noise reduction
    BLUR_KERNEL_SIZE = 3       # Small kernel for speed
    ENHANCE_CONTRAST = True    # CLAHE for better classification (critical for accuracy)
    
    # Image quality validation
    CHECK_IMAGE_QUALITY = True         # Enable basic quality checks
    MIN_IMAGE_BRIGHTNESS = 20          # Minimum acceptable brightness
    MAX_IMAGE_BRIGHTNESS = 235         # Maximum acceptable brightness
    
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
    
    # Operating Modes
    # - NORMAL: Full AI (YOLO + MobileNet) required
    # - DETECTION_ONLY: Only YOLO required, simple classification rules
    # - MANUAL: No AI required, servo controlled by timer/manual
    OPERATING_MODE = 'NORMAL'  # Options: 'NORMAL', 'DETECTION_ONLY', 'MANUAL'
    
    # Classification behavior
    REQUIRE_CLASSIFICATION = True  # If False, servo works even without classification
    DEFAULT_CLASSIFICATION = 'FRESH'  # Fallback when classification fails: 'FRESH', 'SPOILED', 'SKIP'
    
    # Manual mode settings
    MANUAL_SORT_INTERVAL = 5.0  # Seconds between sorts in manual mode
    MANUAL_DEFAULT_FRESH = True  # Default classification in manual mode
    
    # Operation modes
    AUTO_START = False  # Auto-start conveyor on system startup
    CONTINUOUS_MODE = True  # Continuous operation vs single-shot
    VISUAL_DEBUG = False  # Save annotated images for debugging
    
    # Performance (optimized for Raspberry Pi 4)
    MAX_FPS = 30  # Maximum processing FPS
    SKIP_FRAMES = 0  # Process every frame
    USE_THREADING = True  # Enable multi-threading where possible
    
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

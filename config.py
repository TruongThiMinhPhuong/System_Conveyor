"""
Cấu hình toàn bộ hệ thống
"""

import os
from dataclasses import dataclass
from typing import Tuple

@dataclass
class HardwareConfig:
    """Cấu hình phần cứng"""
    # GPIO pins
    SERVO_PIN: int = 17
    MOTOR_IN1_PIN: int = 23
    MOTOR_IN2_PIN: int = 24
    MOTOR_ENA_PIN: int = 25
    SENSOR_PIN: int = 18  # Cảm biến IR
    
    # Thông số vật lý (mm)
    CAMERA_TO_SERVO_DISTANCE: float = 250.0  # Khoảng cách camera đến servo
    CONVEYOR_WIDTH: float = 200.0  # Chiều rộng băng chuyền
    FRUIT_DIAMETER_MAX: float = 80.0  # Đường kính trái cây lớn nhất
    
    # Thông số động cơ
    SERVO_ANGLE_PASS: int = 0      # Góc cho qua
    SERVO_ANGLE_REJECT: int = 90   # Góc đẩy bỏ
    SERVO_MOVE_TIME: float = 0.15  # Thời gian servo di chuyển (giây)
    
    # PWM
    MOTOR_PWM_FREQ: int = 1000     # Tần số PWM cho động cơ
    SERVO_PWM_FREQ: int = 50       # Tần số PWM cho servo
    MOTOR_SPEED_NORMAL: int = 70   # Tốc độ động cơ bình thường (0-100%)

@dataclass
class CameraConfig:
    """Cấu hình camera"""
    RESOLUTION: Tuple[int, int] = (640, 480)
    FPS: int = 30
    BRIGHTNESS: int = 50
    CONTRAST: int = 50
    ROTATION: int = 0  # Xoay ảnh nếu cần
    
    # Vùng quan tâm (ROI) - cắt phần ảnh cần xử lý
    ROI_X: int = 100
    ROI_Y: int = 100
    ROI_WIDTH: int = 440
    ROI_HEIGHT: int = 280

@dataclass
class AIConfig:
    """Cấu hình AI"""
    MODEL_PATH: str = "models/yolov8n_fruit.pt"
    CONFIDENCE_THRESHOLD: float = 0.65  # Ngưỡng tin cậy
    IOU_THRESHOLD: float = 0.45         # Ngưỡng IoU cho NMS
    
    # Các lớp phân loại
    CLASS_NAMES: dict = None
    
    # Hiệu suất
    USE_GPU: bool = False
    HALF_PRECISION: bool = False  # Sử dụng float16

@dataclass
class SystemConfig:
    """Cấu hình hệ thống"""
    LOG_LEVEL: str = "INFO"
    SAVE_IMAGES: bool = False
    IMAGE_SAVE_PATH: str = "data/processed"
    
    # Định thời
    PROCESSING_INTERVAL: float = 0.1  # Thời gian giữa các lần xử lý (giây)
    TRACKING_BUFFER_SIZE: int = 10    # Kích thước buffer theo dõi
    
    # Hiệu chuẩn
    PIXELS_PER_MM: float = 5.0  # Tỉ lệ pixel/mm
    CONVEYOR_SPEED_MM_PER_S: float = 100.0  # Tốc độ băng chuyền (mm/s)

@dataclass
class UIConfig:
    """Cấu hình giao diện"""
    WEB_HOST: str = "0.0.0.0"
    WEB_PORT: int = 5000
    DEBUG_MODE: bool = False
    SHOW_FPS: bool = True

# Khởi tạo cấu hình
hardware_config = HardwareConfig()
camera_config = CameraConfig()
ai_config = AIConfig()
system_config = SystemConfig()
ui_config = UIConfig()

# Ánh xạ nhãn
ai_config.CLASS_NAMES = {
    0: "apple_good",
    1: "apple_bad", 
    2: "orange_good",
    3: "orange_bad",
    4: "banana_good",
    5: "banana_bad"
}
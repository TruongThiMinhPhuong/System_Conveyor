"""
GPIO Configuration for AI Fruit Sorting Conveyor System
Defines all GPIO pin assignments for hardware components
"""

# ============================================
# GPIO Pin Assignments
# ============================================

# Camera - Uses dedicated CSI port (no GPIO pins)
CAMERA_CSI_PORT = 0  # Primary CSI camera port

# Servo Motor (MG996R) - PWM Control
SERVO_PIN = 18  # GPIO 18 (Physical Pin 12) - PWM capable

# L298N Motor Driver - DC Motor Control
# Motor A (Left side or primary motor)
MOTOR_ENA = 22  # GPIO 22 (Physical Pin 15) - Enable A (PWM for speed)
MOTOR_IN1 = 23  # GPIO 23 (Physical Pin 16) - Input 1 (Direction)
MOTOR_IN2 = 24  # GPIO 24 (Physical Pin 18) - Input 2 (Direction)

# Motor B (Right side or secondary motor if needed)
MOTOR_ENB = 17  # GPIO 17 (Physical Pin 11) - Enable B (PWM for speed)
MOTOR_IN3 = 27  # GPIO 27 (Physical Pin 13) - Input 3 (Direction)
MOTOR_IN4 = 25  # GPIO 25 (Physical Pin 22) - Input 4 (Direction)

# ============================================
# PWM Configuration
# ============================================

# Servo PWM Settings
SERVO_PWM_FREQUENCY = 50  # 50 Hz for standard servo
SERVO_MIN_DUTY = 2.5      # Minimum duty cycle (0 degrees)
SERVO_MAX_DUTY = 12.5     # Maximum duty cycle (180 degrees)
SERVO_CENTER_DUTY = 7.5   # Center position (90 degrees)

# Motor PWM Settings
MOTOR_PWM_FREQUENCY = 1000  # 1 kHz for motor speed control

# ============================================
# Servo Angles for Sorting (3-Way Classification)
# ============================================
# LEFT = Non-fruit objects (reject bin 1)
# CENTER = Fresh fruit (good bin - straight path)
# RIGHT = Spoiled fruit (reject bin 2)

SERVO_ANGLE_LEFT = 45      # Non-fruit objects → LEFT (reject bin 1)
SERVO_ANGLE_CENTER = 90    # Fresh fruit → CENTER (good bin - straight)
SERVO_ANGLE_RIGHT = 135    # Spoiled fruit → RIGHT (reject bin 2)

# ============================================
# Motor Speed Settings
# ============================================

CONVEYOR_SPEED_DEFAULT = 60  # Default speed (0-100%)
CONVEYOR_SPEED_FAST = 80     # Fast speed
CONVEYOR_SPEED_SLOW = 30     # Slow speed for detection

# ============================================
# Timing Configuration
# ============================================

DETECTION_ZONE_DELAY = 1.0   # Seconds to wait in detection zone
SERVO_MOVE_DELAY = 0.5       # Seconds for servo to complete movement
CONVEYOR_STOP_DELAY = 0.3    # Seconds to pause for sorting

# ============================================
# Safety Limits
# ============================================

MAX_SERVO_ANGLE = 180        # Maximum servo angle
MIN_SERVO_ANGLE = 0          # Minimum servo angle
EMERGENCY_STOP_PIN = None    # Optional emergency stop button

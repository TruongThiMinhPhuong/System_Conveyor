"""
Servo Motor Control Module
Controls MG996R servo for fruit sorting mechanism
"""

import time
import RPi.GPIO as GPIO
from typing import Optional
from . import gpio_config


class ServoControl:
    """
    Servo motor control class for MG996R
    Controls sorting gate mechanism
    """
    
    def __init__(self, pin: int = gpio_config.SERVO_PIN):
        """
        Initialize servo motor controller
        
        Args:
            pin: GPIO pin number for servo control
        """
        self.pin = pin
        self.pwm = None
        self.current_angle = gpio_config.SERVO_ANGLE_CENTER
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """
        Initialize servo motor with PWM
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"🔧 Initializing servo on GPIO {self.pin}...")
            
            # Setup GPIO mode
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.OUT)
            
            # Initialize PWM
            self.pwm = GPIO.PWM(self.pin, gpio_config.SERVO_PWM_FREQUENCY)
            self.pwm.start(gpio_config.SERVO_CENTER_DUTY)
            
            # Move to center position
            time.sleep(0.5)
            self.move_to_center()
            
            self.is_initialized = True
            print("✅ Servo initialized at center position")
            return True
            
        except Exception as e:
            print(f"❌ Servo initialization failed: {e}")
            self.is_initialized = False
            return False
    
    def _angle_to_duty_cycle(self, angle: float) -> float:
        """
        Convert angle to PWM duty cycle
        
        Args:
            angle: Servo angle (0-180 degrees)
            
        Returns:
            Duty cycle percentage
        """
        # Clamp angle to valid range
        angle = max(gpio_config.MIN_SERVO_ANGLE, 
                   min(gpio_config.MAX_SERVO_ANGLE, angle))
        
        # Linear mapping: 0° = 2.5%, 180° = 12.5%
        duty_cycle = gpio_config.SERVO_MIN_DUTY + (
            (angle / 180.0) * (gpio_config.SERVO_MAX_DUTY - gpio_config.SERVO_MIN_DUTY)
        )
        
        return duty_cycle
    
    def move_to_angle(self, angle: float, speed: float = 0.5) -> bool:
        """
        Move servo to specific angle
        
        Args:
            angle: Target angle (0-180 degrees)
            speed: Movement delay in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_initialized:
            print("⚠️ Servo not initialized.")
            return False
        
        try:
            duty_cycle = self._angle_to_duty_cycle(angle)
            self.pwm.ChangeDutyCycle(duty_cycle)
            time.sleep(speed)
            
            # Stop sending PWM signal to reduce jitter
            self.pwm.ChangeDutyCycle(0)
            
            self.current_angle = angle
            print(f"🔄 Servo moved to {angle}° (duty: {duty_cycle:.2f}%)")
            return True
            
        except Exception as e:
            print(f"❌ Servo movement failed: {e}")
            return False
    
    def move_to_left(self) -> bool:
        """
        Move servo to left position (fresh fruit)
        
        Returns:
            True if successful
        """
        print("⬅️ Sorting LEFT (Fresh)")
        return self.move_to_angle(
            gpio_config.SERVO_ANGLE_LEFT,
            gpio_config.SERVO_MOVE_DELAY
        )
    
    def move_to_right(self) -> bool:
        """
        Move servo to right position (spoiled fruit)
        
        Returns:
            True if successful
        """
        print("➡️ Sorting RIGHT (Spoiled)")
        return self.move_to_angle(
            gpio_config.SERVO_ANGLE_RIGHT,
            gpio_config.SERVO_MOVE_DELAY
        )
    
    def move_to_center(self) -> bool:
        """
        Move servo to center/neutral position
        
        Returns:
            True if successful
        """
        print("↔️ Servo to CENTER")
        return self.move_to_angle(
            gpio_config.SERVO_ANGLE_CENTER,
            gpio_config.SERVO_MOVE_DELAY
        )
    
    def sort_fruit(self, is_fresh: bool) -> bool:
        """
        Sort fruit based on classification result
        
        Args:
            is_fresh: True if fruit is fresh, False if spoiled
            
        Returns:
            True if successful
        """
        if is_fresh:
            return self.move_to_left()
        else:
            return self.move_to_right()
    
    def test_movement(self):
        """
        Test servo by moving through all positions
        """
        if not self.is_initialized:
            self.initialize()
        
        print("\n🧪 Testing servo movement...")
        
        positions = [
            ("Center", gpio_config.SERVO_ANGLE_CENTER),
            ("Left", gpio_config.SERVO_ANGLE_LEFT),
            ("Center", gpio_config.SERVO_ANGLE_CENTER),
            ("Right", gpio_config.SERVO_ANGLE_RIGHT),
            ("Center", gpio_config.SERVO_ANGLE_CENTER)
        ]
        
        for name, angle in positions:
            print(f"Moving to {name} ({angle}°)...")
            self.move_to_angle(angle, 1.0)
            time.sleep(1)
        
        print("✅ Servo test complete!")
    
    def cleanup(self):
        """
        Stop PWM and cleanup GPIO
        """
        if self.pwm:
            try:
                self.pwm.stop()
                print("🛑 Servo stopped")
            except Exception as e:
                print(f"⚠️ Error stopping servo: {e}")
        
        try:
            GPIO.cleanup(self.pin)
            self.is_initialized = False
            print("🧹 Servo GPIO cleaned up")
        except Exception as e:
            print(f"⚠️ Error cleaning up GPIO: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()


# Test code
if __name__ == "__main__":
    print("=== Servo Motor Test ===")
    
    try:
        with ServoControl() as servo:
            servo.test_movement()
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    print("\n✅ Test complete!")

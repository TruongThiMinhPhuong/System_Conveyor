"""
Servo Motor Control Module
Controls MG996R servo for fruit sorting mechanism
"""

import time
from typing import Optional
from . import gpio_config

# Try to import RPi.GPIO - may fail if not on Raspberry Pi
GPIO_AVAILABLE = False
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    print("⚠️ RPi.GPIO not available - running in simulation mode")
    GPIO = None


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
        self.simulation_mode = not GPIO_AVAILABLE
        
    def initialize(self) -> bool:
        """
        Initialize servo motor with PWM
        
        Returns:
            True if successful, False otherwise
        """
        if self.simulation_mode:
            print(f"🔧 Servo running in SIMULATION mode (GPIO {self.pin})")
            self.is_initialized = True
            return True
            
        try:
            print(f"🔧 Initializing servo on GPIO {self.pin}...")
            
            # Setup GPIO mode
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
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
            print("   Switching to simulation mode...")
            self.simulation_mode = True
            self.is_initialized = True
            return True
    
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
        
        if self.simulation_mode:
            self.current_angle = angle
            print(f"🔄 [SIM] Servo moved to {angle}°")
            time.sleep(speed * 0.1)  # Shorter delay in simulation
            return True
        
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
    
    def move_to_fresh(self) -> bool:
        """
        Move servo to fresh position (0° - straight through)

        Returns:
            True if successful
        """
        print("✅ Sorting FRESH (Straight - 0°)")
        return self.move_to_angle(
            gpio_config.SERVO_ANGLE_FRESH,
            gpio_config.SERVO_MOVE_DELAY
        )

    def move_to_spoiled(self) -> bool:
        """
        Move servo to spoiled position (180° - push right)

        Returns:
            True if successful
        """
        print("❌ Sorting SPOILED (Right - 180°)")
        return self.move_to_angle(
            gpio_config.SERVO_ANGLE_SPOILED,
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
            return self.move_to_fresh()
        else:
            return self.move_to_spoiled()
    
    def test_movement(self):
        """
        Test servo by moving through all positions
        """
        if not self.is_initialized:
            self.initialize()
        
        print("\n🧪 Testing servo movement...")
        
        positions = [
            ("Center", gpio_config.SERVO_ANGLE_CENTER),
            ("Fresh (0°)", gpio_config.SERVO_ANGLE_FRESH),
            ("Center", gpio_config.SERVO_ANGLE_CENTER),
            ("Spoiled (180°)", gpio_config.SERVO_ANGLE_SPOILED),
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
        if self.simulation_mode:
            self.is_initialized = False
            print("🧹 [SIM] Servo cleaned up")
            return
            
        if self.pwm:
            try:
                self.pwm.stop()
                print("🛑 Servo stopped")
            except Exception as e:
                print(f"⚠️ Error stopping servo: {e}")
        
        if GPIO_AVAILABLE:
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

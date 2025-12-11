"""
DC Motor Control Module
Controls conveyor belt motor via L298N driver
"""

import time
import RPi.GPIO as GPIO
from typing import Optional
from . import gpio_config


class MotorControl:
    """
    DC Motor control class for conveyor belt
    Uses L298N motor driver module
    """
    
    def __init__(
        self,
        ena_pin: int = gpio_config.MOTOR_ENA,
        in1_pin: int = gpio_config.MOTOR_IN1,
        in2_pin: int = gpio_config.MOTOR_IN2
    ):
        """
        Initialize motor controller
        
        Args:
            ena_pin: Enable pin (PWM for speed control)
            in1_pin: Input 1 (direction control)
            in2_pin: Input 2 (direction control)
        """
        self.ena_pin = ena_pin
        self.in1_pin = in1_pin
        self.in2_pin = in2_pin
        self.pwm = None
        self.current_speed = 0
        self.is_running = False
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """
        Initialize motor controller with GPIO pins
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"🔧 Initializing motor controller...")
            
            # Setup GPIO mode
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.ena_pin, GPIO.OUT)
            GPIO.setup(self.in1_pin, GPIO.OUT)
            GPIO.setup(self.in2_pin, GPIO.OUT)
            
            # Initialize PWM for speed control
            self.pwm = GPIO.PWM(self.ena_pin, gpio_config.MOTOR_PWM_FREQUENCY)
            self.pwm.start(0)  # Start with 0 speed
            
            # Ensure motor is stopped
            GPIO.output(self.in1_pin, GPIO.LOW)
            GPIO.output(self.in2_pin, GPIO.LOW)
            
            self.is_initialized = True
            print("✅ Motor controller initialized")
            return True
            
        except Exception as e:
            print(f"❌ Motor initialization failed: {e}")
            self.is_initialized = False
            return False
    
    def set_speed(self, speed: int):
        """
        Set motor speed
        
        Args:
            speed: Speed percentage (0-100)
        """
        if not self.is_initialized:
            print("⚠️ Motor not initialized.")
            return
        
        # Clamp speed to valid range
        speed = max(0, min(100, speed))
        
        try:
            self.pwm.ChangeDutyCycle(speed)
            self.current_speed = speed
            
            if speed > 0:
                print(f"⚡ Motor speed set to {speed}%")
        except Exception as e:
            print(f"❌ Failed to set speed: {e}")
    
    def start_forward(self, speed: int = gpio_config.CONVEYOR_SPEED_DEFAULT):
        """
        Start motor moving forward
        
        Args:
            speed: Speed percentage (0-100)
        """
        if not self.is_initialized:
            print("⚠️ Motor not initialized.")
            return
        
        try:
            # Set direction to forward
            GPIO.output(self.in1_pin, GPIO.HIGH)
            GPIO.output(self.in2_pin, GPIO.LOW)
            
            # Set speed
            self.set_speed(speed)
            
            self.is_running = True
            print(f"▶️ Motor started FORWARD at {speed}%")
            
        except Exception as e:
            print(f"❌ Failed to start motor: {e}")
    
    def start_reverse(self, speed: int = gpio_config.CONVEYOR_SPEED_DEFAULT):
        """
        Start motor moving in reverse
        
        Args:
            speed: Speed percentage (0-100)
        """
        if not self.is_initialized:
            print("⚠️ Motor not initialized.")
            return
        
        try:
            # Set direction to reverse
            GPIO.output(self.in1_pin, GPIO.LOW)
            GPIO.output(self.in2_pin, GPIO.HIGH)
            
            # Set speed
            self.set_speed(speed)
            
            self.is_running = True
            print(f"◀️ Motor started REVERSE at {speed}%")
            
        except Exception as e:
            print(f"❌ Failed to start motor: {e}")
    
    def stop(self):
        """
        Stop the motor
        """
        if not self.is_initialized:
            return
        
        try:
            # Stop motor by setting both inputs LOW
            GPIO.output(self.in1_pin, GPIO.LOW)
            GPIO.output(self.in2_pin, GPIO.LOW)
            
            # Set speed to 0
            self.set_speed(0)
            
            self.is_running = False
            print("⏹️ Motor stopped")
            
        except Exception as e:
            print(f"❌ Failed to stop motor: {e}")
    
    def brake(self):
        """
        Brake the motor (both inputs HIGH for quick stop)
        """
        if not self.is_initialized:
            return
        
        try:
            # Brake by setting both inputs HIGH
            GPIO.output(self.in1_pin, GPIO.HIGH)
            GPIO.output(self.in2_pin, GPIO.HIGH)
            
            self.is_running = False
            print("🛑 Motor braked")
            
            # After brief brake, set to stop
            time.sleep(0.1)
            self.stop()
            
        except Exception as e:
            print(f"❌ Failed to brake motor: {e}")
    
    def test(self):
        """
        Test motor by running through various speeds and directions
        """
        if not self.is_initialized:
            self.initialize()
        
        print("\n🧪 Testing motor...")
        
        # Test forward at different speeds
        speeds = [30, 50, 70, 100]
        
        for speed in speeds:
            print(f"\nForward at {speed}%...")
            self.start_forward(speed)
            time.sleep(2)
        
        print("\nStopping...")
        self.stop()
        time.sleep(1)
        
        # Test reverse
        print("\nReverse at 50%...")
        self.start_reverse(50)
        time.sleep(2)
        
        print("\nBraking...")
        self.brake()
        time.sleep(1)
        
        print("✅ Motor test complete!")
    
    def cleanup(self):
        """
        Stop motor and cleanup GPIO
        """
        self.stop()
        
        if self.pwm:
            try:
                self.pwm.stop()
            except Exception as e:
                print(f"⚠️ Error stopping PWM: {e}")
        
        try:
            GPIO.cleanup([self.ena_pin, self.in1_pin, self.in2_pin])
            self.is_initialized = False
            print("🧹 Motor GPIO cleaned up")
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
    print("=== DC Motor Test ===")
    
    try:
        with MotorControl() as motor:
            motor.test()
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    print("\n✅ Test complete!")

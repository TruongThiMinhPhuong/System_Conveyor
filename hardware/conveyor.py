"""
Conveyor System Orchestration
High-level control for the complete conveyor belt system
"""

import time
from typing import Optional
from .camera import Camera
from .servo_control import ServoControl
from .motor_control import MotorControl
from . import gpio_config


class ConveyorSystem:
    """
    High-level conveyor system controller
    Orchestrates camera, servo, and motor operations
    """
    
    def __init__(self):
        """Initialize conveyor system components"""
        self.camera = Camera()
        self.servo = ServoControl()
        self.motor = MotorControl()
        self.is_initialized = False
        self.is_running = False
        
    def initialize(self) -> bool:
        """
        Initialize all hardware components
        
        Returns:
            True if all components initialized successfully
        """
        print("🚀 Initializing conveyor system...")
        
        success = True
        
        # Initialize camera
        if not self.camera.initialize():
            print("❌ Camera initialization failed")
            success = False
        
        # Initialize servo
        if not self.servo.initialize():
            print("❌ Servo initialization failed")
            success = False
        
        # Initialize motor
        if not self.motor.initialize():
            print("❌ Motor initialization failed")
            success = False
        
        if success:
            self.is_initialized = True
            print("✅ Conveyor system initialized successfully!")
        else:
            print("❌ Conveyor system initialization failed")
            
        return success
    
    def start_conveyor(self, speed: int = gpio_config.CONVEYOR_SPEED_DEFAULT):
        """
        Start the conveyor belt moving
        
        Args:
            speed: Conveyor speed (0-100%)
        """
        if not self.is_initialized:
            print("⚠️ System not initialized")
            return
        
        self.motor.start_forward(speed)
        self.servo.move_to_center()
        self.is_running = True
        print(f"▶️ Conveyor system started at {speed}% speed")
    
    def stop_conveyor(self):
        """Stop the conveyor belt"""
        self.motor.stop()
        self.is_running = False
        print("⏹️ Conveyor system stopped")
    
    def pause_for_sorting(self):
        """Temporarily pause conveyor for sorting"""
        if self.motor.is_running:
            self.motor.stop()
            time.sleep(gpio_config.CONVEYOR_STOP_DELAY)
    
    def resume_after_sorting(self, speed: int = gpio_config.CONVEYOR_SPEED_DEFAULT):
        """Resume conveyor after sorting"""
        self.servo.move_to_center()
        time.sleep(gpio_config.SERVO_MOVE_DELAY)
        self.motor.start_forward(speed)
    
    def sort_fruit(
        self,
        is_fresh: bool = None,
        is_fruit: bool = True,
        pause_conveyor: bool = True
    ):
        """
        Execute sorting action based on object type
        
        Args:
            is_fresh: True if fresh, False if spoiled (ignored if not fruit)
            is_fruit: Whether the detected object is a fruit
            pause_conveyor: Whether to pause conveyor during sorting
        """
        if pause_conveyor:
            self.pause_for_sorting()
        
        # Move servo based on object type
        if not is_fruit:
            # Non-fruit object → Turn LEFT (reject bin 1)
            print("⚠️ Non-fruit object → LEFT")
            self.servo.move_to_left()
        elif is_fresh:
            # Fresh fruit → Go STRAIGHT (center - good bin)
            print("🍎 Fresh fruit → STRAIGHT")
            self.servo.move_to_center()
        else:
            # Spoiled fruit → Turn RIGHT (reject bin 2)
            print("🍂 Spoiled fruit → RIGHT")
            self.servo.move_to_right()
        
        # Wait for mechanical action to complete
        time.sleep(gpio_config.SERVO_MOVE_DELAY)
        
        if pause_conveyor:
            self.resume_after_sorting()
    
    def capture_image(self, save_path: Optional[str] = None):
        """
        Capture image from camera
        
        Args:
            save_path: Optional path to save image
            
        Returns:
            Captured frame as numpy array
        """
        return self.camera.capture_image(save_path)
    
    def emergency_stop(self):
        """Emergency stop - halt all operations immediately"""
        print("🚨 EMERGENCY STOP!")
        self.motor.brake()
        self.servo.move_to_center()
        self.is_running = False
    
    def run_test_cycle(self):
        """
        Run a complete test cycle of the conveyor system
        """
        print("\n🧪 Running conveyor system test cycle...\n")
        
        if not self.is_initialized:
            self.initialize()
        
        # Test 1: Start conveyor
        print("Test 1: Starting conveyor...")
        self.start_conveyor(50)
        time.sleep(3)
        
        # Test 2: Sort fresh fruit
        print("\nTest 2: Sorting FRESH fruit...")
        self.sort_fruit(is_fresh=True, pause_conveyor=True)
        time.sleep(2)
        
        # Test 3: Sort spoiled fruit
        print("\nTest 3: Sorting SPOILED fruit...")
        self.sort_fruit(is_fresh=False, pause_conveyor=True)
        time.sleep(2)
        
        # Test 4: Capture image
        print("\nTest 4: Capturing image...")
        self.capture_image("./test_conveyor_image.jpg")
        time.sleep(1)
        
        # Test 5: Stop
        print("\nTest 5: Stopping conveyor...")
        self.stop_conveyor()
        
        print("\n✅ Test cycle complete!")
    
    def cleanup(self):
        """Cleanup all hardware components"""
        print("🧹 Cleaning up conveyor system...")
        
        self.stop_conveyor()
        
        self.camera.close()
        self.servo.cleanup()
        self.motor.cleanup()
        
        self.is_initialized = False
        print("✅ Cleanup complete")
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()


# Test code
if __name__ == "__main__":
    print("=== Conveyor System Test ===")
    
    try:
        with ConveyorSystem() as conveyor:
            conveyor.run_test_cycle()
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    print("\n✅ All tests complete!")

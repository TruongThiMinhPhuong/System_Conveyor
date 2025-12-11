"""
Camera Module for Raspberry Pi
Handles camera initialization and image capture using picamera2
Falls back to simulation mode if camera not available
"""

import time
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image

# Try to import picamera2 - may fail if libcamera not installed
PICAMERA_AVAILABLE = False
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ picamera2 not available: {e}")
    print("   Running in simulation mode (no real camera)")
    Picamera2 = None


class Camera:
    """
    Camera control class for Raspberry Pi Camera Module v2 (5MP 1080p)
    """
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (1920, 1080),
        framerate: int = 30,
        rotation: int = 0
    ):
        """
        Initialize camera module
        
        Args:
            resolution: Camera resolution (width, height)
            framerate: Frames per second
            rotation: Camera rotation (0, 90, 180, 270)
        """
        self.resolution = resolution
        self.framerate = framerate
        self.rotation = rotation
        self.camera = None
        self.is_initialized = False
        self.simulation_mode = not PICAMERA_AVAILABLE
        
    def initialize(self) -> bool:
        """
        Initialize and configure the camera
        
        Returns:
            True if successful, False otherwise
        """
        if self.simulation_mode:
            print("🎥 Camera running in SIMULATION mode (no real hardware)")
            self.is_initialized = True
            return True
            
        try:
            print("🎥 Initializing camera...")
            self.camera = Picamera2()
            
            # Configure camera
            config = self.camera.create_still_configuration(
                main={"size": self.resolution, "format": "RGB888"}
            )
            self.camera.configure(config)
            
            # Start camera
            self.camera.start()
            time.sleep(2)  # Allow camera to warm up
            
            self.is_initialized = True
            print(f"✅ Camera initialized: {self.resolution[0]}x{self.resolution[1]}")
            return True
            
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            print("   Switching to simulation mode...")
            self.simulation_mode = True
            self.is_initialized = True
            return True
    
    def _generate_simulation_frame(self) -> np.ndarray:
        """Generate a simulated frame for testing"""
        frame = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        
        # Add gradient background
        for y in range(frame.shape[0]):
            frame[y, :, 0] = int(255 * y / frame.shape[0])
            frame[y, :, 2] = int(255 * (1 - y / frame.shape[0]))
        
        # Try to add text
        try:
            import cv2
            h, w = frame.shape[:2]
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, "SIMULATION MODE", (w//4, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.putText(frame, timestamp, (w//4, h//2 + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        except ImportError:
            pass
        
        return frame
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from camera
        
        Returns:
            Numpy array of the captured frame (RGB), or None if failed
        """
        if not self.is_initialized:
            print("⚠️ Camera not initialized. Call initialize() first.")
            return None
        
        if self.simulation_mode:
            return self._generate_simulation_frame()
        
        try:
            # Capture frame as numpy array
            frame = self.camera.capture_array()
            return frame
            
        except Exception as e:
            print(f"❌ Frame capture failed: {e}")
            return None
    
    def capture_image(self, save_path: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Capture and optionally save an image
        
        Args:
            save_path: Optional path to save the captured image
            
        Returns:
            Captured frame as numpy array, or None if failed
        """
        frame = self.capture_frame()
        
        if frame is not None and save_path:
            try:
                # Convert to PIL Image and save
                image = Image.fromarray(frame)
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                image.save(save_path)
                print(f"💾 Image saved: {save_path}")
            except Exception as e:
                print(f"❌ Failed to save image: {e}")
        
        return frame
    
    def set_camera_settings(
        self,
        brightness: Optional[float] = None,
        contrast: Optional[float] = None,
        saturation: Optional[float] = None
    ):
        """
        Adjust camera settings
        
        Args:
            brightness: Brightness level (-1.0 to 1.0)
            contrast: Contrast level (0.0 to 2.0)
            saturation: Saturation level (0.0 to 2.0)
        """
        if not self.is_initialized:
            print("⚠️ Camera not initialized.")
            return
        
        try:
            controls = {}
            
            if brightness is not None:
                controls["Brightness"] = brightness
            
            if contrast is not None:
                controls["Contrast"] = contrast
                
            if saturation is not None:
                controls["Saturation"] = saturation
            
            if controls:
                self.camera.set_controls(controls)
                print(f"📸 Camera settings updated: {controls}")
                
        except Exception as e:
            print(f"❌ Failed to update camera settings: {e}")
    
    def capture_test(self, save_dir: str = "./test_images") -> bool:
        """
        Capture a test image to verify camera is working
        
        Args:
            save_dir: Directory to save test image
            
        Returns:
            True if test successful, False otherwise
        """
        if not self.is_initialized:
            self.initialize()
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = f"{save_dir}/test_{timestamp}.jpg"
        
        frame = self.capture_image(save_path)
        
        if frame is not None:
            print(f"✅ Camera test successful! Image shape: {frame.shape}")
            return True
        else:
            print("❌ Camera test failed!")
            return False
    
    def close(self):
        """Stop and close the camera"""
        if self.simulation_mode:
            self.is_initialized = False
            print("📴 Simulation camera closed")
            return
            
        if self.camera:
            try:
                self.camera.stop()
                self.camera.close()
                self.is_initialized = False
                print("📴 Camera closed")
            except Exception as e:
                print(f"⚠️ Error closing camera: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Test code
if __name__ == "__main__":
    print("=== Camera Test ===")
    
    # Test camera with context manager
    with Camera() as cam:
        print("\n📷 Capturing test image...")
        cam.capture_test()
        
        print("\n📷 Capturing frames...")
        for i in range(3):
            frame = cam.capture_frame()
            if frame is not None:
                print(f"Frame {i+1}: {frame.shape}, dtype: {frame.dtype}")
            time.sleep(0.5)
    
    print("\n✅ Camera test complete!")

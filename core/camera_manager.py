"""
Quản lý camera Raspberry Pi
"""

import cv2
import numpy as np
import threading
import time
from queue import Queue
from typing import Optional, Tuple
import logging

from config import camera_config

logger = logging.getLogger(__name__)

class CameraManager:
    """Quản lý camera và thu nhận hình ảnh"""
    
    def __init__(self, use_picamera: bool = True):
        """
        Khởi tạo camera manager
        
        Args:
            use_picamera: Sử dụng Raspberry Pi Camera (True) hoặc USB Webcam (False)
        """
        self.use_picamera = use_picamera
        self.camera = None
        self.is_running = False
        self.current_frame = None
        self.frame_queue = Queue(maxsize=2)
        self.capture_thread = None
        self.lock = threading.Lock()
        
        # Thống kê
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        
    def initialize(self) -> bool:
        """Khởi tạo camera"""
        try:
            if self.use_picamera:
                # Sử dụng Pi Camera
                from picamera2 import Picamera2
                from libcamera import controls
                
                self.camera = Picamera2()
                
                # Cấu hình preview
                config = self.camera.create_preview_configuration(
                    main={"size": camera_config.RESOLUTION, "format": "RGB888"},
                    controls={
                        "AfMode": controls.AfModeEnum.Continuous,  # Auto focus
                        "ExposureTime": 20000,  # Thời gian phơi sáng
                        "AnalogueGain": 1.0,    # Độ khuếch đại
                    }
                )
                self.camera.configure(config)
                self.camera.start()
                
                logger.info("Pi Camera initialized successfully")
            else:
                # Sử dụng USB Webcam
                self.camera = cv2.VideoCapture(0)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.RESOLUTION[0])
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.RESOLUTION[1])
                self.camera.set(cv2.CAP_PROP_FPS, camera_config.FPS)
                self.camera.set(cv2.CAP_PROP_BRIGHTNESS, camera_config.BRIGHTNESS)
                self.camera.set(cv2.CAP_PROP_CONTRAST, camera_config.CONTRAST)
                
                if not self.camera.isOpened():
                    logger.error("Cannot open USB camera")
                    return False
                    
                logger.info("USB Webcam initialized successfully")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            return False
    
    def start_capture(self):
        """Bắt đầu luồng thu nhận hình ảnh"""
        if self.camera is None:
            if not self.initialize():
                return
        
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        logger.info("Camera capture started")
    
    def _capture_loop(self):
        """Vòng lặp thu nhận hình ảnh"""
        while self.is_running:
            try:
                frame = self._capture_single_frame()
                if frame is not None:
                    with self.lock:
                        self.current_frame = frame
                    
                    # Đưa vào queue nếu có consumer
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame.copy())
                    
                    # Tính FPS
                    self.frame_count += 1
                    current_time = time.time()
                    if current_time - self.last_fps_time >= 1.0:
                        self.fps = self.frame_count
                        self.frame_count = 0
                        self.last_fps_time = current_time
                
                time.sleep(0.01)  # Tránh chiếm CPU
                
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                time.sleep(0.1)
    
    def _capture_single_frame(self) -> Optional[np.ndarray]:
        """Thu nhận một frame"""
        try:
            if self.use_picamera and self.camera:
                # Pi Camera
                frame = self.camera.capture_array()
                # Chuyển RGB sang BGR cho OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif not self.use_picamera and self.camera:
                # USB Webcam
                ret, frame = self.camera.read()
                if not ret:
                    return None
            else:
                return None
            
            # Áp dụng xoay nếu cần
            if camera_config.ROTATION != 0:
                frame = self._rotate_frame(frame, camera_config.ROTATION)
            
            # Cắt ROI
            frame = self._apply_roi(frame)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None
    
    def _apply_roi(self, frame: np.ndarray) -> np.ndarray:
        """Áp dụng vùng quan tâm (ROI)"""
        height, width = frame.shape[:2]
        
        roi_x = max(0, min(camera_config.ROI_X, width - 1))
        roi_y = max(0, min(camera_config.ROI_Y, height - 1))
        roi_w = max(1, min(camera_config.ROI_WIDTH, width - roi_x))
        roi_h = max(1, min(camera_config.ROI_HEIGHT, height - roi_y))
        
        return frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    
    def _rotate_frame(self, frame: np.ndarray, angle: int) -> np.ndarray:
        """Xoay frame"""
        if angle == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Lấy frame hiện tại"""
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None
    
    def get_frame_from_queue(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Lấy frame từ queue"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except:
            return None
    
    def get_fps(self) -> float:
        """Lấy FPS hiện tại"""
        return self.fps
    
    def stop(self):
        """Dừng camera"""
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        
        if self.use_picamera and self.camera:
            self.camera.stop()
        elif not self.use_picamera and self.camera:
            self.camera.release()
        
        logger.info("Camera stopped")
    
    def capture_test_image(self, save_path: str = "test_image.jpg"):
        """Chụp ảnh test và lưu"""
        frame = self._capture_single_frame()
        if frame is not None:
            cv2.imwrite(save_path, frame)
            logger.info(f"Test image saved to {save_path}")
            return True
        return False
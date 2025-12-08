"""
Bộ xử lý AI chính - Kết hợp tất cả thành phần
"""

import time
import threading
import logging
from typing import Optional, List
import numpy as np

from core.camera_manager import CameraManager
from core.fruit_detector import FruitDetector, FruitDetection
from core.object_tracker import ObjectTracker, TrackedFruit
from core.motor_controller import MotorController
from config import system_config

logger = logging.getLogger(__name__)

class AIProcessor:
    """Bộ xử lý AI chính điều phối toàn bộ hệ thống"""
    
    def __init__(self):
        """Khởi tạo tất cả thành phần"""
        self.camera = CameraManager(use_picamera=True)
        self.detector = FruitDetector()
        self.tracker = ObjectTracker()
        self.motor_controller = MotorController()
        
        self.is_running = False
        self.processing_thread = None
        self.last_processing_time = 0
        
        # Thống kê hệ thống
        self.stats = {
            "total_frames_processed": 0,
            "total_fruits_detected": 0,
            "system_uptime": 0,
            "processing_fps": 0,
            "last_processing_duration": 0
        }
        
        # Callback cho UI
        self.on_frame_processed = None
        self.on_fruit_classified = None
        
    def initialize(self) -> bool:
        """Khởi tạo toàn bộ hệ thống"""
        try:
            logger.info("Initializing AI Processor...")
            
            # Khởi tạo camera
            if not self.camera.initialize():
                logger.error("Failed to initialize camera")
                return False
            
            # Khởi tạo AI detector
            if not self.detector.initialize():
                logger.error("Failed to initialize fruit detector")
                return False
            
            # Khởi tạo motor controller
            if not self.motor_controller.initialize():
                logger.error("Failed to initialize motor controller")
                return False
            
            # Khởi động camera capture
            self.camera.start_capture()
            
            # Khởi động băng chuyền
            self.motor_controller.start_conveyor()
            
            logger.info("AI Processor initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize AI Processor: {e}")
            return False
    
    def start_processing(self):
        """Bắt đầu xử lý"""
        if self.is_running:
            logger.warning("Processing already running")
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        logger.info("AI processing started")
    
    def _processing_loop(self):
        """Vòng lặp xử lý chính"""
        start_time = time.time()
        frame_count = 0
        
        while self.is_running:
            try:
                loop_start = time.time()
                
                # Lấy frame từ camera
                frame = self.camera.get_frame_from_queue(timeout=0.05)
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Xử lý frame
                processed_frame, classifications = self._process_frame(frame)
                
                # Cập nhật thống kê
                frame_count += 1
                self.stats["total_frames_processed"] = frame_count
                self.stats["system_uptime"] = time.time() - start_time
                
                # Tính FPS xử lý
                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    self.stats["processing_fps"] = frame_count / elapsed if elapsed > 0 else 0
                
                # Gọi callback cho UI
                if self.on_frame_processed and processed_frame is not None:
                    self.on_frame_processed(processed_frame, self._get_system_status())
                
                # Xử lý phân loại
                for classification in classifications:
                    self._handle_classification(classification)
                
                # Điều chỉnh tốc độ nếu cần
                self._adjust_conveyor_speed()
                
                # Đảm bảo không xử lý quá nhanh
                processing_time = time.time() - loop_start
                self.stats["last_processing_duration"] = processing_time
                
                sleep_time = max(0, system_config.PROCESSING_INTERVAL - processing_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(0.1)
    
    def _process_frame(self, frame: np.ndarray):
        """
        Xử lý một frame
        
        Returns:
            Tuple[processed_frame, classifications]
        """
        # Phát hiện trái cây
        detections = self.detector.detect(frame)
        self.stats["total_fruits_detected"] += len(detections)
        
        # Lọc detections
        detections = self.detector.filter_by_size(detections)
        detections = self.detector.filter_by_position(detections)
        
        # Cập nhật tracker
        active_tracks = self.tracker.update(detections)
        
        # Vẽ kết quả lên frame
        processed_frame = self._visualize_results(frame, detections, active_tracks)
        
        # Lấy các track cần phân loại
        classifications = []
        for track in active_tracks:
            if track.is_classified and not track.classification_time:
                classifications.append(track)
        
        return processed_frame, classifications
    
    def _visualize_results(self, frame: np.ndarray, 
                          detections: List[FruitDetection],
                          tracks: List[TrackedFruit]) -> np.ndarray:
        """Vẽ kết quả lên frame"""
        result_frame = frame.copy()
        
        # Vẽ bounding boxes cho detections
        for det in detections:
            color = (0, 255, 0) if det.is_good else (0, 0, 255)  # Xanh cho tốt, Đỏ cho xấu
            
            # Vẽ bounding box
            x1, y1, x2, y2 = map(int, det.bbox)
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            # Vẽ label
            label = f"{det.class_name}: {det.confidence:.2f}"
            if det.track_id:
                label = f"ID{det.track_id}: {label}"
            
            # Background cho text
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(result_frame, (x1, y1 - text_height - 4), 
                         (x1 + text_width, y1), color, -1)
            
            # Text
            cv2.putText(result_frame, label, (x1, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Vẽ tâm
            cx, cy = map(int, det.centroid)
            cv2.circle(result_frame, (cx, cy), 3, color, -1)
        
        # Vẽ đường phân loại
        height, width = frame.shape[:2]
        classification_line_y = 400  # Vị trí đường phân loại
        cv2.line(result_frame, (0, classification_line_y), 
                (width, classification_line_y), (255, 255, 0), 2)
        
        # Vẽ thông tin hệ thống
        self._draw_system_info(result_frame)
        
        return result_frame
    
    def _draw_system_info(self, frame: np.ndarray):
        """Vẽ thông tin hệ thống lên frame"""
        info_lines = [
            f"FPS: {self.camera.get_fps():.1f}",
            f"Proc FPS: {self.stats['processing_fps']:.1f}",
            f"Detections: {self.stats['total_fruits_detected']}",
            f"Tracks: {self.tracker.get_track_count()}",
            f"Motor: {self.motor_controller.current_motor_speed}%",
            f"Inference: {self.detector.get_average_inference_time():.1f}ms"
        ]
        
        y_offset = 30
        for line in info_lines:
            cv2.putText(frame, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25
    
    def _handle_classification(self, track: TrackedFruit):
        """Xử lý phân loại trái cây"""
        logger.info(f"Classifying track {track.track_id}: {track.class_name} "
                   f"(Good: {track.is_good})")
        
        # Điều khiển servo phân loại
        self.motor_controller.classify_fruit(track.is_good)
        
        # Gọi callback
        if self.on_fruit_classified:
            self.on_fruit_classified(track)
    
    def _adjust_conveyor_speed(self):
        """Tự động điều chỉnh tốc độ băng chuyền"""
        # Đơn giản: giữ tốc độ cố định
        # Có thể phức tạp hơn dựa trên số lượng detection
        pass
    
    def _get_system_status(self) -> dict:
        """Lấy trạng thái toàn bộ hệ thống"""
        status = {
            **self.stats,
            **self.detector.get_stats(),
            **self.tracker.get_stats(),
            **self.motor_controller.get_status(),
            "camera_fps": self.camera.get_fps(),
            "is_running": self.is_running
        }
        return status
    
    def stop(self):
        """Dừng toàn bộ hệ thống"""
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        
        self.camera.stop()
        self.motor_controller.cleanup()
        
        logger.info("AI Processor stopped")
    
    def set_motor_speed(self, speed: int):
        """Đặt tốc độ động cơ"""
        self.motor_controller.set_motor_speed(speed)
    
    def get_status(self) -> dict:
        """Lấy trạng thái hệ thống"""
        return self._get_system_status()
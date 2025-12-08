"""
Phát hiện và phân loại trái cây bằng YOLOv8
"""

import cv2
import numpy as np
import torch
import time
from typing import List, Tuple, Optional, Dict
import logging
from ultralytics import YOLO
from dataclasses import dataclass

from config import ai_config, system_config

logger = logging.getLogger(__name__)

@dataclass
class FruitDetection:
    """Lưu thông tin phát hiện trái cây"""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    centroid: Tuple[float, float]  # Tâm của bounding box
    timestamp: float
    track_id: Optional[int] = None
    
    @property
    def is_good(self) -> bool:
        """Kiểm tra xem trái cây có tốt không"""
        return "good" in self.class_name.lower()
    
    @property
    def width(self) -> float:
        """Chiều rộng bounding box"""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        """Chiều cao bounding box"""
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self) -> float:
        """Diện tích bounding box"""
        return self.width * self.height

class FruitDetector:
    """Phát hiện và phân loại trái cây"""
    
    def __init__(self):
        """Khởi tạo mô hình YOLOv8"""
        self.model = None
        self.device = None
        self.is_initialized = False
        self.class_names = {}
        
        # Thống kê
        self.inference_times = []
        self.total_detections = 0
        self.last_inference_time = 0
        
    def initialize(self) -> bool:
        """Khởi tạo mô hình AI"""
        try:
            # Kiểm tra GPU
            if torch.cuda.is_available() and ai_config.USE_GPU:
                self.device = torch.device("cuda")
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU")
            
            # Tải mô hình
            logger.info(f"Loading model from {ai_config.MODEL_PATH}")
            self.model = YOLO(ai_config.MODEL_PATH)
            
            # Đẩy mô hình lên device
            self.model.to(self.device)
            
            # Lấy tên các lớp
            if self.model.names:
                self.class_names = self.model.names
            else:
                self.class_names = ai_config.CLASS_NAMES
            
            # Kiểm tra mô hình
            self._test_model()
            
            self.is_initialized = True
            logger.info("Fruit detector initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize fruit detector: {e}")
            return False
    
    def _test_model(self):
        """Kiểm tra mô hình với ảnh dummy"""
        try:
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            results = self.model(dummy_image, verbose=False)
            logger.info("Model test passed")
        except Exception as e:
            logger.error(f"Model test failed: {e}")
    
    def detect(self, image: np.ndarray) -> List[FruitDetection]:
        """
        Phát hiện trái cây trong ảnh
        
        Args:
            image: Ảnh đầu vào (BGR format)
            
        Returns:
            Danh sách các phát hiện
        """
        if not self.is_initialized or self.model is None:
            logger.error("Fruit detector not initialized")
            return []
        
        start_time = time.time()
        
        try:
            # Chuyển đổi ảnh nếu cần
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # Chạy inference
            results = self.model(
                image,
                conf=ai_config.CONFIDENCE_THRESHOLD,
                iou=ai_config.IOU_THRESHOLD,
                verbose=False,
                device=self.device
            )
            
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
                    confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
                    
                    for box, conf, cls_id in zip(boxes, confidences, class_ids):
                        # Chuyển đổi sang định dạng của chúng ta
                        x1, y1, x2, y2 = box
                        
                        # Lấy tên lớp
                        class_name = self.class_names.get(cls_id, f"class_{cls_id}")
                        
                        # Tính tâm
                        centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                        
                        # Tạo detection object
                        detection = FruitDetection(
                            bbox=(float(x1), float(y1), float(x2), float(y2)),
                            confidence=float(conf),
                            class_id=int(cls_id),
                            class_name=class_name,
                            centroid=centroid,
                            timestamp=time.time()
                        )
                        
                        detections.append(detection)
                        self.total_detections += 1
            
            # Tính thời gian inference
            inference_time = (time.time() - start_time) * 1000  # ms
            self.inference_times.append(inference_time)
            self.last_inference_time = inference_time
            
            # Giữ chỉ 100 giá trị gần nhất
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)
            
            logger.debug(f"Detection completed: {len(detections)} fruits found "
                        f"in {inference_time:.1f} ms")
            
            return detections
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return []
    
    def filter_by_size(self, detections: List[FruitDetection], 
                      min_area: float = 1000, 
                      max_area: float = 50000) -> List[FruitDetection]:
        """Lọc detection theo kích thước"""
        filtered = []
        for det in detections:
            if min_area <= det.area <= max_area:
                filtered.append(det)
        return filtered
    
    def filter_by_position(self, detections: List[FruitDetection],
                          y_threshold: float = 100) -> List[FruitDetection]:
        """Lọc detection theo vị trí (chỉ lấy những cái ở giữa ảnh)"""
        filtered = []
        for det in detections:
            # Chỉ xử lý trái cây ở phần dưới ảnh (gần camera)
            if det.centroid[1] > y_threshold:
                filtered.append(det)
        return filtered
    
    def get_average_inference_time(self) -> float:
        """Lấy thời gian inference trung bình"""
        if not self.inference_times:
            return 0
        return sum(self.inference_times) / len(self.inference_times)
    
    def get_stats(self) -> Dict:
        """Lấy thống kê"""
        return {
            "total_detections": self.total_detections,
            "avg_inference_time_ms": self.get_average_inference_time(),
            "last_inference_time_ms": self.last_inference_time,
            "device": str(self.device),
            "is_initialized": self.is_initialized
        }
"""
YOLOv8 Detection Module
Fruit detection using YOLOv8-nano from Ultralytics
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from ultralytics import YOLO


class YOLODetector:
    """
    YOLOv8-nano object detection wrapper
    Detects fruits in images
    """
    
    def __init__(
        self,
        model_path: str = "./models/yolov8n_fruit.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45
    ):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to trained YOLO model
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """
        Load YOLO model
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"🤖 Loading YOLO model from {self.model_path}...")
            
            if not Path(self.model_path).exists():
                print(f"⚠️ Model file not found: {self.model_path}")
                print("   Using YOLOv8n pretrained model for testing...")
                # Use pretrained model for testing
                self.model = YOLO('yolov8n.pt')
            else:
                self.model = YOLO(self.model_path)
            
            self.is_loaded = True
            print("✅ YOLO model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            self.is_loaded = False
            return False
    
    def detect(
        self,
        image: np.ndarray,
        verbose: bool = False
    ) -> List[dict]:
        """
        Detect fruits in image
        
        Args:
            image: Input RGB image (numpy array)
            verbose: Whether to print detection details
            
        Returns:
            List of detections, each containing:
            - bbox: (x1, y1, x2, y2)
            - confidence: detection confidence
            - class_id: class ID
            - class_name: class name
        """
        if not self.is_loaded:
            print("⚠️ Model not loaded. Call load_model() first.")
            return []
        
        try:
            # Run inference with optimizations for speed
            results = self.model(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=verbose,
                imgsz=320,  # Force 320x320 input for speed
                half=False,  # FP16 not supported on CPU, keep False
                device='cpu'  # Explicit CPU device
            )
            
            detections = []
            
            # Parse results
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Get confidence and class
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = result.names[class_id]
                    
                    detection = {
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name
                    }
                    
                    detections.append(detection)
                    
                    if verbose:
                        print(f"🎯 Detected: {class_name} "
                              f"(conf: {confidence:.2f}, "
                              f"bbox: {x1},{y1},{x2},{y2})")
            
            return detections
            
        except Exception as e:
            print(f"❌ Detection failed: {e}")
            return []
    
    def detect_fruits(
        self,
        image: np.ndarray,
        fruit_classes: Optional[List[str]] = None
    ) -> List[dict]:
        """
        Detect specific fruit classes
        
        Args:
            image: Input image
            fruit_classes: List of fruit class names to detect (None = all)
            
        Returns:
            List of fruit detections
        """
        all_detections = self.detect(image)
        
        if fruit_classes is None:
            return all_detections
        
        # Filter by fruit classes
        filtered = [
            det for det in all_detections
            if det['class_name'] in fruit_classes
        ]
        
        return filtered
    
    def get_highest_confidence_detection(
        self,
        image: np.ndarray
    ) -> Optional[dict]:
        """
        Get the detection with highest confidence
        
        Args:
            image: Input image
            
        Returns:
            Detection with highest confidence, or None if no detections
        """
        detections = self.detect(image)
        
        if not detections:
            return None
        
        # Sort by confidence and return highest
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        return detections[0]
    
    def visualize_detections(
        self,
        image: np.ndarray,
        detections: List[dict],
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Draw detections on image
        
        Args:
            image: Input image
            detections: List of detections
            save_path: Optional path to save visualization
            
        Returns:
            Image with drawn detections
        """
        import cv2
        
        image_viz = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']
            
            # Draw bounding box
            cv2.rectangle(image_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(
                image_viz,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(image_viz, cv2.COLOR_RGB2BGR))
            print(f"💾 Visualization saved: {save_path}")
        
        return image_viz
    
    def test(self, test_image_path: Optional[str] = None):
        """
        Test detector with an image
        
        Args:
            test_image_path: Path to test image (uses random if None)
        """
        if not self.is_loaded:
            self.load_model()
        
        # Create or load test image
        if test_image_path and Path(test_image_path).exists():
            import cv2
            image = cv2.imread(test_image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Create random test image
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        print(f"\n🧪 Testing YOLO detector...")
        print(f"   Image shape: {image.shape}")
        
        # Run detection
        detections = self.detect(image, verbose=True)
        
        print(f"\n✅ Found {len(detections)} detections")
        
        if detections:
            # Visualize
            self.visualize_detections(image, detections, "./test_yolo_detection.jpg")


# Test code
if __name__ == "__main__":
    print("=== YOLO Detector Test ===")
    
    detector = YOLODetector()
    detector.test()
    
    print("\n✅ YOLO detector test complete!")

"""
Image Preprocessing Module
OpenCV-based preprocessing for fruit images
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class ImagePreprocessor:
    """
    Image preprocessing pipeline using OpenCV
    Prepares images for AI model inference
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        blur_kernel: int = 5,
        fast_mode: bool = True
    ):
        """
        Initialize preprocessor
        
        Args:
            target_size: Target image size for model input (width, height)
            blur_kernel: Gaussian blur kernel size (odd number)
            fast_mode: Use faster preprocessing (reduced quality for real-time)
        """
        self.target_size = target_size
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.fast_mode = fast_mode
        
        # CLAHE settings optimized for fast mode
        if fast_mode:
            self.clahe_tile_size = (4, 4)  # Smaller tiles = faster
            self.clahe_clip_limit = 2.0    # Lower clip = faster
        else:
            self.clahe_tile_size = (8, 8)  # Better quality
            self.clahe_clip_limit = 3.0
        
    def extract_roi(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        padding: int = 10
    ) -> Optional[np.ndarray]:
        """
        Extract Region of Interest (ROI) from image using bounding box
        
        Args:
            image: Input image (numpy array)
            bbox: Bounding box (x1, y1, x2, y2)
            padding: Padding around bounding box in pixels
            
        Returns:
            Extracted ROI as numpy array, or None if invalid
        """
        try:
            x1, y1, x2, y2 = bbox
            
            # Add padding
            h, w = image.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # Extract ROI
            roi = image[y1:y2, x1:x2]
            
            if roi.size == 0:
                return None
                
            return roi
            
        except Exception as e:
            print(f"⚠️ ROI extraction failed: {e}")
            return None
    
    def apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur to reduce noise
        
        Args:
            image: Input image
            
        Returns:
            Blurred image
        """
        return cv2.GaussianBlur(image, (self.blur_kernel, self.blur_kernel), 0)
    
    def filter_hsv_color(
        self,
        image: np.ndarray,
        lower_bound: Tuple[int, int, int],
        upper_bound: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Filter image by HSV color range
        
        Args:
            image: Input RGB image
            lower_bound: Lower HSV bound (H, S, V)
            upper_bound: Upper HSV bound (H, S, V)
            
        Returns:
            Binary mask where 255 = color in range, 0 = out of range
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Create mask
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        return mask
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using CLAHE (optimized for fast_mode)
        
        Args:
            image: Input image
            
        Returns:
            Contrast-enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel with optimized settings
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_size
        )
        l = clahe.apply(l)
        
        # Merge channels
        enhanced = cv2.merge([l, a, b])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def resize_image(
        self,
        image: np.ndarray,
        size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Resize image to target size
        
        Args:
            image: Input image
            size: Target size (width, height), defaults to self.target_size
            
        Returns:
            Resized image
        """
        if size is None:
            size = self.target_size
        
        return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image for neural network input
        Scales pixel values to [0, 1]
        
        Args:
            image: Input image (0-255)
            
        Returns:
            Normalized image (0.0-1.0)
        """
        return image.astype(np.float32) / 255.0
    
    def preprocess_for_classification(
        self,
        image: np.ndarray,
        apply_blur: bool = True,
        enhance: bool = True
    ) -> np.ndarray:
        """
        Complete preprocessing pipeline for MobileNetV2 classification
        
        Args:
            image: Input RGB image
            apply_blur: Whether to apply Gaussian blur
            enhance: Whether to enhance contrast
            
        Returns:
            Preprocessed image ready for model input
        """
        # Apply blur if requested
        if apply_blur:
            image = self.apply_gaussian_blur(image)
        
        # Enhance contrast if requested
        if enhance:
            image = self.enhance_contrast(image)
        
        # Resize to target size
        image = self.resize_image(image)
        
        # Normalize
        image = self.normalize_image(image)
        
        return image
    
    def preprocess_complete_pipeline(
        self,
        full_image: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Complete preprocessing: ROI extraction + preprocessing
        
        Args:
            full_image: Full captured image
            bbox: Bounding box from YOLO detection
            
        Returns:
            Preprocessed ROI ready for classification, or None if failed
        """
        # Extract ROI
        roi = self.extract_roi(full_image, bbox)
        
        if roi is None:
            return None
        
        # Preprocess for classification
        preprocessed = self.preprocess_for_classification(roi)
        
        return preprocessed
    
    def draw_bbox(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        label: str = "",
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw bounding box on image for visualization
        
        Args:
            image: Input image
            bbox: Bounding box (x1, y1, x2, y2)
            label: Label text to display
            color: Box color (R, G, B)
            thickness: Line thickness
            
        Returns:
            Image with drawn bounding box
        """
        image_copy = image.copy()
        x1, y1, x2, y2 = bbox
        
        # Draw rectangle
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            
            # Get text size
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # Draw background rectangle for text
            cv2.rectangle(
                image_copy,
                (x1, y1 - text_h - 10),
                (x1 + text_w, y1),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                image_copy,
                label,
                (x1, y1 - 5),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness
            )
        
        return image_copy


# Test code
if __name__ == "__main__":
    print("=== Image Preprocessor Test ===")
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    preprocessor = ImagePreprocessor()
    
    # Test ROI extraction
    bbox = (100, 100, 300, 300)
    roi = preprocessor.extract_roi(test_image, bbox)
    print(f"✅ ROI extracted: {roi.shape}")
    
    # Test preprocessing
    preprocessed = preprocessor.preprocess_for_classification(roi)
    print(f"✅ Preprocessed: {preprocessed.shape}, dtype: {preprocessed.dtype}")
    print(f"   Value range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")
    
    # Test complete pipeline
    result = preprocessor.preprocess_complete_pipeline(test_image, bbox)
    print(f"✅ Complete pipeline: {result.shape}")
    
    print("\n✅ Preprocessing test complete!")

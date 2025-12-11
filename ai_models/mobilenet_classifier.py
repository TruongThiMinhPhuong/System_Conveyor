"""
MobileNetV2 Classification Module
Fresh/Spoiled fruit classification using TensorFlow Lite
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite


class MobileNetClassifier:
    """
    MobileNetV2 TFLite classifier for fresh/spoiled fruit classification
    """
    
    def __init__(
        self,
        model_path: str = "./models/mobilenet_classifier.tflite",
        input_size: int = 224,
        class_names: Tuple[str, str] = ("Fresh", "Spoiled")
    ):
        """
        Initialize MobileNetV2 classifier
        
        Args:
            model_path: Path to TFLite model
            input_size: Model input size (224 for MobileNetV2)
            class_names: Tuple of (fresh_label, spoiled_label)
        """
        self.model_path = model_path
        self.input_size = input_size
        self.class_names = class_names
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """
        Load TFLite model
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"🤖 Loading MobileNetV2 model from {self.model_path}...")
            
            if not Path(self.model_path).exists():
                print(f"⚠️ Model file not found: {self.model_path}")
                print("   Please train the model first using training/mobilenet/train_mobilenet.py")
                return False
            
            # Load TFLite model
            self.interpreter = tflite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            self.is_loaded = True
            
            print("✅ MobileNetV2 model loaded successfully")
            print(f"   Input shape: {self.input_details[0]['shape']}")
            print(f"   Output shape: {self.output_details[0]['shape']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load MobileNetV2 model: {e}")
            self.is_loaded = False
            return False
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input
        
        Args:
            image: Input image (already normalized to [0, 1])
            
        Returns:
            Preprocessed batch ready for model
        """
        # Ensure correct shape
        if len(image.shape) == 3:
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
        
        # Ensure float32 type
        image = image.astype(np.float32)
        
        return image
    
    def classify(
        self,
        image: np.ndarray,
        return_probabilities: bool = False
    ) -> Tuple[str, float]:
        """
        Classify fruit as fresh or spoiled
        
        Args:
            image: Preprocessed input image (224x224x3, normalized)
            return_probabilities: Whether to return all probabilities
            
        Returns:
            Tuple of (class_name, confidence)
            If return_probabilities=True: (class_name, confidence, [fresh_prob, spoiled_prob])
        """
        if not self.is_loaded:
            print("⚠️ Model not loaded. Call load_model() first.")
            return ("Unknown", 0.0)
        
        try:
            # Preprocess image
            input_data = self.preprocess_image(image)
            
            # Set input tensor
            self.interpreter.set_tensor(
                self.input_details[0]['index'],
                input_data
            )
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            output_data = self.interpreter.get_tensor(
                self.output_details[0]['index']
            )[0]
            
            # Get prediction
            if len(output_data) == 2:
                # Binary classification: [fresh_prob, spoiled_prob]
                fresh_prob = float(output_data[0])
                spoiled_prob = float(output_data[1])
            else:
                # Single output (sigmoid): fresh_prob
                fresh_prob = float(output_data[0])
                spoiled_prob = 1.0 - fresh_prob
            
            # Determine class
            if fresh_prob > spoiled_prob:
                class_name = self.class_names[0]  # Fresh
                confidence = fresh_prob
            else:
                class_name = self.class_names[1]  # Spoiled
                confidence = spoiled_prob
            
            if return_probabilities:
                return (class_name, confidence, [fresh_prob, spoiled_prob])
            else:
                return (class_name, confidence)
            
        except Exception as e:
            print(f"❌ Classification failed: {e}")
            return ("Error", 0.0)
    
    def is_fresh(self, image: np.ndarray, threshold: float = 0.5) -> bool:
        """
        Check if fruit is fresh
        
        Args:
            image: Preprocessed input image
            threshold: Confidence threshold
            
        Returns:
            True if fresh, False if spoiled
        """
        class_name, confidence = self.classify(image)
        
        return (class_name == self.class_names[0] and confidence >= threshold)
    
    def classify_with_details(self, image: np.ndarray) -> dict:
        """
        Classify and return detailed results
        
        Args:
            image: Preprocessed input image
            
        Returns:
            Dictionary with classification details
        """
        class_name, confidence, probs = self.classify(
            image,
            return_probabilities=True
        )
        
        return {
            'predicted_class': class_name,
            'confidence': confidence,
            'is_fresh': class_name == self.class_names[0],
            'fresh_probability': probs[0],
            'spoiled_probability': probs[1]
        }
    
    def test(self):
        """
        Test classifier with random input
        """
        if not self.is_loaded:
            if not self.load_model():
                print("❌ Cannot run test without model")
                return
        
        print(f"\n🧪 Testing MobileNetV2 classifier...")
        
        # Create random test image
        test_image = np.random.rand(self.input_size, self.input_size, 3).astype(np.float32)
        
        print(f"   Input shape: {test_image.shape}")
        
        # Run classification
        result = self.classify_with_details(test_image)
        
        print(f"\n📊 Classification Results:")
        print(f"   Predicted: {result['predicted_class']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Fresh probability: {result['fresh_probability']:.2%}")
        print(f"   Spoiled probability: {result['spoiled_probability']:.2%}")
        print(f"   Is Fresh: {result['is_fresh']}")
        
        print("\n✅ Classifier test complete!")


# Test code
if __name__ == "__main__":
    print("=== MobileNetV2 Classifier Test ===")
    
    classifier = MobileNetClassifier()
    
    # Try to load model (will fail if not trained yet)
    if classifier.load_model():
        classifier.test()
    else:
        print("\n⚠️ Model not found. Please train the model first.")
        print("   Run: python training/mobilenet/train_mobilenet.py")
    
    print("\n✅ Test complete!")

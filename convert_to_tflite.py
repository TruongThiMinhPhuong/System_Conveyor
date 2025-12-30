"""
Convert Keras model to TensorFlow Lite for Raspberry Pi deployment
"""

import tensorflow as tf
import os
from pathlib import Path

def convert_keras_to_tflite(keras_path, tflite_path, optimize=True):
    """
    Convert Keras model to TFLite with optimizations

    Args:
        keras_path: Path to Keras model (.keras)
        tflite_path: Path to save TFLite model (.tflite)
        optimize: Whether to apply optimizations
    """
    print(f"📦 Loading Keras model: {keras_path}")

    # Load the Keras model
    model = tf.keras.models.load_model(keras_path)
    print("✅ Keras model loaded")

    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if optimize:
        # Apply optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Use float16 for smaller size and faster inference
        converter.target_spec.supported_types = [tf.float16]

        print("⚡ Applying optimizations (DEFAULT + float16)")

    # Convert to TFLite
    print("🔄 Converting to TFLite...")
    tflite_model = converter.convert()

    # Save TFLite model
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    # Get file sizes
    keras_size = os.path.getsize(keras_path) / (1024 * 1024)  # MB
    tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)  # MB

    print("✅ Conversion complete!")
    print(f"   Keras model size: {keras_size:.2f} MB")
    print(f"   TFLite model size: {tflite_size:.2f} MB")
    print(f"   Size reduction: {(1 - tflite_size/keras_size)*100:.1f}%")

    return tflite_size

def main():
    """Main conversion function"""
    print("🍎 Keras to TFLite Converter")
    print("=" * 40)

    # Paths
    models_dir = Path("mobilenet_training")
    keras_model = models_dir / "mobilenet_fruit_classifier_best.keras"
    tflite_model = Path("models") / "mobilenet_classifier.tflite"

    # Ensure models directory exists
    tflite_model.parent.mkdir(exist_ok=True)

    # Check if Keras model exists
    if not keras_model.exists():
        print(f"❌ Keras model not found: {keras_model}")
        print("   Please train the model first using Train_MobileNet_Colab.ipynb")
        return

    print(f"🔍 Found Keras model: {keras_model}")

    # Convert to TFLite
    try:
        convert_keras_to_tflite(keras_model, tflite_model, optimize=True)

        print("\n🎉 Success!")
        print(f"   TFLite model saved to: {tflite_model}")
        print("   Ready for deployment on Raspberry Pi!")
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

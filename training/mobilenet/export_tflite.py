"""
Export MobileNetV2 Model to TensorFlow Lite
Optimize for Raspberry Pi inference
"""

import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import numpy as np


def convert_to_tflite(
    model_path: str,
    output_path: str = '../../models/mobilenet_classifier.tflite',
    quantize: bool = True
):
    """
    Convert Keras model to TensorFlow Lite
    
    Args:
        model_path: Path to Keras model file
        output_path: Output path for TFLite model
        quantize: Whether to apply quantization
    """
    print("=" * 60)
    print("Converting Model to TensorFlow Lite")
    print("=" * 60)
    
    # Load Keras model
    print(f"\n📦 Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    
    print("✅ Model loaded")
    print(f"\n📊 Model Summary:")
    model.summary()
    
    # Convert to TFLite
    print(f"\n🔄 Converting to TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply optimizations
    if quantize:
        print("   Applying dynamic range quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert
    tflite_model = converter.convert()
    
    # Save TFLite model
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✅ TFLite model saved to {output_path}")
    
    # Get file sizes
    keras_size = Path(model_path).stat().st_size / (1024 * 1024)
    tflite_size = output_file.stat().st_size / (1024 * 1024)
    
    print(f"\n📊 Model Sizes:")
    print(f"   Keras model: {keras_size:.2f} MB")
    print(f"   TFLite model: {tflite_size:.2f} MB")
    print(f"   Compression: {(1 - tflite_size/keras_size) * 100:.1f}%")
    
    # Test TFLite model
    test_tflite_model(output_path)
    
    return output_path


def test_tflite_model(model_path: str):
    """
    Test TFLite model inference
    
    Args:
        model_path: Path to TFLite model
    """
    print(f"\n🧪 Testing TFLite model...")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Input dtype: {input_details[0]['dtype']}")
    print(f"   Output shape: {output_details[0]['shape']}")
    print(f"   Output dtype: {output_details[0]['dtype']}")
    
    # Create random test input
    input_shape = input_details[0]['shape']
    test_input = np.random.rand(*input_shape).astype(np.float32)
    test_input = (test_input - 0.5) * 2  # Normalize to [-1, 1]
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    
    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"   Output: {output[0]}")
    print(f"   Predicted class: {np.argmax(output[0])}")
    print(f"   Confidence: {np.max(output[0]):.2%}")
    
    print("✅ TFLite model test successful!")


def batch_convert_models(models_dir: str, output_dir: str):
    """
    Convert multiple Keras models to TFLite
    
    Args:
        models_dir: Directory containing Keras models
        output_dir: Output directory for TFLite models
    """
    models_path = Path(models_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all .keras files
    keras_models = list(models_path.glob('*.keras')) + list(models_path.glob('*.h5'))
    
    print(f"Found {len(keras_models)} models to convert")
    
    for model_file in keras_models:
        print(f"\n{'='*60}")
        print(f"Converting {model_file.name}...")
        
        output_file = output_path / f"{model_file.stem}.tflite"
        
        try:
            convert_to_tflite(
                model_path=str(model_file),
                output_path=str(output_file),
                quantize=True
            )
        except Exception as e:
            print(f"❌ Failed to convert {model_file.name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Export Keras model to TensorFlow Lite'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to Keras model file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../../models/mobilenet_classifier.tflite',
        help='Output path for TFLite model'
    )
    parser.add_argument(
        '--quantize',
        action='store_true',
        default=True,
        help='Apply quantization (default: True)'
    )
    parser.add_argument(
        '--batch',
        type=str,
        help='Convert all models in directory'
    )
    
    args = parser.parse_args()
    
    if args.batch:
        batch_convert_models(args.batch, args.output)
    else:
        convert_to_tflite(
            model_path=args.model,
            output_path=args.output,
            quantize=args.quantize
        )
    
    print(f"\n🎉 Conversion complete!")
    print(f"📦 Copy the TFLite model to Raspberry Pi:")
    print(f"   scp {args.output} pi@<raspberry-pi-ip>:~/System_Conveyor/models/")


if __name__ == '__main__':
    main()

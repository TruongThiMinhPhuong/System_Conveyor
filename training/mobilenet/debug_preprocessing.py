"""
Preprocessing Debug and Test Script
Test the complete preprocessing pipeline to ensure consistency
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ai_models import MobileNetClassifier, ImagePreprocessor
import matplotlib.pyplot as plt


def test_preprocessing_pipeline(image_path: str, output_dir: str = './debug_output'):
    """
    Test preprocessing pipeline and show visualizations
    
    Args:
        image_path: Path to test image
        output_dir: Output directory for debug images
    """
    print("=" * 60)
    print("Preprocessing Pipeline Debug Test")
    print("=" * 60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load image
    print(f"\n📷 Loading image from {image_path}...")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Failed to load image from {image_path}")
        return
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"✅ Image loaded: {image_rgb.shape}")
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(target_size=(224, 224))
    
    # Test different preprocessing steps
    print(f"\n🔍 Testing preprocessing steps...")
    
    # Step 1: Gaussian blur
    blurred = preprocessor.apply_gaussian_blur(image_rgb)
    print(f"   1. Gaussian blur: {blurred.shape}")
    
    # Step 2: Contrast enhancement
    enhanced = preprocessor.enhance_contrast(blurred)
    print(f"   2. Contrast enhanced: {enhanced.shape}")
    
    # Step 3: Resize
    resized = preprocessor.resize_image(enhanced)
    print(f"   3. Resized: {resized.shape}")
    
    # Step 4: Normalize
    normalized = preprocessor.normalize_image(resized)
    print(f"   4. Normalized: {normalized.shape}, range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    # MobileNetV2 preprocessing (convert to [-1, 1])
    mobilenet_input = normalized * 2.0 - 1.0
    print(f"   5. MobileNetV2 input: range: [{mobilenet_input.min():.3f}, {mobilenet_input.max():.3f}]")
    
    # Visualize preprocessing steps
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('1. Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(blurred)
    axes[0, 1].set_title('2. Gaussian Blur')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(enhanced)
    axes[0, 2].set_title('3. Contrast Enhanced')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(resized)
    axes[1, 0].set_title('4. Resized (224x224)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(normalized)
    axes[1, 1].set_title('5. Normalized [0, 1]')
    axes[1, 1].axis('off')
    
    # Show MobileNetV2 input (denormalized for visualization)
    mobilenet_viz = (mobilenet_input + 1.0) / 2.0
    axes[1, 2].imshow(mobilenet_viz)
    axes[1, 2].set_title('6. MobileNetV2 Input [-1, 1]\n(denormalized for viz)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / 'preprocessing_steps.png', dpi=150)
    print(f"\n💾 Preprocessing visualization saved to {output_path / 'preprocessing_steps.png'}")
    plt.close()
    
    # Test with MobileNet classifier
    print(f"\n🤖 Testing with MobileNet classifier...")
    
    classifier = MobileNetClassifier(
        model_path="./models/mobilenet_classifier.tflite",
        input_size=224
    )
    
    if classifier.load_model():
        # Classify
        result = classifier.classify_with_details(normalized)
        
        print(f"\n📊 Classification Results:")
        print(f"   Predicted: {result['predicted_class']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Fresh probability: {result['fresh_probability']:.2%}")
        print(f"   Spoiled probability: {result['spoiled_probability']:.2%}")
        print(f"   Is Fresh: {result['is_fresh']}")
        
        # Save result visualization
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(normalized)
        
        # Add classification info
        color = 'green' if result['is_fresh'] else 'red'
        title = f"{result['predicted_class']} ({result['confidence']:.1%})"
        ax.set_title(title, fontsize=16, color=color, fontweight='bold')
        ax.axis('off')
        
        # Add probability bar
        fresh_prob = result['fresh_probability']
        spoiled_prob = result['spoiled_probability']
        
        fig.text(0.5, 0.05, f"Fresh: {fresh_prob:.1%} | Spoiled: {spoiled_prob:.1%}", 
                ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path / 'classification_result.png', dpi=150)
        print(f"💾 Classification result saved to {output_path / 'classification_result.png'}")
        plt.close()
    else:
        print("⚠️ Model not loaded. Skipping classification test.")
    
    print(f"\n✅ Debug test complete!")
    print(f"📁 Results saved to {output_path}")


def test_batch_images(image_dir: str, output_dir: str = './debug_output'):
    """
    Test preprocessing on a batch of images
    
    Args:
        image_dir: Directory containing test images
        output_dir: Output directory for debug results
    """
    print("=" * 60)
    print("Batch Image Preprocessing Test")
    print("=" * 60)
    
    image_path = Path(image_dir)
    
    if not image_path.exists():
        print(f"❌ Directory not found: {image_dir}")
        return
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(list(image_path.glob(ext)))
        image_files.extend(list(image_path.glob(ext.upper())))
    
    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return
    
    print(f"\n📷 Found {len(image_files)} images")
    
    # Initialize classifier
    classifier = MobileNetClassifier(
        model_path="./models/mobilenet_classifier.tflite",
        input_size=224
    )
    
    if not classifier.load_model():
        print("❌ Failed to load model")
        return
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(target_size=(224, 224))
    
    # Process each image
    results = []
    
    for i, img_file in enumerate(image_files[:20]):  # Limit to 20 images
        print(f"\nProcessing {i+1}/{min(20, len(image_files))}: {img_file.name}")
        
        # Load and preprocess
        image = cv2.imread(str(img_file))
        if image is None:
            print(f"   ❌ Failed to load")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preprocessed = preprocessor.preprocess_for_classification(image_rgb)
        
        # Classify
        result = classifier.classify_with_details(preprocessed)
        
        print(f"   {result['predicted_class']}: {result['confidence']:.1%}")
        
        results.append({
            'filename': img_file.name,
            'predicted_class': result['predicted_class'],
            'confidence': result['confidence'],
            'fresh_prob': result['fresh_probability'],
            'spoiled_prob': result['spoiled_probability']
        })
    
    # Print summary
    print(f"\n📊 Summary:")
    print("=" * 60)
    
    fresh_count = sum(1 for r in results if r['predicted_class'] == 'Fresh')
    spoiled_count = len(results) - fresh_count
    avg_confidence = np.mean([r['confidence'] for r in results])
    
    print(f"Total images: {len(results)}")
    print(f"Fresh: {fresh_count} ({fresh_count/len(results)*100:.1f}%)")
    print(f"Spoiled: {spoiled_count} ({spoiled_count/len(results)*100:.1f}%)")
    print(f"Average confidence: {avg_confidence:.2%}")
    
    print(f"\n✅ Batch test complete!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug preprocessing pipeline')
    parser.add_argument(
        '--image',
        type=str,
        help='Path to single test image'
    )
    parser.add_argument(
        '--batch',
        type=str,
        help='Directory containing batch of images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./debug_output',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    if args.image:
        test_preprocessing_pipeline(args.image, args.output)
    elif args.batch:
        test_batch_images(args.batch, args.output)
    else:
        print("Please provide --image or --batch argument")
        parser.print_help()

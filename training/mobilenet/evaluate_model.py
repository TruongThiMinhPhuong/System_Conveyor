"""
Model Evaluation and Performance Analysis
Evaluate trained MobileNetV2 model with detailed metrics and confusion matrix
"""

import argparse
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from augmentation import create_data_generator


def evaluate_model(
    model_path: str,
    dataset_dir: str = './datasets/fruit_classification',
    image_size: int = 224,
    batch_size: int = 32,
    output_dir: str = './evaluation'
):
    """
    Comprehensive model evaluation
    
    Args:
        model_path: Path to trained model
        dataset_dir: Path to dataset directory
        image_size: Input image size
        batch_size: Batch size for evaluation
        output_dir: Output directory for reports
    """
    print("=" * 60)
    print("MobileNetV2 Model Evaluation")
    print("=" * 60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\n📦 Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    print("✅ Model loaded")
    
    # Load test dataset
    print(f"\n📊 Loading test dataset from {dataset_dir}/test...")
    test_dataset = create_data_generator(
        directory=f"{dataset_dir}/test",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
        augment=False
    )
    
    # Evaluate model
    print(f"\n🔍 Evaluating model...")
    results = model.evaluate(test_dataset, verbose=1)
    
    print(f"\n📈 Test Metrics:")
    print(f"   Loss: {results[0]:.4f}")
    print(f"   Accuracy: {results[1]:.2%}")
    print(f"   Precision: {results[2]:.2%}")
    print(f"   Recall: {results[3]:.2%}")
    
    # Calculate F1 score
    precision = results[2]
    recall = results[3]
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"   F1 Score: {f1_score:.2%}")
    
    # Get predictions
    print(f"\n🎯 Generating predictions...")
    y_true = []
    y_pred = []
    y_pred_probs = []
    
    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0)
        y_pred_probs.extend(predictions)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(np.argmax(labels, axis=1))
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_probs = np.array(y_pred_probs)
    
    # Confusion matrix
    print(f"\n📊 Creating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    
    class_names = ['Fresh', 'Spoiled']
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - Fresh vs Spoiled Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path / 'confusion_matrix.png', dpi=150)
    print(f"💾 Confusion matrix saved to {output_path / 'confusion_matrix.png'}")
    plt.close()
    
    # Normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage'}
    )
    plt.title('Normalized Confusion Matrix - Fresh vs Spoiled Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path / 'confusion_matrix_normalized.png', dpi=150)
    print(f"💾 Normalized confusion matrix saved to {output_path / 'confusion_matrix_normalized.png'}")
    plt.close()
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print("=" * 60)
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
    print(report)
    
    # Save report to file
    with open(output_path / 'classification_report.txt', 'w') as f:
        f.write("Classification Report - Fresh vs Spoiled\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
        f.write("\n\nTest Metrics:\n")
        f.write(f"Loss: {results[0]:.4f}\n")
        f.write(f"Accuracy: {results[1]:.2%}\n")
        f.write(f"Precision: {results[2]:.2%}\n")
        f.write(f"Recall: {results[3]:.2%}\n")
        f.write(f"F1 Score: {f1_score:.2%}\n")
    
    print(f"💾 Report saved to {output_path / 'classification_report.txt'}")
    
    # Analyze prediction confidence
    print(f"\n📊 Prediction Confidence Analysis:")
    print("=" * 60)
    
    # Get confidence values
    confidences = np.max(y_pred_probs, axis=1)
    
    print(f"Average confidence: {np.mean(confidences):.2%}")
    print(f"Min confidence: {np.min(confidences):.2%}")
    print(f"Max confidence: {np.max(confidences):.2%}")
    print(f"Median confidence: {np.median(confidences):.2%}")
    
    # Plot confidence distribution
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(confidences, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(np.mean(confidences), color='red', linestyle='--', label=f'Mean: {np.mean(confidences):.2%}')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.title('Overall Confidence Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Confidence by class
    plt.subplot(1, 2, 2)
    fresh_confidences = confidences[y_true == 0]
    spoiled_confidences = confidences[y_true == 1]
    
    plt.hist(fresh_confidences, bins=30, alpha=0.6, label='Fresh', color='green')
    plt.hist(spoiled_confidences, bins=30, alpha=0.6, label='Spoiled', color='red')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'confidence_distribution.png', dpi=150)
    print(f"💾 Confidence distribution saved to {output_path / 'confidence_distribution.png'}")
    plt.close()
    
    # Analyze misclassifications
    misclassified = y_true != y_pred
    num_misclassified = np.sum(misclassified)
    
    print(f"\n❌ Misclassification Analysis:")
    print(f"   Total misclassified: {num_misclassified} / {len(y_true)} ({num_misclassified/len(y_true):.2%})")
    
    if num_misclassified > 0:
        print(f"   Average confidence on misclassified: {np.mean(confidences[misclassified]):.2%}")
        print(f"   Average confidence on correct: {np.mean(confidences[~misclassified]):.2%}")
        
        # Count specific misclassifications
        fresh_as_spoiled = np.sum((y_true == 0) & (y_pred == 1))
        spoiled_as_fresh = np.sum((y_true == 1) & (y_pred == 0))
        
        print(f"\n   Fresh classified as Spoiled: {fresh_as_spoiled}")
        print(f"   Spoiled classified as Fresh: {spoiled_as_fresh}")
    
    print(f"\n✅ Evaluation complete!")
    print(f"📁 Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate MobileNetV2 model'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model (.keras)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='./datasets/fruit_classification',
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='Image size'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./evaluation',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        dataset_dir=args.dataset,
        image_size=args.image_size,
        batch_size=args.batch,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()

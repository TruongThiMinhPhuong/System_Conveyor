"""
YOLOv8-nano Training Script for Fruit Detection
Train on PC/Laptop with GPU, then deploy to Raspberry Pi
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train_yolo(
    data_yaml: str,
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = '0',
    project_name: str = 'fruit_detection',
    name: str = 'yolov8n_fruit'
):
    """
    Train YOLOv8-nano model for fruit detection
    
    Args:
        data_yaml: Path to dataset YAML file
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
        device: Device to use ('0' for GPU, 'cpu' for CPU)
        project_name: Project directory name
        name: Experiment name
    """
    print("=" * 60)
    print("YOLOv8-nano Fruit Detection Training")
    print("=" * 60)
    
    # Load YOLOv8-nano model
    print(f"\n📦 Loading YOLOv8-nano model...")
    model = YOLO('yolov8n.pt')  # Start from pretrained weights
    
    print(f"✅ Model loaded successfully")
    print(f"\n📊 Training Configuration:")
    print(f"   Dataset: {data_yaml}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Image size: {img_size}")
    print(f"   Device: {device}")
    
    # Train model
    print(f"\n🚀 Starting training...")
    print("-" * 60)
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project_name,
        name=name,
        patience=20,          # Early stopping patience
        save=True,            # Save checkpoints
        save_period=10,       # Save every N epochs
        cache=False,          # Don't cache images (use if low RAM)
        plots=True,           # Save training plots
        verbose=True,
        # Optimization
        optimizer='Adam',
        lr0=0.001,           # Initial learning rate
        momentum=0.9,
        weight_decay=0.0005,
        # Augmentation
        hsv_h=0.015,         # HSV-Hue augmentation
        hsv_s=0.7,           # HSV-Saturation augmentation
        hsv_v=0.4,           # HSV-Value augmentation
        degrees=10.0,        # Rotation
        translate=0.1,       # Translation
        scale=0.5,           # Scale
        fliplr=0.5,          # Horizontal flip probability
        mosaic=1.0,          # Mosaic augmentation probability
    )
    
    print("-" * 60)
    print("✅ Training complete!")
    
    # Validation
    print(f"\n📊 Running validation...")
    metrics = model.val()
    
    print(f"\n📈 Validation Metrics:")
    print(f"   mAP50: {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    print(f"   Precision: {metrics.box.mp:.4f}")
    print(f"   Recall: {metrics.box.mr:.4f}")
    
    # Export model
    output_path = f"./{project_name}/{name}/weights/best.pt"
    export_path = "../../models/yolov8n_fruit.pt"
    
    print(f"\n💾 Saving best model:")
    print(f"   From: {output_path}")
    print(f"   To: {export_path}")
    
    # Create models directory
    Path(export_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Copy best model
    import shutil
    if Path(output_path).exists():
        shutil.copy(output_path, export_path)
        print("✅ Model saved successfully!")
    else:
        print("⚠️ Best model file not found")
    
    print(f"\n🎉 Training complete! Model ready for deployment.")
    print(f"📦 Copy '{export_path}' to Raspberry Pi")


def main():
    parser = argparse.ArgumentParser(
        description='Train YOLOv8-nano for fruit detection'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='./dataset.yaml',
        help='Path to dataset YAML file'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='Device (0 for GPU, cpu for CPU)'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='fruit_detection',
        help='Project name'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='yolov8n_fruit',
        help='Experiment name'
    )
    
    args = parser.parse_args()
    
    train_yolo(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        device=args.device,
        project_name=args.project,
        name=args.name
    )


if __name__ == '__main__':
    main()

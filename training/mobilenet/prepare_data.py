"""
Data Preparation for MobileNetV2 Classification
Organize fresh/spoiled fruit images for training
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import Tuple
import random


def prepare_classification_dataset(
    source_dir: str,
    output_dir: str = './datasets/fruit_classification',
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15
) -> dict:
    """
    Prepare and organize dataset for binary classification
    
    Expected source directory structure:
    source_dir/
    ├── fresh/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    └── spoiled/
        ├── img1.jpg
        ├── img2.jpg
        └── ...
    
    Output directory structure:
    output_dir/
    ├── train/
    │   ├── fresh/
    │   └── spoiled/
    ├── val/
    │   ├── fresh/
    │   └── spoiled/
    └── test/
        ├── fresh/
        └── spoiled/
    
    Args:
        source_dir: Directory containing fresh/ and spoiled/ subdirectories
        output_dir: Output directory for organized dataset
        train_split: Training set proportion
        val_split: Validation set proportion
        test_split: Test set proportion
        
    Returns:
        Dictionary with dataset statistics
    """
    print("=" * 60)
    print("Preparing Classification Dataset")
    print("=" * 60)
    
    # Validate splits
    assert abs(train_split + val_split + test_split - 1.0) < 0.01, \
        "Splits must sum to 1.0"
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Check source directory
    if not source_path.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return {}
    
    # Create output directories
    splits = ['train', 'val', 'test']
    classes = ['fresh', 'spoiled']
    
    for split in splits:
        for cls in classes:
            (output_path / split / cls).mkdir(parents=True, exist_ok=True)
    
    stats = {'fresh': 0, 'spoiled': 0, 'total': 0}
    
    # Process each class
    for cls in classes:
        class_dir = source_path / cls
        
        if not class_dir.exists():
            print(f"⚠️ Class directory not found: {class_dir}")
            continue
        
        # Get all images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        images = []
        for ext in image_extensions:
            images.extend(list(class_dir.glob(ext)))
            images.extend(list(class_dir.glob(ext.upper())))
        
        if not images:
            print(f"⚠️ No images found in {class_dir}")
            continue
        
        # Shuffle images
        random.shuffle(images)
        
        # Calculate split indices
        n_images = len(images)
        n_train = int(n_images * train_split)
        n_val = int(n_images * val_split)
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Copy images to respective directories
        print(f"\n📂 Processing {cls}...")
        print(f"   Total: {n_images}")
        print(f"   Train: {len(train_images)}")
        print(f"   Val: {len(val_images)}")
        print(f"   Test: {len(test_images)}")
        
        for img in train_images:
            shutil.copy(img, output_path / 'train' / cls / img.name)
        
        for img in val_images:
            shutil.copy(img, output_path / 'val' / cls / img.name)
        
        for img in test_images:
            shutil.copy(img, output_path / 'test' / cls / img.name)
        
        stats[cls] = n_images
        stats['total'] += n_images
    
    print(f"\n✅ Dataset prepared successfully!")
    print(f"\n📊 Dataset Statistics:")
    print(f"   Fresh images: {stats['fresh']}")
    print(f"   Spoiled images: {stats['spoiled']}")
    print(f"   Total images: {stats['total']}")
    print(f"\n📁 Output directory: {output_dir}")
    
    return stats


def verify_dataset(dataset_dir: str):
    """
    Verify dataset structure and count images
    
    Args:
        dataset_dir: Path to dataset directory
    """
    dataset_path = Path(dataset_dir)
    
    print(f"\n🔍 Verifying dataset: {dataset_dir}")
    print("-" * 60)
    
    splits = ['train', 'val', 'test']
    classes = ['fresh', 'spoiled']
    
    for split in splits:
        print(f"\n{split.upper()}:")
        for cls in classes:
            cls_dir = dataset_path / split / cls
            if cls_dir.exists():
                n_images = len(list(cls_dir.glob('*.[jp][pn][g]*')))
                print(f"   {cls}: {n_images} images")
            else:
                print(f"   {cls}: Directory not found")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare dataset for fruit classification'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Source directory with fresh/ and spoiled/ subdirectories'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./datasets/fruit_classification',
        help='Output directory for organized dataset'
    )
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.7,
        help='Training set proportion (default: 0.7)'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.15,
        help='Validation set proportion (default: 0.15)'
    )
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.15,
        help='Test set proportion (default: 0.15)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify dataset after preparation'
    )
    
    args = parser.parse_args()
    
    # Prepare dataset
    stats = prepare_classification_dataset(
        source_dir=args.source,
        output_dir=args.output,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split
    )
    
    # Verify if requested
    if args.verify and stats:
        verify_dataset(args.output)
    
    print("\n🎉 Done!")


if __name__ == '__main__':
    main()

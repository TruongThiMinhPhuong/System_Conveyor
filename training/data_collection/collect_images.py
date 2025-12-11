"""
Data Collection Script
Capture images from Raspberry Pi camera for training dataset
"""

import time
import argparse
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from hardware.camera import Camera


def collect_images(
    output_dir: str,
    num_images: int = 100,
    interval: float = 1.0,
    prefix: str = "fruit",
    preview: bool = True
):
    """
    Collect images from camera for training dataset
    
    Args:
        output_dir: Directory to save collected images
        num_images: Number of images to capture
        interval: Time interval between captures (seconds)
        prefix: Filename prefix
        preview: Whether to show preview instructions
    """
    print("=" * 60)
    print("Image Collection for Training Dataset")
    print("=" * 60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir}")
    print(f"📷 Number of images: {num_images}")
    print(f"⏱️ Interval: {interval} seconds")
    
    if preview:
        print(f"\n💡 Tips for good dataset:")
        print("   - Vary fruit positions and orientations")
        print("   - Include different lighting conditions")
        print("   - Capture from different angles")
        print("   - Mix backgrounds")
        print("   - For spoiled fruits: various stages of decay")
        
        input("\nPress ENTER to start capturing...")
    
    # Initialize camera
    with Camera() as cam:
        print(f"\n🎬 Starting image capture...")
        print("-" * 60)
        
        for i in range(num_images):
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}_{i:04d}.jpg"
            filepath = output_path / filename
            
            # Capture image
            frame = cam.capture_image(str(filepath))
            
            if frame is not None:
                print(f"✅ [{i+1}/{num_images}] Captured: {filename}")
            else:
                print(f"❌ [{i+1}/{num_images}] Failed to capture")
            
            # Wait for interval (except for last image)
            if i < num_images - 1:
                time.sleep(interval)
        
        print("-" * 60)
        print(f"✅ Image collection complete!")
        print(f"📊 Captured {num_images} images to {output_dir}")


def collect_classification_dataset(
    base_dir: str = "./raw_images",
    fresh_count: int = 100,
    spoiled_count: int = 100,
    interval: float = 1.0
):
    """
    Collect images for both fresh and spoiled classes
    
    Args:
        base_dir: Base directory for dataset
        fresh_count: Number of fresh fruit images
        spoiled_count: Number of spoiled fruit images
        interval: Capture interval
    """
    print("=" * 60)
    print("Classification Dataset Collection")
    print("=" * 60)
    
    print(f"\n📊 Collection Plan:")
    print(f"   Fresh fruits: {fresh_count} images")
    print(f"   Spoiled fruits: {spoiled_count} images")
    print(f"   Total: {fresh_count + spoiled_count} images")
    
    # Collect fresh fruits
    print(f"\n{'='*60}")
    print("Phase 1: Fresh Fruits")
    print(f"{'='*60}")
    print("\n🍎 Prepare FRESH fruits (good quality, no damage)")
    input("Press ENTER when ready...")
    
    collect_images(
        output_dir=f"{base_dir}/fresh",
        num_images=fresh_count,
        interval=interval,
        prefix="fresh",
        preview=False
    )
    
    # Collect spoiled fruits
    print(f"\n{'='*60}")
    print("Phase 2: Spoiled Fruits")
    print(f"{'='*60}")
    print("\n🍂 Prepare SPOILED fruits (damaged, moldy, rotten)")
    input("Press ENTER when ready...")
    
    collect_images(
        output_dir=f"{base_dir}/spoiled",
        num_images=spoiled_count,
        interval=interval,
        prefix="spoiled",
        preview=False
    )
    
    print(f"\n🎉 Dataset collection complete!")
    print(f"📁 Images saved to: {base_dir}/")
    print(f"\n📝 Next steps:")
    print(f"   1. Review and clean images")
    print(f"   2. For YOLO: Annotate with LabelImg")
    print(f"   3. For Classification: Run prepare_data.py")


def main():
    parser = argparse.ArgumentParser(
        description='Collect images for training dataset'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['single', 'classification'],
        default='classification',
        help='Collection mode'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./raw_images',
        help='Output directory'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=100,
        help='Number of images per class'
    )
    parser.add_argument(
        '--interval',
        type=float,
        default=1.0,
        help='Capture interval in seconds'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='fruit',
        help='Filename prefix'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'classification':
        collect_classification_dataset(
            base_dir=args.output,
            fresh_count=args.count,
            spoiled_count=args.count,
            interval=args.interval
        )
    else:
        collect_images(
            output_dir=args.output,
            num_images=args.count,
            interval=args.interval,
            prefix=args.prefix
        )


if __name__ == '__main__':
    main()

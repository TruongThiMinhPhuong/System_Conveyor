"""
Prepare project for Raspberry Pi deployment
Clean up unnecessary files for lightweight deployment
"""

import shutil
from pathlib import Path
import os

def get_size(path):
    """Get folder size in MB"""
    total = 0
    for entry in Path(path).rglob('*'):
        if entry.is_file():
            total += entry.stat().st_size
    return total / 1024 / 1024

def prepare_for_deployment():
    """Remove unnecessary files for Pi deployment"""
    
    print("=" * 60)
    print("🧹 Preparing Project for Raspberry Pi Deployment")
    print("=" * 60)
    
    # Folders to remove
    folders_to_remove = [
        'dataset',
        'raw_images',
        'improved_training',
        '__pycache__',
        '.pytest_cache',
        'venv',
        'env',
    ]
    
    # File patterns to remove
    files_to_remove = [
        '*.h5',          # Keras models (keep .tflite only)
        '*.keras',       # Keras models
        '*.pyc',         # Python cache
        '*.pyo',         # Python optimized
        '*.log',         # Log files
        'training*.txt', # Training logs
    ]
    
    # Training scripts to remove
    training_scripts = [
        'retrain_model.py',
        'simple_train.py',
        'quick_train.py',
        'train_yolo.py',
        'collect_data.py',
        'organize_images.py',
        'dataset_quality_checker.py',
        'clean_dataset.py',
        'convert_to_tflite.py',
    ]
    
    total_saved = 0
    
    # Remove folders
    print("\n📂 Removing unnecessary folders...")
    for folder in folders_to_remove:
        folder_path = Path(folder)
        if folder_path.exists():
            size = get_size(folder_path)
            shutil.rmtree(folder_path, ignore_errors=True)
            total_saved += size
            print(f"  ✅ Removed: {folder} ({size:.2f} MB)")
    
    # Remove files by pattern
    print("\n📄 Removing unnecessary files...")
    for pattern in files_to_remove:
        for file in Path('.').rglob(pattern):
            if file.is_file():
                size = file.stat().st_size / 1024 / 1024
                file.unlink()
                total_saved += size
                print(f"  ✅ Removed: {file} ({size:.2f} MB)")
    
    # Remove training scripts
    print("\n🐍 Removing training scripts...")
    for script in training_scripts:
        script_path = Path(script)
        if script_path.exists():
            size = script_path.stat().st_size / 1024 / 1024
            script_path.unlink()
            total_saved += size
            print(f"  ✅ Removed: {script}")
    
    # Keep only requirements-rpi.txt
    print("\n📋 Cleaning requirements files...")
    if Path('requirements-pc.txt').exists():
        Path('requirements-pc.txt').unlink()
        print("  ✅ Removed: requirements-pc.txt")
    
    if Path('requirements-laptop.txt').exists():
        Path('requirements-laptop.txt').unlink()
        print("  ✅ Removed: requirements-laptop.txt")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"✅ Cleanup Complete!")
    print(f"💾 Total space saved: {total_saved:.2f} MB")
    print("=" * 60)
    
    # List essential files
    print("\n📦 Essential files kept:")
    essential = [
        'ai_models/*.tflite',
        'hardware/',
        'web/',
        'run_web.py',
        'requirements-rpi.txt',
        'config.yaml',
        'utils/',
    ]
    for item in essential:
        print(f"  ✅ {item}")
    
    print("\n🚀 Project is now ready for Raspberry Pi deployment!")
    print("\nNext steps:")
    print("1. Compress: tar -czf conveyor_pi.tar.gz .")
    print("2. Transfer to Pi: scp conveyor_pi.tar.gz pi@raspberrypi:~")
    print("3. Extract on Pi: tar -xzf conveyor_pi.tar.gz")
    print("4. Install dependencies: pip install -r requirements-rpi.txt")
    print("5. Run: python run_web.py")

if __name__ == "__main__":
    try:
        prepare_for_deployment()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please run this script from the project root directory.")

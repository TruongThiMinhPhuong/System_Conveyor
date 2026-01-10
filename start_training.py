"""
START TRAINING MOBILENETV2 - Complete Automated Workflow
Run this script to train model with automatic dataset preparation
"""

import os
import sys
from pathlib import Path

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_step(step_num, message):
    print(f"\n{Colors.BOLD}{Colors.BLUE}[Step {step_num}]{Colors.END} {message}")

def print_success(message):
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_warning(message):
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def print_error(message):
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def check_dataset():
    """Check if dataset exists and has proper structure"""
    print_step(1, "Checking dataset structure...")
    
    data_dir = Path("data/fruits")
    
    required_dirs = [
        data_dir / "train" / "fresh",
        data_dir / "train" / "spoiled",
        data_dir / "validation" / "fresh",
        data_dir / "validation" / "spoiled"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if not dir_path.exists():
            print_error(f"Missing: {dir_path}")
            all_exist = False
        else:
            # Count images
            images = list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.png"))
            print_success(f"Found {len(images)} images in {dir_path.name}")
    
    if not all_exist:
        print_warning("\nDataset structure incomplete!")
        print("\nRequired structure:")
        print("data/fruits/")
        print("├── train/")
        print("│   ├── fresh/     (at least 100 images)")
        print("│   └── spoiled/   (at least 100 images)")
        print("└── validation/")
        print("    ├── fresh/     (at least 20 images)")
        print("    └── spoiled/   (at least 20 images)")
        print("\nSee HUONG_DAN_NHAP_ANH.md for data collection guide")
        return False
    
    return True

def check_dependencies():
    """Check required packages"""
    print_step(2, "Checking dependencies...")
    
    required_packages = {
        'tensorflow': 'tensorflow>=2.10.0',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'pillow': 'Pillow'
    }
    
    missing = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package.lower())
            print_success(f"{package} installed")
        except ImportError:
            missing.append(pip_name)
            print_error(f"{package} NOT installed")
    
    if missing:
        print_warning(f"\nInstall missing packages:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    return True

def start_training():
    """Start MobileNetV2 training"""
    print_step(3, "Starting MobileNetV2 training...")
    
    print("\n" + "="*60)
    print(f"{Colors.BOLD}TRAINING CONFIGURATION{Colors.END}")
    print("="*60)
    print(f"Dataset:          data/fruits/")
    print(f"Epochs:           50")
    print(f"Batch Size:       32")
    print(f"Learning Rate:    0.001")
    print(f"Image Size:       224x224")
    print(f"Dropout:          0.5")
    print(f"Base Model:       MobileNetV2 (frozen)")
    print("="*60)
    
    input("\nPress Enter to start training or Ctrl+C to cancel...")
    
    # Change to training directory
    os.chdir(Path("training/mobilenet"))
    
    # Run training
    print(f"\n{Colors.BOLD}Training started...{Colors.END}\n")
    exit_code = os.system(
        "python train_mobilenet.py "
        "--dataset ../../data/fruits "
        "--epochs 50 "
        "--batch-size 32 "
        "--learning-rate 0.001 "
        "--dropout 0.5"
    )
    
    if exit_code == 0:
        print_success("\nTraining completed successfully!")
        return True
    else:
        print_error("\nTraining failed!")
        return False

def export_to_tflite():
    """Export trained model to TensorFlow Lite"""
    print_step(4, "Exporting to TensorFlow Lite...")
    
    os.chdir(Path("training/mobilenet"))
    
    exit_code = os.system("python export_tflite.py")
    
    if exit_code == 0:
        print_success("TFLite export completed!")
        return True
    else:
        print_error("TFLite export failed!")
        return False

def verify_model():
    """Verify exported model"""
    print_step(5, "Verifying model files...")
    
    model_h5 = Path("training/mobilenet/mobilenet_training/mobilenet_fruit_classifier.h5")
    model_tflite = Path("models/mobilenet_classifier.tflite")
    
    if model_h5.exists():
        size_mb = model_h5.stat().st_size / (1024 * 1024)
        print_success(f"Keras model: {model_h5} ({size_mb:.2f} MB)")
    else:
        print_error(f"Keras model not found: {model_h5}")
    
    if model_tflite.exists():
        size_mb = model_tflite.stat().st_size / (1024 * 1024)
        print_success(f"TFLite model: {model_tflite} ({size_mb:.2f} MB)")
    else:
        print_error(f"TFLite model not found: {model_tflite}")
    
    return model_h5.exists() and model_tflite.exists()

def show_next_steps():
    """Show next steps after training"""
    print("\n" + "="*60)
    print(f"{Colors.BOLD}{Colors.GREEN}TRAINING COMPLETE!{Colors.END}")
    print("="*60)
    
    print("\n📋 Next Steps:")
    print("\n1. Review training results:")
    print("   - Check training/mobilenet/mobilenet_training/training_history.png")
    print("   - Validation accuracy should be > 85%")
    
    print("\n2. Transfer to Raspberry Pi:")
    print("   python prepare_for_pi.py")
    print("   scp conveyor_pi_deploy.zip pi@<pi_ip>:~/")
    
    print("\n3. Test on Raspberry Pi:")
    print("   cd ~/System_Conveyor")
    print("   python3 fruit_sorter.py")
    
    print("\n4. Web interface:")
    print("   python3 run_web.py")
    print("   Access: http://<pi_ip>:5001")
    
    print("\n" + "="*60)

def main():
    """Main workflow"""
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}  MobileNetV2 Training - Automated Workflow{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}\n")
    
    # Check dataset
    if not check_dataset():
        print_error("\nDataset check failed! Fix issues and try again.")
        return 1
    
    # Check dependencies
    if not check_dependencies():
        print_error("\nDependency check failed! Install packages and try again.")
        return 1
    
    # Start training
    if not start_training():
        print_error("\nTraining failed!")
        return 1
    
    # Export to TFLite
    if not export_to_tflite():
        print_warning("\nTFLite export failed, but Keras model is available")
    
    # Verify
    verify_model()
    
    # Show next steps
    show_next_steps()
    
    return 0

if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Training cancelled by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print_error(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

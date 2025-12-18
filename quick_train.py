"""
Quick Training Script - Train and Deploy MobileNet Model
Run this on your PC to quickly train and prepare for deployment
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and report progress"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed!")
        return False
    
    print(f"\n✅ {description} completed successfully!")
    return True

def main():
    print("="*60)
    print("🚀 Quick Training and Deployment Pipeline")
    print("="*60)
    
    # Check if we're in the right directory
    if not Path("training/mobilenet/train_mobilenet.py").exists():
        print("\n❌ Error: Please run this from the System_Conveyor directory")
        sys.exit(1)
    
    # Dataset path
    dataset_path = "training/mobilenet/datasets/fruit_classification"
    
    # Check dataset exists
    if not Path(f"{dataset_path}/train").exists():
        print(f"\n⚠️  Dataset not found at: {dataset_path}")
        print("\nPlease prepare your dataset first:")
        print("  python training/mobilenet/prepare_data.py --source <your_images> --output training/mobilenet/datasets/fruit_classification")
        sys.exit(1)
    
    # Count images
    train_fresh = len(list(Path(f"{dataset_path}/train/fresh").glob("*.[jp][pn][g]*")))
    train_spoiled = len(list(Path(f"{dataset_path}/train/spoiled").glob("*.[jp][pn][g]*")))
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Fresh: {train_fresh} images")
    print(f"   Spoiled: {train_spoiled} images")
    print(f"   Total: {train_fresh + train_spoiled} images")
    
    if train_fresh < 20 or train_spoiled < 20:
        print(f"\n⚠️  Warning: Low image count! Recommended minimum: 50 per class")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Ask for epochs
    print(f"\n🎯 Training Configuration:")
    epochs = input("Number of epochs (default: 50): ").strip()
    epochs = epochs if epochs else "50"
    
    batch_size = input("Batch size (default: 32): ").strip()
    batch_size = batch_size if batch_size else "32"
    
    # Pipeline steps
    steps = [
        {
            "cmd": f"python training/mobilenet/train_mobilenet.py --dataset {dataset_path} --epochs {epochs} --batch {batch_size}",
            "desc": "Training MobileNet Model"
        },
        {
            "cmd": f"python training/mobilenet/evaluate_model.py --model training/mobilenet/mobilenet_training/mobilenet_fruit_classifier_best.keras --dataset {dataset_path}",
            "desc": "Evaluating Model Performance"
        },
        {
            "cmd": f"python training/mobilenet/export_tflite.py --model training/mobilenet/mobilenet_training/mobilenet_fruit_classifier_best.keras --output models/mobilenet_classifier.tflite",
            "desc": "Converting to TensorFlow Lite"
        }
    ]
    
    # Execute pipeline
    for i, step in enumerate(steps, 1):
        print(f"\n\n📍 Step {i}/{len(steps)}")
        if not run_command(step["cmd"], step["desc"]):
            print(f"\n❌ Pipeline failed at step {i}")
            sys.exit(1)
    
    # Success!
    print("\n" + "="*60)
    print("🎉 Training Pipeline Complete!")
    print("="*60)
    
    print("\n📦 Model Location:")
    print(f"   TFLite model: models/mobilenet_classifier.tflite")
    print(f"   Keras model: training/mobilenet/mobilenet_training/mobilenet_fruit_classifier_best.keras")
    
    print("\n📊 Evaluation Results:")
    print(f"   Check: training/mobilenet/evaluation/")
    print(f"   - confusion_matrix.png")
    print(f"   - classification_report.txt")
    print(f"   - confidence_distribution.png")
    
    print("\n🚀 Next Step: Deploy to Raspberry Pi")
    print("="*60)
    
    # Ask about deployment
    deploy = input("\nDeploy to Raspberry Pi now? (y/n): ").strip().lower()
    
    if deploy == 'y':
        pi_address = input("Raspberry Pi address (default: pi@192.168.137.177): ").strip()
        pi_address = pi_address if pi_address else "pi@192.168.137.177"
        
        deploy_cmd = f"scp models/mobilenet_classifier.tflite {pi_address}:~/System_Conveyor/models/"
        
        print(f"\n📤 Deploying to {pi_address}...")
        if run_command(deploy_cmd, "Deploying Model"):
            print("\n✅ Model deployed successfully!")
            print(f"\nOn Raspberry Pi, run:")
            print(f"  cd ~/System_Conveyor")
            print(f"  python3 fruit_sorter.py")
    else:
        print("\n💡 To deploy manually:")
        print(f"   scp models/mobilenet_classifier.tflite pi@192.168.137.177:~/System_Conveyor/models/")
    
    print("\n🎉 All done!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

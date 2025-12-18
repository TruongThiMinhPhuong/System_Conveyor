# Complete System Setup - PC Training Environment
# Windows PowerShell Script

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "AI Fruit Sorter - PC Setup (Training)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check Python
Write-Host "`n[1/5] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Success: Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: Python not found! Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# Install PC requirements (for training)
Write-Host "`n[2/5] Installing PC requirements (TensorFlow, etc.)..." -ForegroundColor Yellow

$pcRequirements = @"
tensorflow>=2.10.0
opencv-python>=4.8.0
numpy>=1.23.0
matplotlib>=3.7.0
Pillow>=9.5.0
scikit-learn>=1.3.0
seaborn>=0.12.0
ultralytics>=8.0.0
"@

Set-Content -Path "requirements-pc.txt" -Value $pcRequirements
Write-Host "Created requirements-pc.txt" -ForegroundColor Cyan

python -m pip install --upgrade pip
pip install -r requirements-pc.txt

# Verify TensorFlow
Write-Host "`n[3/5] Verifying TensorFlow installation..." -ForegroundColor Yellow
try {
    $tfCheck = python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')" 2>&1
    Write-Host "Success: $tfCheck" -ForegroundColor Green
} catch {
    Write-Host "Warning: TensorFlow verification failed" -ForegroundColor Yellow
}

# Create necessary directories
Write-Host "`n[4/5] Creating directories..." -ForegroundColor Yellow
$dirs = @(
    "models",
    "logs",  
    "data",
    "training\mobilenet\datasets",
    "training\mobilenet\mobilenet_training",
    "training\yolo\datasets"
)

foreach ($dir in $dirs) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created: $dir" -ForegroundColor Cyan
    }
}
Write-Host "Success: Directories created" -ForegroundColor Green

# Check for dataset
Write-Host "`n[5/5] Checking dataset..." -ForegroundColor Yellow
$datasetPath = "training\mobilenet\datasets\fruit_classification"

if (Test-Path "$datasetPath\train") {
    $trainFresh = (Get-ChildItem "$datasetPath\train\fresh" -File -ErrorAction SilentlyContinue).Count
    $trainSpoiled = (Get-ChildItem "$datasetPath\train\spoiled" -File -ErrorAction SilentlyContinue).Count
    
    if ($trainFresh -gt 0 -and $trainSpoiled -gt 0) {
        Write-Host "Success: Dataset found:" -ForegroundColor Green
        Write-Host "  Fresh: $trainFresh images" -ForegroundColor Cyan
        Write-Host "  Spoiled: $trainSpoiled images" -ForegroundColor Cyan
    } else {
        Write-Host "Warning: Dataset folder exists but no images found" -ForegroundColor Yellow
        Write-Host "  Please prepare your dataset using:" -ForegroundColor Yellow
        Write-Host "  python training\mobilenet\prepare_data.py --source YOUR_IMAGES" -ForegroundColor Cyan
    }
} else {
    Write-Host "Warning: Dataset not found at: $datasetPath" -ForegroundColor Yellow
    Write-Host "  Please prepare your dataset:" -ForegroundColor Yellow
    Write-Host "  1. Collect images (fresh and spoiled fruits)" -ForegroundColor Cyan
    Write-Host "  2. Run: python training\mobilenet\prepare_data.py --source YOUR_PATH" -ForegroundColor Cyan
}

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "PC Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`nNext Steps:" -ForegroundColor Yellow
Write-Host "1. Prepare dataset (if not done):" -ForegroundColor White
Write-Host "   python training\mobilenet\prepare_data.py --source YOUR_IMAGES" -ForegroundColor Cyan
Write-Host "`n2. Train the model:" -ForegroundColor White
Write-Host "   python quick_train.py" -ForegroundColor Cyan
Write-Host "   OR" -ForegroundColor White
Write-Host "   python training\mobilenet\train_mobilenet.py --dataset training/mobilenet/datasets/fruit_classification --epochs 50" -ForegroundColor Cyan
Write-Host "`n3. Deploy to Raspberry Pi:" -ForegroundColor White
Write-Host "   scp models\mobilenet_classifier.tflite pi@192.168.137.177:~/System_Conveyor/models/" -ForegroundColor Cyan

Write-Host "`nSuccess: Ready to train!" -ForegroundColor Green

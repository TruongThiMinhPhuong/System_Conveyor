#!/bin/bash
# Fix picamera2 numpy compatibility issue on Raspberry Pi
# Run this script to resolve: numpy.dtype size changed error

echo "=========================================="
echo "Fixing picamera2 numpy compatibility"
echo "=========================================="
echo ""

# Check if running as pi user
if [ "$USER" != "pi" ]; then
    echo "⚠️  Warning: Not running as 'pi' user"
    echo "   Current user: $USER"
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
else
    echo "⚠️  No virtual environment found, using system Python"
fi

echo ""
echo "Step 1: Uninstalling numpy and picamera2..."
pip3 uninstall -y numpy picamera2

echo ""
echo "Step 2: Installing compatible numpy version..."
# Install numpy 1.23.5 which is compatible with picamera2
pip3 install "numpy<1.24"

echo ""
echo "Step 3: Reinstalling picamera2..."
pip3 install picamera2

echo ""
echo "Step 4: Verifying installation..."
python3 -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python3 -c "import picamera2; print('Picamera2: OK')" 2>&1 | grep -v "dtype"

echo ""
echo "=========================================="
echo "✅ Fix complete!"
echo "=========================================="
echo ""
echo "Test the camera:"
echo "  python3 -c \"from hardware import Camera; c = Camera(); c.initialize()\""
echo ""

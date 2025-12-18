#!/bin/bash
# Script Tự Động - Upload Notebook Lên Google Colab
# Chạy script này trên Raspberry Pi

echo "=========================================="
echo "🚀 Setup Google Colab Training"
echo "=========================================="

# Check if chromium is installed
if ! command -v chromium-browser &> /dev/null; then
    echo "⚠️ Chromium browser not found. Installing..."
    sudo apt-get update
    sudo apt-get install -y chromium-browser
fi

echo -e "\n📋 Chuẩn Bị Training Trên Google Colab"
echo "======================================"

# Instructions
cat << 'EOF'

✅ CÁC BƯỚC THỰC HIỆN:

1️⃣ Chuẩn Bị Dataset
   - Bạn cần có ảnh fresh và spoiled
   - Tối thiểu: 50 ảnh mỗi loại
   - Tổ chức thành folder: train/fresh, train/spoiled, val/fresh, val/spoiled

2️⃣ Nén Dataset
   Chạy lệnh sau để tạo file ZIP:
   
   cd ~/System_Conveyor
   zip -r dataset.zip raw_images/
   
   Hoặc nếu đã có folder khác:
   cd /path/to/your/images
   zip -r ~/dataset.zip train/ val/ test/

3️⃣ Mở Google Colab
   
   Tôi sẽ mở browser cho bạn...
   
EOF

read -p "Bạn đã có dataset.zip chưa? (y/n): " has_dataset

if [ "$has_dataset" != "y" ]; then
    echo ""
    echo "⚠️ Hãy chuẩn bị dataset trước!"
    echo ""
    echo "Tạo folder ảnh:"
    echo "  mkdir -p ~/my_dataset/train/fresh"
    echo "  mkdir -p ~/my_dataset/train/spoiled"
    echo "  mkdir -p ~/my_dataset/val/fresh"
    echo "  mkdir -p ~/my_dataset/val/spoiled"
    echo ""
    echo "Copy ảnh vào:"
    echo "  cp /path/to/fresh/*.jpg ~/my_dataset/train/fresh/"
    echo "  cp /path/to/spoiled/*.jpg ~/my_dataset/train/spoiled/"
    echo ""
    echo "Nén dataset:"
    echo "  cd ~"
    echo "  zip -r dataset.zip my_dataset/"
    echo ""
    exit 0
fi

echo ""
echo "✅ Tốt! Bắt đầu mở Colab..."
echo ""

# Open Google Colab in Chromium
echo "🌐 Mở Google Colab trong trình duyệt..."
chromium-browser "https://colab.research.google.com" &

sleep 3

cat << 'EOF'

📋 Hướng Dẫn Tiếp Theo (Trong Browser):

1. Đăng nhập Gmail (nếu chưa)

2. Upload Notebook:
   - Click: File → Upload notebook
   - Chọn file: ~/System_Conveyor/Train_MobileNet_Colab.ipynb
   
3. Chọn GPU MIỄN PHÍ:
   - Click: Runtime → Change runtime type
   - Hardware accelerator → Choose "T4 GPU"
   - Click Save
   
4. Chạy Lần Lượt:
   - Click vào cell đầu tiên
   - Nhấn Shift + Enter để chạy
   - Chờ xong, chạy cell tiếp theo
   
5. Upload Dataset:
   - Cell "Upload Dataset" sẽ có nút "Choose Files"
   - Chọn file dataset.zip của bạn
   - Chờ upload xong
   
6. Đợi Training:
   - Training sẽ mất ~15-20 phút
   - Theo dõi progress bar
   - val_accuracy > 0.90 là tốt!
   
7. Download Model:
   - Cell cuối sẽ tự động download
   - File tải về: mobilenet_classifier.tflite
   - Lưu vào ~/Downloads/
   
8. Copy Model:
   Quay lại terminal này và chạy:
   
   cp ~/Downloads/mobilenet_classifier.tflite ~/System_Conveyor/models/
   python3 ~/System_Conveyor/fruit_sorter.py

🎉 Xong! Hệ thống sẽ chạy với model mới!

EOF

echo ""
echo "✅ Browser đã mở!"
echo "📖 Làm theo hướng dẫn bên trên"
echo ""
echo "⚡ Tip: Copy nội dung trên để tham khảo khi cần!"
echo ""

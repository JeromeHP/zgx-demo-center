#!/bin/bash

echo "======================================"
echo "Document Translation Demo Installer"
echo "======================================"
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed."
    echo "Please install with: sudo apt-get install python3 python3-pip python3-venv"
    exit 1
fi

echo "✔ Python 3 found: $(python3 --version)"

# Use existing virtual environment
echo ""
echo "Using existing virtual environment: document-translate-env"
if [ ! -d "document-translate-env" ]; then
    echo "❌ Virtual environment 'document-translate-env' not found!"
    echo "Please ensure 'document-translate-env' exists in the current directory"
    exit 1
else
    echo "✔ Found virtual environment 'document-translate-env'"
fi

# Activate virtual environment
source document-translate-env/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install PyTorch for whichever accelerator this box actually has.
# This was pinned to the cu130 index, which on an AMD box installs a torch that
# cannot see the GPU -- backend/main.py uses device_map="auto" and fp16, so it
# would have quietly fallen back to CPU and nothing would have said so.
echo ""
echo "Installing PyTorch..."
TORCH_INDEX="$(bash "$(dirname "${BASH_SOURCE[0]}")/../scripts/torch_index.sh")"
echo "  wheel index: $TORCH_INDEX"
pip install --pre torch torchvision torchaudio --index-url "$TORCH_INDEX" --timeout 60 --retries 10

# Install backend dependencies
echo ""
echo "Installing dependencies..."
cd backend
pip install -r requirements.txt

echo ""
echo "======================================"
echo "⚠️  Model Setup Information"
echo "======================================"
echo ""
echo "This demo uses the Seed-X translation model:"
echo "  - Model: ByteDance-Seed/Seed-X-PPO-7B"
echo "  - Tokenizer: mistralai/Mistral-7B-v0.1"
echo ""
echo "IMPORTANT:"
echo "- Models will be loaded when you click 'Load Models' in the web interface"
echo "- The model is ~7B parameters (~14GB in float16)"
echo "- Loading may take several minutes on first run"
echo "- Ensure you have sufficient GPU memory (recommended: 16GB+ VRAM)"
echo "- Models will use float16 precision to reduce memory requirements"
echo ""
echo "Supported Language Pairs:"
echo "  - English → French, Spanish, German, Italian, Portuguese, Chinese"
echo "  - French, Spanish, German, Chinese → English"
echo ""

cd ..

echo ""
echo "======================================"
echo "✅ Installation Complete!"
echo "======================================"
echo ""
echo "To start the demo:"
echo "  chmod +x start_demo_remote.sh"
echo "  ./start_demo_remote.sh"
echo ""
echo "Then access from your Windows laptop:"
echo "  http://YOUR_SERVER_IP:8080"
echo ""
echo "The script will automatically detect your server IP address"
echo ""

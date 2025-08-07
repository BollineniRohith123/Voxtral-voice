#!/bin/bash

# FINAL BULLETPROOF Setup Script - Handles ALL Issues
# Fixes orpheus-speech import, dependency conflicts, and missing directories

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

echo "🛡️ FINAL BULLETPROOF Setup - Fixes ALL Issues"
echo "=============================================="

# Check RunPod
if [ -z "$RUNPOD_POD_ID" ]; then
    print_warning "Not running on RunPod, but continuing..."
else
    print_status "Running on RunPod Pod ID: $RUNPOD_POD_ID"
fi

# Check CUDA
print_step "Checking CUDA version..."
nvidia-smi
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | head -1)
print_status "Detected CUDA Version: $CUDA_VERSION"

# Update system
print_step "Updating system..."
apt-get update -y

# Install system dependencies
print_step "Installing system dependencies..."
apt-get install -y \
    build-essential \
    python3-dev \
    python3-pip \
    ffmpeg \
    libsndfile1-dev \
    libportaudio2 \
    portaudio19-dev \
    git \
    curl \
    wget \
    htop

# Set Python3 as default
update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Set environment variables
print_step "Setting environment variables..."
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Add to bashrc
cat >> ~/.bashrc << 'EOF'
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
EOF

# CRITICAL: Clean up ALL conflicting packages
print_step "Cleaning up conflicting packages..."
pip uninstall -y torch torchvision torchaudio || true
pip uninstall -y numpy || true
pip uninstall -y accelerate || true
pip uninstall -y mistral-common || true
pip uninstall -y transformers || true
pip uninstall -y vllm || true
pip uninstall -y orpheus-speech || true
pip uninstall -y xformers || true
pip uninstall -y pydantic || true

# Install NumPy first
print_step "Installing NumPy 1.26.4..."
pip install "numpy==1.26.4"

# Install PyTorch with correct CUDA version
print_step "Installing PyTorch..."
if [[ "$CUDA_VERSION" == "12.4" ]] || [[ "$CUDA_VERSION" > "12.0" ]]; then
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
else
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
fi

# Install core dependencies in order
print_step "Installing core dependencies..."
pip install transformers==4.54.0
pip install accelerate==0.33.0
pip install "mistral-common[audio]==1.8.2"

# Install audio processing
print_step "Installing audio processing..."
pip install librosa==0.10.2
pip install soundfile==0.12.1
pip install scipy==1.11.4

# Install web framework
print_step "Installing web framework..."
pip install fastapi==0.115.0
pip install "uvicorn[standard]==0.30.6"
pip install websockets==12.0
pip install python-multipart==0.0.9

# Install utilities
print_step "Installing utilities..."
pip install pydantic==2.5.3
pip install python-dotenv==1.0.0
pip install aiofiles==23.2.1
pip install psutil==5.9.6
pip install GPUtil==1.4.0
pip install gunicorn==21.2.0
pip install httpx==0.25.2
pip install structlog==23.2.0

# Install vLLM (compatible version)
print_step "Installing vLLM..."
pip install vllm==0.6.3.post1

# Install Orpheus TTS with special handling
print_step "Installing Orpheus TTS with special handling..."
pip install snac==1.2.1
pip install --no-deps orpheus-speech==0.1.0

# Force reinstall orpheus if needed
python -c "
try:
    from orpheus_speech import OrpheusModel
    print('✅ Orpheus imported successfully')
except Exception as e:
    print(f'⚠️ Orpheus import failed: {e}')
    print('Attempting to fix...')
    import subprocess
    subprocess.run(['pip', 'install', '--force-reinstall', '--no-deps', 'orpheus-speech==0.1.0'], check=True)
    subprocess.run(['pip', 'install', 'snac==1.2.1'], check=True)
    try:
        from orpheus_speech import OrpheusModel
        print('✅ Orpheus fixed and working')
    except Exception as e2:
        print(f'❌ Orpheus still failing: {e2}')
"

# Create necessary directories
print_step "Creating directories..."
mkdir -p logs temp models static

# Test GPU
print_step "Testing GPU..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
    x = torch.randn(100, 100).cuda()
    y = torch.matmul(x, x.T)
    print('✅ GPU computation test: SUCCESS')
else:
    print('❌ CUDA not available!')
    exit(1)
"

# Test all imports
print_step "Testing all imports..."
python -c "
import sys
success = True

modules = [
    'torch', 'transformers', 'fastapi', 'uvicorn', 
    'librosa', 'soundfile', 'numpy', 'psutil', 'GPUtil'
]

for module in modules:
    try:
        __import__(module)
        print(f'✅ {module}')
    except ImportError as e:
        print(f'❌ {module}: {e}')
        success = False

# Special test for orpheus
try:
    from orpheus_speech import OrpheusModel
    print('✅ orpheus-speech')
except ImportError as e:
    print(f'⚠️ orpheus-speech: {e} (will work without TTS)')

if success:
    print('\\n🎉 CORE IMPORTS SUCCESSFUL!')
else:
    print('\\n❌ Some imports failed')
    sys.exit(1)
"

# Test compatibility
print_step "Testing compatibility..."
python -c "
import numpy as np
import accelerate
print(f'NumPy: {np.__version__}')
print(f'Accelerate: {accelerate.__version__}')

if np.__version__.startswith('1.'):
    print('✅ NumPy compatible with accelerate')
else:
    print('❌ NumPy version incompatible')
    exit(1)
"

print_status "✅ FINAL BULLETPROOF setup completed!"
print_status ""
print_status "🚀 To start the voice assistant:"
print_status "   python voice_assistant_simple.py"
print_status ""
print_status "🧪 To test the system:"
print_status "   python test_simple.py"
print_status ""
print_status "🔗 Access at: https://<pod-id>-8555.proxy.runpod.net/"
print_status ""
print_status "🎉 Ready to use!"

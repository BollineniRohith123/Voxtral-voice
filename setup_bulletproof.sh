#!/bin/bash

# BULLETPROOF Setup Script for RunPod A40 - Handles ALL Conflicts
# This script fixes the exact issues you encountered

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

echo "🛡️ BULLETPROOF RunPod Setup - Fixes ALL Dependency Conflicts"
echo "============================================================="

# Check if running on RunPod
if [ -z "$RUNPOD_POD_ID" ]; then
    print_warning "Not running on RunPod, but continuing..."
else
    print_status "Running on RunPod Pod ID: $RUNPOD_POD_ID"
fi

# Check CUDA version
print_step "Checking CUDA version..."
nvidia-smi
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | head -1)
print_status "Detected CUDA Version: $CUDA_VERSION"

# Update system
print_step "Updating system packages..."
apt-get update -y

# Install system dependencies
print_step "Installing system dependencies..."
apt-get install -y \
    build-essential \
    software-properties-common \
    wget \
    curl \
    git \
    ffmpeg \
    libsndfile1-dev \
    libportaudio2 \
    portaudio19-dev \
    python3-dev \
    python3-pip \
    python3-venv \
    htop \
    tmux \
    vim

# Set Python3 as default
update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Set environment variables
print_step "Setting up environment variables..."
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

# CRITICAL: Clean up existing installations to avoid conflicts
print_step "Cleaning up existing installations..."
pip uninstall -y torch torchvision torchaudio || true
pip uninstall -y numpy || true
pip uninstall -y accelerate || true
pip uninstall -y mistral-common || true
pip uninstall -y transformers || true
pip uninstall -y vllm || true
pip uninstall -y orpheus-speech || true

# Install NumPy first with EXACT compatible version
print_step "Installing NumPy with compatible version..."
pip install "numpy==1.26.4"

# Install PyTorch with correct CUDA version
print_step "Installing PyTorch with CUDA support..."
if [[ "$CUDA_VERSION" == "12.4" ]] || [[ "$CUDA_VERSION" > "12.0" ]]; then
    print_status "Installing PyTorch for CUDA 12.4+"
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
else
    print_status "Installing PyTorch for CUDA 11.8"
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
fi

# Install dependencies in EXACT order to avoid conflicts
print_step "Installing dependencies in order..."

# Core ML libraries
pip install transformers==4.54.0
pip install accelerate==0.33.0
pip install "mistral-common[audio]==1.8.2"

# Audio processing
pip install librosa==0.10.2
pip install soundfile==0.12.1
pip install scipy==1.11.4

# Web framework
pip install fastapi==0.115.0
pip install "uvicorn[standard]==0.30.6"
pip install websockets==12.0
pip install python-multipart==0.0.9

# Additional utilities
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

# Install Orpheus TTS (EXACT version)
print_step "Installing Orpheus TTS..."
pip install orpheus-speech==0.1.0

# Verify installations
print_step "Verifying installations..."
python -c "
import sys
try:
    import torch
    print(f'✅ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')
    
    import numpy as np
    print(f'✅ NumPy {np.__version__}')
    
    import transformers
    print(f'✅ Transformers {transformers.__version__}')
    
    import accelerate
    print(f'✅ Accelerate {accelerate.__version__}')
    
    import vllm
    print(f'✅ vLLM {vllm.__version__}')
    
    from orpheus_speech import OrpheusModel
    print('✅ Orpheus Speech imported')
    
    import fastapi
    print(f'✅ FastAPI {fastapi.__version__}')
    
    import librosa
    print(f'✅ Librosa {librosa.__version__}')
    
    import GPUtil
    print('✅ GPUtil imported')
    
    print('\\n🎉 ALL IMPORTS SUCCESSFUL!')
    
except Exception as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

# Test GPU
print_step "Testing GPU..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
    
    # Test GPU computation
    x = torch.randn(100, 100).cuda()
    y = torch.matmul(x, x.T)
    print('✅ GPU computation test: SUCCESS')
else:
    print('❌ CUDA not available!')
    exit(1)
"

# Test dependency compatibility
print_step "Testing dependency compatibility..."
python -c "
import numpy as np
import accelerate

print(f'NumPy version: {np.__version__}')
print(f'Accelerate version: {accelerate.__version__}')

# Check compatibility
if np.__version__.startswith('1.'):
    print('✅ NumPy version compatible with accelerate')
else:
    print('❌ NumPy version incompatible')
    exit(1)
"

# Create directories
print_step "Creating directories..."
mkdir -p logs temp models static

print_status "✅ BULLETPROOF setup completed successfully!"
print_status ""
print_status "🚀 To start the SIMPLE voice assistant (recommended):"
print_status "   python voice_assistant_simple.py"
print_status ""
print_status "🧪 To test the system:"
print_status "   python test_runpod.py"
print_status ""
print_status "🔗 Don't forget to expose port 8555 in RunPod!"
print_status ""
print_status "🎉 Your bulletproof voice assistant is ready!"

#!/bin/bash

# BULLETPROOF RunPod Setup Script for Voxtral + Orpheus Voice Assistant
# Fixed for CUDA 12.4 and dependency conflicts

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

echo "🚀 BULLETPROOF Voxtral + Orpheus Setup for RunPod A40 (CUDA 12.4)"
echo "=================================================================="

# Check if running on RunPod
if [ -z "$RUNPOD_POD_ID" ]; then
    print_warning "Not running on RunPod, but continuing..."
else
    print_status "Running on RunPod Pod ID: $RUNPOD_POD_ID"
fi

# Check CUDA version
print_step "Checking CUDA version..."
nvidia-smi
nvcc --version || print_warning "nvcc not found, but nvidia-smi works"

# Update system packages
print_step "Updating system packages..."
apt-get update -y

# Install essential system dependencies
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
    screen \
    vim \
    nano

# Set Python3 as default
print_step "Setting up Python environment..."
update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Set critical environment variables for A40 optimization
print_step "Setting up environment variables..."
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export NVIDIA_TF32_OVERRIDE=0
export TORCH_CUDNN_V8_API_ENABLED=1

# Add to bashrc for persistence
cat >> ~/.bashrc << 'EOF'
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export NVIDIA_TF32_OVERRIDE=0
export TORCH_CUDNN_V8_API_ENABLED=1
EOF

# CRITICAL: Fix dependency conflicts first
print_step "Fixing dependency conflicts..."

# Uninstall conflicting packages
pip uninstall -y numpy || true
pip uninstall -y accelerate || true
pip uninstall -y mistral-common || true

# Install NumPy first with correct version
print_step "Installing NumPy with correct version..."
pip install "numpy==1.26.4"

# Install PyTorch with CUDA 12.4 support (RunPod default)
print_step "Installing PyTorch with CUDA 12.4 support..."
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Install core dependencies in specific order to avoid conflicts
print_step "Installing core dependencies in order..."
pip install transformers==4.54.0
pip install accelerate==0.33.0
pip install "mistral-common[audio]==1.8.2"

# Install vLLM with compatible version
print_step "Installing vLLM..."
pip install vllm==0.6.3.post1

# Install Orpheus TTS (CRITICAL - exact version)
print_step "Installing Orpheus TTS..."
pip install orpheus-speech==0.1.0

# Install remaining requirements one by one to catch conflicts
print_step "Installing remaining requirements..."
pip install fastapi==0.115.0
pip install "uvicorn[standard]==0.30.6"
pip install websockets==12.0
pip install python-multipart==0.0.9
pip install librosa==0.10.2
pip install soundfile==0.12.1
pip install scipy==1.11.4
pip install pydantic==2.5.3
pip install python-dotenv==1.0.0
pip install aiofiles==23.2.1
pip install psutil==5.9.6
pip install GPUtil==1.4.0
pip install gunicorn==21.2.0
pip install httpx==0.25.2
pip install structlog==23.2.0

# Verify GPU and CUDA
print_step "Verifying GPU setup..."
python -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')

    # Test GPU computation
    x = torch.randn(100, 100).cuda()
    y = torch.matmul(x, x.T)
    print(f'GPU computation test: SUCCESS')
else:
    print('ERROR: CUDA not available!')
    exit(1)
"

# Test model imports
print_step "Testing model imports..."
python -c "
try:
    import transformers
    print('✅ transformers imported')

    import vllm
    print('✅ vllm imported')

    from orpheus_speech import OrpheusModel
    print('✅ orpheus-speech imported')

    import fastapi
    print('✅ fastapi imported')

    import librosa
    print('✅ librosa imported')

    import GPUtil
    print('✅ GPUtil imported')

    print('✅ All imports successful')
except Exception as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

# Create necessary directories
print_step "Creating directories..."
mkdir -p logs temp models static

# Test dependency compatibility
print_step "Testing dependency compatibility..."
python -c "
import numpy as np
import accelerate
print(f'NumPy version: {np.__version__}')
print(f'Accelerate version: {accelerate.__version__}')

# Check if versions are compatible
if np.__version__.startswith('1.'):
    print('✅ NumPy version compatible with accelerate')
else:
    print('❌ NumPy version may cause issues')
    exit(1)
"

# Download and cache models (optional but recommended)
print_step "Pre-downloading models (this may take 10-15 minutes)..."
python -c "
import torch
from transformers import AutoProcessor, VoxtralForConditionalGeneration

print('Downloading Voxtral Mini 3B...')
try:
    processor = AutoProcessor.from_pretrained('mistralai/Voxtral-Mini-3B-2507')
    print('✅ Voxtral processor downloaded')

    # Just download, don't load to GPU yet
    model = VoxtralForConditionalGeneration.from_pretrained(
        'mistralai/Voxtral-Mini-3B-2507',
        torch_dtype=torch.bfloat16,
        device_map='cpu'
    )
    print('✅ Voxtral model downloaded')

    # Clear memory
    del model, processor
    torch.cuda.empty_cache()

except Exception as e:
    print(f'⚠️ Voxtral download failed: {e}')
    print('Models will be downloaded on first use')
"

print_status "✅ RunPod setup completed successfully!"
print_status ""
print_status "🚀 To start the voice assistant:"
print_status "   python voice_assistant_runpod.py"
print_status ""
print_status "🧪 To test the system:"
print_status "   python test_runpod.py"
print_status ""
print_status "🔗 Don't forget to expose port 8555 in RunPod!"
print_status ""
print_status "🎉 Your voice assistant is ready!"

# Verify GPU and CUDA
print_step "Verifying GPU setup..."
python -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
else:
    print('ERROR: CUDA not available!')
    exit(1)
"

# Test model imports
print_step "Testing model imports..."
python -c "
try:
    import transformers
    import vllm
    from orpheus_speech import OrpheusModel
    print('✅ All imports successful')
except Exception as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

# Create necessary directories
print_step "Creating directories..."
mkdir -p logs temp models static

# Download and cache models
print_step "Pre-downloading models (this may take 10-15 minutes)..."
python -c "
import torch
from transformers import AutoProcessor, VoxtralForConditionalGeneration

print('Downloading Voxtral Mini 3B...')
try:
    processor = AutoProcessor.from_pretrained('mistralai/Voxtral-Mini-3B-2507')
    print('✅ Voxtral processor downloaded')
    
    # Just download, don't load to GPU yet
    model = VoxtralForConditionalGeneration.from_pretrained(
        'mistralai/Voxtral-Mini-3B-2507',
        torch_dtype=torch.bfloat16,
        device_map='cpu'
    )
    print('✅ Voxtral model downloaded')
    
    # Clear memory
    del model, processor
    torch.cuda.empty_cache()
    
except Exception as e:
    print(f'⚠️ Voxtral download failed: {e}')
    print('Models will be downloaded on first use')
"

print_status "✅ RunPod setup completed successfully!"
print_status ""
print_status "🚀 To start the voice assistant:"
print_status "   python voice_assistant_runpod.py"
print_status ""
print_status "🔗 Don't forget to expose port 8555 in RunPod!"
print_status ""
print_status "🎉 Your voice assistant is ready!"

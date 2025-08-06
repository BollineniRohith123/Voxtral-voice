#!/bin/bash

# RunPod A40 Deployment Script - Production Ready
set -e

echo "🚀 DEPLOYING Voxtral + Orpheus Voice Assistant on RunPod A40"
echo "============================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if running on RunPod
if [ -z "$RUNPOD_POD_ID" ]; then
    print_warning "Not running on RunPod, but continuing deployment..."
else
    print_status "Running on RunPod Pod ID: $RUNPOD_POD_ID"
fi

# Update system
print_step "Updating system packages..."
apt-get update -y
apt-get upgrade -y

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
    nvidia-utils-525 \
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
echo 'export CUDA_VISIBLE_DEVICES=0' >> ~/.bashrc
echo 'export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"' >> ~/.bashrc
echo 'export TOKENIZERS_PARALLELISM=false' >> ~/.bashrc
echo 'export PYTHONUNBUFFERED=1' >> ~/.bashrc

# Install PyTorch with CUDA support (EXACT VERSION for A40)
print_step "Installing PyTorch with CUDA support..."
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

# Install exact transformers version with Voxtral support
print_step "Installing Transformers with Voxtral support..."
pip install transformers==4.54.0

# Install mistral-common with audio support
print_step "Installing Mistral Common..."
pip install "mistral-common[audio]>=1.8.1"

# Install vLLM with specific version for Voxtral
print_step "Installing vLLM for Voxtral..."
pip install "vllm[audio]==0.10.0" --extra-index-url https://wheels.vllm.ai/

# Install Orpheus TTS (CRITICAL - exact version)
print_step "Installing Orpheus TTS..."
pip install orpheus-speech==0.1.0

# Install remaining requirements
print_step "Installing remaining requirements..."
pip install -r requirements.txt

# Verify GPU and CUDA
print_step "Verifying GPU setup..."
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(f'CUDA Version: {torch.version.cuda}');"

# Create necessary directories
print_step "Creating directories..."
mkdir -p logs temp models static

# Download models to cache (optional but recommended)
print_step "Pre-downloading models (this may take a while)..."
python -c "
from transformers import AutoProcessor, VoxtralForConditionalGeneration
import torch

print('Downloading Voxtral Mini 3B...')
try:
    processor = AutoProcessor.from_pretrained('mistralai/Voxtral-Mini-3B-2507')
    model = VoxtralForConditionalGeneration.from_pretrained(
        'mistralai/Voxtral-Mini-3B-2507',
        torch_dtype=torch.bfloat16,
        device_map='cpu'  # Just download, don't load to GPU yet
    )
    print('✅ Voxtral downloaded')
    
    # Clear memory
    del model, processor
    torch.cuda.empty_cache()
except Exception as e:
    print(f'⚠️ Voxtral download failed: {e}')
"

# Create startup script
print_step "Creating startup script..."
cat > start_voice_assistant.sh << 'EOF'
#!/bin/bash

echo "🎤 Starting Voxtral + Orpheus Voice Assistant..."

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Check GPU
echo "🔍 Checking GPU..."
nvidia-smi

# Start the application
echo "🚀 Launching voice assistant..."
python voice_assistant.py
EOF

chmod +x start_voice_assistant.sh

# Create system monitoring script
print_step "Creating monitoring script..."
cat > monitor_system.sh << 'EOF'
#!/bin/bash

echo "📊 System Monitor for Voice Assistant"
echo "====================================="

while true; do
    clear
    echo "📊 System Monitor - $(date)"
    echo "====================================="
    
    echo ""
    echo "🖥️  GPU Status:"
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
    
    echo ""
    echo "💾 Memory Usage:"
    free -h
    
    echo ""
    echo "🔄 CPU Usage:"
    top -bn1 | grep "Cpu(s)" | awk '{print $2 $3 $4 $5 $6 $7 $8}'
    
    echo ""
    echo "🌐 Network Connections:"
    netstat -an | grep :8555 | wc -l | awk '{print "Active connections: " $1}'
    
    echo ""
    echo "📝 Recent Logs:"
    tail -n 5 voice_assistant.log 2>/dev/null || echo "No logs yet"
    
    echo ""
    echo "🧪 Health Check:"
    curl -s http://localhost:8555/health | python -m json.tool 2>/dev/null || echo "Service not responding"
    
    echo ""
    echo "Press Ctrl+C to exit"
    sleep 5
done
EOF

chmod +x monitor_system.sh

print_status "✅ RunPod A40 deployment completed successfully!"
echo ""
print_status "🚀 To start the voice assistant:"
echo "   ./start_voice_assistant.sh"
echo ""
print_status "🧪 To test the system:"
echo "   python test_system.py"
echo ""
print_status "📊 To monitor system:"
echo "   ./monitor_system.sh"
echo ""
print_status "🔗 Don't forget to expose port 8555 in RunPod!"
echo ""
print_status "🎉 Your production voice assistant is ready!"

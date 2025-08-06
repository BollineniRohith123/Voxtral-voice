#!/bin/bash

# Production Startup Script for Voxtral + Orpheus Voice Assistant
# Optimized for RunPod A40 deployment

set -e

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

echo "🎤 Starting Voxtral + Orpheus Voice Assistant..."
echo "=============================================="

# Set environment variables for optimal performance
print_step "Setting environment variables..."
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export NVIDIA_TF32_OVERRIDE=0
export TORCH_CUDNN_V8_API_ENABLED=1

# Check if running in RunPod
if [ ! -z "$RUNPOD_POD_ID" ]; then
    print_status "Running on RunPod Pod ID: $RUNPOD_POD_ID"
    
    # Get RunPod public URL if available
    if [ ! -z "$RUNPOD_PUBLIC_IP" ]; then
        print_status "Public URL: https://$RUNPOD_PUBLIC_IP-8555.proxy.runpod.net/"
    fi
fi

# Check GPU availability
print_step "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    
    # Check GPU memory
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ "$GPU_MEMORY" -lt 30000 ]; then
        print_warning "GPU has less than 30GB memory. Performance may be limited."
    else
        print_status "GPU memory sufficient: ${GPU_MEMORY}MB"
    fi
else
    print_error "nvidia-smi not found. GPU may not be available."
fi

# Check Python and dependencies
print_step "Checking Python environment..."
python --version
pip list | grep -E "(torch|transformers|vllm|fastapi)" || print_warning "Some dependencies may be missing"

# Check disk space
print_step "Checking disk space..."
df -h . | tail -1 | awk '{print "Available disk space: " $4}'

# Create necessary directories
print_step "Creating directories..."
mkdir -p logs temp models static

# Check if models are cached
print_step "Checking model cache..."
if [ -d "$HOME/.cache/huggingface" ]; then
    CACHE_SIZE=$(du -sh $HOME/.cache/huggingface 2>/dev/null | cut -f1)
    print_status "HuggingFace cache size: $CACHE_SIZE"
else
    print_warning "No model cache found. First run will download models."
fi

# Pre-flight checks
print_step "Running pre-flight checks..."

# Check if port 8555 is available
if lsof -Pi :8555 -sTCP:LISTEN -t >/dev/null ; then
    print_error "Port 8555 is already in use!"
    print_status "Killing existing process..."
    kill $(lsof -t -i:8555) 2>/dev/null || true
    sleep 2
fi

# Check Python imports
python -c "
import sys
try:
    import torch
    import transformers
    import fastapi
    import uvicorn
    print('✅ Core dependencies available')
    
    if torch.cuda.is_available():
        print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        print('⚠️ CUDA not available')
        
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    print_error "Pre-flight checks failed!"
    exit 1
fi

print_status "✅ Pre-flight checks passed!"

# Start the application with proper logging
print_step "Starting voice assistant..."
print_status "🌐 Web interface will be available at:"
print_status "   Local: http://localhost:8555"
if [ ! -z "$RUNPOD_PUBLIC_IP" ]; then
    print_status "   Public: https://$RUNPOD_PUBLIC_IP-8555.proxy.runpod.net/"
fi

print_status "📊 Health check: http://localhost:8555/health"
print_status "📚 API docs: http://localhost:8555/docs"
print_status ""
print_status "🔄 Starting application... (this may take a few minutes for model loading)"

# Start with proper error handling
if [ "$1" = "--background" ] || [ "$1" = "-d" ]; then
    print_status "Starting in background mode..."
    nohup python voice_assistant.py > logs/voice_assistant.log 2>&1 &
    echo $! > voice_assistant.pid
    print_status "Application started with PID: $(cat voice_assistant.pid)"
    print_status "Monitor logs with: tail -f logs/voice_assistant.log"
else
    print_status "Starting in foreground mode..."
    print_status "Press Ctrl+C to stop the application"
    python voice_assistant.py
fi

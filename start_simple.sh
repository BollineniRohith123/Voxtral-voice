#!/bin/bash

# Simple start script with error handling

echo "🎤 Starting Simple Voice Assistant..."
echo "===================================="

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Create necessary directories
mkdir -p logs temp models static

# Check if running on RunPod
if [ ! -z "$RUNPOD_POD_ID" ]; then
    echo "✅ Running on RunPod Pod ID: $RUNPOD_POD_ID"
    
    if [ ! -z "$RUNPOD_PUBLIC_IP" ]; then
        echo "🌐 Public URL: https://$RUNPOD_PUBLIC_IP-8555.proxy.runpod.net/"
    fi
else
    echo "⚠️ Not running on RunPod (local development mode)"
fi

# Check GPU
echo "🔍 Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️ nvidia-smi not found"
fi

# Check Python imports
echo "🔍 Checking Python imports..."
python -c "
import sys
try:
    import torch
    print(f'✅ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')
    
    import fastapi
    print(f'✅ FastAPI {fastapi.__version__}')
    
    import librosa
    print('✅ Librosa')
    
    import numpy as np
    print(f'✅ NumPy {np.__version__}')
    
    try:
        from orpheus_speech import OrpheusModel
        print('✅ Orpheus TTS available')
    except ImportError:
        print('⚠️ Orpheus TTS not available (will run without TTS)')
    
    print('✅ Core dependencies ready')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
    print('Please run setup_final.sh first')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Dependency check failed!"
    echo "Please run: ./setup_final.sh"
    exit 1
fi

# Start the application
echo "🚀 Starting voice assistant..."
echo "📊 Health check: http://localhost:8555/health"
echo "🌐 Web interface: http://localhost:8555"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python voice_assistant_simple.py

#!/bin/bash

# Simple start script for RunPod Voice Assistant

echo "🎤 Starting Voxtral + Orpheus Voice Assistant for RunPod..."
echo "=========================================================="

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Check if running on RunPod
if [ ! -z "$RUNPOD_POD_ID" ]; then
    echo "✅ Running on RunPod Pod ID: $RUNPOD_POD_ID"
    
    if [ ! -z "$RUNPOD_PUBLIC_IP" ]; then
        echo "🌐 Public URL: https://$RUNPOD_PUBLIC_IP-8555.proxy.runpod.net/"
    fi
else
    echo "⚠️  Not running on RunPod (local development mode)"
fi

# Check GPU
echo "🔍 Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Create logs directory
mkdir -p logs

# Start the application
echo "🚀 Starting voice assistant..."
echo "📊 Health check: http://localhost:8555/health"
echo "🌐 Web interface: http://localhost:8555"
echo ""

python voice_assistant_runpod.py

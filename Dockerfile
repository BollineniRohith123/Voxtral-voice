# Production-Ready Dockerfile for Voxtral + Orpheus Voice Assistant
# Optimized for RunPod A40 GPU deployment

FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# GPU and CUDA optimizations
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
ENV TOKENIZERS_PARALLELISM=false
ENV NVIDIA_TF32_OVERRIDE=0
ENV TORCH_CUDNN_V8_API_ENABLED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    software-properties-common \
    wget \
    curl \
    git \
    ffmpeg \
    libsndfile1-dev \
    libportaudio2 \
    portaudio19-dev \
    libasound2-dev \
    libssl-dev \
    libffi-dev \
    htop \
    tmux \
    screen \
    vim \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Set Python3 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Upgrade pip and install build tools
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support first (critical for GPU)
RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install vLLM with audio support (specific version for Voxtral)
RUN pip install "vllm[audio]==0.10.0" --extra-index-url https://wheels.vllm.ai/

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs temp models static

# Set proper permissions
RUN chmod +x *.py
RUN chmod +x *.sh 2>/dev/null || true

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8555/health || exit 1

# Expose port
EXPOSE 8555

# Default command
CMD ["python", "voice_assistant.py"]

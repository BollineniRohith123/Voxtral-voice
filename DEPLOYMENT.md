# 🚀 Deployment Guide

Complete deployment guide for the Voxtral + Orpheus Voice Assistant across different platforms.

## 📋 Pre-Deployment Checklist

- [ ] GPU with 30GB+ VRAM (A40 recommended)
- [ ] 32GB+ system RAM
- [ ] 80GB+ free disk space
- [ ] Ubuntu 22.04+ or compatible Linux distribution
- [ ] CUDA 11.8+ drivers installed
- [ ] Python 3.8+ installed
- [ ] Git installed

## 🎯 RunPod Deployment (Recommended)

### Step 1: Create RunPod Instance

1. **Login to RunPod**:
   - Go to [runpod.io](https://runpod.io)
   - Sign in to your account

2. **Create New Pod**:
   - Click "Deploy" → "GPU Pod"
   - Select **A40** GPU (40GB VRAM)
   - Choose **Ubuntu 22.04 + PyTorch** template
   - Set **Disk**: 80GB minimum
   - **Expose TCP Port**: 8555
   - **Container Disk**: 50GB minimum

3. **Pod Configuration**:
   ```
   GPU: A40 (40GB VRAM)
   vCPU: 16+ cores
   RAM: 64GB
   Storage: 80GB+
   Network: Expose port 8555
   ```

### Step 2: Connect and Deploy

1. **Connect to Pod**:
   ```bash
   # Use RunPod's web terminal or SSH
   ssh root@<pod-ip>
   ```

2. **Clone Repository**:
   ```bash
   cd /workspace
   git clone <your-repository-url>
   cd voxtral-orpheus-assistant
   ```

3. **Run Deployment Script**:
   ```bash
   chmod +x deploy_runpod.sh
   ./deploy_runpod.sh
   ```

4. **Test Installation**:
   ```bash
   python test_system.py
   ```

5. **Start Application**:
   ```bash
   ./start_voice_assistant.sh
   ```

### Step 3: Access Your Application

- **Web Interface**: `https://<pod-id>-8555.proxy.runpod.net/`
- **Health Check**: `https://<pod-id>-8555.proxy.runpod.net/health`
- **API Documentation**: `https://<pod-id>-8555.proxy.runpod.net/docs`

## 🐳 Docker Deployment

### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA Container Toolkit

### Step 1: Install NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Step 2: Deploy with Docker Compose

```bash
# Clone repository
git clone <your-repository-url>
cd voxtral-orpheus-assistant

# Build and start
docker-compose up --build -d

# Check logs
docker-compose logs -f

# Stop
docker-compose down
```

### Step 3: Manual Docker Build

```bash
# Build image
docker build -t voxtral-orpheus .

# Run container
docker run -d \
  --name voxtral-orpheus \
  --gpus all \
  -p 8555:8555 \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/models:/app/models \
  voxtral-orpheus

# Check status
docker ps
docker logs voxtral-orpheus
```

## 💻 Local Development Deployment

### Step 1: System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
  python3 python3-pip python3-dev \
  build-essential \
  ffmpeg libsndfile1-dev \
  libportaudio2 portaudio19-dev \
  git curl wget

# Set Python3 as default
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 1
```

### Step 2: CUDA Installation

```bash
# Install CUDA 11.8 (adjust version as needed)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Step 3: Python Environment

```bash
# Clone repository
git clone <your-repository-url>
cd voxtral-orpheus-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r requirements.txt

# Install vLLM
pip install "vllm[audio]==0.10.0" --extra-index-url https://wheels.vllm.ai/
```

### Step 4: Configuration and Testing

```bash
# Copy environment file
cp .env.example .env

# Edit configuration (optional)
nano .env

# Test system
python test_system.py

# Start application
python voice_assistant.py
```

## ☁️ Cloud Platform Deployment

### AWS EC2

1. **Launch Instance**:
   - Instance Type: `p3.2xlarge` or `p4d.xlarge`
   - AMI: Deep Learning AMI (Ubuntu 22.04)
   - Storage: 100GB+ EBS
   - Security Group: Allow port 8555

2. **Setup**:
   ```bash
   ssh -i your-key.pem ubuntu@<instance-ip>
   git clone <your-repository-url>
   cd voxtral-orpheus-assistant
   ./deploy_runpod.sh  # Works on EC2 too
   ```

### Google Cloud Platform

1. **Create VM**:
   ```bash
   gcloud compute instances create voxtral-assistant \
     --zone=us-central1-a \
     --machine-type=n1-standard-8 \
     --accelerator=type=nvidia-tesla-v100,count=1 \
     --image-family=pytorch-latest-gpu \
     --image-project=deeplearning-platform-release \
     --boot-disk-size=100GB \
     --maintenance-policy=TERMINATE
   ```

2. **Deploy**:
   ```bash
   gcloud compute ssh voxtral-assistant
   # Follow local deployment steps
   ```

### Azure

1. **Create VM**:
   - Size: Standard_NC6s_v3 or similar
   - Image: Data Science Virtual Machine - Ubuntu 22.04
   - Disk: 100GB+ Premium SSD

2. **Deploy**:
   ```bash
   ssh azureuser@<vm-ip>
   # Follow local deployment steps
   ```

## 🔧 Production Configuration

### Environment Variables

```bash
# Production settings
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
export TOKENIZERS_PARALLELISM=false

# Optional: API security
export API_KEY="your-secure-api-key"
export JWT_SECRET="your-jwt-secret"
```

### Systemd Service (Linux)

Create `/etc/systemd/system/voxtral-assistant.service`:

```ini
[Unit]
Description=Voxtral Orpheus Voice Assistant
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/voxtral-orpheus-assistant
Environment=PYTHONUNBUFFERED=1
Environment=CUDA_VISIBLE_DEVICES=0
ExecStart=/home/ubuntu/voxtral-orpheus-assistant/venv/bin/python voice_assistant.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable voxtral-assistant
sudo systemctl start voxtral-assistant
sudo systemctl status voxtral-assistant
```

### Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8555;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
```

## 📊 Monitoring Setup

### Prometheus Metrics

The application exposes metrics at `/metrics` endpoint. Configure Prometheus:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'voxtral-assistant'
    static_configs:
      - targets: ['localhost:8555']
```

### Grafana Dashboard

Import the provided Grafana dashboard configuration for monitoring:
- GPU utilization
- Memory usage
- Request latency
- Active connections
- Error rates

### Log Management

```bash
# Centralized logging with rsyslog
echo "*.* @@your-log-server:514" >> /etc/rsyslog.conf
systemctl restart rsyslog

# Log rotation
cat > /etc/logrotate.d/voxtral-assistant << EOF
/path/to/voice_assistant.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 ubuntu ubuntu
}
EOF
```

## 🔒 Security Hardening

### Firewall Configuration

```bash
# UFW (Ubuntu)
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 8555/tcp  # Application
sudo ufw enable

# iptables
sudo iptables -A INPUT -p tcp --dport 8555 -j ACCEPT
sudo iptables-save > /etc/iptables/rules.v4
```

### SSL/TLS Setup

```bash
# Let's Encrypt with Certbot
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### API Security

Enable API key authentication in `.env`:
```bash
API_KEY=your-secure-random-api-key-here
JWT_SECRET=your-jwt-secret-key-here
```

## 🚨 Troubleshooting

### Common Deployment Issues

1. **CUDA Not Found**:
   ```bash
   # Check CUDA installation
   nvcc --version
   nvidia-smi
   
   # Reinstall CUDA drivers
   sudo apt purge nvidia-*
   sudo apt install nvidia-driver-525
   sudo reboot
   ```

2. **Out of Memory**:
   ```bash
   # Reduce GPU memory utilization
   export GPU_MEMORY_UTILIZATION=0.7
   
   # Clear GPU memory
   python -c "import torch; torch.cuda.empty_cache()"
   ```

3. **Port Conflicts**:
   ```bash
   # Find process using port
   sudo lsof -i :8555
   
   # Kill process
   sudo kill $(sudo lsof -t -i:8555)
   ```

4. **Model Download Issues**:
   ```bash
   # Clear cache and retry
   rm -rf ~/.cache/huggingface
   export HF_HUB_DISABLE_PROGRESS_BARS=1
   ```

### Performance Issues

1. **Slow Response Times**:
   - Check GPU utilization: `nvidia-smi`
   - Monitor system resources: `htop`
   - Review logs for errors
   - Reduce concurrent connections

2. **Memory Leaks**:
   - Monitor memory usage over time
   - Restart service periodically
   - Check for unclosed connections

### Getting Help

- Check logs: `tail -f voice_assistant.log`
- Run diagnostics: `python test_system.py`
- Monitor system: `./monitor_system.sh`
- Review health endpoint: `curl http://localhost:8555/health`

## 📈 Scaling Considerations

### Horizontal Scaling

- Deploy multiple instances behind a load balancer
- Use Redis for session management
- Implement sticky sessions for WebSocket connections

### Vertical Scaling

- Upgrade to larger GPU instances (A100, H100)
- Increase system memory
- Use faster storage (NVMe SSDs)

### Cost Optimization

- Use spot instances for development
- Implement auto-scaling based on demand
- Monitor and optimize GPU utilization
- Use model quantization for reduced memory usage

---

**Need help?** Check our [troubleshooting guide](TROUBLESHOOTING.md) or open an issue on GitHub.

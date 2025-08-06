# 🎤 Voxtral + Orpheus Real-Time Voice Assistant

A production-ready, real-time voice assistant powered by **Voxtral** (speech recognition) and **Orpheus** (text-to-speech) models, optimized for RunPod A40 GPU deployment.

## ✨ Features

- **🎯 Real-Time Processing**: Ultra-low latency voice interaction (<600ms end-to-end)
- **🧠 Advanced Models**: Voxtral Mini 3B for speech recognition + Orpheus 3B for emotional TTS
- **🎭 Emotional TTS**: 8 different voices with emotion detection and synthesis
- **🚀 Production Ready**: Comprehensive error handling, monitoring, and logging
- **🔧 RunPod Optimized**: Specifically tuned for A40 GPU performance
- **🌐 Web Interface**: Beautiful, responsive web client with real-time stats
- **📊 Monitoring**: Real-time system monitoring and health checks
- **🐳 Docker Support**: Complete containerization for easy deployment

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Client    │◄──►│   FastAPI +      │◄──►│   GPU Models    │
│   (Browser)     │    │   WebSocket      │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Connection     │    │   Voxtral +     │
                       │   Manager        │    │   Orpheus       │
                       └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Option 1: RunPod Deployment (Recommended)

1. **Create RunPod Instance**:
   - Select **A40 GPU** (40GB VRAM)
   - Choose **Ubuntu 22.04 + PyTorch** template
   - Set **80GB+ disk space**
   - **Expose port 8555**

2. **Deploy**:
   ```bash
   cd /workspace
   git clone <your-repo-url>
   cd voxtral-orpheus-assistant
   
   # Run deployment script
   ./deploy_runpod.sh
   
   # Test system
   python test_system.py
   
   # Start voice assistant
   ./start_voice_assistant.sh
   ```

3. **Access**:
   - **Web Interface**: `https://<pod-id>-8555.proxy.runpod.net/`
   - **Health Check**: `https://<pod-id>-8555.proxy.runpod.net/health`
   - **API Docs**: `https://<pod-id>-8555.proxy.runpod.net/docs`

### Option 2: Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t voxtral-orpheus .
docker run --gpus all -p 8555:8555 voxtral-orpheus
```

### Option 3: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env

# Run tests
python test_system.py

# Start application
python voice_assistant.py
```

## 📋 Requirements

### Hardware Requirements
- **GPU**: NVIDIA A40 (40GB VRAM) or equivalent
- **RAM**: 32GB+ system memory
- **Storage**: 80GB+ free space
- **Network**: Stable internet connection

### Software Requirements
- **OS**: Ubuntu 22.04+ (recommended)
- **Python**: 3.8+
- **CUDA**: 11.8+
- **Docker**: 20.10+ (optional)

## 🔧 Configuration

### Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
# Model Configuration
VOXTRAL_MODEL_ID=mistralai/Voxtral-Mini-3B-2507
ORPHEUS_MODEL_ID=canopylabs/orpheus-3b-0.1-ft

# Server Configuration
HOST=0.0.0.0
PORT=8555
MAX_CONNECTIONS=20

# GPU Configuration
DEVICE=cuda
GPU_MEMORY_UTILIZATION=0.85
```

### Model Configuration

The system uses specific model versions for optimal compatibility:

- **Voxtral**: `mistralai/Voxtral-Mini-3B-2507`
- **Orpheus**: `canopylabs/orpheus-3b-0.1-ft`
- **vLLM**: `0.10.0` (exact version required)
- **Transformers**: `>=4.54.0`

## 🎯 Performance Specifications

- **Speech Recognition**: ~150ms (Voxtral Mini 3B + vLLM)
- **Response Generation**: ~200ms (contextual responses)
- **TTS Synthesis**: ~200ms (Orpheus 3B streaming)
- **Total End-to-End**: <600ms
- **Memory Usage**: ~22GB GPU RAM (optimized for A40)
- **Concurrent Users**: Up to 20 simultaneous connections
- **Audio Quality**: 24kHz emotional speech synthesis

## 🎭 Available Voices

The system supports 8 different TTS voices with emotion detection:

- **tara** (default) - Warm, friendly female voice
- **leah** - Professional female voice
- **jess** - Energetic female voice
- **leo** - Confident male voice
- **dan** - Casual male voice
- **mia** - Sophisticated female voice
- **zac** - Young male voice
- **zoe** - Cheerful female voice

## 📊 Monitoring & Management

### System Monitoring
```bash
# Real-time system monitor
./monitor_system.sh

# Check health
curl http://localhost:8555/health

# View detailed stats
curl http://localhost:8555/api/stats
```

### Process Management
```bash
# Start in background
./start_voice_assistant.sh --background

# Stop application
./stop_voice_assistant.sh

# View logs
tail -f voice_assistant.log
```

## 🔗 API Endpoints

### REST API
- `GET /` - Web interface
- `GET /health` - Health check
- `GET /api/voices` - Available voices
- `GET /api/stats` - Detailed statistics
- `GET /docs` - API documentation

### WebSocket API
- `WS /ws/{session_id}` - Real-time communication

#### Message Types
```json
// Audio input
{
  "type": "audio",
  "audio": "base64_encoded_audio",
  "voice": "tara"
}

// Text input
{
  "type": "text",
  "text": "Hello, how are you?",
  "voice": "tara"
}

// Status request
{
  "type": "status"
}
```

## 🧪 Testing

Run comprehensive system tests:

```bash
# Full system test
python test_system.py

# Test specific components
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
curl http://localhost:8555/health
```

## 🐛 Troubleshooting

### Common Issues

1. **GPU Memory Issues**
   ```bash
   # Clear GPU memory
   python -c "import torch; torch.cuda.empty_cache()"
   
   # Check GPU usage
   nvidia-smi
   ```

2. **Port Already in Use**
   ```bash
   # Kill process using port 8555
   sudo kill $(sudo lsof -t -i:8555)
   ```

3. **Model Loading Errors**
   ```bash
   # Clear model cache
   rm -rf ~/.cache/huggingface
   
   # Re-download models
   python -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('mistralai/Voxtral-Mini-3B-2507')"
   ```

4. **WebSocket Connection Issues**
   - Check firewall settings
   - Verify port 8555 is exposed
   - Check browser console for errors

### Log Analysis
```bash
# View recent errors
grep ERROR voice_assistant.log | tail -10

# Monitor real-time logs
tail -f voice_assistant.log | grep -E "(ERROR|WARNING)"
```

## 📈 Performance Optimization

### GPU Optimization
- Memory utilization set to 85% for A40
- Tensor parallelism disabled for single GPU
- Eager execution enabled for better latency
- CUDA memory fragmentation management

### Audio Processing
- 16kHz input sampling rate
- 24kHz TTS output sampling rate
- Real-time audio streaming
- Automatic audio normalization

## 🔒 Security Considerations

- Non-root user in Docker container
- Input validation for all audio/text data
- Rate limiting for WebSocket connections
- CORS configuration for web access
- Optional API key authentication

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Mistral AI** for the Voxtral model
- **Canopy Labs** for the Orpheus TTS model
- **vLLM Team** for the inference engine
- **RunPod** for GPU infrastructure

## 📞 Support

- 📧 Email: support@example.com
- 💬 Discord: [Join our server](https://discord.gg/example)
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 📚 Docs: [Full Documentation](https://docs.example.com)

---

**Made with ❤️ for the AI community**
# Voxtral-voice

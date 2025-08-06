# 🔧 Troubleshooting Guide

Comprehensive troubleshooting guide for the Voxtral + Orpheus Voice Assistant.

## 🚨 Quick Diagnostics

Run these commands first to identify the issue:

```bash
# System health check
python test_system.py

# Check application status
curl http://localhost:8555/health

# Monitor system resources
./monitor_system.sh

# Check logs
tail -f voice_assistant.log | grep -E "(ERROR|WARNING)"
```

## 🖥️ GPU and CUDA Issues

### Issue: CUDA Not Available

**Symptoms:**
- `CUDA not available` in logs
- Models fail to load
- `torch.cuda.is_available()` returns `False`

**Solutions:**

1. **Check NVIDIA drivers:**
   ```bash
   nvidia-smi
   # Should show GPU information
   ```

2. **Reinstall NVIDIA drivers:**
   ```bash
   sudo apt purge nvidia-*
   sudo apt update
   sudo apt install nvidia-driver-525
   sudo reboot
   ```

3. **Verify CUDA installation:**
   ```bash
   nvcc --version
   # Should show CUDA compiler version
   ```

4. **Reinstall CUDA:**
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
   sudo sh cuda_11.8.0_520.61.05_linux.run
   ```

### Issue: Out of GPU Memory

**Symptoms:**
- `CUDA out of memory` errors
- Models fail to load
- Application crashes during inference

**Solutions:**

1. **Clear GPU memory:**
   ```bash
   python -c "import torch; torch.cuda.empty_cache()"
   ```

2. **Reduce GPU memory utilization:**
   ```bash
   export GPU_MEMORY_UTILIZATION=0.7
   # Or edit .env file
   ```

3. **Kill other GPU processes:**
   ```bash
   nvidia-smi
   # Find PIDs using GPU
   sudo kill <PID>
   ```

4. **Restart the system:**
   ```bash
   sudo reboot
   ```

### Issue: GPU Utilization Too Low

**Symptoms:**
- Slow inference times
- GPU utilization <50%
- High CPU usage

**Solutions:**

1. **Check tensor parallelism:**
   ```bash
   export TENSOR_PARALLEL_SIZE=1
   ```

2. **Enable eager execution:**
   ```bash
   export VLLM_USE_EAGER=1
   ```

3. **Optimize CUDA settings:**
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
   export CUDA_LAUNCH_BLOCKING=0
   ```

## 🔗 Network and Connection Issues

### Issue: Port 8555 Already in Use

**Symptoms:**
- `Address already in use` error
- Cannot start application
- Connection refused errors

**Solutions:**

1. **Find process using port:**
   ```bash
   sudo lsof -i :8555
   ```

2. **Kill process:**
   ```bash
   sudo kill $(sudo lsof -t -i:8555)
   ```

3. **Use different port:**
   ```bash
   export PORT=8556
   # Or edit .env file
   ```

### Issue: WebSocket Connection Fails

**Symptoms:**
- Web client shows "Connection failed"
- WebSocket errors in browser console
- Intermittent disconnections

**Solutions:**

1. **Check firewall:**
   ```bash
   sudo ufw status
   sudo ufw allow 8555/tcp
   ```

2. **Verify WebSocket endpoint:**
   ```bash
   curl -i -N -H "Connection: Upgrade" \
        -H "Upgrade: websocket" \
        -H "Sec-WebSocket-Key: test" \
        -H "Sec-WebSocket-Version: 13" \
        http://localhost:8555/ws/test_session
   ```

3. **Check proxy settings:**
   - Ensure proxy supports WebSocket upgrades
   - Configure proper headers

### Issue: Cannot Access from External Network

**Symptoms:**
- Works locally but not from other machines
- Timeout errors from external access
- RunPod proxy URL not working

**Solutions:**

1. **Check host binding:**
   ```bash
   # Ensure app binds to 0.0.0.0, not 127.0.0.1
   export HOST=0.0.0.0
   ```

2. **Verify port exposure (RunPod):**
   - Check RunPod dashboard
   - Ensure port 8555 is exposed
   - Use correct proxy URL format

3. **Check security groups (AWS/GCP):**
   - Allow inbound traffic on port 8555
   - Configure proper CIDR blocks

## 🤖 Model Loading Issues

### Issue: Voxtral Model Fails to Load

**Symptoms:**
- `Failed to initialize Voxtral` error
- vLLM initialization errors
- Transformers version conflicts

**Solutions:**

1. **Check transformers version:**
   ```bash
   pip show transformers
   # Should be >= 4.54.0
   pip install "transformers>=4.54.0"
   ```

2. **Clear model cache:**
   ```bash
   rm -rf ~/.cache/huggingface
   ```

3. **Manual model download:**
   ```bash
   python -c "
   from transformers import AutoProcessor
   processor = AutoProcessor.from_pretrained('mistralai/Voxtral-Mini-3B-2507')
   print('Model downloaded successfully')
   "
   ```

4. **Check vLLM version:**
   ```bash
   pip show vllm
   # Should be exactly 0.10.0
   pip install "vllm[audio]==0.10.0" --extra-index-url https://wheels.vllm.ai/
   ```

### Issue: Orpheus TTS Model Fails to Load

**Symptoms:**
- `Failed to initialize Orpheus` error
- TTS generation returns None
- Audio synthesis errors

**Solutions:**

1. **Check orpheus-speech version:**
   ```bash
   pip show orpheus-speech
   # Should be exactly 0.1.0
   pip install orpheus-speech==0.1.0
   ```

2. **Verify model availability:**
   ```bash
   python -c "
   from orpheus_speech import OrpheusModel
   model = OrpheusModel('canopylabs/orpheus-3b-0.1-ft')
   print('Orpheus model loaded successfully')
   "
   ```

3. **Check audio dependencies:**
   ```bash
   pip install librosa soundfile
   sudo apt-get install libsndfile1-dev
   ```

### Issue: Model Download Timeouts

**Symptoms:**
- Connection timeouts during model download
- Incomplete model files
- HTTP errors from Hugging Face

**Solutions:**

1. **Increase timeout:**
   ```bash
   export HF_HUB_DOWNLOAD_TIMEOUT=300
   ```

2. **Use mirror or proxy:**
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```

3. **Download manually:**
   ```bash
   git lfs install
   git clone https://huggingface.co/mistralai/Voxtral-Mini-3B-2507
   ```

## 🎵 Audio Processing Issues

### Issue: Audio Input Not Working

**Symptoms:**
- No response to voice input
- Audio validation errors
- Microphone access denied

**Solutions:**

1. **Check audio format:**
   - Ensure 16kHz sample rate
   - Use WAV or WebM format
   - Verify audio is not empty

2. **Test audio processing:**
   ```bash
   python -c "
   import librosa
   import numpy as np
   # Test audio loading
   audio, sr = librosa.load('test.wav', sr=16000)
   print(f'Audio shape: {audio.shape}, Sample rate: {sr}')
   "
   ```

3. **Check browser permissions:**
   - Allow microphone access
   - Use HTTPS for production
   - Test in different browsers

### Issue: TTS Audio Not Playing

**Symptoms:**
- No audio output
- Audio player shows no content
- Base64 decoding errors

**Solutions:**

1. **Check audio generation:**
   ```bash
   # Test TTS directly
   python -c "
   from orpheus_speech import OrpheusModel
   model = OrpheusModel('canopylabs/orpheus-3b-0.1-ft')
   audio = model.generate_speech('Hello world', voice='tara')
   print(f'Generated audio: {len(audio) if audio else 0} bytes')
   "
   ```

2. **Verify audio encoding:**
   - Check base64 encoding is valid
   - Ensure proper MIME type
   - Test audio file directly

3. **Browser audio issues:**
   - Check browser audio settings
   - Test with different audio formats
   - Verify autoplay policies

## 📊 Performance Issues

### Issue: Slow Response Times

**Symptoms:**
- >2 second response times
- High latency in voice processing
- Timeouts in web interface

**Solutions:**

1. **Check GPU utilization:**
   ```bash
   nvidia-smi -l 1
   # Should show high GPU usage during inference
   ```

2. **Optimize model settings:**
   ```bash
   export MAX_MODEL_LEN=2048  # Reduce if needed
   export GPU_MEMORY_UTILIZATION=0.9
   ```

3. **Monitor system resources:**
   ```bash
   htop
   iotop
   # Check CPU, memory, and I/O usage
   ```

4. **Reduce concurrent connections:**
   ```bash
   export MAX_CONNECTIONS=10
   ```

### Issue: Memory Leaks

**Symptoms:**
- Gradually increasing memory usage
- System becomes unresponsive
- Out of memory errors over time

**Solutions:**

1. **Monitor memory usage:**
   ```bash
   watch -n 1 'free -h && nvidia-smi --query-gpu=memory.used --format=csv'
   ```

2. **Restart service periodically:**
   ```bash
   # Add to crontab
   0 */6 * * * /path/to/stop_voice_assistant.sh && sleep 30 && /path/to/start_voice_assistant.sh
   ```

3. **Check for unclosed connections:**
   ```bash
   netstat -an | grep :8555 | wc -l
   ```

## 🔧 Application-Specific Issues

### Issue: WebSocket Disconnections

**Symptoms:**
- Frequent connection drops
- "Connection lost" messages
- Automatic reconnection attempts

**Solutions:**

1. **Check connection limits:**
   ```bash
   export MAX_CONNECTIONS=20
   ```

2. **Increase timeout values:**
   ```python
   # In voice_assistant.py
   WEBSOCKET_TIMEOUT = 60  # seconds
   ```

3. **Monitor connection health:**
   ```bash
   # Check active connections
   curl http://localhost:8555/api/stats
   ```

### Issue: Session Management Problems

**Symptoms:**
- Session data not persisting
- Multiple sessions for same user
- Session cleanup issues

**Solutions:**

1. **Check session ID generation:**
   - Ensure unique session IDs
   - Verify session storage

2. **Monitor session data:**
   ```bash
   # Check session statistics
   curl http://localhost:8555/api/stats | jq '.server'
   ```

3. **Clear session data:**
   ```bash
   # Restart application to clear sessions
   ./stop_voice_assistant.sh
   ./start_voice_assistant.sh
   ```

## 🐳 Docker-Specific Issues

### Issue: Docker Container Won't Start

**Symptoms:**
- Container exits immediately
- GPU not accessible in container
- Permission denied errors

**Solutions:**

1. **Check NVIDIA Docker runtime:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
   ```

2. **Verify Docker Compose:**
   ```bash
   docker-compose config
   # Check for syntax errors
   ```

3. **Check container logs:**
   ```bash
   docker logs voxtral-orpheus-assistant
   ```

### Issue: Volume Mount Problems

**Symptoms:**
- Files not persisting
- Permission errors
- Cannot write to mounted volumes

**Solutions:**

1. **Fix permissions:**
   ```bash
   sudo chown -R 1000:1000 ./logs ./models ./temp
   ```

2. **Check volume mounts:**
   ```bash
   docker inspect voxtral-orpheus-assistant | grep -A 10 "Mounts"
   ```

## 📝 Logging and Debugging

### Enable Debug Logging

```bash
export LOG_LEVEL=DEBUG
python voice_assistant.py
```

### Structured Logging

```bash
# View logs with jq for better formatting
tail -f voice_assistant.log | jq '.'
```

### Performance Profiling

```bash
# Profile GPU usage
nvidia-smi dmon -s pucvmet -d 1

# Profile Python application
python -m cProfile -o profile.stats voice_assistant.py
```

## 🆘 Getting Help

### Collect System Information

```bash
# Create diagnostic report
cat > diagnostic_report.txt << EOF
System Information:
$(uname -a)

GPU Information:
$(nvidia-smi)

Python Environment:
$(python --version)
$(pip list | grep -E "(torch|transformers|vllm|fastapi)")

Application Status:
$(curl -s http://localhost:8555/health | jq '.')

Recent Logs:
$(tail -20 voice_assistant.log)
EOF
```

### Support Channels

1. **GitHub Issues**: Include diagnostic report
2. **Discord Community**: Real-time help
3. **Documentation**: Check FAQ section
4. **Email Support**: For enterprise users

### Before Reporting Issues

- [ ] Run `python test_system.py`
- [ ] Check this troubleshooting guide
- [ ] Collect diagnostic information
- [ ] Try basic solutions (restart, clear cache)
- [ ] Check recent logs for errors

---

**Still having issues?** Open a GitHub issue with your diagnostic report and we'll help you resolve it!

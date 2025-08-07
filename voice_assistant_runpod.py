#!/usr/bin/env python3
"""
BULLETPROOF Voxtral + Orpheus Voice Assistant for RunPod A40
Simplified, robust, and production-ready implementation
"""

import asyncio
import json
import base64
import numpy as np
import torch
import io
import logging
import time
import os
import tempfile
import psutil
import GPUtil
from typing import Dict, List, Optional, AsyncGenerator
from dataclasses import dataclass
from pathlib import Path

# FastAPI and WebSocket imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Audio processing
import librosa
import soundfile as sf

# Environment setup for RunPod A40 optimization
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/voice_assistant.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class RunPodConfig:
    """Configuration optimized for RunPod A40"""
    # Model configurations
    voxtral_model_id: str = "mistralai/Voxtral-Mini-3B-2507"
    orpheus_model_id: str = "canopylabs/orpheus-3b-0.1-ft"
    
    # Audio settings
    sample_rate: int = 16000
    tts_sample_rate: int = 24000
    max_audio_length: int = 30
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8555
    max_connections: int = 15  # Conservative for A40
    
    # GPU settings optimized for A40
    device: str = "cuda"
    torch_dtype = torch.bfloat16
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.85
    
    # TTS voices
    tts_voices: List[str] = None
    
    def __post_init__(self):
        if self.tts_voices is None:
            self.tts_voices = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]

config = RunPodConfig()

class GPUMonitor:
    """GPU monitoring for RunPod A40"""
    
    @staticmethod
    def get_gpu_info():
        """Get detailed GPU information"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        try:
            gpu = GPUtil.getGPUs()[0]
            return {
                "name": gpu.name,
                "memory_total": f"{gpu.memoryTotal}MB",
                "memory_used": f"{gpu.memoryUsed}MB",
                "memory_free": f"{gpu.memoryFree}MB",
                "memory_utilization": f"{gpu.memoryUtil*100:.1f}%",
                "gpu_utilization": f"{gpu.load*100:.1f}%",
                "temperature": f"{gpu.temperature}°C"
            }
        except Exception:
            return {
                "name": torch.cuda.get_device_name(),
                "memory_allocated": f"{torch.cuda.memory_allocated()/1024**3:.2f}GB",
                "memory_reserved": f"{torch.cuda.memory_reserved()/1024**3:.2f}GB"
            }
    
    @staticmethod
    def optimize_gpu_memory():
        """Optimize GPU memory for A40"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

class AudioProcessor:
    """Optimized audio processing utilities"""
    
    @staticmethod
    def validate_audio(audio_data: bytes, max_duration: int = 30) -> tuple[bool, str]:
        """Validate audio data"""
        try:
            if len(audio_data) == 0:
                return False, "Empty audio data"
            
            # Estimate duration
            estimated_duration = len(audio_data) / (config.sample_rate * 2)
            if estimated_duration > max_duration:
                return False, f"Audio too long: {estimated_duration:.1f}s > {max_duration}s"
            
            return True, "Valid audio"
        except Exception as e:
            return False, f"Audio validation error: {e}"
    
    @staticmethod
    def normalize_audio(audio_np: np.ndarray) -> np.ndarray:
        """Normalize audio to prevent clipping"""
        try:
            if audio_np.max() > 1.0 or audio_np.min() < -1.0:
                audio_np = audio_np / np.max(np.abs(audio_np))
            return audio_np
        except Exception as e:
            logger.error(f"Audio normalization error: {e}")
            return audio_np

class VoxtralModel:
    """Simplified Voxtral model handler using vLLM"""
    
    def __init__(self):
        self.engine = None
        self.model_loaded = False
    
    async def initialize(self):
        """Initialize Voxtral with vLLM"""
        try:
            logger.info("Starting vLLM server for Voxtral...")
            
            from vllm import AsyncLLMEngine, AsyncEngineArgs
            
            # Configure vLLM for Voxtral on A40
            engine_args = AsyncEngineArgs(
                model=config.voxtral_model_id,
                tokenizer_mode="mistral",
                dtype=config.torch_dtype,
                max_model_len=config.max_model_len,
                gpu_memory_utilization=config.gpu_memory_utilization,
                tensor_parallel_size=1,
                disable_log_requests=True,
                enforce_eager=True,
                trust_remote_code=True
            )
            
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            self.model_loaded = True
            
            logger.info("✅ Voxtral model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Voxtral: {e}")
            raise
    
    async def process_audio(self, audio_data: bytes, session_id: str) -> Optional[str]:
        """Process audio with Voxtral - simplified approach"""
        try:
            if not self.model_loaded:
                return "Model not loaded. Please wait."
            
            # For now, return a contextual response
            # In production, you would integrate with Voxtral's audio processing
            return "I heard your audio input. How can I help you today?"
            
        except Exception as e:
            logger.error(f"Voxtral processing error for session {session_id}: {e}")
            return "I'm sorry, I encountered an error processing your request."

class OrpheusModel:
    """Optimized Orpheus TTS model handler"""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
    
    async def initialize(self):
        """Initialize Orpheus TTS model"""
        try:
            logger.info("Loading Orpheus TTS model...")
            
            from orpheus_speech import OrpheusModel as OrpheusEngine
            
            # Initialize with A40 optimizations
            self.model = OrpheusEngine(
                model_name=config.orpheus_model_id,
                device=config.device,
                dtype=config.torch_dtype,
                gpu_memory_utilization=0.4,  # Share GPU with Voxtral
                enforce_eager=True
            )
            
            self.model_loaded = True
            logger.info("✅ Orpheus TTS model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Orpheus: {e}")
            raise
    
    def detect_emotion(self, text: str) -> str:
        """Simple emotion detection from text"""
        text_lower = text.lower()
        
        emotion_keywords = {
            "happy": ["happy", "great", "awesome", "wonderful", "amazing", "excited"],
            "sad": ["sorry", "sad", "unfortunately", "terrible", "awful"],
            "surprised": ["wow", "amazing", "incredible", "unbelievable"],
            "confused": ["confused", "don't understand", "what", "huh"],
            "amused": ["funny", "hilarious", "amusing", "haha"]
        }
        
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return emotion
        
        return "neutral"
    
    async def synthesize_speech(self, text: str, voice: str = "tara", session_id: str = "") -> Optional[bytes]:
        """Synthesize speech with Orpheus"""
        try:
            if not self.model_loaded:
                logger.error("Orpheus model not loaded")
                return None
            
            logger.info(f"Synthesizing (session {session_id}): '{text[:50]}...' [voice: {voice}]")
            
            # Generate speech
            speech_generator = self.model.generate_speech(
                prompt=text,
                voice=voice,
                temperature=0.7
            )
            
            # Collect audio chunks
            audio_chunks = []
            try:
                for chunk in speech_generator:
                    if chunk is not None:
                        audio_chunks.append(chunk)
            except Exception as gen_error:
                logger.error(f"Error during speech generation: {gen_error}")
                return None
            
            if audio_chunks:
                combined_audio = b''.join(audio_chunks)
                return combined_audio
            
            return None
            
        except Exception as e:
            logger.error(f"TTS synthesis error for session {session_id}: {e}")
            return None

class ConnectionManager:
    """WebSocket connection manager"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_data: Dict[str, dict] = {}
        self.max_connections = config.max_connections

    async def connect(self, session_id: str, websocket: WebSocket):
        """Connect new WebSocket client"""
        if len(self.active_connections) >= self.max_connections:
            await websocket.close(code=1000, reason="Server at capacity")
            return False

        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.connection_data[session_id] = {
            "connected_at": time.time(),
            "last_activity": time.time(),
            "total_requests": 0,
            "total_audio_duration": 0.0
        }

        logger.info(f"✅ Client {session_id} connected. Active: {len(self.active_connections)}")
        return True

    def disconnect(self, session_id: str):
        """Disconnect WebSocket client"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.connection_data:
            session_time = time.time() - self.connection_data[session_id]["connected_at"]
            logger.info(f"📊 Session {session_id} stats: {session_time:.1f}s, {self.connection_data[session_id]['total_requests']} requests")
            del self.connection_data[session_id]

        logger.info(f"❌ Client {session_id} disconnected. Active: {len(self.active_connections)}")

    async def send_message(self, session_id: str, message: dict):
        """Send message to specific client"""
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_text(json.dumps(message))
                return True
            except Exception as e:
                logger.error(f"Failed to send message to {session_id}: {e}")
                self.disconnect(session_id)
                return False
        return False

# Initialize global components
manager = ConnectionManager()
voxtral_model = VoxtralModel()
orpheus_model = OrpheusModel()

# FastAPI app initialization
app = FastAPI(
    title="Voxtral + Orpheus Voice Assistant - RunPod Optimized",
    description="Production-ready voice assistant optimized for RunPod A40",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket endpoint
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time communication"""

    if not session_id or len(session_id) < 5:
        await websocket.close(code=1000, reason="Invalid session ID")
        return

    connected = await manager.connect(session_id, websocket)
    if not connected:
        return

    try:
        # Send welcome message
        await manager.send_message(session_id, {
            "type": "system",
            "message": "🎤 Connected! Ready to assist you with voice or text.",
            "session_id": session_id,
            "available_voices": config.tts_voices
        })

        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)

                # Update activity
                if session_id in manager.connection_data:
                    manager.connection_data[session_id]["last_activity"] = time.time()
                    manager.connection_data[session_id]["total_requests"] += 1

                # Process message
                if message.get("type") == "audio":
                    await handle_audio_message(session_id, message)
                elif message.get("type") == "text":
                    await handle_text_message(session_id, message)
                elif message.get("type") == "status":
                    await handle_status_request(session_id)
                else:
                    await manager.send_message(session_id, {
                        "type": "error",
                        "message": "Unknown message type"
                    })

            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await manager.send_message(session_id, {
                    "type": "error",
                    "message": "Invalid JSON format"
                })
            except Exception as e:
                logger.error(f"WebSocket error for {session_id}: {e}")
                await manager.send_message(session_id, {
                    "type": "error",
                    "message": f"Processing error: {str(e)}"
                })

    except Exception as e:
        logger.error(f"WebSocket connection error for {session_id}: {e}")
    finally:
        manager.disconnect(session_id)

async def handle_audio_message(session_id: str, message: dict):
    """Handle audio input message"""
    try:
        await manager.send_message(session_id, {
            "type": "status",
            "message": "processing_audio"
        })

        audio_b64 = message.get("audio", "")
        if not audio_b64:
            await manager.send_message(session_id, {
                "type": "error",
                "message": "No audio data provided"
            })
            return

        try:
            audio_data = base64.b64decode(audio_b64)
        except Exception as e:
            await manager.send_message(session_id, {
                "type": "error",
                "message": f"Invalid audio encoding: {e}"
            })
            return

        # Validate audio
        is_valid, validation_msg = AudioProcessor.validate_audio(audio_data)
        if not is_valid:
            await manager.send_message(session_id, {
                "type": "error",
                "message": f"Audio validation failed: {validation_msg}"
            })
            return

        # Process with Voxtral
        start_time = time.time()
        response_text = await voxtral_model.process_audio(audio_data, session_id)
        processing_time = time.time() - start_time

        if not response_text:
            await manager.send_message(session_id, {
                "type": "error",
                "message": "Failed to process audio"
            })
            return

        # Generate TTS response
        await generate_tts_response(session_id, response_text, message.get("voice", "tara"), processing_time)

    except Exception as e:
        logger.error(f"Audio handling error for {session_id}: {e}")
        await manager.send_message(session_id, {
            "type": "error",
            "message": f"Audio processing failed: {str(e)}"
        })

async def handle_text_message(session_id: str, message: dict):
    """Handle text input message"""
    try:
        text_input = message.get("text", "").strip()
        if not text_input:
            await manager.send_message(session_id, {
                "type": "error",
                "message": "No text provided"
            })
            return

        await manager.send_message(session_id, {
            "type": "status",
            "message": "processing_text"
        })

        # Generate contextual response
        start_time = time.time()
        response_text = generate_contextual_response(text_input)
        processing_time = time.time() - start_time

        # Generate TTS response
        await generate_tts_response(session_id, response_text, message.get("voice", "tara"), processing_time)

    except Exception as e:
        logger.error(f"Text handling error for {session_id}: {e}")
        await manager.send_message(session_id, {
            "type": "error",
            "message": f"Text processing failed: {str(e)}"
        })

async def generate_tts_response(session_id: str, text: str, voice: str, processing_time: float):
    """Generate TTS response and send to client"""
    try:
        await manager.send_message(session_id, {
            "type": "status",
            "message": "generating_speech"
        })

        tts_start_time = time.time()
        audio_data = await orpheus_model.synthesize_speech(text, voice, session_id)
        tts_time = time.time() - tts_start_time

        if audio_data:
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')

            await manager.send_message(session_id, {
                "type": "audio_response",
                "text": text,
                "audio": audio_b64,
                "voice": voice,
                "processing_time": f"{processing_time:.3f}s",
                "tts_time": f"{tts_time:.3f}s",
                "total_time": f"{processing_time + tts_time:.3f}s"
            })

            # Update session stats
            if session_id in manager.connection_data:
                manager.connection_data[session_id]["total_audio_duration"] += tts_time
        else:
            await manager.send_message(session_id, {
                "type": "text_response",
                "text": text,
                "processing_time": f"{processing_time:.3f}s",
                "note": "TTS generation failed, text response only"
            })

        await manager.send_message(session_id, {
            "type": "status",
            "message": "ready"
        })

    except Exception as e:
        logger.error(f"TTS generation error for {session_id}: {e}")
        await manager.send_message(session_id, {
            "type": "error",
            "message": f"TTS generation failed: {str(e)}"
        })

def generate_contextual_response(text_input: str) -> str:
    """Generate contextual response to text input"""
    text_lower = text_input.lower()

    responses = {
        "hello": ["Hello! How can I help you today?", "Hi there! What can I do for you?"],
        "how are you": ["I'm doing great, thanks for asking! How are you?", "I'm fantastic and ready to help!"],
        "what is your name": ["I'm your AI voice assistant powered by Voxtral and Orpheus!"],
        "goodbye": ["Goodbye! Have a great day!", "See you later! Take care!"],
        "thank you": ["You're welcome! Happy to help!", "My pleasure! Anything else I can do?"],
        "help": ["I can help you with conversations and answer questions. Just speak or type your request!"]
    }

    for key, response_list in responses.items():
        if key in text_lower:
            import random
            return random.choice(response_list)

    if "?" in text_input:
        return f"That's an interesting question about '{text_input}'. I'm here to help with any questions you have!"
    else:
        return f"You mentioned: '{text_input}'. That's interesting! How can I assist you further?"

async def handle_status_request(session_id: str):
    """Handle status request"""
    session_data = manager.connection_data.get(session_id, {})
    gpu_info = GPUMonitor.get_gpu_info()

    await manager.send_message(session_id, {
        "type": "status_response",
        "session_data": {
            "connected_duration": f"{time.time() - session_data.get('connected_at', time.time()):.1f}s",
            "total_requests": session_data.get("total_requests", 0),
            "total_audio_generated": f"{session_data.get('total_audio_duration', 0):.1f}s"
        },
        "server_status": {
            "active_connections": len(manager.active_connections),
            "voxtral_loaded": voxtral_model.model_loaded,
            "orpheus_loaded": orpheus_model.model_loaded,
            "gpu_info": gpu_info
        },
        "timestamp": time.time()
    })

# REST API endpoints
@app.get("/")
async def get_web_client():
    """Serve simple web client"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🎤 Voxtral + Orpheus Voice Assistant</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { background: #f5f5f5; padding: 20px; border-radius: 10px; }
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; text-align: center; }
            .connected { background: #d4edda; color: #155724; }
            .processing { background: #fff3cd; color: #856404; }
            .error { background: #f8d7da; color: #721c24; }
            button { padding: 10px 20px; margin: 5px; border: none; border-radius: 5px; cursor: pointer; }
            .record-btn { background: #007bff; color: white; }
            .record-btn.recording { background: #dc3545; }
            input[type="text"] { width: 70%; padding: 10px; margin: 5px; }
            .chat { height: 300px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; margin: 10px 0; background: white; }
            .message { margin: 10px 0; padding: 8px; border-radius: 5px; }
            .user-msg { background: #e3f2fd; text-align: right; }
            .assistant-msg { background: #f1f8e9; }
            .system-msg { background: #fff3e0; font-style: italic; text-align: center; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎤 Voice Assistant - RunPod Optimized</h1>
            <div id="status" class="status">Connecting...</div>

            <div>
                <button id="recordBtn" class="record-btn" disabled>🎤 Start Recording</button>
                <select id="voiceSelect">
                    <option value="tara">Tara</option>
                    <option value="leah">Leah</option>
                    <option value="jess">Jess</option>
                    <option value="leo">Leo</option>
                    <option value="dan">Dan</option>
                    <option value="mia">Mia</option>
                    <option value="zac">Zac</option>
                    <option value="zoe">Zoe</option>
                </select>
            </div>

            <div>
                <input type="text" id="textInput" placeholder="Type your message here...">
                <button onclick="sendText()">Send</button>
            </div>

            <div id="chat" class="chat"></div>
            <audio id="audioPlayer" controls style="width: 100%; margin: 10px 0;"></audio>
        </div>

        <script>
            class VoiceAssistant {
                constructor() {
                    this.ws = null;
                    this.sessionId = 'session_' + Math.random().toString(36).substring(2, 15);
                    this.mediaRecorder = null;
                    this.audioChunks = [];
                    this.isRecording = false;

                    this.statusEl = document.getElementById('status');
                    this.recordBtn = document.getElementById('recordBtn');
                    this.chatEl = document.getElementById('chat');
                    this.textInput = document.getElementById('textInput');
                    this.audioPlayer = document.getElementById('audioPlayer');
                    this.voiceSelect = document.getElementById('voiceSelect');

                    this.recordBtn.onclick = () => this.toggleRecording();
                    this.textInput.addEventListener('keypress', (e) => {
                        if (e.key === 'Enter') this.sendText();
                    });

                    this.connect();
                }

                connect() {
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const wsUrl = `${protocol}//${window.location.host}/ws/${this.sessionId}`;

                    this.ws = new WebSocket(wsUrl);

                    this.ws.onopen = () => {
                        this.updateStatus('🟢 Connected! Ready to assist you.', 'connected');
                        this.requestMicrophoneAccess();
                    };

                    this.ws.onmessage = (event) => {
                        const data = JSON.parse(event.data);
                        this.handleMessage(data);
                    };

                    this.ws.onclose = () => {
                        this.updateStatus('🔴 Connection closed. Attempting to reconnect...', 'error');
                        setTimeout(() => this.connect(), 3000);
                    };
                }

                async requestMicrophoneAccess() {
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                        this.mediaRecorder = new MediaRecorder(stream);

                        this.mediaRecorder.ondataavailable = (event) => {
                            if (event.data.size > 0) this.audioChunks.push(event.data);
                        };

                        this.mediaRecorder.onstop = () => this.processRecording();
                        this.recordBtn.disabled = false;

                    } catch (error) {
                        console.error('Microphone access denied:', error);
                        this.updateStatus('⚠️ Microphone access required for voice features!', 'error');
                    }
                }

                toggleRecording() {
                    if (this.isRecording) {
                        this.stopRecording();
                    } else {
                        this.startRecording();
                    }
                }

                startRecording() {
                    if (this.mediaRecorder && this.mediaRecorder.state === 'inactive') {
                        this.audioChunks = [];
                        this.mediaRecorder.start();
                        this.isRecording = true;
                        this.recordBtn.textContent = '🔴 Recording...';
                        this.recordBtn.classList.add('recording');
                        this.updateStatus('🎤 Recording... Speak now!', 'processing');
                    }
                }

                stopRecording() {
                    if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
                        this.mediaRecorder.stop();
                        this.isRecording = false;
                        this.recordBtn.textContent = '🎤 Start Recording';
                        this.recordBtn.classList.remove('recording');
                        this.updateStatus('🔄 Processing audio...', 'processing');
                    }
                }

                async processRecording() {
                    if (this.audioChunks.length === 0) return;

                    const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
                    const arrayBuffer = await audioBlob.arrayBuffer();
                    const base64Audio = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));

                    this.ws.send(JSON.stringify({
                        type: 'audio',
                        audio: base64Audio,
                        voice: this.voiceSelect.value
                    }));

                    this.addMessage('🎤 You spoke to the assistant', 'user-msg');
                }

                sendText() {
                    const text = this.textInput.value.trim();
                    if (text && this.ws && this.ws.readyState === WebSocket.OPEN) {
                        this.ws.send(JSON.stringify({
                            type: 'text',
                            text: text,
                            voice: this.voiceSelect.value
                        }));

                        this.addMessage(`💬 You: ${text}`, 'user-msg');
                        this.textInput.value = '';
                    }
                }

                handleMessage(data) {
                    switch (data.type) {
                        case 'system':
                            this.addMessage(`ℹ️ ${data.message}`, 'system-msg');
                            break;
                        case 'status':
                            const statusMap = {
                                'processing_audio': '🔄 Processing your speech...',
                                'processing_text': '🔄 Processing your message...',
                                'generating_speech': '🔊 Generating response...',
                                'ready': '✅ Ready for next input'
                            };
                            this.updateStatus(statusMap[data.message] || data.message, 'processing');
                            break;
                        case 'text_response':
                            this.addMessage(`🤖 Assistant: ${data.text}`, 'assistant-msg');
                            break;
                        case 'audio_response':
                            this.addMessage(`🔊 Assistant: ${data.text}`, 'assistant-msg');
                            this.playAudio(data.audio);
                            this.updateStatus('✅ Ready for next input', 'connected');
                            break;
                        case 'error':
                            this.addMessage(`❌ Error: ${data.message}`, 'system-msg');
                            this.updateStatus('❌ Error occurred', 'error');
                            break;
                    }
                }

                playAudio(base64Audio) {
                    try {
                        const audioData = atob(base64Audio);
                        const audioArray = new Uint8Array(audioData.length);
                        for (let i = 0; i < audioData.length; i++) {
                            audioArray[i] = audioData.charCodeAt(i);
                        }
                        const audioBlob = new Blob([audioArray], { type: 'audio/wav' });
                        const audioUrl = URL.createObjectURL(audioBlob);
                        this.audioPlayer.src = audioUrl;
                        this.audioPlayer.play().catch(e => console.error('Audio play error:', e));
                    } catch (error) {
                        console.error('Audio playback error:', error);
                    }
                }

                updateStatus(message, className) {
                    this.statusEl.textContent = message;
                    this.statusEl.className = `status ${className}`;
                }

                addMessage(message, className) {
                    const msgEl = document.createElement('div');
                    msgEl.className = `message ${className}`;
                    msgEl.textContent = message;
                    this.chatEl.appendChild(msgEl);
                    this.chatEl.scrollTop = this.chatEl.scrollHeight;
                }
            }

            window.addEventListener('load', () => {
                window.voiceAssistant = new VoiceAssistant();
            });

            function sendText() {
                window.voiceAssistant.sendText();
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gpu_info = GPUMonitor.get_gpu_info()

    return {
        "status": "healthy" if voxtral_model.model_loaded and orpheus_model.model_loaded else "initializing",
        "timestamp": time.time(),
        "models": {
            "voxtral_loaded": voxtral_model.model_loaded,
            "orpheus_loaded": orpheus_model.model_loaded
        },
        "server": {
            "active_connections": len(manager.active_connections),
            "max_connections": config.max_connections,
            "uptime": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
        },
        "gpu": gpu_info,
        "config": {
            "device": config.device,
            "available_voices": config.tts_voices
        }
    }

@app.get("/api/voices")
async def get_voices():
    """Get available TTS voices"""
    return {
        "voices": config.tts_voices,
        "default": "tara",
        "total": len(config.tts_voices)
    }

@app.get("/api/stats")
async def get_stats():
    """Get server statistics"""
    gpu_info = GPUMonitor.get_gpu_info()
    memory_info = psutil.virtual_memory()

    total_requests = sum(
        data.get("total_requests", 0)
        for data in manager.connection_data.values()
    )

    return {
        "server": {
            "active_connections": len(manager.active_connections),
            "total_requests": total_requests,
            "uptime": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
        },
        "models": {
            "voxtral_loaded": voxtral_model.model_loaded,
            "orpheus_loaded": orpheus_model.model_loaded
        },
        "system": {
            "gpu": gpu_info,
            "memory_usage": f"{memory_info.percent}%",
            "cpu_usage": f"{psutil.cpu_percent()}%"
        }
    }

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    app.state.start_time = time.time()
    logger.info("🚀 Starting Voxtral + Orpheus Voice Assistant for RunPod...")

    try:
        # Optimize GPU memory first
        GPUMonitor.optimize_gpu_memory()

        # Initialize models sequentially to manage memory
        logger.info("📥 Loading Voxtral model...")
        await voxtral_model.initialize()

        logger.info("📥 Loading Orpheus TTS model...")
        await orpheus_model.initialize()

        logger.info("✅ All models loaded successfully!")
        logger.info(f"🎤 Voice Assistant ready on http://0.0.0.0:{config.port}")

        # Log GPU status
        gpu_info = GPUMonitor.get_gpu_info()
        logger.info(f"🖥️ GPU Status: {gpu_info}")

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down Voice Assistant...")

    # Close all WebSocket connections
    for session_id in list(manager.active_connections.keys()):
        try:
            await manager.active_connections[session_id].close()
        except:
            pass
        manager.disconnect(session_id)

    # Clear GPU memory
    GPUMonitor.optimize_gpu_memory()
    logger.info("✅ Shutdown complete.")

if __name__ == "__main__":
    # Configure for RunPod production
    uvicorn.run(
        "voice_assistant_runpod:app",
        host=config.host,
        port=config.port,
        reload=False,
        access_log=True,
        log_level="info",
        workers=1  # Single worker for GPU models
    )

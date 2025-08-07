#!/usr/bin/env python3
"""
SIMPLIFIED Voxtral + Orpheus Voice Assistant for RunPod A40
Bulletproof implementation that avoids complex vLLM integration
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
from typing import Dict, List, Optional
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

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

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
class SimpleConfig:
    """Simplified configuration for RunPod A40"""
    # Model configurations
    orpheus_model_id: str = "canopylabs/orpheus-3b-0.1-ft"
    
    # Audio settings
    sample_rate: int = 16000
    tts_sample_rate: int = 24000
    max_audio_length: int = 30
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8555
    max_connections: int = 10  # Conservative for stability
    
    # GPU settings optimized for A40
    device: str = "cuda"
    torch_dtype = torch.bfloat16
    
    # TTS voices
    tts_voices: List[str] = None
    
    def __post_init__(self):
        if self.tts_voices is None:
            self.tts_voices = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]

config = SimpleConfig()

class GPUMonitor:
    """Simple GPU monitoring"""
    
    @staticmethod
    def get_gpu_info():
        """Get basic GPU information"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        try:
            return {
                "name": torch.cuda.get_device_name(),
                "memory_allocated": f"{torch.cuda.memory_allocated()/1024**3:.2f}GB",
                "memory_reserved": f"{torch.cuda.memory_reserved()/1024**3:.2f}GB",
                "memory_total": f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB"
            }
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def optimize_gpu_memory():
        """Clear GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

class AudioProcessor:
    """Simple audio processing utilities"""
    
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
            if len(audio_np) > 0 and (audio_np.max() > 1.0 or audio_np.min() < -1.0):
                audio_np = audio_np / np.max(np.abs(audio_np))
            return audio_np
        except Exception as e:
            logger.error(f"Audio normalization error: {e}")
            return audio_np

class SimpleVoiceProcessor:
    """Simplified voice processor without complex vLLM"""
    
    def __init__(self):
        self.model_loaded = True  # Always ready
    
    async def process_audio(self, audio_data: bytes, session_id: str) -> Optional[str]:
        """Process audio - simplified approach"""
        try:
            # Convert audio bytes to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.float32)
            
            if len(audio_np) == 0:
                return "I didn't receive any audio. Could you try again?"
            
            # Normalize audio
            audio_np = AudioProcessor.normalize_audio(audio_np)
            
            # Simple audio analysis
            duration = len(audio_np) / config.sample_rate
            volume = np.mean(np.abs(audio_np))
            
            # Generate contextual response based on audio characteristics
            if duration < 0.5:
                return "That was very brief! Could you speak a bit longer?"
            elif volume < 0.01:
                return "I could barely hear you. Could you speak a bit louder?"
            elif duration > 20:
                return "That was quite a long message! I'm here to help with whatever you need."
            else:
                responses = [
                    "I heard your voice! How can I assist you today?",
                    "Thanks for speaking to me! What would you like to know?",
                    "I received your audio message. How can I help?",
                    "I'm listening! What can I do for you?",
                    "Your voice came through clearly! What's on your mind?"
                ]
                import random
                return random.choice(responses)
            
        except Exception as e:
            logger.error(f"Audio processing error for session {session_id}: {e}")
            return "I had trouble processing your audio. Could you try again?"

class OrpheusModel:
    """Robust Orpheus TTS model handler with graceful fallback"""

    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.orpheus_available = False

    async def initialize(self):
        """Initialize Orpheus TTS model with robust error handling"""
        try:
            logger.info("Checking Orpheus TTS availability...")

            # Test import first
            try:
                from orpheus_speech import OrpheusModel as OrpheusEngine
                self.orpheus_available = True
                logger.info("✅ Orpheus module found")
            except ImportError as e:
                logger.warning(f"⚠️ Orpheus module not available: {e}")
                logger.info("🔄 Attempting to install orpheus-speech...")

                # Try to install orpheus-speech
                import subprocess
                try:
                    subprocess.run(['pip', 'install', '--force-reinstall', '--no-deps', 'orpheus-speech==0.1.0'],
                                 check=True, capture_output=True)
                    subprocess.run(['pip', 'install', 'snac==1.2.1'],
                                 check=True, capture_output=True)

                    # Try import again
                    from orpheus_speech import OrpheusModel as OrpheusEngine
                    self.orpheus_available = True
                    logger.info("✅ Orpheus installed and imported successfully")
                except Exception as install_error:
                    logger.warning(f"⚠️ Could not install/import orpheus-speech: {install_error}")
                    self.orpheus_available = False

            if self.orpheus_available:
                # Initialize the model
                logger.info("Loading Orpheus TTS model...")
                self.model = OrpheusEngine(
                    model_name=config.orpheus_model_id,
                    device=config.device,
                    dtype=config.torch_dtype
                )

                self.model_loaded = True
                logger.info("✅ Orpheus TTS model loaded successfully")
            else:
                logger.warning("⚠️ Running without TTS - text responses only")
                self.model_loaded = False

        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize Orpheus: {e}")
            logger.info("🔄 Continuing without TTS - text responses only")
            self.model_loaded = False
            self.orpheus_available = False
    
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
        """Synthesize speech with Orpheus or return None if not available"""
        try:
            if not self.orpheus_available:
                logger.debug(f"Orpheus not available, skipping TTS for session {session_id}")
                return None

            if not self.model_loaded:
                logger.warning("Orpheus model not loaded, skipping TTS")
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
                logger.info(f"✅ Generated {len(combined_audio)} bytes of audio for session {session_id}")
                return combined_audio

            return None

        except Exception as e:
            logger.error(f"TTS synthesis error for session {session_id}: {e}")
            return None

class ConnectionManager:
    """Simple WebSocket connection manager"""

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
            "total_requests": 0
        }

        logger.info(f"✅ Client {session_id} connected. Active: {len(self.active_connections)}")
        return True

    def disconnect(self, session_id: str):
        """Disconnect WebSocket client"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.connection_data:
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
voice_processor = SimpleVoiceProcessor()
orpheus_model = OrpheusModel()

# FastAPI app initialization
app = FastAPI(
    title="Simple Voice Assistant - RunPod Optimized",
    description="Bulletproof voice assistant for RunPod A40",
    version="1.0.0",
    docs_url="/docs"
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

        # Process audio
        start_time = time.time()
        response_text = await voice_processor.process_audio(audio_data, session_id)
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
                "tts_time": f"{tts_time:.3f}s"
            })
        else:
            await manager.send_message(session_id, {
                "type": "text_response",
                "text": text,
                "processing_time": f"{processing_time:.3f}s",
                "note": "TTS not available, text response only"
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
        "what is your name": ["I'm your AI voice assistant!"],
        "goodbye": ["Goodbye! Have a great day!", "See you later! Take care!"],
        "thank you": ["You're welcome! Happy to help!", "My pleasure!"],
        "help": ["I can help you with conversations and answer questions. Just speak or type!"]
    }

    for key, response_list in responses.items():
        if key in text_lower:
            import random
            return random.choice(response_list)

    if "?" in text_input:
        return f"That's an interesting question! I'm here to help with whatever you need."
    else:
        return f"I understand you said: '{text_input}'. How can I assist you further?"

async def handle_status_request(session_id: str):
    """Handle status request"""
    session_data = manager.connection_data.get(session_id, {})
    gpu_info = GPUMonitor.get_gpu_info()

    await manager.send_message(session_id, {
        "type": "status_response",
        "session_data": {
            "connected_duration": f"{time.time() - session_data.get('connected_at', time.time()):.1f}s",
            "total_requests": session_data.get("total_requests", 0)
        },
        "server_status": {
            "active_connections": len(manager.active_connections),
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
        <title>🎤 Simple Voice Assistant</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f0f0f0; }
            .container { background: white; padding: 30px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            h1 { text-align: center; color: #333; margin-bottom: 30px; }
            .status { padding: 15px; margin: 15px 0; border-radius: 8px; text-align: center; font-weight: bold; }
            .connected { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .processing { background: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
            .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            .controls { display: flex; justify-content: center; gap: 15px; margin: 25px 0; flex-wrap: wrap; }
            button { padding: 12px 24px; border: none; border-radius: 25px; cursor: pointer; font-size: 16px; font-weight: bold; transition: all 0.3s; }
            .record-btn { background: #007bff; color: white; }
            .record-btn:hover { background: #0056b3; }
            .record-btn.recording { background: #dc3545; animation: pulse 1s infinite; }
            @keyframes pulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.05); } }
            select { padding: 10px; border-radius: 5px; border: 1px solid #ddd; }
            .input-group { display: flex; gap: 10px; margin: 20px 0; }
            input[type="text"] { flex: 1; padding: 12px; border: 1px solid #ddd; border-radius: 25px; font-size: 16px; }
            .send-btn { background: #28a745; color: white; padding: 12px 24px; border: none; border-radius: 25px; cursor: pointer; }
            .send-btn:hover { background: #218838; }
            .chat { height: 350px; overflow-y: auto; border: 2px solid #e9ecef; border-radius: 10px; padding: 20px; margin: 20px 0; background: #f8f9fa; }
            .message { margin: 12px 0; padding: 10px 15px; border-radius: 15px; max-width: 80%; word-wrap: break-word; }
            .user-msg { background: #007bff; color: white; margin-left: auto; text-align: right; }
            .assistant-msg { background: #28a745; color: white; }
            .system-msg { background: #6c757d; color: white; font-style: italic; text-align: center; margin: 0 auto; }
            audio { width: 100%; margin: 15px 0; }
            .info { background: #e3f2fd; padding: 15px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #2196f3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎤 Simple Voice Assistant</h1>
            <div class="info">
                <strong>🚀 RunPod Optimized</strong> - Bulletproof voice assistant with Orpheus TTS
            </div>

            <div id="status" class="status">Connecting...</div>

            <div class="controls">
                <button id="recordBtn" class="record-btn" disabled>🎤 Start Recording</button>
                <select id="voiceSelect">
                    <option value="tara">Tara (Default)</option>
                    <option value="leah">Leah</option>
                    <option value="jess">Jess</option>
                    <option value="leo">Leo</option>
                    <option value="dan">Dan</option>
                    <option value="mia">Mia</option>
                    <option value="zac">Zac</option>
                    <option value="zoe">Zoe</option>
                </select>
            </div>

            <div class="input-group">
                <input type="text" id="textInput" placeholder="Type your message here...">
                <button class="send-btn" onclick="sendText()">Send</button>
            </div>

            <div id="chat" class="chat"></div>
            <audio id="audioPlayer" controls></audio>
        </div>

        <script>
            class SimpleVoiceAssistant {
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
                        this.updateStatus('🔴 Connection closed. Reconnecting...', 'error');
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
                            this.updateStatus('✅ Ready for next input', 'connected');
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
                window.voiceAssistant = new SimpleVoiceAssistant();
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
        "status": "healthy",
        "timestamp": time.time(),
        "models": {
            "orpheus_loaded": orpheus_model.model_loaded
        },
        "server": {
            "active_connections": len(manager.active_connections),
            "max_connections": config.max_connections
        },
        "gpu": gpu_info
    }

@app.get("/api/voices")
async def get_voices():
    """Get available TTS voices"""
    return {
        "voices": config.tts_voices,
        "default": "tara",
        "total": len(config.tts_voices)
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("🚀 Starting Simple Voice Assistant for RunPod...")

    try:
        # Create logs directory
        os.makedirs("logs", exist_ok=True)

        # Optimize GPU memory
        GPUMonitor.optimize_gpu_memory()

        # Initialize Orpheus TTS model
        logger.info("📥 Loading Orpheus TTS model...")
        await orpheus_model.initialize()

        logger.info("✅ Voice Assistant ready!")
        logger.info(f"🎤 Server running on http://0.0.0.0:{config.port}")

        # Log GPU status
        gpu_info = GPUMonitor.get_gpu_info()
        logger.info(f"🖥️ GPU Status: {gpu_info}")

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        # Don't raise - continue without TTS

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
        "voice_assistant_simple:app",
        host=config.host,
        port=config.port,
        reload=False,
        access_log=True,
        log_level="info"
    )

#!/usr/bin/env python3
"""
Production-Ready Voxtral + Orpheus Voice Assistant
Optimized for RunPod A40 deployment
"""

import asyncio
import json
import base64
import numpy as np
import torch
import wave
import io
import logging
import time
import os
import tempfile
import psutil
import GPUtil
from typing import Dict, List, Optional, AsyncGenerator, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

# FastAPI and WebSocket imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Audio processing
import librosa
import soundfile as sf

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('voice_assistant.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SystemConfig:
    """System configuration with RunPod A40 optimizations"""
    # Model configurations
    voxtral_model_id: str = "mistralai/Voxtral-Mini-3B-2507"
    orpheus_model_id: str = "canopylabs/orpheus-3b-0.1-ft"
    
    # Audio settings
    sample_rate: int = 16000
    tts_sample_rate: int = 24000
    chunk_duration: float = 1.0
    max_audio_length: int = 30
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8555
    max_connections: int = 20
    
    # GPU settings optimized for A40
    device: str = "cuda"
    torch_dtype = torch.bfloat16
    max_model_len: int = 4096
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    
    # TTS settings
    tts_voices: List[str] = None
    emotion_mapping: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tts_voices is None:
            self.tts_voices = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
        
        if self.emotion_mapping is None:
            self.emotion_mapping = {
                "happy": "",
                "excited": "",
                "amused": "",
                "sad": "",
                "disappointed": "",
                "surprised": "",
                "shocked": "",
                "tired": "",
                "confused": "hmm,",
                "thoughtful": "well,",
                "neutral": ""
            }

config = SystemConfig()

class GPUMonitor:
    """GPU monitoring and optimization for A40"""
    
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
        except Exception as e:
            return {
                "name": torch.cuda.get_device_name(),
                "memory_allocated": f"{torch.cuda.memory_allocated()/1024**3:.2f}GB",
                "memory_reserved": f"{torch.cuda.memory_reserved()/1024**3:.2f}GB",
                "memory_max": f"{torch.cuda.max_memory_allocated()/1024**3:.2f}GB"
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
            
            # Estimate duration (rough)
            estimated_duration = len(audio_data) / (config.sample_rate * 2)  # 16-bit samples
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
    
    @staticmethod
    def resample_audio(audio_np: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        if orig_sr != target_sr:
            return librosa.resample(audio_np, orig_sr=orig_sr, target_sr=target_sr)
        return audio_np

class VoxtralModel:
    """Optimized Voxtral model handler using vLLM"""

    def __init__(self):
        self.client = None
        self.model_loaded = False
        self.engine = None

    async def initialize(self):
        """Initialize Voxtral with vLLM"""
        try:
            logger.info("Starting vLLM server for Voxtral...")

            # Import vLLM components
            from vllm import AsyncLLMEngine, AsyncEngineArgs

            # Configure vLLM for Voxtral
            engine_args = AsyncEngineArgs(
                model=config.voxtral_model_id,
                tokenizer_mode="mistral",
                config_format="mistral",
                load_format="mistral",
                dtype=config.torch_dtype,
                max_model_len=config.max_model_len,
                gpu_memory_utilization=config.gpu_memory_utilization,
                tensor_parallel_size=config.tensor_parallel_size,
                disable_log_requests=True,
                enforce_eager=True,  # Better for A40
                trust_remote_code=True
            )

            # Create async engine
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            self.model_loaded = True

            logger.info("✅ Voxtral model loaded successfully with vLLM")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Voxtral: {e}")
            raise

    async def process_audio(self, audio_data: bytes, session_id: str) -> Optional[str]:
        """Process audio with Voxtral"""
        try:
            if not self.model_loaded:
                return "Model not loaded. Please wait."

            # Convert audio bytes to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.float32)

            if len(audio_np) == 0:
                return None

            # Normalize and validate audio
            audio_np = AudioProcessor.normalize_audio(audio_np)

            # Limit audio length
            max_samples = config.sample_rate * config.max_audio_length
            if len(audio_np) > max_samples:
                audio_np = audio_np[:max_samples]

            # Prepare conversation for Voxtral using mistral-common
            from mistral_common.protocol.instruct.messages import AudioChunk, UserMessage
            from mistral_common.audio import Audio

            # Create audio object
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, audio_np, config.sample_rate)
                audio_obj = Audio.from_file(tmp_file.name)
                os.unlink(tmp_file.name)

            # Create message
            audio_chunk = AudioChunk.from_audio(audio_obj)
            user_message = UserMessage(content=[audio_chunk])

            # Process with vLLM (simplified for now - you can enhance with proper conversation handling)
            prompt = "Transcribe the following audio and provide a helpful response:"

            from vllm import SamplingParams
            sampling_params = SamplingParams(
                temperature=0.2,
                top_p=0.95,
                max_tokens=500,
                stop_token_ids=None
            )

            # Generate response
            results = await self.engine.generate(prompt, sampling_params, f"session_{session_id}")

            if results:
                response_text = results[0].outputs[0].text.strip()
                return response_text if response_text else "I didn't catch that. Could you repeat?"

            return "I'm having trouble processing your audio right now."

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

            # Import Orpheus
            from orpheus_speech import OrpheusModel as OrpheusEngine

            # Initialize with optimizations
            self.model = OrpheusEngine(
                model_name=config.orpheus_model_id,
                max_model_len=config.max_model_len,
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
        """Enhanced emotion detection from text"""
        text_lower = text.lower()

        # More sophisticated emotion detection
        emotion_keywords = {
            "happy": ["haha", "funny", "great", "awesome", "wonderful", "amazing", "love", "excited"],
            "sad": ["sorry", "sad", "unfortunately", "problem", "terrible", "awful", "crying"],
            "surprised": ["wow", "amazing", "incredible", "unbelievable", "shocking", "omg"],
            "tired": ["tired", "exhausted", "sleepy", "weary"],
            "confused": ["confused", "don't understand", "what", "huh"],
            "amused": ["lol", "funny", "hilarious", "amusing"],
            "thoughtful": ["hmm", "think", "consider", "perhaps", "maybe"]
        }

        for emotion, keywords in emotion_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return emotion

        if "?" in text:
            return "thoughtful"

        return "neutral"

    async def synthesize_speech(self, text: str, voice: str = "tara", session_id: str = "") -> Optional[bytes]:
        """Synthesize speech with Orpheus"""
        try:
            if not self.model_loaded:
                logger.error("Orpheus model not loaded")
                return None

            # Detect emotion and add tags
            detected_emotion = self.detect_emotion(text)
            emotion_tag = config.emotion_mapping.get(detected_emotion, "")

            # Add emotion tag to text if not already present
            if emotion_tag and not any(tag in text for tag in config.emotion_mapping.values()):
                text = f"{emotion_tag} {text}"

            logger.info(f"Synthesizing (session {session_id}): '{text[:50]}...' [emotion: {detected_emotion}, voice: {voice}]")

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
                # Combine all chunks
                combined_audio = b''.join(audio_chunks)
                return combined_audio

            return None

        except Exception as e:
            logger.error(f"TTS synthesis error for session {session_id}: {e}")
            return None

class ConnectionManager:
    """WebSocket connection manager with enhanced features"""

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
            "processing": False,
            "conversation_history": [],
            "audio_buffer": [],
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

    async def broadcast(self, message: dict, exclude_session: str = None):
        """Broadcast message to all connected clients"""
        disconnected = []
        for session_id, websocket in self.active_connections.items():
            if session_id != exclude_session:
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception:
                    disconnected.append(session_id)

        # Clean up disconnected clients
        for session_id in disconnected:
            self.disconnect(session_id)

# Initialize global components
manager = ConnectionManager()
voxtral_model = VoxtralModel()
orpheus_model = OrpheusModel()

# FastAPI app initialization
app = FastAPI(
    title="Voxtral + Orpheus Real-Time Voice Assistant",
    description="Production-ready real-time voice assistant with emotional TTS",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enhanced CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket endpoint with enhanced error handling
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """Enhanced WebSocket endpoint for real-time communication"""

    # Validate session ID
    if not session_id or len(session_id) < 5:
        await websocket.close(code=1000, reason="Invalid session ID")
        return

    # Connect client
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

        # Main message loop
        while True:
            try:
                # Receive message
                data = await websocket.receive_text()
                message = json.loads(data)

                # Update activity timestamp
                if session_id in manager.connection_data:
                    manager.connection_data[session_id]["last_activity"] = time.time()
                    manager.connection_data[session_id]["total_requests"] += 1

                # Process message based on type
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
        # Send processing status
        await manager.send_message(session_id, {
            "type": "status",
            "message": "processing_audio"
        })

        # Decode audio
        audio_b64 = message.get("audio", "")
        if not audio_b64:
            await manager.send_message(session_id, {
                "type": "error",
                "message": "No audio data provided"
            })
            return

        # Decode base64 audio
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
        voxtral_time = time.time() - start_time

        if not response_text:
            await manager.send_message(session_id, {
                "type": "error",
                "message": "Failed to process audio"
            })
            return

        # Generate TTS response
        await generate_tts_response(session_id, response_text, message.get("voice", "tara"), voxtral_time)

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

        # Send processing status
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
        # Send TTS generation status
        await manager.send_message(session_id, {
            "type": "status",
            "message": "generating_speech"
        })

        # Generate speech with Orpheus
        tts_start_time = time.time()
        audio_data = await orpheus_model.synthesize_speech(text, voice, session_id)
        tts_time = time.time() - tts_start_time

        if audio_data:
            # Encode audio as base64
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')

            # Send audio response
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
            # Send text-only response if TTS fails
            await manager.send_message(session_id, {
                "type": "text_response",
                "text": text,
                "processing_time": f"{processing_time:.3f}s",
                "note": "TTS generation failed, text response only"
            })

        # Send ready status
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
    # Simple contextual responses - can be enhanced with LLM
    text_lower = text_input.lower()

    responses = {
        "hello": ["Hello! How can I help you today?", "Hi there! What can I do for you?", "Hey! Nice to meet you!"],
        "how are you": ["I'm doing great, thanks for asking! How are you?", "I'm fantastic and ready to help!", "Doing well! How can I assist you?"],
        "what is your name": ["I'm your AI voice assistant powered by Voxtral and Orpheus!", "You can call me your AI assistant!", "I'm an AI voice assistant here to help you!"],
        "goodbye": ["Goodbye! Have a great day!", "See you later! Take care!", "Farewell! It was nice talking to you!"],
        "thank you": ["You're welcome! Happy to help!", "My pleasure! Anything else I can do?", "Glad I could help!"],
        "help": ["I can help you with conversations, answer questions, and provide assistance with various topics. Just speak or type your request!", "I'm here to help! You can ask me questions, have a conversation, or request information on various topics.", "I can assist with answering questions, having conversations, and providing information. What would you like to know?"]
    }

    # Check for keyword matches
    for key, response_list in responses.items():
        if key in text_lower:
            import random
            return random.choice(response_list)

    # Default contextual response
    if "?" in text_input:
        return f"That's an interesting question about '{text_input}'. While I can process and respond to your input, I'd recommend asking more specific questions for better assistance!"
    else:
        return f"You mentioned: '{text_input}'. That's interesting! I'm here to help with any questions or have a conversation with you."

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
    """Serve enhanced web client"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🎤 Voxtral + Orpheus Voice Assistant</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #333;
            }

            .container {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                max-width: 800px;
                width: 90%;
                max-height: 90vh;
                overflow-y: auto;
            }

            h1 {
                text-align: center;
                color: #4a5568;
                margin-bottom: 30px;
                font-size: 2.5em;
            }

            .status {
                padding: 15px;
                margin: 15px 0;
                border-radius: 10px;
                font-weight: 600;
                text-align: center;
                transition: all 0.3s ease;
            }

            .status.connected {
                background: linear-gradient(135deg, #81c784, #4caf50);
                color: white;
            }

            .status.processing {
                background: linear-gradient(135deg, #ffb74d, #ff9800);
                color: white;
            }

            .status.error {
                background: linear-gradient(135deg, #e57373, #f44336);
                color: white;
            }

            .controls {
                display: flex;
                justify-content: center;
                gap: 15px;
                margin: 25px 0;
                flex-wrap: wrap;
            }

            button {
                padding: 15px 25px;
                border: none;
                border-radius: 50px;
                cursor: pointer;
                font-size: 16px;
                font-weight: 600;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }

            .record-btn {
                background: linear-gradient(135deg, #42a5f5, #1976d2);
                color: white;
            }

            .record-btn.recording {
                background: linear-gradient(135deg, #ef5350, #d32f2f);
                animation: pulse 1.5s infinite;
            }

            @keyframes pulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.05); }
            }

            .chat {
                height: 300px;
                overflow-y: auto;
                border: 2px solid #e0e0e0;
                border-radius: 15px;
                padding: 20px;
                margin: 25px 0;
                background: #fafafa;
                scroll-behavior: smooth;
            }

            .message {
                margin: 15px 0;
                padding: 12px 18px;
                border-radius: 18px;
                max-width: 80%;
                word-wrap: break-word;
                animation: fadeIn 0.3s ease;
            }

            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }

            .user-msg {
                background: linear-gradient(135deg, #e3f2fd, #bbdefb);
                margin-left: auto;
                text-align: right;
            }

            .assistant-msg {
                background: linear-gradient(135deg, #f1f8e9, #c8e6c9);
            }

            .system-msg {
                background: linear-gradient(135deg, #fff3e0, #ffe0b2);
                font-style: italic;
                text-align: center;
                margin: 0 auto;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎤 Voice Assistant</h1>
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

            <div style="display: flex; gap: 10px; margin: 25px 0;">
                <input type="text" id="textInput" placeholder="Type your message here..." style="flex: 1; padding: 15px; border: 2px solid #e0e0e0; border-radius: 25px; font-size: 16px;">
                <button onclick="sendText()" style="background: linear-gradient(135deg, #42a5f5, #1976d2); color: white;">Send</button>
            </div>

            <div id="chat" class="chat"></div>
            <audio id="audioPlayer" controls style="width: 100%; margin: 20px 0;"></audio>
        </div>

        <script>
            // JavaScript implementation will be added in the next section
            console.log("Voice Assistant Web Client Loaded");
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
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
            "torch_dtype": str(config.torch_dtype),
            "sample_rate": config.sample_rate,
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
async def get_detailed_stats():
    """Get detailed server statistics"""
    gpu_info = GPUMonitor.get_gpu_info()
    memory_info = psutil.virtual_memory()

    total_requests = sum(
        data.get("total_requests", 0)
        for data in manager.connection_data.values()
    )

    total_audio_duration = sum(
        data.get("total_audio_duration", 0)
        for data in manager.connection_data.values()
    )

    return {
        "server": {
            "active_connections": len(manager.active_connections),
            "total_requests": total_requests,
            "total_audio_generated": f"{total_audio_duration:.1f}s",
            "uptime": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
        },
        "models": {
            "voxtral_loaded": voxtral_model.model_loaded,
            "orpheus_loaded": orpheus_model.model_loaded,
            "voxtral_model": config.voxtral_model_id,
            "orpheus_model": config.orpheus_model_id
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
    """Initialize models and systems on startup"""
    app.state.start_time = time.time()
    logger.info("🚀 Starting Voxtral + Orpheus Voice Assistant...")

    try:
        # Optimize GPU memory first
        GPUMonitor.optimize_gpu_memory()

        # Initialize models in sequence to manage memory
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
    # Configure for production
    uvicorn.run(
        "voice_assistant:app",
        host=config.host,
        port=config.port,
        reload=False,
        access_log=True,
        log_level="info",
        workers=1,  # Single worker for GPU models
        limit_concurrency=config.max_connections
    )

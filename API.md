# 🔌 API Documentation

Complete API reference for the Voxtral + Orpheus Voice Assistant.

## 📋 Overview

The Voice Assistant provides both REST API endpoints and WebSocket connections for real-time communication.

**Base URL**: `http://localhost:8555` (or your deployed URL)
**WebSocket URL**: `ws://localhost:8555/ws/{session_id}`

## 🌐 REST API Endpoints

### Health Check

**GET** `/health`

Returns the current health status of the application.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1703123456.789,
  "models": {
    "voxtral_loaded": true,
    "orpheus_loaded": true
  },
  "server": {
    "active_connections": 3,
    "max_connections": 20,
    "uptime": 3600.5
  },
  "gpu": {
    "name": "NVIDIA A40",
    "memory_used": "15000MB",
    "memory_total": "40000MB",
    "memory_utilization": "37.5%",
    "gpu_utilization": "85.2%",
    "temperature": "65°C"
  },
  "config": {
    "device": "cuda",
    "torch_dtype": "torch.bfloat16",
    "sample_rate": 16000,
    "available_voices": ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
  }
}
```

### Available Voices

**GET** `/api/voices`

Returns list of available TTS voices.

**Response:**
```json
{
  "voices": ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"],
  "default": "tara",
  "total": 8
}
```

### Detailed Statistics

**GET** `/api/stats`

Returns comprehensive server and application statistics.

**Response:**
```json
{
  "server": {
    "active_connections": 5,
    "total_requests": 1250,
    "total_audio_generated": "3600.5s",
    "uptime": 7200.3
  },
  "models": {
    "voxtral_loaded": true,
    "orpheus_loaded": true,
    "voxtral_model": "mistralai/Voxtral-Mini-3B-2507",
    "orpheus_model": "canopylabs/orpheus-3b-0.1-ft"
  },
  "system": {
    "gpu": {
      "name": "NVIDIA A40",
      "memory_used": "18000MB",
      "memory_total": "40000MB"
    },
    "memory_usage": "45.2%",
    "cpu_usage": "23.1%"
  }
}
```

### Web Interface

**GET** `/`

Returns the web client interface for browser-based interaction.

### API Documentation

**GET** `/docs`

Interactive API documentation (Swagger UI).

**GET** `/redoc`

Alternative API documentation (ReDoc).

## 🔌 WebSocket API

### Connection

**WebSocket** `/ws/{session_id}`

Establishes a WebSocket connection for real-time communication.

**Parameters:**
- `session_id` (string): Unique session identifier (minimum 5 characters)

**Connection Flow:**
1. Client connects to WebSocket endpoint
2. Server sends welcome message
3. Client can send audio/text messages
4. Server responds with processed results

### Message Types

#### Welcome Message (Server → Client)

Sent immediately after connection establishment.

```json
{
  "type": "system",
  "message": "🎤 Connected! Ready to assist you with voice or text.",
  "session_id": "session_abc123",
  "available_voices": ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
}
```

#### Audio Input (Client → Server)

Send audio data for speech recognition and response.

```json
{
  "type": "audio",
  "audio": "base64_encoded_audio_data",
  "voice": "tara",
  "timestamp": 1703123456789
}
```

**Parameters:**
- `audio` (string): Base64-encoded audio data (WAV/WebM format)
- `voice` (string, optional): TTS voice for response (default: "tara")
- `timestamp` (number, optional): Client timestamp

#### Text Input (Client → Server)

Send text message for processing and TTS response.

```json
{
  "type": "text",
  "text": "Hello, how are you today?",
  "voice": "leah",
  "timestamp": 1703123456789
}
```

**Parameters:**
- `text` (string): Text message to process
- `voice` (string, optional): TTS voice for response (default: "tara")
- `timestamp` (number, optional): Client timestamp

#### Status Request (Client → Server)

Request current session and server status.

```json
{
  "type": "status"
}
```

#### Status Update (Server → Client)

Indicates current processing status.

```json
{
  "type": "status",
  "message": "processing_audio"
}
```

**Status Messages:**
- `processing_audio`: Processing audio input
- `processing_text`: Processing text input
- `generating_speech`: Generating TTS response
- `ready`: Ready for next input

#### Audio Response (Server → Client)

Response with both text and synthesized audio.

```json
{
  "type": "audio_response",
  "text": "Hello! I'm doing great, thank you for asking.",
  "audio": "base64_encoded_audio_data",
  "voice": "tara",
  "processing_time": "0.150s",
  "tts_time": "0.200s",
  "total_time": "0.350s"
}
```

#### Text Response (Server → Client)

Text-only response (when TTS fails or is disabled).

```json
{
  "type": "text_response",
  "text": "Hello! I'm doing great, thank you for asking.",
  "processing_time": "0.150s",
  "note": "TTS generation failed, text response only"
}
```

#### Status Response (Server → Client)

Response to status request with detailed information.

```json
{
  "type": "status_response",
  "session_data": {
    "connected_duration": "120.5s",
    "total_requests": 15,
    "total_audio_generated": "45.2s"
  },
  "server_status": {
    "active_connections": 8,
    "voxtral_loaded": true,
    "orpheus_loaded": true,
    "gpu_info": {
      "name": "NVIDIA A40",
      "memory_used": "16000MB",
      "memory_total": "40000MB"
    }
  },
  "timestamp": 1703123456.789
}
```

#### Error Message (Server → Client)

Error information when processing fails.

```json
{
  "type": "error",
  "message": "Audio validation failed: Audio too long: 35.2s > 30s"
}
```

#### Warning Message (Server → Client)

Warning information for non-critical issues.

```json
{
  "type": "warning",
  "message": "High GPU memory usage detected"
}
```

## 🎵 Audio Format Requirements

### Input Audio
- **Format**: WAV, WebM, or MP3
- **Sample Rate**: 16kHz (will be resampled if different)
- **Channels**: Mono (stereo will be converted)
- **Duration**: Maximum 30 seconds
- **Encoding**: Base64 for WebSocket transmission

### Output Audio
- **Format**: WAV
- **Sample Rate**: 24kHz
- **Channels**: Mono
- **Encoding**: Base64 for WebSocket transmission

## 🔒 Authentication (Optional)

If API key authentication is enabled:

### Header Authentication
```http
Authorization: Bearer your_api_key_here
```

### Query Parameter Authentication
```
GET /health?api_key=your_api_key_here
```

### WebSocket Authentication
```json
{
  "type": "auth",
  "api_key": "your_api_key_here"
}
```

## 📊 Rate Limiting

- **WebSocket Connections**: 20 concurrent connections per server
- **Message Rate**: No explicit limit, but processing queue applies
- **Audio Duration**: 30 seconds maximum per message
- **Session Timeout**: 1 hour of inactivity

## 🚨 Error Codes

### HTTP Status Codes
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `401`: Unauthorized (invalid API key)
- `429`: Too Many Requests (rate limited)
- `500`: Internal Server Error
- `503`: Service Unavailable (models not loaded)

### WebSocket Close Codes
- `1000`: Normal closure
- `1001`: Going away (server shutdown)
- `1002`: Protocol error
- `1003`: Unsupported data type
- `1011`: Server error

## 🔧 Client Examples

### JavaScript WebSocket Client

```javascript
class VoiceAssistantClient {
    constructor(baseUrl = 'ws://localhost:8555') {
        this.baseUrl = baseUrl;
        this.sessionId = this.generateSessionId();
        this.ws = null;
    }
    
    generateSessionId() {
        return 'session_' + Math.random().toString(36).substring(2, 15);
    }
    
    connect() {
        this.ws = new WebSocket(`${this.baseUrl}/ws/${this.sessionId}`);
        
        this.ws.onopen = () => {
            console.log('Connected to voice assistant');
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };
        
        this.ws.onclose = () => {
            console.log('Disconnected from voice assistant');
        };
    }
    
    sendText(text, voice = 'tara') {
        this.ws.send(JSON.stringify({
            type: 'text',
            text: text,
            voice: voice
        }));
    }
    
    sendAudio(audioBlob, voice = 'tara') {
        const reader = new FileReader();
        reader.onload = () => {
            const base64Audio = reader.result.split(',')[1];
            this.ws.send(JSON.stringify({
                type: 'audio',
                audio: base64Audio,
                voice: voice
            }));
        };
        reader.readAsDataURL(audioBlob);
    }
    
    handleMessage(data) {
        switch (data.type) {
            case 'audio_response':
                this.playAudio(data.audio);
                console.log('Assistant:', data.text);
                break;
            case 'text_response':
                console.log('Assistant:', data.text);
                break;
            case 'error':
                console.error('Error:', data.message);
                break;
        }
    }
    
    playAudio(base64Audio) {
        const audioData = atob(base64Audio);
        const audioArray = new Uint8Array(audioData.length);
        for (let i = 0; i < audioData.length; i++) {
            audioArray[i] = audioData.charCodeAt(i);
        }
        const audioBlob = new Blob([audioArray], { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        audio.play();
    }
}

// Usage
const client = new VoiceAssistantClient();
client.connect();
client.sendText('Hello, how are you?');
```

### Python Client

```python
import asyncio
import websockets
import json
import base64

class VoiceAssistantClient:
    def __init__(self, base_url="ws://localhost:8555"):
        self.base_url = base_url
        self.session_id = f"session_{hash(str(asyncio.get_event_loop()))}"
    
    async def connect(self):
        uri = f"{self.base_url}/ws/{self.session_id}"
        async with websockets.connect(uri) as websocket:
            # Wait for welcome message
            welcome = await websocket.recv()
            print("Connected:", json.loads(welcome))
            
            # Send text message
            await websocket.send(json.dumps({
                "type": "text",
                "text": "Hello, how are you?",
                "voice": "tara"
            }))
            
            # Wait for response
            response = await websocket.recv()
            data = json.loads(response)
            
            if data["type"] == "audio_response":
                print("Response:", data["text"])
                # Save audio to file
                audio_data = base64.b64decode(data["audio"])
                with open("response.wav", "wb") as f:
                    f.write(audio_data)

# Usage
async def main():
    client = VoiceAssistantClient()
    await client.connect()

asyncio.run(main())
```

## 📈 Performance Considerations

- **Concurrent Connections**: Limit to 20 for optimal performance
- **Audio Size**: Keep audio under 30 seconds for best response times
- **Message Frequency**: Allow processing to complete before sending next message
- **Connection Pooling**: Reuse connections when possible

## 🔍 Monitoring and Debugging

### Health Monitoring
```bash
# Check health status
curl http://localhost:8555/health

# Monitor statistics
watch -n 5 'curl -s http://localhost:8555/api/stats | jq .'
```

### WebSocket Debugging
```javascript
// Enable WebSocket debugging in browser
localStorage.debug = 'websocket*';
```

---

**Need more examples?** Check our [GitHub repository](https://github.com/your-repo) for complete client implementations in multiple languages.

#!/usr/bin/env python3
"""
Simple test script for the bulletproof voice assistant
"""

import asyncio
import json
import requests
import websockets
import torch
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gpu():
    """Test GPU availability"""
    logger.info("🔍 Testing GPU...")
    
    try:
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        # Test GPU computation
        x = torch.randn(100, 100).cuda()
        y = torch.matmul(x, x.T)
        
        return True, f"GPU {gpu_name} with {gpu_memory:.1f}GB memory - computation test passed"
        
    except Exception as e:
        return False, f"GPU test failed: {e}"

def test_dependencies():
    """Test all dependencies"""
    logger.info("📦 Testing dependencies...")
    
    required_packages = [
        "torch", "transformers", "fastapi", "uvicorn",
        "librosa", "soundfile", "numpy", "psutil", "GPUtil"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package}")
    
    # Test orpheus-speech separately
    try:
        from orpheus_speech import OrpheusModel
        logger.info("✅ orpheus-speech")
    except ImportError:
        missing_packages.append("orpheus-speech")
        logger.error("❌ orpheus-speech")
    
    if missing_packages:
        return False, f"Missing packages: {', '.join(missing_packages)}"
    
    return True, "All dependencies available"

def test_compatibility():
    """Test dependency compatibility"""
    logger.info("🔧 Testing compatibility...")
    
    try:
        import numpy as np
        import accelerate
        
        logger.info(f"NumPy version: {np.__version__}")
        logger.info(f"Accelerate version: {accelerate.__version__}")
        
        # Check NumPy compatibility
        if not np.__version__.startswith('1.'):
            return False, f"NumPy {np.__version__} incompatible with accelerate (needs 1.x)"
        
        return True, "All dependencies compatible"
        
    except Exception as e:
        return False, f"Compatibility test failed: {e}"

def test_health_endpoint():
    """Test health endpoint"""
    logger.info("🏥 Testing health endpoint...")
    
    try:
        response = requests.get("http://localhost:8555/health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Health check: {data.get('status', 'unknown')}")
            return True, f"Health check passed: {data.get('status', 'unknown')}"
        else:
            return False, f"Health check failed: HTTP {response.status_code}"
            
    except Exception as e:
        return False, f"Health check error: {e}"

async def test_websocket():
    """Test WebSocket connection"""
    logger.info("🔌 Testing WebSocket...")
    
    try:
        session_id = f"test_session_{int(time.time())}"
        uri = f"ws://localhost:8555/ws/{session_id}"
        
        async with websockets.connect(uri, timeout=10) as websocket:
            # Wait for welcome message
            welcome_msg = await asyncio.wait_for(websocket.recv(), timeout=5)
            welcome_data = json.loads(welcome_msg)
            
            if welcome_data.get("type") == "system":
                logger.info("✅ WebSocket connection established")
                return True, "WebSocket connection working"
            else:
                return False, "Invalid welcome message"
                
    except Exception as e:
        return False, f"WebSocket test failed: {e}"

async def test_text_processing():
    """Test text processing"""
    logger.info("💬 Testing text processing...")
    
    try:
        session_id = f"test_text_{int(time.time())}"
        uri = f"ws://localhost:8555/ws/{session_id}"
        
        async with websockets.connect(uri, timeout=10) as websocket:
            # Wait for welcome message
            await asyncio.wait_for(websocket.recv(), timeout=5)
            
            # Send text message
            test_message = {
                "type": "text",
                "text": "Hello, this is a test message",
                "voice": "tara"
            }
            
            await websocket.send(json.dumps(test_message))
            
            # Wait for response
            response_received = False
            timeout_count = 0
            max_timeout = 15  # Reduced timeout for simple version
            
            while not response_received and timeout_count < max_timeout:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=1)
                    data = json.loads(response)
                    
                    if data.get("type") in ["audio_response", "text_response"]:
                        logger.info("✅ Text processing working")
                        return True, "Text processing successful"
                    elif data.get("type") == "error":
                        return False, f"Text processing error: {data.get('message')}"
                        
                except asyncio.TimeoutError:
                    timeout_count += 1
                    logger.info(f"⏳ Waiting for response... ({timeout_count}s)")
            
            return False, "Text processing timeout"
            
    except Exception as e:
        return False, f"Text processing test failed: {e}"

async def run_all_tests():
    """Run all tests"""
    logger.info("🚀 Starting Simple Voice Assistant tests...")
    
    results = {}
    
    # Test 1: GPU
    results["GPU Availability"] = test_gpu()
    
    # Test 2: Dependencies
    results["Dependencies"] = test_dependencies()
    
    # Test 3: Compatibility
    results["Compatibility"] = test_compatibility()
    
    # Test 4: Health Endpoint
    results["Health Endpoint"] = test_health_endpoint()
    
    # Test 5: WebSocket
    results["WebSocket Connection"] = await test_websocket()
    
    # Test 6: Text Processing
    results["Text Processing"] = await test_text_processing()
    
    # Generate report
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result[0])
    
    print("\n" + "="*60)
    print("🧪 SIMPLE VOICE ASSISTANT TEST REPORT")
    print("="*60)
    
    for test_name, (success, message) in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {message}")
    
    print("-"*60)
    print(f"📈 SUMMARY: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! System ready for use.")
        return True
    else:
        print("⚠️  SOME TESTS FAILED! Check the issues above.")
        return False

async def main():
    """Main test function"""
    print("🧪 Simple Voice Assistant - Test Suite")
    print("="*60)
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8555/health", timeout=5)
        print("✅ Server is running, starting tests...")
    except:
        print("❌ Server not running! Please start the voice assistant first:")
        print("   python voice_assistant_simple.py")
        return
    
    success = await run_all_tests()
    
    if success:
        print("\n🎉 System is ready for production use!")
        print("\n🌐 Access your voice assistant at:")
        print("   http://localhost:8555")
        if os.getenv("RUNPOD_PUBLIC_IP"):
            print(f"   https://{os.getenv('RUNPOD_PUBLIC_IP')}-8555.proxy.runpod.net/")
    else:
        print("\n⚠️  Please fix the failing tests.")

if __name__ == "__main__":
    import os
    asyncio.run(main())

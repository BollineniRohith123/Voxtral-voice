#!/usr/bin/env python3
"""
Simple test script for RunPod Voice Assistant
Tests all components before deployment
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

class RunPodTester:
    def __init__(self, base_url="http://localhost:8555"):
        self.base_url = base_url
        self.ws_url = base_url.replace("http", "ws")
    
    def test_gpu_availability(self):
        """Test GPU availability"""
        logger.info("🔍 Testing GPU availability...")
        
        try:
            if not torch.cuda.is_available():
                return False, "CUDA not available"
            
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            logger.info(f"✅ GPU: {gpu_name}")
            logger.info(f"✅ GPU Memory: {gpu_memory:.1f}GB")
            
            return True, f"GPU {gpu_name} with {gpu_memory:.1f}GB memory"
            
        except Exception as e:
            return False, f"GPU test failed: {e}"
    
    def test_dependencies(self):
        """Test required dependencies"""
        logger.info("📦 Testing dependencies...")
        
        required_packages = [
            "torch", "transformers", "vllm", "fastapi", "uvicorn",
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
        
        if missing_packages:
            return False, f"Missing packages: {', '.join(missing_packages)}"
        
        return True, "All dependencies available"
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        logger.info("🏥 Testing health endpoint...")
        
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Health check: {data.get('status', 'unknown')}")
                return True, f"Health check passed: {data.get('status', 'unknown')}"
            else:
                return False, f"Health check failed: HTTP {response.status_code}"
                
        except Exception as e:
            return False, f"Health check error: {e}"
    
    async def test_websocket_connection(self):
        """Test WebSocket connection"""
        logger.info("🔌 Testing WebSocket connection...")
        
        try:
            session_id = f"test_session_{int(time.time())}"
            uri = f"{self.ws_url}/ws/{session_id}"
            
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
    
    async def test_text_processing(self):
        """Test text processing via WebSocket"""
        logger.info("💬 Testing text processing...")
        
        try:
            session_id = f"test_text_{int(time.time())}"
            uri = f"{self.ws_url}/ws/{session_id}"
            
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
                max_timeout = 30
                
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
    
    async def run_all_tests(self):
        """Run all tests"""
        logger.info("🚀 Starting RunPod Voice Assistant tests...")
        
        results = {}
        
        # Test 1: GPU Availability
        results["GPU Availability"] = self.test_gpu_availability()
        
        # Test 2: Dependencies
        results["Dependencies"] = self.test_dependencies()
        
        # Test 3: Health Endpoint
        results["Health Endpoint"] = self.test_health_endpoint()
        
        # Test 4: WebSocket Connection
        results["WebSocket Connection"] = await self.test_websocket_connection()
        
        # Test 5: Text Processing
        results["Text Processing"] = await self.test_text_processing()
        
        # Generate report
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result[0])
        
        print("\n" + "="*60)
        print("🧪 RUNPOD VOICE ASSISTANT TEST REPORT")
        print("="*60)
        
        for test_name, (success, message) in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} {test_name}: {message}")
        
        print("-"*60)
        print(f"📈 SUMMARY: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! System ready for deployment.")
            return True
        else:
            print("⚠️  SOME TESTS FAILED! Please fix issues before deployment.")
            return False

async def main():
    """Main test function"""
    print("🧪 RunPod Voice Assistant - Test Suite")
    print("="*60)
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8555/health", timeout=5)
        print("✅ Server is running, starting tests...")
    except:
        print("❌ Server not running! Please start the voice assistant first:")
        print("   python voice_assistant_runpod.py")
        return
    
    tester = RunPodTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎉 System is ready for production deployment!")
    else:
        print("\n⚠️  Please fix the failing tests before deployment.")

if __name__ == "__main__":
    asyncio.run(main())

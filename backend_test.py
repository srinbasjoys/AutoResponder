#!/usr/bin/env python3
"""
Comprehensive Backend API Testing for AutoResponder AI Assistant
Tests all FastAPI endpoints systematically
"""

import requests
import json
import base64
import uuid
import time
import asyncio
import websockets
from datetime import datetime
import os
import sys

# Get backend URL from frontend env
BACKEND_URL = "http://localhost:8001"
API_BASE = f"{BACKEND_URL}/api"

class BackendTester:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.test_results = []
        
    def log_test(self, test_name, success, message="", response_data=None):
        """Log test results"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {message}")
        self.test_results.append({
            "test": test_name,
            "success": success,
            "message": message,
            "response_data": response_data
        })
        
    def test_health_endpoint(self):
        """Test GET /api/health"""
        try:
            response = requests.get(f"{API_BASE}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "status" in data and data["status"] == "healthy":
                    self.log_test("Health Check", True, f"Status: {data['status']}")
                    return True
                else:
                    self.log_test("Health Check", False, f"Invalid response format: {data}")
                    return False
            else:
                self.log_test("Health Check", False, f"HTTP {response.status_code}: {response.text}")
                return False
        except Exception as e:
            self.log_test("Health Check", False, f"Exception: {str(e)}")
            return False
            
    def test_models_endpoint(self):
        """Test GET /api/models"""
        try:
            response = requests.get(f"{API_BASE}/models", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "models" in data and isinstance(data["models"], dict):
                    expected_providers = ["openai", "anthropic", "gemini", "groq"]
                    available_providers = list(data["models"].keys())
                    
                    if all(provider in available_providers for provider in expected_providers):
                        self.log_test("Models Endpoint", True, f"Found {len(available_providers)} providers")
                        return True
                    else:
                        missing = [p for p in expected_providers if p not in available_providers]
                        self.log_test("Models Endpoint", False, f"Missing providers: {missing}")
                        return False
                else:
                    self.log_test("Models Endpoint", False, f"Invalid response format: {data}")
                    return False
            else:
                self.log_test("Models Endpoint", False, f"HTTP {response.status_code}: {response.text}")
                return False
        except Exception as e:
            self.log_test("Models Endpoint", False, f"Exception: {str(e)}")
            return False
            
    def test_save_provider(self):
        """Test POST /api/providers"""
        try:
            # Test with Groq provider (API key is pre-configured)
            provider_data = {
                "name": "groq",
                "api_key": "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",
                "model": "llama-3.1-8b-instant"
            }
            
            response = requests.post(f"{API_BASE}/providers", 
                                   json=provider_data, 
                                   timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if "message" in data and "Provider saved successfully" in data["message"]:
                    self.log_test("Save Provider", True, f"Groq provider saved successfully")
                    return True
                else:
                    self.log_test("Save Provider", False, f"Unexpected response: {data}")
                    return False
            else:
                self.log_test("Save Provider", False, f"HTTP {response.status_code}: {response.text}")
                return False
        except Exception as e:
            self.log_test("Save Provider", False, f"Exception: {str(e)}")
            return False
            
    def test_get_providers(self):
        """Test GET /api/providers"""
        try:
            response = requests.get(f"{API_BASE}/providers", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "providers" in data and isinstance(data["providers"], list):
                    # Check if Groq provider is configured
                    groq_provider = next((p for p in data["providers"] if p["name"] == "groq"), None)
                    if groq_provider and groq_provider.get("configured"):
                        self.log_test("Get Providers", True, f"Found {len(data['providers'])} providers, Groq configured")
                        return True
                    else:
                        self.log_test("Get Providers", False, "Groq provider not found or not configured")
                        return False
                else:
                    self.log_test("Get Providers", False, f"Invalid response format: {data}")
                    return False
            else:
                self.log_test("Get Providers", False, f"HTTP {response.status_code}: {response.text}")
                return False
        except Exception as e:
            self.log_test("Get Providers", False, f"Exception: {str(e)}")
            return False
            
    def create_mock_audio_base64(self):
        """Create a mock base64 audio data for testing"""
        # Create a simple WAV header for a 1-second silent audio
        sample_rate = 44100
        duration = 1  # 1 second
        num_samples = sample_rate * duration
        
        # WAV header (44 bytes)
        wav_header = bytearray([
            0x52, 0x49, 0x46, 0x46,  # "RIFF"
            0x00, 0x00, 0x00, 0x00,  # File size (will be filled)
            0x57, 0x41, 0x56, 0x45,  # "WAVE"
            0x66, 0x6D, 0x74, 0x20,  # "fmt "
            0x10, 0x00, 0x00, 0x00,  # Subchunk1Size (16 for PCM)
            0x01, 0x00,              # AudioFormat (1 for PCM)
            0x01, 0x00,              # NumChannels (1 for mono)
            0x44, 0xAC, 0x00, 0x00,  # SampleRate (44100)
            0x88, 0x58, 0x01, 0x00,  # ByteRate (44100 * 1 * 16/8)
            0x02, 0x00,              # BlockAlign (1 * 16/8)
            0x10, 0x00,              # BitsPerSample (16)
            0x64, 0x61, 0x74, 0x61,  # "data"
            0x00, 0x00, 0x00, 0x00   # Subchunk2Size (will be filled)
        ])
        
        # Add silent audio data (zeros)
        audio_data = bytearray(num_samples * 2)  # 16-bit samples
        
        # Update file size in header
        file_size = len(wav_header) + len(audio_data) - 8
        wav_header[4:8] = file_size.to_bytes(4, 'little')
        wav_header[40:44] = len(audio_data).to_bytes(4, 'little')
        
        # Combine header and data
        full_audio = wav_header + audio_data
        
        # Convert to base64
        return base64.b64encode(full_audio).decode('utf-8')
        
    def test_process_audio(self):
        """Test POST /api/process-audio"""
        try:
            # Create mock audio data
            mock_audio = self.create_mock_audio_base64()
            
            audio_request = {
                "audio_data": mock_audio,
                "session_id": self.session_id,
                "provider": "groq",
                "model": "llama-3.1-8b-instant"
            }
            
            response = requests.post(f"{API_BASE}/process-audio", 
                                   json=audio_request, 
                                   timeout=30)  # Longer timeout for AI processing
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ["id", "user_input", "ai_response", "provider", "model"]
                if all(field in data for field in required_fields):
                    self.log_test("Process Audio", True, f"Audio processed successfully, AI response: {data['ai_response'][:50]}...")
                    return True
                else:
                    missing_fields = [f for f in required_fields if f not in data]
                    self.log_test("Process Audio", False, f"Missing fields: {missing_fields}")
                    return False
            else:
                self.log_test("Process Audio", False, f"HTTP {response.status_code}: {response.text}")
                return False
        except Exception as e:
            self.log_test("Process Audio", False, f"Exception: {str(e)}")
            return False
            
    def test_get_conversation(self):
        """Test GET /api/conversations/{session_id}"""
        try:
            response = requests.get(f"{API_BASE}/conversations/{self.session_id}", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "conversations" in data and isinstance(data["conversations"], list):
                    if len(data["conversations"]) > 0:
                        # Check if conversation has required fields
                        conv = data["conversations"][0]
                        required_fields = ["id", "user_input", "ai_response", "timestamp", "provider"]
                        if all(field in conv for field in required_fields):
                            self.log_test("Get Conversation", True, f"Found {len(data['conversations'])} conversations")
                            return True
                        else:
                            missing_fields = [f for f in required_fields if f not in conv]
                            self.log_test("Get Conversation", False, f"Missing fields in conversation: {missing_fields}")
                            return False
                    else:
                        self.log_test("Get Conversation", True, "No conversations found (expected for new session)")
                        return True
                else:
                    self.log_test("Get Conversation", False, f"Invalid response format: {data}")
                    return False
            else:
                self.log_test("Get Conversation", False, f"HTTP {response.status_code}: {response.text}")
                return False
        except Exception as e:
            self.log_test("Get Conversation", False, f"Exception: {str(e)}")
            return False
            
    def test_clear_conversation(self):
        """Test DELETE /api/conversations/{session_id}"""
        try:
            response = requests.delete(f"{API_BASE}/conversations/{self.session_id}", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "message" in data and "Cleared" in data["message"]:
                    self.log_test("Clear Conversation", True, data["message"])
                    return True
                else:
                    self.log_test("Clear Conversation", False, f"Unexpected response: {data}")
                    return False
            else:
                self.log_test("Clear Conversation", False, f"HTTP {response.status_code}: {response.text}")
                return False
        except Exception as e:
            self.log_test("Clear Conversation", False, f"Exception: {str(e)}")
            return False
            
    async def test_websocket(self):
        """Test WebSocket /ws/{session_id}"""
        try:
            ws_url = f"ws://localhost:8001/ws/{self.session_id}"
            
            async with websockets.connect(ws_url) as websocket:
                # Send a test message
                test_message = {
                    "type": "audio_data",
                    "audio_data": self.create_mock_audio_base64(),
                    "provider": "groq",
                    "model": "llama-3.1-8b-instant"
                }
                
                await websocket.send(json.dumps(test_message))
                
                # Wait for responses
                responses = []
                timeout_count = 0
                max_timeout = 10  # 10 seconds max wait
                
                while len(responses) < 2 and timeout_count < max_timeout:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(response)
                        responses.append(data)
                        
                        if data.get("type") == "error":
                            self.log_test("WebSocket", False, f"WebSocket error: {data.get('message')}")
                            return False
                            
                    except asyncio.TimeoutError:
                        timeout_count += 1
                        continue
                
                # Check if we got transcription and AI response
                transcription_received = any(r.get("type") == "transcription" for r in responses)
                ai_response_received = any(r.get("type") == "ai_response" for r in responses)
                
                if transcription_received and ai_response_received:
                    self.log_test("WebSocket", True, f"Received {len(responses)} messages including transcription and AI response")
                    return True
                elif len(responses) > 0:
                    self.log_test("WebSocket", False, f"Partial response: got {len(responses)} messages but missing transcription or AI response")
                    return False
                else:
                    self.log_test("WebSocket", False, "No response received within timeout")
                    return False
                    
        except Exception as e:
            self.log_test("WebSocket", False, f"Exception: {str(e)}")
            return False
            
    def run_all_tests(self):
        """Run all backend tests"""
        print("🚀 Starting AutoResponder Backend API Tests")
        print("=" * 60)
        
        # Test sequence
        tests = [
            ("Health Check", self.test_health_endpoint),
            ("Models Endpoint", self.test_models_endpoint),
            ("Save Provider", self.test_save_provider),
            ("Get Providers", self.test_get_providers),
            ("Process Audio", self.test_process_audio),
            ("Get Conversation", self.test_get_conversation),
            ("Clear Conversation", self.test_clear_conversation),
        ]
        
        # Run synchronous tests
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n🧪 Running {test_name}...")
            if test_func():
                passed += 1
            time.sleep(1)  # Brief pause between tests
            
        # Run WebSocket test
        print(f"\n🧪 Running WebSocket Test...")
        try:
            ws_result = asyncio.run(self.test_websocket())
            if ws_result:
                passed += 1
            total += 1
        except Exception as e:
            self.log_test("WebSocket", False, f"Failed to run WebSocket test: {str(e)}")
            total += 1
            
        # Summary
        print("\n" + "=" * 60)
        print(f"📊 TEST SUMMARY: {passed}/{total} tests passed")
        print("=" * 60)
        
        # Detailed results
        print("\n📋 DETAILED RESULTS:")
        for result in self.test_results:
            status = "✅" if result["success"] else "❌"
            print(f"{status} {result['test']}: {result['message']}")
            
        return passed, total

if __name__ == "__main__":
    print("AutoResponder AI Assistant - Backend API Testing")
    print(f"Testing backend at: {BACKEND_URL}")
    print(f"Session ID: {str(uuid.uuid4())}")
    
    tester = BackendTester()
    passed, total = tester.run_all_tests()
    
    if passed == total:
        print(f"\n🎉 All tests passed! Backend is fully functional.")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} tests failed. Backend needs attention.")
        sys.exit(1)
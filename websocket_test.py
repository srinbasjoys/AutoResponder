#!/usr/bin/env python3
"""
Comprehensive WebSocket Testing for AutoResponder AI Assistant
Focus on WebSocket endpoint functionality as requested in review
"""

import asyncio
import websockets
import json
import base64
import uuid
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get backend URL from frontend env
try:
    with open('/app/frontend/.env', 'r') as f:
        for line in f:
            if line.startswith('REACT_APP_BACKEND_URL='):
                BACKEND_URL = line.split('=', 1)[1].strip()
                break
        else:
            BACKEND_URL = "http://localhost:8001"  # fallback
except:
    BACKEND_URL = "http://localhost:8001"  # fallback

class WebSocketTester:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.test_results = []
        
    def log_test(self, test_name, success, message="", details=None):
        """Log test results"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {message}")
        self.test_results.append({
            "test": test_name,
            "success": success,
            "message": message,
            "details": details
        })
        
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

    async def test_websocket_connection(self):
        """Test WebSocket Connection to /ws/{session_id} endpoint"""
        try:
            # Convert HTTP URL to WebSocket URL
            ws_url = BACKEND_URL.replace('https://', 'wss://').replace('http://', 'ws://') + f"/ws/{self.session_id}"
            logger.info(f"Attempting WebSocket connection to: {ws_url}")
            
            # Test connection with timeout
            async with websockets.connect(ws_url, timeout=10) as websocket:
                logger.info("WebSocket connection established successfully")
                
                # Wait for connection confirmation message
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(response)
                    
                    if data.get("type") == "connection_established" and data.get("session_id") == self.session_id:
                        self.log_test("WebSocket Connection", True, f"Connection established for session {self.session_id}")
                        return True
                    else:
                        self.log_test("WebSocket Connection", False, f"Unexpected connection response: {data}")
                        return False
                        
                except asyncio.TimeoutError:
                    self.log_test("WebSocket Connection", False, "No connection confirmation received within timeout")
                    return False
                    
        except websockets.exceptions.InvalidHandshake as e:
            self.log_test("WebSocket Connection", False, f"Handshake failed: {str(e)}")
            return False
        except websockets.exceptions.ConnectionClosed as e:
            self.log_test("WebSocket Connection", False, f"Connection closed: {str(e)}")
            return False
        except Exception as e:
            self.log_test("WebSocket Connection", False, f"Connection failed: {str(e)}")
            return False

    async def test_websocket_ping_message(self):
        """Test ping message type for health check"""
        try:
            ws_url = BACKEND_URL.replace('https://', 'wss://').replace('http://', 'ws://') + f"/ws/{self.session_id}"
            
            async with websockets.connect(ws_url, timeout=10) as websocket:
                # Wait for connection confirmation
                await websocket.recv()
                
                # Send ping message
                ping_message = {
                    "type": "ping",
                    "timestamp": datetime.now().isoformat()
                }
                
                await websocket.send(json.dumps(ping_message))
                logger.info("Sent ping message")
                
                # Wait for pong response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(response)
                    
                    if data.get("type") == "pong":
                        self.log_test("WebSocket Ping Message", True, "Received pong response to ping")
                        return True
                    else:
                        self.log_test("WebSocket Ping Message", False, f"Expected pong, got: {data}")
                        return False
                        
                except asyncio.TimeoutError:
                    self.log_test("WebSocket Ping Message", False, "No pong response received within timeout")
                    return False
                    
        except Exception as e:
            self.log_test("WebSocket Ping Message", False, f"Exception: {str(e)}")
            return False

    async def test_websocket_audio_data_message(self):
        """Test audio_data message type for audio processing"""
        try:
            ws_url = BACKEND_URL.replace('https://', 'wss://').replace('http://', 'ws://') + f"/ws/{self.session_id}"
            
            async with websockets.connect(ws_url, timeout=10) as websocket:
                # Wait for connection confirmation
                await websocket.recv()
                
                # Send audio_data message
                audio_message = {
                    "type": "audio_data",
                    "audio_data": self.create_mock_audio_base64(),
                    "provider": "groq",
                    "model": "llama-3.1-8b-instant"
                }
                
                await websocket.send(json.dumps(audio_message))
                logger.info("Sent audio_data message")
                
                # Collect responses
                responses = []
                timeout_count = 0
                max_timeout = 15  # 15 seconds max wait for audio processing
                
                while timeout_count < max_timeout:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(response)
                        responses.append(data)
                        logger.info(f"Received response: {data.get('type')}")
                        
                        # Check for error
                        if data.get("type") == "error":
                            self.log_test("WebSocket Audio Data Message", False, f"Audio processing error: {data.get('message')}")
                            return False
                        
                        # Check if we have both transcription and AI response
                        transcription_received = any(r.get("type") == "transcription" for r in responses)
                        ai_response_received = any(r.get("type") == "ai_response" for r in responses)
                        
                        if transcription_received and ai_response_received:
                            self.log_test("WebSocket Audio Data Message", True, f"Audio processed successfully - received transcription and AI response")
                            return True
                            
                    except asyncio.TimeoutError:
                        timeout_count += 1
                        continue
                
                # Check what we got
                if len(responses) > 0:
                    response_types = [r.get("type") for r in responses]
                    self.log_test("WebSocket Audio Data Message", False, f"Incomplete audio processing - received: {response_types}")
                else:
                    self.log_test("WebSocket Audio Data Message", False, "No response received for audio processing")
                return False
                    
        except Exception as e:
            self.log_test("WebSocket Audio Data Message", False, f"Exception: {str(e)}")
            return False

    async def test_websocket_unknown_message_type(self):
        """Test error handling for unknown message types"""
        try:
            ws_url = BACKEND_URL.replace('https://', 'wss://').replace('http://', 'ws://') + f"/ws/{self.session_id}"
            
            async with websockets.connect(ws_url, timeout=10) as websocket:
                # Wait for connection confirmation
                await websocket.recv()
                
                # Send unknown message type
                unknown_message = {
                    "type": "unknown_message_type",
                    "data": "test data"
                }
                
                await websocket.send(json.dumps(unknown_message))
                logger.info("Sent unknown message type")
                
                # Wait for error response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(response)
                    
                    if data.get("type") == "error" and "Unknown message type" in data.get("message", ""):
                        self.log_test("WebSocket Unknown Message Type", True, "Properly handled unknown message type with error response")
                        return True
                    else:
                        self.log_test("WebSocket Unknown Message Type", False, f"Unexpected response to unknown message: {data}")
                        return False
                        
                except asyncio.TimeoutError:
                    self.log_test("WebSocket Unknown Message Type", False, "No error response received for unknown message type")
                    return False
                    
        except Exception as e:
            self.log_test("WebSocket Unknown Message Type", False, f"Exception: {str(e)}")
            return False

    async def test_websocket_invalid_message_format(self):
        """Test invalid message formats"""
        try:
            ws_url = BACKEND_URL.replace('https://', 'wss://').replace('http://', 'ws://') + f"/ws/{self.session_id}"
            
            async with websockets.connect(ws_url, timeout=10) as websocket:
                # Wait for connection confirmation
                await websocket.recv()
                
                # Send invalid JSON
                await websocket.send("invalid json message")
                logger.info("Sent invalid JSON message")
                
                # Wait for response or connection close
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    # If we get a response, check if it's an error
                    try:
                        data = json.loads(response)
                        if data.get("type") == "error":
                            self.log_test("WebSocket Invalid Message Format", True, "Properly handled invalid JSON with error response")
                            return True
                        else:
                            self.log_test("WebSocket Invalid Message Format", False, f"Unexpected response to invalid JSON: {data}")
                            return False
                    except json.JSONDecodeError:
                        self.log_test("WebSocket Invalid Message Format", False, "Received non-JSON response to invalid message")
                        return False
                        
                except asyncio.TimeoutError:
                    self.log_test("WebSocket Invalid Message Format", True, "Connection handled invalid JSON gracefully (no response)")
                    return True
                    
        except websockets.exceptions.ConnectionClosed:
            self.log_test("WebSocket Invalid Message Format", True, "Connection closed gracefully after invalid message")
            return True
        except Exception as e:
            self.log_test("WebSocket Invalid Message Format", False, f"Exception: {str(e)}")
            return False

    async def test_websocket_connection_timeout(self):
        """Test connection timeout handling"""
        try:
            ws_url = BACKEND_URL.replace('https://', 'wss://').replace('http://', 'ws://') + f"/ws/{self.session_id}"
            
            async with websockets.connect(ws_url, timeout=10) as websocket:
                # Wait for connection confirmation
                await websocket.recv()
                
                # Stay idle and wait for server ping
                logger.info("Waiting for server ping due to inactivity...")
                
                timeout_count = 0
                max_timeout = 35  # Wait longer than server timeout (30s)
                ping_received = False
                
                while timeout_count < max_timeout:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(response)
                        
                        if data.get("type") == "ping":
                            ping_received = True
                            logger.info("Received server ping")
                            
                            # Send pong back
                            pong_message = {
                                "type": "pong",
                                "timestamp": datetime.now().isoformat()
                            }
                            await websocket.send(json.dumps(pong_message))
                            logger.info("Sent pong response")
                            break
                            
                    except asyncio.TimeoutError:
                        timeout_count += 1
                        continue
                
                if ping_received:
                    self.log_test("WebSocket Connection Timeout", True, "Server sent ping after timeout period")
                    return True
                else:
                    self.log_test("WebSocket Connection Timeout", False, "No server ping received during timeout period")
                    return False
                    
        except websockets.exceptions.ConnectionClosed:
            self.log_test("WebSocket Connection Timeout", True, "Connection closed due to timeout (expected behavior)")
            return True
        except Exception as e:
            self.log_test("WebSocket Connection Timeout", False, f"Exception: {str(e)}")
            return False

    async def test_websocket_streaming_audio_chunks(self):
        """Test streaming audio processing with audio chunks"""
        try:
            ws_url = BACKEND_URL.replace('https://', 'wss://').replace('http://', 'ws://') + f"/ws/{self.session_id}"
            
            async with websockets.connect(ws_url, timeout=10) as websocket:
                # Wait for connection confirmation
                await websocket.recv()
                
                # Send audio chunk (not final)
                audio_chunk_message = {
                    "type": "audio_chunk",
                    "audio_data": self.create_mock_audio_base64(),
                    "is_final": False,
                    "provider": "groq",
                    "model": "llama-3.1-8b-instant"
                }
                
                await websocket.send(json.dumps(audio_chunk_message))
                logger.info("Sent audio chunk (not final)")
                
                # Wait for processing status
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(response)
                    
                    if data.get("type") == "processing_audio" and not data.get("is_final"):
                        logger.info("Received processing status for non-final chunk")
                        
                        # Send final audio chunk
                        final_chunk_message = {
                            "type": "audio_chunk",
                            "audio_data": self.create_mock_audio_base64(),
                            "is_final": True,
                            "provider": "groq",
                            "model": "llama-3.1-8b-instant"
                        }
                        
                        await websocket.send(json.dumps(final_chunk_message))
                        logger.info("Sent final audio chunk")
                        
                        # Collect responses for final chunk
                        responses = []
                        timeout_count = 0
                        max_timeout = 15
                        
                        while timeout_count < max_timeout:
                            try:
                                response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                                data = json.loads(response)
                                responses.append(data)
                                
                                if data.get("type") == "error":
                                    self.log_test("WebSocket Streaming Audio Chunks", False, f"Error in streaming: {data.get('message')}")
                                    return False
                                
                                # Check for complete processing
                                transcription_received = any(r.get("type") == "transcription" for r in responses)
                                ai_response_received = any(r.get("type") == "ai_response" for r in responses)
                                
                                if transcription_received and ai_response_received:
                                    self.log_test("WebSocket Streaming Audio Chunks", True, "Streaming audio chunks processed successfully")
                                    return True
                                    
                            except asyncio.TimeoutError:
                                timeout_count += 1
                                continue
                        
                        self.log_test("WebSocket Streaming Audio Chunks", False, "Incomplete streaming processing")
                        return False
                        
                    else:
                        self.log_test("WebSocket Streaming Audio Chunks", False, f"Unexpected response to audio chunk: {data}")
                        return False
                        
                except asyncio.TimeoutError:
                    self.log_test("WebSocket Streaming Audio Chunks", False, "No response to audio chunk")
                    return False
                    
        except Exception as e:
            self.log_test("WebSocket Streaming Audio Chunks", False, f"Exception: {str(e)}")
            return False

    async def run_all_websocket_tests(self):
        """Run all WebSocket tests"""
        print("🚀 Starting AutoResponder WebSocket Tests")
        print("=" * 60)
        print(f"Testing WebSocket at: {BACKEND_URL}")
        print(f"Session ID: {self.session_id}")
        print("=" * 60)
        
        # Test sequence
        tests = [
            ("WebSocket Connection", self.test_websocket_connection),
            ("WebSocket Ping Message", self.test_websocket_ping_message),
            ("WebSocket Audio Data Message", self.test_websocket_audio_data_message),
            ("WebSocket Unknown Message Type", self.test_websocket_unknown_message_type),
            ("WebSocket Invalid Message Format", self.test_websocket_invalid_message_format),
            ("WebSocket Connection Timeout", self.test_websocket_connection_timeout),
            ("WebSocket Streaming Audio Chunks", self.test_websocket_streaming_audio_chunks),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n🧪 Running {test_name}...")
            try:
                if await test_func():
                    passed += 1
            except Exception as e:
                self.log_test(test_name, False, f"Test execution failed: {str(e)}")
            
            await asyncio.sleep(1)  # Brief pause between tests
            
        # Summary
        print("\n" + "=" * 60)
        print(f"📊 WEBSOCKET TEST SUMMARY: {passed}/{total} tests passed")
        print("=" * 60)
        
        # Detailed results
        print("\n📋 DETAILED RESULTS:")
        for result in self.test_results:
            status = "✅" if result["success"] else "❌"
            print(f"{status} {result['test']}: {result['message']}")
            
        return passed, total

async def main():
    """Main function to run WebSocket tests"""
    tester = WebSocketTester()
    passed, total = await tester.run_all_websocket_tests()
    
    if passed == total:
        print(f"\n🎉 All WebSocket tests passed! WebSocket functionality is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} WebSocket tests failed. WebSocket functionality needs attention.")
        return 1

if __name__ == "__main__":
    import sys
    result = asyncio.run(main())
    sys.exit(result)
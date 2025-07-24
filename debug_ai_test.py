#!/usr/bin/env python3
"""
Focused test for AI response generation issue
"""

import requests
import json
import base64
import uuid

BACKEND_URL = "http://localhost:8001"
API_BASE = f"{BACKEND_URL}/api"

def create_mock_audio_base64():
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

def test_ai_response():
    """Test AI response generation in detail"""
    session_id = str(uuid.uuid4())
    
    # First, ensure Groq provider is configured
    provider_data = {
        "name": "groq",
        "api_key": "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",
        "model": "mixtral-8x7b-32768"
    }
    
    print("1. Configuring Groq provider...")
    response = requests.post(f"{API_BASE}/providers", json=provider_data, timeout=10)
    print(f"   Provider config response: {response.status_code} - {response.json()}")
    
    # Check providers
    print("2. Checking configured providers...")
    response = requests.get(f"{API_BASE}/providers", timeout=10)
    providers_data = response.json()
    print(f"   Providers: {json.dumps(providers_data, indent=2)}")
    
    # Test audio processing
    print("3. Testing audio processing...")
    mock_audio = create_mock_audio_base64()
    
    audio_request = {
        "audio_data": mock_audio,
        "session_id": session_id,
        "provider": "groq",
        "model": "mixtral-8x7b-32768"
    }
    
    response = requests.post(f"{API_BASE}/process-audio", json=audio_request, timeout=30)
    print(f"   Audio processing response: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   Response data: {json.dumps(data, indent=2)}")
    else:
        print(f"   Error response: {response.text}")

if __name__ == "__main__":
    test_ai_response()
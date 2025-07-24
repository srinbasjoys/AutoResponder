#!/usr/bin/env python3
"""
Test with updated Groq model
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

def test_with_updated_model():
    """Test with updated Groq model"""
    session_id = str(uuid.uuid4())
    
    # Configure with updated model
    provider_data = {
        "name": "groq",
        "api_key": "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",
        "model": "llama-3.1-8b-instant"
    }
    
    print("1. Configuring Groq provider with updated model...")
    response = requests.post(f"{API_BASE}/providers", json=provider_data, timeout=10)
    print(f"   Provider config response: {response.status_code} - {response.json()}")
    
    # Test audio processing
    print("2. Testing audio processing with updated model...")
    mock_audio = create_mock_audio_base64()
    
    audio_request = {
        "audio_data": mock_audio,
        "session_id": session_id,
        "provider": "groq",
        "model": "llama-3.1-8b-instant"
    }
    
    response = requests.post(f"{API_BASE}/process-audio", json=audio_request, timeout=30)
    print(f"   Audio processing response: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   User input: {data['user_input']}")
        print(f"   AI response: {data['ai_response']}")
        
        # Check if AI response is valid (not an error)
        if "Error generating response" not in data['ai_response']:
            print("   ✅ AI response generation successful!")
            return True
        else:
            print("   ❌ AI response generation failed!")
            return False
    else:
        print(f"   Error response: {response.text}")
        return False

if __name__ == "__main__":
    test_with_updated_model()
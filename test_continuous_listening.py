#!/usr/bin/env python3
"""
Focused test for continuous listening functionality
"""

import requests
import json
import base64
import uuid
import time
import numpy as np
import wave
import io

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

API_BASE = f"{BACKEND_URL}/api"

def create_realistic_audio_base64():
    """Create a more realistic base64 audio data with actual sound for testing"""
    try:
        # Generate a simple spoken-like audio pattern (sine waves at speech frequencies)
        sample_rate = 16000  # Common for speech recognition
        duration = 2.0  # 2 seconds
        
        # Generate time array
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create a pattern that mimics speech (multiple frequencies)
        # Fundamental frequency around 150Hz (typical male voice)
        f1 = 150  # Fundamental
        f2 = 300  # First harmonic
        f3 = 450  # Second harmonic
        
        # Create audio signal with envelope to simulate speech
        audio_signal = (
            0.3 * np.sin(2 * np.pi * f1 * t) +
            0.2 * np.sin(2 * np.pi * f2 * t) +
            0.1 * np.sin(2 * np.pi * f3 * t)
        )
        
        # Add envelope to make it more speech-like (fade in/out)
        envelope = np.exp(-((t - duration/2) ** 2) / (2 * (duration/4) ** 2))
        audio_signal *= envelope
        
        # Add some noise to make it more realistic
        noise = np.random.normal(0, 0.05, len(audio_signal))
        audio_signal += noise
        
        # Normalize and convert to 16-bit PCM
        audio_signal = np.clip(audio_signal, -1, 1)
        audio_data = (audio_signal * 32767).astype(np.int16)
        
        # Create WAV file in memory
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        wav_io.seek(0)
        wav_data = wav_io.getvalue()
        
        # Convert to base64
        return base64.b64encode(wav_data).decode('utf-8')
        
    except Exception as e:
        print(f"Warning: Could not create realistic audio: {e}")
        return None

def test_continuous_listening_full_flow():
    """Test the complete continuous listening flow"""
    session_id = str(uuid.uuid4()) + "_continuous_test"
    
    print(f"🧪 Testing Continuous Listening Full Flow")
    print(f"Session ID: {session_id}")
    print("=" * 60)
    
    # Step 1: Start listening session
    print("1. Starting listening session...")
    start_request = {
        "session_id": session_id,
        "provider": "perplexity",  # Use Perplexity since it's working
        "model": "sonar",
        "continuous_mode": True,
        "max_duration": 300,
        "voice_activity_threshold": 0.3,
        "silence_timeout": 3
    }
    
    start_response = requests.post(f"{API_BASE}/start-listening", 
                                 json=start_request, 
                                 timeout=10)
    
    if start_response.status_code == 200:
        print("✅ Listening session started successfully")
        print(f"   Response: {start_response.json()}")
    else:
        print(f"❌ Failed to start listening: {start_response.status_code} - {start_response.text}")
        return False
    
    # Step 2: Check initial conversation state
    print("\n2. Checking initial conversation state...")
    state_response = requests.get(f"{API_BASE}/conversation-state/{session_id}", timeout=10)
    
    if state_response.status_code == 200:
        state_data = state_response.json()
        print("✅ Initial state retrieved successfully")
        print(f"   Is listening: {state_data.get('is_listening')}")
        print(f"   Is processing: {state_data.get('is_processing')}")
        print(f"   Chunk count: {state_data.get('chunk_count')}")
    else:
        print(f"❌ Failed to get conversation state: {state_response.status_code} - {state_response.text}")
        return False
    
    # Step 3: Send audio chunk for processing
    print("\n3. Sending audio chunk for processing...")
    realistic_audio = create_realistic_audio_base64()
    
    if not realistic_audio:
        print("❌ Failed to create audio data")
        return False
    
    audio_request = {
        "session_id": session_id,
        "provider": "perplexity",
        "model": "sonar",
        "audio_chunk": realistic_audio,
        "chunk_index": 1,
        "is_final": True,
        "voice_activity_detected": True,
        "noise_reduction": True,
        "noise_reduction_strength": 0.7,
        "auto_gain_control": True,
        "high_pass_filter": True
    }
    
    audio_response = requests.post(f"{API_BASE}/continuous-audio", 
                                 json=audio_request, 
                                 timeout=45)
    
    if audio_response.status_code == 200:
        audio_data = audio_response.json()
        print("✅ Audio chunk sent successfully")
        print(f"   Response: {audio_data}")
    else:
        print(f"❌ Failed to process audio: {audio_response.status_code} - {audio_response.text}")
        return False
    
    # Step 4: Wait for processing and check state multiple times
    print("\n4. Monitoring processing state...")
    for i in range(5):
        time.sleep(2)
        state_response = requests.get(f"{API_BASE}/conversation-state/{session_id}", timeout=10)
        
        if state_response.status_code == 200:
            state_data = state_response.json()
            print(f"   Check {i+1}: listening={state_data.get('is_listening')}, "
                  f"processing={state_data.get('is_processing')}, "
                  f"responding={state_data.get('is_responding')}, "
                  f"chunks={state_data.get('chunk_count')}")
            
            # Check if we have transcription or AI response
            transcription = state_data.get('current_transcription', '')
            ai_response = state_data.get('ai_response', '')
            
            if transcription:
                print(f"   📝 Transcription: {transcription[:100]}...")
            if ai_response:
                print(f"   🤖 AI Response: {ai_response[:100]}...")
                break
        else:
            print(f"   ❌ Failed to get state: {state_response.status_code}")
    
    # Step 5: Check conversation persistence
    print("\n5. Checking conversation persistence...")
    time.sleep(2)  # Extra wait for database write
    
    conv_response = requests.get(f"{API_BASE}/conversations/{session_id}", timeout=10)
    
    if conv_response.status_code == 200:
        conv_data = conv_response.json()
        conversations = conv_data.get('conversations', [])
        
        print(f"✅ Found {len(conversations)} conversations")
        
        for i, conv in enumerate(conversations):
            print(f"   Conversation {i+1}:")
            print(f"     User input: {conv.get('user_input', 'N/A')[:50]}...")
            print(f"     AI response: {conv.get('ai_response', 'N/A')[:50]}...")
            print(f"     Provider: {conv.get('provider', 'N/A')}")
            print(f"     Continuous listening: {conv.get('continuous_listening', 'N/A')}")
        
        if len(conversations) > 0:
            print("✅ Conversation persistence working")
        else:
            print("⚠️  No conversations found (may be due to transcription issues)")
    else:
        print(f"❌ Failed to get conversations: {conv_response.status_code} - {conv_response.text}")
    
    # Step 6: Stop listening session
    print("\n6. Stopping listening session...")
    stop_request = {
        "session_id": session_id,
        "reason": "test_completed"
    }
    
    stop_response = requests.post(f"{API_BASE}/stop-listening", 
                                json=stop_request, 
                                timeout=10)
    
    if stop_response.status_code == 200:
        print("✅ Listening session stopped successfully")
        print(f"   Response: {stop_response.json()}")
    else:
        print(f"❌ Failed to stop listening: {stop_response.status_code} - {stop_response.text}")
    
    print("\n" + "=" * 60)
    print("🎯 Continuous Listening Test Complete")
    
    return True

if __name__ == "__main__":
    print("AutoResponder AI Assistant - Continuous Listening Test")
    print(f"Testing backend at: {BACKEND_URL}")
    print()
    
    test_continuous_listening_full_flow()
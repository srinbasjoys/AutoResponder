from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import os
import logging
from typing import Dict, List, Optional
import asyncio
from datetime import datetime
import uuid
import base64
import tempfile
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from openai import OpenAI
from emergentintegrations.llm.chat import LlmChat, UserMessage
import speech_recognition as sr
from pydub import AudioSegment
import io
from duckduckgo_search import DDGS
import pyttsx3
import threading
import librosa
import noisereduce as nr
import soundfile as sf
from scipy import signal
import numpy as np

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AutoResponder AI Assistant", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017/autoresponder")
client = AsyncIOMotorClient(MONGO_URL)
db = client.autoresponder

# OpenAI client for Whisper (optional)
openai_client = None
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    openai_client = OpenAI(api_key=openai_api_key)
    logger.info("OpenAI Whisper client initialized")
else:
    logger.warning("OpenAI API key not found. Whisper transcription will use fallback method.")

# Available models for each provider
AVAILABLE_MODELS = {
    "openai": [
        "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "o4-mini", "o3-mini", "o3",
        "o1-mini", "gpt-4o-mini", "gpt-4.5-preview", "gpt-4o", "o1", "o1-pro"
    ],
    "anthropic": [
        "claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-3-7-sonnet-20250219",
        "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022"
    ],
    "gemini": [
        "gemini-2.5-flash-preview-04-17", "gemini-2.5-pro-preview-05-06", "gemini-2.0-flash",
        "gemini-2.0-flash-preview-image-generation", "gemini-2.0-flash-lite", "gemini-1.5-flash",
        "gemini-1.5-flash-8b", "gemini-1.5-pro"
    ],
    "groq": [
        "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it", "llama-guard-3-8b"
    ],
    "perplexity": [
        "sonar", "sonar-pro", "sonar-reasoning", "sonar-reasoning-pro", 
        "sonar-deep-research", "r1-1776"
    ]
}

# Data models
class LLMProvider(BaseModel):
    name: str
    api_key: str
    model: str

class ConversationMessage(BaseModel):
    id: str
    session_id: str
    user_input: str
    ai_response: str
    timestamp: datetime
    provider: str
    model: str

class AudioProcessRequest(BaseModel):
    audio_data: str  # base64 encoded audio
    session_id: str
    provider: str
    model: str
    noise_reduction: bool = True  # Enable noise reduction by default
    noise_reduction_strength: float = 0.7  # 0.0 to 1.0
    auto_gain_control: bool = True
    high_pass_filter: bool = True

class ModelSelectionRequest(BaseModel):
    provider: str
    model: str

class WebSearchRequest(BaseModel):
    query: str
    max_results: int = 5

class WebSearchWithAIRequest(BaseModel):
    query: str
    session_id: str
    provider: str
    model: str
    max_results: int = 5
    include_search: bool = True

class TextToSpeechRequest(BaseModel):
    text: str
    voice_speed: int = 150  # Words per minute
    voice_pitch: int = 0    # -50 to 50

class AudioEnhancementConfig(BaseModel):
    session_id: str
    noise_reduction: bool = True
    noise_reduction_strength: float = 0.7  # 0.0 to 1.0  
    auto_gain_control: bool = True
    high_pass_filter: bool = True

# Active WebSocket connections
active_connections: Dict[str, WebSocket] = {}

# Chat instances for maintaining context
chat_instances: Dict[str, LlmChat] = {}

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("Starting AutoResponder AI Assistant...")
    # Ensure MongoDB indexes
    await db.conversations.create_index("session_id")
    await db.providers.create_index("user_id")
    
    # Set up Perplexity provider with environment API key
    perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
    if perplexity_api_key:
        try:
            perplexity_provider = {
                "name": "perplexity",
                "api_key": perplexity_api_key,
                "model": "sonar",
                "created_at": datetime.now(),
                "user_id": "default"
            }
            
            await db.providers.update_one(
                {"name": "perplexity", "user_id": "default"},
                {"$set": perplexity_provider},
                upsert=True
            )
            logger.info("Perplexity provider configured with environment API key")
        except Exception as e:
            logger.error(f"Error setting up Perplexity provider: {e}")

@app.get("/")
async def root():
    return {"message": "AutoResponder AI Assistant API", "status": "running"}

@app.get("/api/test-speech-recognition")
async def test_speech_recognition():
    """Test Google Speech Recognition connectivity"""
    try:
        recognizer = sr.Recognizer()
        
        # Test with a simple audio sample (silence)
        import numpy as np
        
        # Generate a simple sine wave audio sample for testing
        sample_rate = 16000
        duration = 1.0  # 1 second
        frequency = 440  # A4 note
        
        # Generate sine wave
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * frequency * t) * 0.1  # Low volume
        
        # Convert to 16-bit PCM
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Create a WAV file in memory
        wav_io = io.BytesIO()
        import wave
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        wav_io.seek(0)
        
        # Test with speech recognition
        with sr.AudioFile(wav_io) as source:
            audio = recognizer.record(source)
            
            # Try to recognize (will likely fail but tests connectivity)
            try:
                text = recognizer.recognize_google(audio, language="en-US")
                return {
                    "status": "success",
                    "message": "Google Speech Recognition service is working",
                    "recognized_text": text
                }
            except sr.UnknownValueError:
                return {
                    "status": "success",
                    "message": "Google Speech Recognition service is accessible (test audio not recognized, which is expected)",
                    "recognized_text": None
                }
                
    except sr.RequestError as e:
        return {
            "status": "error",
            "message": f"Google Speech Recognition service error: {e}",
            "recognized_text": None
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Test failed: {e}",
            "recognized_text": None
        }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/api/models")
async def get_available_models():
    """Get all available models for each provider"""
    return {"models": AVAILABLE_MODELS}

@app.post("/api/providers")
async def save_provider(provider: LLMProvider):
    """Save LLM provider configuration"""
    try:
        # Validate model is available for the provider
        if provider.name not in AVAILABLE_MODELS:
            raise HTTPException(status_code=400, detail=f"Provider '{provider.name}' not supported")
        
        if provider.model not in AVAILABLE_MODELS[provider.name]:
            raise HTTPException(status_code=400, detail=f"Model '{provider.model}' not available for provider '{provider.name}'")
        
        provider_data = {
            "name": provider.name,
            "api_key": provider.api_key,
            "model": provider.model,
            "created_at": datetime.now(),
            "user_id": "default"  # In real app, this would be user-specific
        }
        
        # Update or insert provider
        await db.providers.update_one(
            {"name": provider.name, "user_id": "default"},
            {"$set": provider_data},
            upsert=True
        )
        
        return {"message": "Provider saved successfully", "provider": provider.name, "model": provider.model}
    except Exception as e:
        logger.error(f"Error saving provider: {e}")
        raise HTTPException(status_code=500, detail="Failed to save provider")

@app.get("/api/providers")
async def get_providers():
    """Get all configured LLM providers"""
    try:
        providers = []
        async for provider in db.providers.find({"user_id": "default"}):
            providers.append({
                "name": provider["name"],
                "model": provider.get("model"),
                "configured": bool(provider.get("api_key")),
                "available_models": AVAILABLE_MODELS.get(provider["name"], [])
            })
        return {"providers": providers}
    except Exception as e:
        logger.error(f"Error fetching providers: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch providers")

@app.get("/api/conversations/{session_id}")
async def get_conversation(session_id: str, limit: int = 5):
    """Get conversation history for a session"""
    try:
        conversations = []
        async for conv in db.conversations.find(
            {"session_id": session_id}
        ).sort("timestamp", -1).limit(limit):
            conversations.append({
                "id": conv["id"],
                "user_input": conv["user_input"],
                "ai_response": conv["ai_response"],
                "timestamp": conv["timestamp"],
                "provider": conv["provider"],
                "model": conv.get("model", "")
            })
        
        # Reverse to get chronological order
        conversations.reverse()
        return {"conversations": conversations}
    except Exception as e:
        logger.error(f"Error fetching conversation: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch conversation")

async def apply_noise_cancellation(audio_data: np.ndarray, sample_rate: int = 16000, 
                                 noise_reduction: bool = True, 
                                 noise_reduction_strength: float = 0.7,
                                 auto_gain_control: bool = True,
                                 high_pass_filter: bool = True) -> np.ndarray:
    """
    Apply comprehensive noise cancellation and audio enhancement
    """
    try:
        logger.info("Applying noise cancellation and audio enhancement...")
        
        # Ensure audio is float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize audio to prevent clipping
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        enhanced_audio = audio_data.copy()
        
        # 1. High-pass filter to remove low-frequency noise (like AC hum)
        if high_pass_filter:
            try:
                # Design high-pass filter (remove frequencies below 80Hz)
                sos = signal.butter(4, 80, btype='highpass', fs=sample_rate, output='sos')
                enhanced_audio = signal.sosfilt(sos, enhanced_audio)
                logger.info("✓ High-pass filter applied")
            except Exception as e:
                logger.warning(f"High-pass filter failed: {e}")
        
        # 2. Spectral noise reduction
        if noise_reduction and noise_reduction_strength > 0:
            try:
                # Use first 0.5 seconds as noise sample, or entire audio if shorter
                noise_duration = min(0.5, len(enhanced_audio) / sample_rate / 2)
                noise_sample_length = int(noise_duration * sample_rate)
                
                if len(enhanced_audio) > noise_sample_length:
                    # Apply noise reduction using noisereduce library
                    enhanced_audio = nr.reduce_noise(
                        y=enhanced_audio, 
                        sr=sample_rate,
                        stationary=True,  # Assume stationary noise
                        prop_decrease=noise_reduction_strength,  # Proportion of noise to reduce
                        n_std_thresh_stationary=1.5,  # Threshold for stationary noise
                        n_fft=1024,  # FFT size
                        n_jobs=1  # Single threaded for stability
                    )
                    logger.info(f"✓ Spectral noise reduction applied (strength: {noise_reduction_strength})")
                else:
                    logger.info("Audio too short for noise reduction")
            except Exception as e:
                logger.warning(f"Noise reduction failed: {e}")
        
        # 3. Auto Gain Control (AGC) to normalize volume
        if auto_gain_control:
            try:
                # Calculate RMS and normalize to target level
                rms = np.sqrt(np.mean(enhanced_audio**2))
                if rms > 0:
                    target_rms = 0.2  # Target RMS level
                    gain = target_rms / rms
                    # Limit gain to prevent excessive amplification
                    gain = min(gain, 5.0)  # Max 5x amplification
                    enhanced_audio = enhanced_audio * gain
                    logger.info(f"✓ Auto gain control applied (gain: {gain:.2f}x)")
            except Exception as e:
                logger.warning(f"Auto gain control failed: {e}")
        
        # 4. Soft limiting to prevent clipping
        enhanced_audio = np.tanh(enhanced_audio * 0.9) * 0.9
        
        # 5. Final normalization
        if np.max(np.abs(enhanced_audio)) > 0:
            enhanced_audio = enhanced_audio / np.max(np.abs(enhanced_audio)) * 0.95
        
        logger.info("✓ Noise cancellation and audio enhancement completed")
        return enhanced_audio.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Error in noise cancellation: {e}")
        return audio_data  # Return original audio if enhancement fails

async def transcribe_audio_with_whisper(audio_data: str, noise_reduction: bool = True, 
                                      noise_reduction_strength: float = 0.7,
                                      auto_gain_control: bool = True,
                                      high_pass_filter: bool = True) -> str:
    """Transcribe audio using OpenAI Whisper or fallback method"""
    if openai_client is None:
        logger.info("OpenAI client not available, using fallback transcription")
        return await transcribe_audio_fallback(audio_data, noise_reduction, 
                                             noise_reduction_strength, 
                                             auto_gain_control, high_pass_filter)
    
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_data)
        
        # Load audio with librosa for noise cancellation
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_bytes)
            temp_file.flush()
            
            # Load audio with librosa
            audio_array, sample_rate = librosa.load(temp_file.name, sr=16000)
            
            # Apply noise cancellation if enabled
            if noise_reduction:
                audio_array = await apply_noise_cancellation(
                    audio_array, sample_rate, noise_reduction, 
                    noise_reduction_strength, auto_gain_control, high_pass_filter
                )
            
            # Save enhanced audio to temporary file
            enhanced_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            sf.write(enhanced_temp.name, audio_array, sample_rate)
            enhanced_temp.close()
            
            # Use OpenAI Whisper for transcription
            with open(enhanced_temp.name, "rb") as audio_file:
                transcript = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            
            # Clean up temp files
            os.unlink(temp_file.name)
            os.unlink(enhanced_temp.name)
            
            return transcript.strip()
    except Exception as e:
        logger.error(f"Error transcribing with Whisper: {e}")
        # Fallback to SpeechRecognition
        return await transcribe_audio_fallback(audio_data, noise_reduction, 
                                             noise_reduction_strength, 
                                             auto_gain_control, high_pass_filter)

async def transcribe_audio_fallback(audio_data: str, noise_reduction: bool = True, 
                                   noise_reduction_strength: float = 0.7,
                                   auto_gain_control: bool = True,
                                   high_pass_filter: bool = True) -> str:
    """Fallback transcription using SpeechRecognition library with noise cancellation"""
    try:
        logger.info("Starting fallback transcription with Google Speech Recognition and noise cancellation")
        
        # Step 1: Decode base64 audio
        try:
            audio_bytes = base64.b64decode(audio_data)
            logger.info(f"Successfully decoded base64 audio data: {len(audio_bytes)} bytes")
        except Exception as e:
            logger.error(f"Error decoding base64 audio: {e}")
            return "Error: Could not decode audio data. Please try recording again."
        
        # Step 2: Load audio with librosa for advanced processing
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(audio_bytes)
                temp_file.flush()
                
                # Load audio with librosa (better than pydub for audio processing)
                audio_array, sample_rate = librosa.load(temp_file.name, sr=16000)
                logger.info(f"Audio loaded - Duration: {len(audio_array)/sample_rate:.2f}s, Sample Rate: {sample_rate}")
                
                # Apply noise cancellation if enabled
                if noise_reduction:
                    logger.info("Applying noise cancellation...")
                    audio_array = await apply_noise_cancellation(
                        audio_array, sample_rate, noise_reduction, 
                        noise_reduction_strength, auto_gain_control, high_pass_filter
                    )
                
                os.unlink(temp_file.name)
                
        except Exception as e:
            logger.error(f"Error loading/processing audio: {e}")
            return "Error: Could not process audio format. Please try recording again."
        
        # Step 3: Convert enhanced audio back to wav for speech recognition
        try:
            wav_io = io.BytesIO()
            sf.write(wav_io, audio_array, sample_rate, format='WAV')
            wav_io.seek(0)
            logger.info(f"Enhanced audio converted to WAV format: {wav_io.getvalue().__len__()} bytes")
        except Exception as e:
            logger.error(f"Error converting enhanced audio to WAV: {e}")
            return "Error: Could not convert enhanced audio. Please try recording again."
        
        # Step 4: Use speech recognition with enhanced audio
        try:
            recognizer = sr.Recognizer()
            
            # Optimized recognizer settings for enhanced audio
            recognizer.energy_threshold = 200  # Lower threshold for cleaner audio
            recognizer.dynamic_energy_threshold = True
            recognizer.pause_threshold = 0.6    # Shorter pause for better response
            recognizer.phrase_threshold = 0.2
            recognizer.non_speaking_duration = 0.6
            
            with sr.AudioFile(wav_io) as source:
                logger.info("Recording enhanced audio for speech recognition...")
                # Adjust for ambient noise (shorter duration for pre-processed audio)
                recognizer.adjust_for_ambient_noise(source, duration=0.2)
                audio = recognizer.record(source)
                
                logger.info("Calling Google Speech Recognition API with enhanced audio...")
                # Use Google Speech Recognition with optimized settings
                text = recognizer.recognize_google(
                    audio, 
                    language="en-US",
                    show_all=False
                )
                
                logger.info(f"✓ Successfully transcribed enhanced audio: '{text}'")
                return text
                
        except sr.UnknownValueError:
            logger.warning("Google Speech Recognition could not understand the enhanced audio")
            return "Could not understand the audio. Please speak clearly and try again."
            
        except sr.RequestError as e:
            logger.error(f"Google Speech Recognition service error: {e}")
            return "Speech recognition service is temporarily unavailable. Please try again later."
            
        except Exception as e:
            logger.error(f"Unexpected error during speech recognition: {e}")
            return "An error occurred during speech recognition. Please try again."
            
    except Exception as e:
        logger.error(f"Unexpected error in fallback transcription: {e}")
        return "Sorry, I couldn't process the audio. Please try again."

async def stream_ai_response(websocket: WebSocket, user_input: str, session_id: str, provider: str, model: str):
    """Stream AI response to WebSocket with real-time updates"""
    try:
        # Get API key for the provider
        provider_config = await db.providers.find_one({"name": provider, "user_id": "default"})
        if not provider_config:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Provider '{provider}' not configured. Please add API key in settings.",
                "timestamp": datetime.now().isoformat()
            }))
            return
        
        api_key = provider_config["api_key"]
        
        # Create or get existing chat instance for this session
        chat_key = f"{session_id}_{provider}_{model}"
        
        if chat_key not in chat_instances:
            # Create new chat instance
            system_message = "You are a helpful AI assistant engaged in a real-time conversation. Provide concise, relevant responses."
            chat_instances[chat_key] = LlmChat(
                api_key=api_key,
                session_id=session_id,
                system_message=system_message
            ).with_model(provider, model).with_max_tokens(1000)
        
        chat = chat_instances[chat_key]
        
        # Send processing status
        await websocket.send_text(json.dumps({
            "type": "ai_thinking",
            "timestamp": datetime.now().isoformat()
        }))
        
        # Send message and get response
        user_message = UserMessage(text=user_input)
        response = await chat.send_message(user_message)
        
        # Send the complete AI response
        await websocket.send_text(json.dumps({
            "type": "ai_response",
            "text": response,
            "timestamp": datetime.now().isoformat()
        }))
        
        # Save conversation to database
        conversation_id = str(uuid.uuid4())
        conversation_data = {
            "id": conversation_id,
            "session_id": session_id,
            "user_input": user_input,
            "ai_response": response,
            "timestamp": datetime.now(),
            "provider": provider,
            "model": model
        }
        
        await db.conversations.insert_one(conversation_data)
        
    except Exception as e:
        logger.error(f"Error streaming AI response: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Error generating response: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }))

@app.post("/api/audio-enhancement-config")
async def save_audio_enhancement_config(config: AudioEnhancementConfig):
    """Save audio enhancement configuration for a session"""
    try:
        config_data = {
            "session_id": config.session_id,
            "noise_reduction": config.noise_reduction,
            "noise_reduction_strength": config.noise_reduction_strength,
            "auto_gain_control": config.auto_gain_control,
            "high_pass_filter": config.high_pass_filter,
            "updated_at": datetime.now()
        }
        
        # Update or insert configuration
        await db.audio_configs.update_one(
            {"session_id": config.session_id},
            {"$set": config_data},
            upsert=True
        )
        
        return {
            "message": "Audio enhancement configuration saved successfully",
            "config": config_data
        }
    except Exception as e:
        logger.error(f"Error saving audio enhancement config: {e}")
        raise HTTPException(status_code=500, detail="Failed to save audio enhancement configuration")

@app.get("/api/audio-enhancement-config/{session_id}")
async def get_audio_enhancement_config(session_id: str):
    """Get audio enhancement configuration for a session"""
    try:
        config = await db.audio_configs.find_one({"session_id": session_id})
        
        if config:
            return {
                "session_id": config["session_id"],
                "noise_reduction": config.get("noise_reduction", True),
                "noise_reduction_strength": config.get("noise_reduction_strength", 0.7),
                "auto_gain_control": config.get("auto_gain_control", True),
                "high_pass_filter": config.get("high_pass_filter", True),
                "updated_at": config.get("updated_at")
            }
        else:
            # Return default configuration
            return {
                "session_id": session_id,
                "noise_reduction": True,
                "noise_reduction_strength": 0.7,
                "auto_gain_control": True,
                "high_pass_filter": True,
                "updated_at": None
            }
            
    except Exception as e:
        logger.error(f"Error fetching audio enhancement config: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch audio enhancement configuration")

@app.get("/api/audio-stats/{session_id}")
async def get_audio_processing_stats(session_id: str):
    """Get audio processing statistics for a session"""
    try:
        # Get recent conversations with audio enhancement data
        conversations = []
        async for conv in db.conversations.find(
            {"session_id": session_id, "audio_enhancement": {"$exists": True}}
        ).sort("timestamp", -1).limit(10):
            conversations.append({
                "timestamp": conv["timestamp"],
                "audio_enhancement": conv.get("audio_enhancement", {}),
                "transcription_success": bool(conv.get("user_input"))
            })
        
        # Calculate statistics
        total_conversations = len(conversations)
        successful_transcriptions = sum(1 for conv in conversations if conv["transcription_success"])
        success_rate = (successful_transcriptions / total_conversations * 100) if total_conversations > 0 else 0
        
        # Audio enhancement usage
        noise_reduction_usage = sum(1 for conv in conversations 
                                  if conv.get("audio_enhancement", {}).get("noise_reduction", False))
        
        return {
            "session_id": session_id,
            "total_audio_processed": total_conversations,
            "transcription_success_rate": round(success_rate, 2),
            "noise_reduction_usage": round((noise_reduction_usage / total_conversations * 100) if total_conversations > 0 else 0, 2),
            "recent_conversations": conversations
        }
        
    except Exception as e:
        logger.error(f"Error fetching audio processing stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch audio processing statistics")

async def get_ai_response(user_input: str, session_id: str, provider: str, model: str) -> str:
    """Generate AI response using the specified provider and model"""
    try:
        # Get API key for the provider
        provider_config = await db.providers.find_one({"name": provider, "user_id": "default"})
        if not provider_config:
            return f"Provider '{provider}' not configured. Please add API key in settings."
        
        api_key = provider_config["api_key"]
        
        # Create or get existing chat instance for this session
        chat_key = f"{session_id}_{provider}_{model}"
        
        if chat_key not in chat_instances:
            # Create new chat instance
            system_message = "You are a helpful AI assistant engaged in a real-time conversation. Provide concise, relevant responses."
            chat_instances[chat_key] = LlmChat(
                api_key=api_key,
                session_id=session_id,
                system_message=system_message
            ).with_model(provider, model).with_max_tokens(1000)
        
        chat = chat_instances[chat_key]
        
        # Send message and get response
        user_message = UserMessage(text=user_input)
        response = await chat.send_message(user_message)
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting AI response: {e}")
        return f"Error generating response: {str(e)}"

async def stream_ai_response(websocket: WebSocket, user_input: str, session_id: str, provider: str, model: str):
    """Stream AI response to WebSocket with real-time updates"""
    try:
        # Get API key for the provider
        provider_config = await db.providers.find_one({"name": provider, "user_id": "default"})
        if not provider_config:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Provider '{provider}' not configured. Please add API key in settings.",
                "timestamp": datetime.now().isoformat()
            }))
            return
        
        api_key = provider_config["api_key"]
        
        # Create or get existing chat instance for this session
        chat_key = f"{session_id}_{provider}_{model}"
        
        if chat_key not in chat_instances:
            # Create new chat instance
            system_message = "You are a helpful AI assistant engaged in a real-time conversation. Provide concise, relevant responses."
            chat_instances[chat_key] = LlmChat(
                api_key=api_key,
                session_id=session_id,
                system_message=system_message
            ).with_model(provider, model).with_max_tokens(1000)
        
        chat = chat_instances[chat_key]
        
        # Send processing status
        await websocket.send_text(json.dumps({
            "type": "ai_thinking",
            "timestamp": datetime.now().isoformat()
        }))
        
        # Send message and get response
        user_message = UserMessage(text=user_input)
        response = await chat.send_message(user_message)
        
        # Send the complete AI response
        await websocket.send_text(json.dumps({
            "type": "ai_response",
            "text": response,
            "timestamp": datetime.now().isoformat()
        }))
        
        # Save conversation to database
        conversation_id = str(uuid.uuid4())
        conversation_data = {
            "id": conversation_id,
            "session_id": session_id,
            "user_input": user_input,
            "ai_response": response,
            "timestamp": datetime.now(),
            "provider": provider,
            "model": model
        }
        
        await db.conversations.insert_one(conversation_data)
        
    except Exception as e:
        logger.error(f"Error streaming AI response: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Error generating response: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }))

@app.post("/api/process-audio")
async def process_audio(request: AudioProcessRequest):
    """Process audio input with noise cancellation and generate AI response"""
    try:
        logger.info(f"Processing audio for session {request.session_id} with {request.provider}/{request.model}")
        logger.info(f"Noise reduction settings - enabled: {request.noise_reduction}, strength: {request.noise_reduction_strength}")
        
        # Transcribe audio with noise cancellation
        user_input = await transcribe_audio_with_whisper(
            request.audio_data,
            noise_reduction=request.noise_reduction,
            noise_reduction_strength=request.noise_reduction_strength,
            auto_gain_control=request.auto_gain_control,
            high_pass_filter=request.high_pass_filter
        )
        logger.info(f"Transcribed text: {user_input}")
        
        if not user_input or user_input.strip() == "":
            raise HTTPException(status_code=400, detail="Could not transcribe audio")
        
        # Get AI response
        ai_response = await get_ai_response(user_input, request.session_id, request.provider, request.model)
        logger.info(f"AI response: {ai_response}")
        
        # Save conversation
        conversation_id = str(uuid.uuid4())
        conversation_data = {
            "id": conversation_id,
            "session_id": request.session_id,
            "user_input": user_input,
            "ai_response": ai_response,
            "timestamp": datetime.now(),
            "provider": request.provider,
            "model": request.model,
            "audio_enhancement": {
                "noise_reduction": request.noise_reduction,
                "noise_reduction_strength": request.noise_reduction_strength,
                "auto_gain_control": request.auto_gain_control,
                "high_pass_filter": request.high_pass_filter
            }
        }
        
        await db.conversations.insert_one(conversation_data)
        
        return {
            "id": conversation_id,
            "user_input": user_input,
            "ai_response": ai_response,
            "provider": request.provider,
            "model": request.model,
            "audio_enhancement": conversation_data["audio_enhancement"]
        }
        
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process audio: {str(e)}")

@app.delete("/api/conversations/{session_id}")
async def clear_conversation(session_id: str):
    """Clear conversation history and chat context"""
    try:
        # Clear database records
        result = await db.conversations.delete_many({"session_id": session_id})
        
        # Clear chat instances for this session
        keys_to_remove = [key for key in chat_instances.keys() if key.startswith(session_id)]
        for key in keys_to_remove:
            del chat_instances[key]
        
        return {"message": f"Cleared {result.deleted_count} conversations for session {session_id}"}
    except Exception as e:
        logger.error(f"Error clearing conversation: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear conversation")

import subprocess

async def generate_speech(text: str, voice_speed: int = 150, voice_pitch: int = 0) -> bytes:
    """Generate speech audio from text using espeak directly"""
    def text_to_speech():
        try:
            # Create temporary file for audio output
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_filename = temp_file.name
            temp_file.close()
            
            # Use espeak directly for more reliable TTS
            cmd = [
                'espeak',
                '-w', temp_filename,  # Write to wav file
                '-s', str(voice_speed),  # Speed (words per minute)
                '-p', str(50 + voice_pitch),  # Pitch (0-99, default ~50)
                '-a', '100',  # Amplitude (volume)
                text
            ]
            
            # Run espeak command
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logger.error(f"espeak failed with return code {result.returncode}: {result.stderr}")
                return None
            
            # Check if file was created and has content
            if not os.path.exists(temp_filename):
                logger.error("TTS audio file was not created")
                return None
                
            file_size = os.path.getsize(temp_filename)
            if file_size == 0:
                logger.error("TTS audio file is empty")
                os.unlink(temp_filename)
                return None
            
            # Read the generated audio file
            with open(temp_filename, 'rb') as f:
                audio_data = f.read()
            
            # Clean up temporary file
            os.unlink(temp_filename)
            
            logger.info(f"Successfully generated {len(audio_data)} bytes of audio using espeak")
            return audio_data
            
        except subprocess.TimeoutExpired:
            logger.error("espeak command timed out")
            return None
        except Exception as e:
            logger.error(f"Error in text-to-speech generation: {e}")
            return None
    
    # Run TTS in a separate thread to avoid blocking
    loop = asyncio.get_event_loop()
    audio_data = await loop.run_in_executor(None, text_to_speech)
    
    return audio_data

@app.post("/api/text-to-speech")
async def text_to_speech_endpoint(request: TextToSpeechRequest):
    """Convert text to speech and return audio data"""
    try:
        logger.info(f"Generating speech for text: {request.text[:50]}...")
        
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Generate speech audio
        audio_data = await generate_speech(request.text, request.voice_speed, request.voice_pitch)
        
        if audio_data is None:
            raise HTTPException(status_code=500, detail="Failed to generate speech")
        
        # Return audio as streaming response
        def generate_audio():
            yield audio_data
        
        return StreamingResponse(
            generate_audio(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav",
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in text-to-speech endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate speech: {str(e)}")

async def perform_web_search(query: str, max_results: int = 5) -> List[Dict]:
    """Perform web search using DuckDuckGo"""
    try:
        with DDGS() as ddgs:
            results = []
            for result in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": result.get("title", ""),
                    "body": result.get("body", ""),
                    "url": result.get("href", ""),
                    "source": "DuckDuckGo"
                })
            return results
    except Exception as e:
        logger.error(f"Error performing web search: {e}")
        return []

@app.post("/api/search")
async def web_search(request: WebSearchRequest):
    """Perform web search and return results"""
    try:
        logger.info(f"Performing web search for query: {request.query}")
        results = await perform_web_search(request.query, request.max_results)
        
        return {
            "query": request.query,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error in web search endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to perform web search: {str(e)}")

@app.post("/api/search-with-ai")
async def web_search_with_ai(request: WebSearchWithAIRequest):
    """Perform web search and get AI response based on results"""
    try:
        logger.info(f"Performing web search with AI for query: {request.query}")
        
        # Perform web search if requested
        search_results = []
        if request.include_search:
            search_results = await perform_web_search(request.query, request.max_results)
        
        # Prepare context for AI
        search_context = ""
        if search_results:
            search_context = "\n\nWeb search results:\n"
            for i, result in enumerate(search_results, 1):
                search_context += f"{i}. {result['title']}\n{result['body']}\nSource: {result['url']}\n\n"
        
        # Create enhanced prompt with search context
        enhanced_query = f"{request.query}{search_context}"
        
        # Get AI response with search context
        ai_response = await get_ai_response(enhanced_query, request.session_id, request.provider, request.model)
        
        # Save conversation with search context
        conversation_id = str(uuid.uuid4())
        conversation_data = {
            "id": conversation_id,
            "session_id": request.session_id,
            "user_input": request.query,
            "ai_response": ai_response,
            "timestamp": datetime.now(),
            "provider": request.provider,
            "model": request.model,
            "search_results": search_results if request.include_search else []
        }
        
        await db.conversations.insert_one(conversation_data)
        
        return {
            "id": conversation_id,
            "query": request.query,
            "ai_response": ai_response,
            "search_results": search_results if request.include_search else [],
            "provider": request.provider,
            "model": request.model
        }
        
    except Exception as e:
        logger.error(f"Error in search with AI endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to perform search with AI: {str(e)}")

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time communication"""
    try:
        await websocket.accept()
        active_connections[session_id] = websocket
        logger.info(f"WebSocket connected for session: {session_id}")
        
        # Send connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }))
        
        while True:
            # Wait for messages from client with timeout
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    # Handle ping/pong for connection health
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }))
                    continue
                
                elif message.get("type") == "audio_chunk":
                    # Process streaming audio chunks
                    try:
                        audio_data = message.get("audio_data")
                        is_final = message.get("is_final", False)
                        provider = message.get("provider", "groq")
                        model = message.get("model", "llama-3.1-8b-instant")
                        
                        # Send processing status
                        await websocket.send_text(json.dumps({
                            "type": "processing_audio",
                            "is_final": is_final,
                            "timestamp": datetime.now().isoformat()
                        }))
                        
                        # Transcribe audio with noise cancellation
                        user_input = await transcribe_audio_with_whisper(
                            audio_data,
                            noise_reduction=message.get("noise_reduction", True),
                            noise_reduction_strength=message.get("noise_reduction_strength", 0.7),
                            auto_gain_control=message.get("auto_gain_control", True),
                            high_pass_filter=message.get("high_pass_filter", True)
                        )
                        
                        # Send transcription update
                        await websocket.send_text(json.dumps({
                            "type": "transcription",
                            "text": user_input,
                            "is_final": is_final,
                            "timestamp": datetime.now().isoformat()
                        }))
                        
                        # Only get AI response if this is the final chunk
                        if is_final and user_input.strip():
                            # Get AI response with streaming
                            await stream_ai_response(websocket, user_input, session_id, provider, model)
                        
                    except Exception as e:
                        logger.error(f"Error processing audio chunk: {e}")
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": f"Error processing audio: {str(e)}",
                            "timestamp": datetime.now().isoformat()
                        }))
                
                elif message.get("type") == "audio_data":
                    # Process complete audio data (backward compatibility)
                    try:
                        audio_data = message.get("audio_data")
                        provider = message.get("provider", "groq")
                        model = message.get("model", "llama-3.1-8b-instant")
                        
                        # Transcribe audio with noise cancellation
                        user_input = await transcribe_audio_with_whisper(
                            audio_data,
                            noise_reduction=message.get("noise_reduction", True),
                            noise_reduction_strength=message.get("noise_reduction_strength", 0.7),
                            auto_gain_control=message.get("auto_gain_control", True),
                            high_pass_filter=message.get("high_pass_filter", True)
                        )
                        
                        # Send transcription update
                        await websocket.send_text(json.dumps({
                            "type": "transcription",
                            "text": user_input,
                            "timestamp": datetime.now().isoformat()
                        }))
                        
                        # Get AI response with streaming
                        if user_input.strip():
                            await stream_ai_response(websocket, user_input, session_id, provider, model)
                        
                    except Exception as e:
                        logger.error(f"Error processing audio data: {e}")
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": f"Error processing audio: {str(e)}",
                            "timestamp": datetime.now().isoformat()
                        }))
                
                else:
                    # Handle unknown message types
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Unknown message type: {message.get('type')}",
                        "timestamp": datetime.now().isoformat()
                    }))
                    
            except asyncio.TimeoutError:
                # Send ping to check connection health
                await websocket.send_text(json.dumps({
                    "type": "ping",
                    "timestamp": datetime.now().isoformat()
                }))
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
        active_connections.pop(session_id, None)
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        active_connections.pop(session_id, None)
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
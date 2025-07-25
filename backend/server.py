from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
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
        "llama-3.1-sonar-small-128k-online", "llama-3.1-sonar-large-128k-online",
        "llama-3.1-sonar-huge-128k-online", "llama-3.1-sonar-small-128k-chat",
        "llama-3.1-sonar-large-128k-chat", "llama-3.1-sonar-huge-128k-chat"
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

class ModelSelectionRequest(BaseModel):
    provider: str
    model: str

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

@app.get("/")
async def root():
    return {"message": "AutoResponder AI Assistant API", "status": "running"}

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

async def transcribe_audio_with_whisper(audio_data: str) -> str:
    """Transcribe audio using OpenAI Whisper or fallback method"""
    if openai_client is None:
        logger.info("OpenAI client not available, using fallback transcription")
        return await transcribe_audio_fallback(audio_data)
    
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_data)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_bytes)
            temp_file.flush()
            
            # Use OpenAI Whisper for transcription
            with open(temp_file.name, "rb") as audio_file:
                transcript = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            
            # Clean up temp file
            os.unlink(temp_file.name)
            
            return transcript.strip()
    except Exception as e:
        logger.error(f"Error transcribing with Whisper: {e}")
        # Fallback to SpeechRecognition
        return await transcribe_audio_fallback(audio_data)

async def transcribe_audio_fallback(audio_data: str) -> str:
    """Fallback transcription using SpeechRecognition library"""
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_data)
        
        # Convert to audio segment
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        
        # Export as wav for speech recognition
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)
        
        # Use speech recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_io) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            return text
    except Exception as e:
        logger.error(f"Error in fallback transcription: {e}")
        return "Sorry, I couldn't understand the audio. Please try again."

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

@app.post("/api/process-audio")
async def process_audio(request: AudioProcessRequest):
    """Process audio input and generate AI response"""
    try:
        logger.info(f"Processing audio for session {request.session_id} with {request.provider}/{request.model}")
        
        # Transcribe audio
        user_input = await transcribe_audio_with_whisper(request.audio_data)
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
            "model": request.model
        }
        
        await db.conversations.insert_one(conversation_data)
        
        return {
            "id": conversation_id,
            "user_input": user_input,
            "ai_response": ai_response,
            "provider": request.provider,
            "model": request.model
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

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    active_connections[session_id] = websocket
    
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "audio_data":
                # Process audio in real-time
                try:
                    audio_data = message.get("audio_data")
                    provider = message.get("provider", "groq")
                    model = message.get("model", "mixtral-8x7b-32768")
                    
                    # Transcribe audio
                    user_input = await transcribe_audio_with_whisper(audio_data)
                    
                    # Send transcription update
                    await websocket.send_text(json.dumps({
                        "type": "transcription",
                        "text": user_input,
                        "timestamp": datetime.now().isoformat()
                    }))
                    
                    # Get AI response
                    ai_response = await get_ai_response(user_input, session_id, provider, model)
                    
                    # Send AI response
                    await websocket.send_text(json.dumps({
                        "type": "ai_response",
                        "text": ai_response,
                        "timestamp": datetime.now().isoformat()
                    }))
                    
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": str(e),
                        "timestamp": datetime.now().isoformat()
                    }))
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
        active_connections.pop(session_id, None)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        active_connections.pop(session_id, None)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
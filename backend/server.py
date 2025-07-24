from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
import logging
from typing import Dict, List, Optional
import asyncio
from datetime import datetime
import uuid
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

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

# Data models
class LLMProvider(BaseModel):
    name: str
    api_key: str
    model: Optional[str] = None

class ConversationMessage(BaseModel):
    id: str
    session_id: str
    user_input: str
    ai_response: str
    timestamp: datetime
    provider: str

class AudioProcessRequest(BaseModel):
    audio_data: str  # base64 encoded audio
    session_id: str
    provider: str

# Active WebSocket connections
active_connections: Dict[str, WebSocket] = {}

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

@app.post("/api/providers")
async def save_provider(provider: LLMProvider):
    """Save LLM provider configuration"""
    try:
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
        
        return {"message": "Provider saved successfully", "provider": provider.name}
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
                "configured": bool(provider.get("api_key"))
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
                "provider": conv["provider"]
            })
        
        # Reverse to get chronological order
        conversations.reverse()
        return {"conversations": conversations}
    except Exception as e:
        logger.error(f"Error fetching conversation: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch conversation")

@app.post("/api/process-audio")
async def process_audio(request: AudioProcessRequest):
    """Process audio input and generate AI response"""
    try:
        # For now, return a mock response
        # This will be implemented with actual speech-to-text and LLM integration
        
        conversation_id = str(uuid.uuid4())
        user_input = "Mock transcribed audio input"  # Will be replaced with actual STT
        ai_response = f"This is a mock AI response from {request.provider}"  # Will be replaced with actual LLM
        
        # Save conversation
        conversation_data = {
            "id": conversation_id,
            "session_id": request.session_id,
            "user_input": user_input,
            "ai_response": ai_response,
            "timestamp": datetime.now(),
            "provider": request.provider
        }
        
        await db.conversations.insert_one(conversation_data)
        
        return {
            "id": conversation_id,
            "user_input": user_input,
            "ai_response": ai_response,
            "provider": request.provider
        }
        
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail="Failed to process audio")

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
            
            # Echo back for now - will implement real-time processing
            response = {
                "type": "response",
                "message": f"Received: {message.get('message', '')}",
                "timestamp": datetime.now().isoformat()
            }
            
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
        active_connections.pop(session_id, None)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        active_connections.pop(session_id, None)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
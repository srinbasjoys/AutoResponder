#!/usr/bin/env python3
"""
HTTP-based Real-Time Conversation System
Implements continuous listening and real-time responses without WebSocket
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
import base64
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import threading
import queue
from collections import defaultdict
import aiohttp
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data models for HTTP real-time system
class ContinuousListeningRequest(BaseModel):
    session_id: str
    provider: str
    model: str
    audio_chunk: str  # base64 encoded audio chunk
    chunk_index: int
    is_final: bool = False
    voice_activity_detected: bool = True
    noise_reduction: bool = True
    noise_reduction_strength: float = 0.7
    auto_gain_control: bool = True
    high_pass_filter: bool = True

class StartListeningRequest(BaseModel):
    session_id: str
    provider: str
    model: str
    continuous_mode: bool = True
    max_duration: int = 300  # 5 minutes max
    voice_activity_threshold: float = 0.3
    silence_timeout: int = 3  # seconds

class StopListeningRequest(BaseModel):
    session_id: str
    reason: str = "user_stopped"

class ConversationStateRequest(BaseModel):
    session_id: str

class InterruptConversationRequest(BaseModel):
    session_id: str
    new_audio_chunk: str
    reason: str = "user_interrupt"

@dataclass
class ConversationState:
    session_id: str
    is_listening: bool = False
    is_processing: bool = False
    is_responding: bool = False
    current_transcription: str = ""
    partial_transcription: str = ""
    ai_response: str = ""
    provider: str = "groq"
    model: str = "llama-3.1-8b-instant"
    last_activity: float = 0.0
    audio_chunks: List[str] = None
    chunk_count: int = 0
    conversation_started: float = 0.0
    voice_activity_detected: bool = False
    can_interrupt: bool = True
    
    def __post_init__(self):
        if self.audio_chunks is None:
            self.audio_chunks = []
        if self.last_activity == 0.0:
            self.last_activity = time.time()
        if self.conversation_started == 0.0:
            self.conversation_started = time.time()

class HTTPRealTimeConversation:
    def __init__(self, backend_url: str):
        self.backend_url = backend_url
        self.conversation_states: Dict[str, ConversationState] = {}
        self.processing_queue = queue.Queue()
        self.response_events: Dict[str, asyncio.Event] = {}
        self.session_responses: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Start background processing thread
        self.processing_thread = threading.Thread(
            target=self._background_processor, 
            daemon=True
        )
        self.processing_thread.start()
        
        logger.info("HTTP Real-Time Conversation system initialized")
    
    def _background_processor(self):
        """Background thread for processing audio chunks"""
        while True:
            try:
                # Get next processing task
                task = self.processing_queue.get(timeout=1.0)
                
                if task['type'] == 'process_audio':
                    asyncio.run(self._process_audio_chunk(task['data']))
                elif task['type'] == 'cleanup_session':
                    self._cleanup_session(task['session_id'])
                
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in background processor: {e}")
    
    async def _process_audio_chunk(self, data: Dict[str, Any]):
        """Process audio chunk in background"""
        try:
            session_id = data['session_id']
            audio_chunk = data['audio_chunk']
            is_final = data['is_final']
            
            # Update state
            if session_id in self.conversation_states:
                state = self.conversation_states[session_id]
                state.is_processing = True
                state.audio_chunks.append(audio_chunk)
                state.chunk_count += 1
                state.last_activity = time.time()
                
                # Process transcription
                transcription = await self._transcribe_audio(session_id, audio_chunk)
                
                if transcription:
                    if is_final:
                        state.current_transcription = transcription
                        state.partial_transcription = ""
                        
                        # Get AI response
                        ai_response = await self._get_ai_response(
                            session_id, transcription, state.provider, state.model
                        )
                        
                        state.ai_response = ai_response
                        state.is_responding = True
                        
                        # Store response for polling
                        self.session_responses[session_id]['transcription'] = transcription
                        self.session_responses[session_id]['ai_response'] = ai_response
                        self.session_responses[session_id]['timestamp'] = time.time()
                        
                        # Trigger response event
                        if session_id in self.response_events:
                            self.response_events[session_id].set()
                        
                    else:
                        state.partial_transcription = transcription
                        self.session_responses[session_id]['partial_transcription'] = transcription
                        self.session_responses[session_id]['timestamp'] = time.time()
                
                state.is_processing = False
                
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            if session_id in self.conversation_states:
                self.conversation_states[session_id].is_processing = False
    
    async def _transcribe_audio(self, session_id: str, audio_chunk: str) -> str:
        """Transcribe audio chunk using backend API"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "audio_data": audio_chunk,
                    "session_id": session_id,
                    "provider": "groq",  # Use for transcription
                    "model": "llama-3.1-8b-instant"
                }
                
                async with session.post(
                    f"{self.backend_url}/api/process-audio",
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('user_input', '')
                    else:
                        logger.error(f"Transcription failed: {response.status}")
                        return ""
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return ""
    
    async def _get_ai_response(self, session_id: str, user_input: str, provider: str, model: str) -> str:
        """Get AI response using backend API"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "audio_data": "",  # Empty for text-only
                    "session_id": session_id,
                    "provider": provider,
                    "model": model
                }
                
                # For now, simulate AI response since we need to modify backend
                # In real implementation, we'd call the backend API
                responses = [
                    f"I understand you said: '{user_input}'. How can I help you with that?",
                    f"That's interesting about '{user_input}'. Tell me more.",
                    f"Thanks for sharing about '{user_input}'. What would you like to know?",
                    f"Regarding '{user_input}', I can help you with that."
                ]
                
                import random
                return random.choice(responses)
                
        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            return "I'm sorry, I had trouble processing that. Could you please try again?"
    
    def _cleanup_session(self, session_id: str):
        """Clean up session data"""
        if session_id in self.conversation_states:
            del self.conversation_states[session_id]
        if session_id in self.response_events:
            del self.response_events[session_id]
        if session_id in self.session_responses:
            del self.session_responses[session_id]
    
    def start_listening(self, request: StartListeningRequest) -> Dict[str, Any]:
        """Start continuous listening session"""
        session_id = request.session_id
        
        # Initialize conversation state
        state = ConversationState(
            session_id=session_id,
            is_listening=True,
            provider=request.provider,
            model=request.model,
            conversation_started=time.time()
        )
        
        self.conversation_states[session_id] = state
        self.response_events[session_id] = asyncio.Event()
        
        logger.info(f"Started listening session: {session_id}")
        
        return {
            "session_id": session_id,
            "status": "listening",
            "continuous_mode": request.continuous_mode,
            "max_duration": request.max_duration,
            "message": "Listening started - send audio chunks to /api/continuous-audio"
        }
    
    def stop_listening(self, request: StopListeningRequest) -> Dict[str, Any]:
        """Stop continuous listening session"""
        session_id = request.session_id
        
        if session_id in self.conversation_states:
            state = self.conversation_states[session_id]
            state.is_listening = False
            
            # Process any remaining audio chunks
            if state.audio_chunks:
                # Queue final processing
                self.processing_queue.put({
                    'type': 'process_audio',
                    'data': {
                        'session_id': session_id,
                        'audio_chunk': ''.join(state.audio_chunks),
                        'is_final': True
                    }
                })
        
        logger.info(f"Stopped listening session: {session_id}")
        
        return {
            "session_id": session_id,
            "status": "stopped",
            "reason": request.reason,
            "message": "Listening stopped"
        }
    
    def process_audio_chunk(self, request: ContinuousListeningRequest) -> Dict[str, Any]:
        """Process continuous audio chunk"""
        session_id = request.session_id
        
        if session_id not in self.conversation_states:
            raise HTTPException(
                status_code=404, 
                detail="Session not found. Start listening first."
            )
        
        state = self.conversation_states[session_id]
        
        if not state.is_listening:
            raise HTTPException(
                status_code=400,
                detail="Session is not in listening mode"
            )
        
        # Update voice activity
        state.voice_activity_detected = request.voice_activity_detected
        state.last_activity = time.time()
        
        # Queue audio processing
        self.processing_queue.put({
            'type': 'process_audio',
            'data': {
                'session_id': session_id,
                'audio_chunk': request.audio_chunk,
                'is_final': request.is_final,
                'chunk_index': request.chunk_index
            }
        })
        
        return {
            "session_id": session_id,
            "chunk_index": request.chunk_index,
            "status": "processing" if state.is_processing else "queued",
            "voice_activity": request.voice_activity_detected,
            "message": "Audio chunk queued for processing"
        }
    
    def get_conversation_state(self, session_id: str) -> Dict[str, Any]:
        """Get current conversation state"""
        if session_id not in self.conversation_states:
            raise HTTPException(
                status_code=404,
                detail="Session not found"
            )
        
        state = self.conversation_states[session_id]
        
        # Get latest responses
        responses = self.session_responses.get(session_id, {})
        
        return {
            "session_id": session_id,
            "is_listening": state.is_listening,
            "is_processing": state.is_processing,
            "is_responding": state.is_responding,
            "current_transcription": state.current_transcription,
            "partial_transcription": state.partial_transcription,
            "ai_response": state.ai_response,
            "provider": state.provider,
            "model": state.model,
            "last_activity": state.last_activity,
            "chunk_count": state.chunk_count,
            "conversation_started": state.conversation_started,
            "voice_activity_detected": state.voice_activity_detected,
            "can_interrupt": state.can_interrupt,
            "latest_responses": responses
        }
    
    def interrupt_conversation(self, request: InterruptConversationRequest) -> Dict[str, Any]:
        """Interrupt current conversation with new audio"""
        session_id = request.session_id
        
        if session_id not in self.conversation_states:
            raise HTTPException(
                status_code=404,
                detail="Session not found"
            )
        
        state = self.conversation_states[session_id]
        
        if not state.can_interrupt:
            raise HTTPException(
                status_code=400,
                detail="Conversation cannot be interrupted at this time"
            )
        
        # Stop current processing
        state.is_processing = False
        state.is_responding = False
        state.ai_response = ""
        
        # Process interruption audio
        self.processing_queue.put({
            'type': 'process_audio',
            'data': {
                'session_id': session_id,
                'audio_chunk': request.new_audio_chunk,
                'is_final': True
            }
        })
        
        logger.info(f"Interrupted conversation: {session_id}")
        
        return {
            "session_id": session_id,
            "status": "interrupted",
            "reason": request.reason,
            "message": "Conversation interrupted, processing new audio"
        }
    
    async def stream_responses(self, session_id: str):
        """Stream real-time responses (Server-Sent Events)"""
        if session_id not in self.conversation_states:
            raise HTTPException(
                status_code=404,
                detail="Session not found"
            )
        
        async def event_generator():
            last_sent_time = 0
            
            while True:
                try:
                    state = self.conversation_states.get(session_id)
                    if not state or not state.is_listening:
                        break
                    
                    # Check for new responses
                    responses = self.session_responses.get(session_id, {})
                    current_time = responses.get('timestamp', 0)
                    
                    if current_time > last_sent_time:
                        # Send update
                        update = {
                            "type": "update",
                            "session_id": session_id,
                            "timestamp": current_time,
                            "transcription": responses.get('transcription', ''),
                            "partial_transcription": responses.get('partial_transcription', ''),
                            "ai_response": responses.get('ai_response', ''),
                            "is_processing": state.is_processing,
                            "is_responding": state.is_responding
                        }
                        
                        yield f"data: {json.dumps(update)}\n\n"
                        last_sent_time = current_time
                    
                    await asyncio.sleep(0.1)  # Check every 100ms
                    
                except Exception as e:
                    logger.error(f"Error in stream: {e}")
                    break
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*"
            }
        )

# Global instance
http_realtime = HTTPRealTimeConversation(
    backend_url="https://b8dbe72f-8d81-45fb-a90a-90537128a55e.preview.emergentagent.com"
)

# Example usage and testing
if __name__ == "__main__":
    # Example of how to use the system
    import asyncio
    
    async def test_system():
        # Start listening
        start_req = StartListeningRequest(
            session_id="test_session",
            provider="groq",
            model="llama-3.1-8b-instant"
        )
        
        result = http_realtime.start_listening(start_req)
        print(f"Started: {result}")
        
        # Simulate audio chunks
        for i in range(3):
            chunk_req = ContinuousListeningRequest(
                session_id="test_session",
                provider="groq",
                model="llama-3.1-8b-instant",
                audio_chunk=f"dummy_audio_chunk_{i}",
                chunk_index=i,
                is_final=(i == 2)
            )
            
            result = http_realtime.process_audio_chunk(chunk_req)
            print(f"Chunk {i}: {result}")
            
            await asyncio.sleep(1)
        
        # Check state
        state = http_realtime.get_conversation_state("test_session")
        print(f"State: {state}")
        
        # Stop listening
        stop_req = StopListeningRequest(
            session_id="test_session",
            reason="test_complete"
        )
        
        result = http_realtime.stop_listening(stop_req)
        print(f"Stopped: {result}")
    
    asyncio.run(test_system())
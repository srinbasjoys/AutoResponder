#!/usr/bin/env python3
"""
Simple WebSocket connection test to diagnose the issue
"""

import asyncio
import websockets
import json
import uuid

# Get backend URL from frontend env
try:
    with open('/app/frontend/.env', 'r') as f:
        for line in f:
            if line.startswith('REACT_APP_BACKEND_URL='):
                BACKEND_URL = line.split('=', 1)[1].strip()
                break
        else:
            BACKEND_URL = "http://localhost:8001"
except:
    BACKEND_URL = "http://localhost:8001"

async def test_simple_connection():
    session_id = str(uuid.uuid4())
    ws_url = BACKEND_URL.replace('https://', 'wss://').replace('http://', 'ws://') + f"/ws/{session_id}"
    
    print(f"Attempting to connect to: {ws_url}")
    
    try:
        # Try with shorter timeout
        async with websockets.connect(ws_url, open_timeout=5) as websocket:
            print("✅ WebSocket connection successful!")
            
            # Try to receive initial message
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                data = json.loads(response)
                print(f"✅ Received initial message: {data}")
                return True
            except asyncio.TimeoutError:
                print("⚠️ No initial message received")
                return False
                
    except websockets.exceptions.InvalidHandshake as e:
        print(f"❌ Handshake failed: {e}")
        return False
    except websockets.exceptions.ConnectionClosed as e:
        print(f"❌ Connection closed: {e}")
        return False
    except asyncio.TimeoutError:
        print("❌ Connection timeout")
        return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_simple_connection())
    print(f"Test result: {'PASS' if result else 'FAIL'}")
#!/usr/bin/env python3
"""
WebSocket Connectivity Test for AutoResponder AI Assistant
Tests WebSocket connection and provides detailed diagnostics
"""

import asyncio
import websockets
import json
import sys
import logging
from datetime import datetime
import ssl
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSocketTester:
    def __init__(self, base_url="b8dbe72f-8d81-45fb-a90a-90537128a55e.preview.emergentagent.com"):
        self.base_url = base_url
        self.session_id = "test_session_" + str(int(datetime.now().timestamp()))
        
        # WebSocket URL
        self.ws_url = f"wss://{base_url}/ws/{self.session_id}"
        
        # HTTP URL for fallback testing
        self.http_url = f"https://{base_url}/api/health"
        
        logger.info(f"WebSocket URL: {self.ws_url}")
        logger.info(f"HTTP URL: {self.http_url}")
        
    async def test_http_connectivity(self):
        """Test basic HTTP connectivity"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(self.http_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ HTTP connectivity successful: {data}")
                        return True
                    else:
                        logger.error(f"❌ HTTP connectivity failed: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"❌ HTTP connectivity test failed: {e}")
            return False
    
    async def test_websocket_connectivity(self):
        """Test WebSocket connectivity with detailed diagnostics"""
        try:
            logger.info("🔍 Testing WebSocket connectivity...")
            
            # Try to connect with SSL context
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Test with multiple timeout values
            for timeout in [5, 10, 30]:
                try:
                    logger.info(f"🔄 Attempting WebSocket connection with {timeout}s timeout...")
                    
                    async with websockets.connect(
                        self.ws_url,
                        ssl=ssl_context,
                        timeout=timeout,
                        ping_interval=20,
                        ping_timeout=10
                    ) as websocket:
                        logger.info(f"✅ WebSocket connected successfully with {timeout}s timeout!")
                        
                        # Test basic message exchange
                        await self.test_websocket_messages(websocket)
                        return True
                        
                except asyncio.TimeoutError:
                    logger.warning(f"⏱️ WebSocket connection timeout after {timeout}s")
                    continue
                except Exception as e:
                    logger.error(f"❌ WebSocket connection failed with {timeout}s timeout: {e}")
                    continue
            
            logger.error("❌ All WebSocket connection attempts failed")
            return False
            
        except Exception as e:
            logger.error(f"❌ WebSocket connectivity test failed: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def test_websocket_messages(self, websocket):
        """Test WebSocket message exchange"""
        try:
            # Send ping message
            ping_message = {
                "type": "ping",
                "timestamp": datetime.now().isoformat()
            }
            
            await websocket.send(json.dumps(ping_message))
            logger.info("📤 Sent ping message")
            
            # Wait for response
            response = await asyncio.wait_for(websocket.recv(), timeout=10)
            data = json.loads(response)
            
            if data.get("type") == "connection_established":
                logger.info("🔗 Connection established message received")
                
                # Wait for pong response
                pong_response = await asyncio.wait_for(websocket.recv(), timeout=10)
                pong_data = json.loads(pong_response)
                
                if pong_data.get("type") == "pong":
                    logger.info("🏓 Pong message received - WebSocket is fully functional!")
                    return True
                else:
                    logger.warning(f"⚠️ Unexpected response: {pong_data}")
                    
            else:
                logger.info(f"📥 Initial message received: {data}")
                
        except asyncio.TimeoutError:
            logger.error("❌ WebSocket message timeout")
        except Exception as e:
            logger.error(f"❌ WebSocket message test failed: {e}")
        
        return False
    
    async def diagnose_websocket_issue(self):
        """Diagnose WebSocket connectivity issues"""
        logger.info("🔍 Diagnosing WebSocket issues...")
        
        # Test different protocols
        protocols = [
            f"wss://{self.base_url}/ws/{self.session_id}",
            f"ws://{self.base_url}/ws/{self.session_id}",
            f"wss://{self.base_url}:443/ws/{self.session_id}",
            f"wss://{self.base_url}:80/ws/{self.session_id}"
        ]
        
        for protocol in protocols:
            try:
                logger.info(f"🔄 Testing protocol: {protocol}")
                
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                async with websockets.connect(
                    protocol,
                    ssl=ssl_context if protocol.startswith("wss") else None,
                    timeout=5
                ) as websocket:
                    logger.info(f"✅ Protocol {protocol} works!")
                    return True
                    
            except Exception as e:
                logger.error(f"❌ Protocol {protocol} failed: {e}")
                continue
        
        return False
    
    async def run_comprehensive_test(self):
        """Run comprehensive WebSocket connectivity test"""
        logger.info("🚀 Starting comprehensive WebSocket connectivity test...")
        
        # Test 1: HTTP connectivity
        http_ok = await self.test_http_connectivity()
        
        # Test 2: WebSocket connectivity
        ws_ok = await self.test_websocket_connectivity()
        
        # Test 3: Diagnose issues if WebSocket fails
        if not ws_ok:
            logger.info("🔍 Running diagnostic tests...")
            await self.diagnose_websocket_issue()
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("📊 CONNECTIVITY TEST SUMMARY")
        logger.info("="*50)
        logger.info(f"HTTP Connectivity: {'✅ PASS' if http_ok else '❌ FAIL'}")
        logger.info(f"WebSocket Connectivity: {'✅ PASS' if ws_ok else '❌ FAIL'}")
        
        if not ws_ok:
            logger.info("\n🔧 RECOMMENDED FIXES:")
            logger.info("1. Add WebSocket ingress annotations:")
            logger.info("   nginx.ingress.kubernetes.io/proxy-read-timeout: '3600'")
            logger.info("   nginx.ingress.kubernetes.io/proxy-send-timeout: '3600'")
            logger.info("   nginx.ingress.kubernetes.io/websocket-services: 'backend-service'")
            logger.info("2. Ensure ingress controller supports WebSocket upgrades")
            logger.info("3. Check firewall rules for WebSocket traffic")
            
        return ws_ok

async def main():
    """Main test function"""
    try:
        tester = WebSocketTester()
        await tester.run_comprehensive_test()
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # Install required packages
    import subprocess
    import sys
    
    try:
        import websockets
        import aiohttp
    except ImportError:
        logger.info("📦 Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets", "aiohttp"])
        import websockets
        import aiohttp
    
    asyncio.run(main())
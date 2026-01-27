#!/usr/bin/env python3
"""Test WebSocket connection"""
import asyncio
import websockets
import json

async def test_ws():
    url = "ws://localhost:8000/api/v1/investigate/ws"
    print(f"🔌 Connecting to: {url}")
    
    try:
        async with websockets.connect(
            url,
            additional_headers={"Origin": "http://localhost:5173"}
        ) as ws:
            print("✅ Connected!")
            
            # Send test request
            request = {
                "query": "Test query",
                "use_reasoning": False
            }
            await ws.send(json.dumps(request))
            print(f"📤 Sent: {request}")
            
            # Receive response
            response = await asyncio.wait_for(ws.recv(), timeout=5.0)
            print(f"📥 Received: {response}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"   Type: {type(e).__name__}")

if __name__ == "__main__":
    asyncio.run(test_ws())

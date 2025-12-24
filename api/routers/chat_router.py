"""
Chat & WebSocket Router

Endpoints for LLM-powered chat and streaming WebSocket connections.
"""

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, WebSocket
from pydantic import BaseModel

from api.services.llm_service import get_llm_service


router = APIRouter(tags=["chat"])


# ============== MODELS ==============

class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    response: str
    suggestions: List[str] = []


# ============== CHAT ENDPOINT ==============

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with LLM about data and causality."""
    llm = get_llm_service()
    
    if not await llm.check_availability():
        raise HTTPException(503, "LLM not available")
    
    # Build context-aware prompt
    context_str = ""
    if request.context:
        context_str = f"\n\nContext:\n{json.dumps(request.context, indent=2)}"
    
    prompt = f"""You are a helpful assistant for causal discovery and data analysis.

User question: {request.message}{context_str}

Provide a clear, concise answer. If suggesting next steps, format them as a list."""
    
    response = await llm.generate(prompt)
    
    # Extract suggestions (simple heuristic)
    suggestions = []
    if "next step" in response.lower() or "you could" in response.lower():
        suggestions = ["Run causal discovery", "View correlations", "Check data quality"]
    
    return ChatResponse(
        response=response,
        suggestions=suggestions,
    )


# ============== WEBSOCKET ENDPOINT ==============

@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket for streaming LLM responses."""
    await websocket.accept()
    llm = get_llm_service()
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if not await llm.check_availability():
                await websocket.send_text(json.dumps({
                    "error": "LLM not available"
                }))
                continue
            
            # Stream response
            async for chunk in llm._generate_stream(message.get("prompt", "")):
                await websocket.send_text(json.dumps({
                    "chunk": chunk,
                    "done": False,
                }))
            
            await websocket.send_text(json.dumps({
                "chunk": "",
                "done": True,
            }))
    
    except Exception as e:
        await websocket.close()

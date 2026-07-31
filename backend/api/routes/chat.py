import json
import logging
from typing import AsyncGenerator
from fastapi import APIRouter, Depends, Body
from fastapi.responses import StreamingResponse
from backend.api.dependencies import get_graph
from backend.state import make_initial_state
from backend.api.schemas.chat import ChatRequest
from backend.api.auth_utils import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["Chat"])

@router.post("/stream")
async def chat_stream(
    request: ChatRequest = Body(...),
    graph = Depends(get_graph),
    current_user: dict = Depends(get_current_user)
):
    """
    POST stream endpoint for chatting.
    Expects JSON body: {"query": "...", "session_id": "...", "uploaded_file_path": null}
    """
    
    # We construct the input state for this turn
    state = make_initial_state(query=request.query, session_id=request.session_id)
    if request.uploaded_file_path:
        state["uploaded_file_path"] = request.uploaded_file_path
        
    config = {"configurable": {"thread_id": request.session_id}}
    
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # LangGraph v2 stream_events
            async for event in graph.astream_events(state, config=config, version="v2"):
                # We specifically look for the Combiner node's LLM chunks
                if event["event"] == "on_chat_model_stream":
                    # Filter by node if needed, though typically only planner/combiner use chat model here.
                    # Planner uses with_structured_output which doesn't stream tokens easily,
                    # Combiner uses normal invoke, which can stream. Wait, combiner uses `.ainvoke`.
                    # In LangGraph, `astream_events` works well with `.ainvoke`.
                    chunk = event["data"]["chunk"].content
                    if chunk:
                        # SSE format: data: {"token": "..."}\n\n
                        yield f"data: {json.dumps({'token': chunk})}\n\n"
                        
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Error during stream: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

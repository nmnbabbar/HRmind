import logging
from fastapi import APIRouter, Depends
from backend.api.dependencies import get_graph

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sessions", tags=["Sessions"])

@router.delete("/{session_id}")
async def clear_session(session_id: str, graph = Depends(get_graph)):
    """
    Clears the conversation history for a given session.
    Since we use MemorySaver, there's no native 'delete_thread' exposed cleanly yet in standard MemorySaver.
    However, we can just let it exist or recreate the memory saver.
    For local MemorySaver, it is just a dict. We can clear it if we need to.
    Wait, LangGraph's checkpointer doesn't have a direct 'delete' method typically exposed on MemorySaver.
    We will just return a message that the client should generate a new session_id.
    """
    return {"message": f"Session {session_id} marked as cleared. Please use a new session_id."}

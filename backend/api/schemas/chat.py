from pydantic import BaseModel

class ChatRequest(BaseModel):
    query: str
    session_id: str

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    error: str | None = None

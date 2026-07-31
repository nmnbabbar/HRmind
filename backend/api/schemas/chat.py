from pydantic import BaseModel

class ChatRequest(BaseModel):
    query: str
    session_id: str
    uploaded_file_path: str | None = None

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    error: str | None = None

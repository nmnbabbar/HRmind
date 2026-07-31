from pydantic import BaseModel

class UploadResponse(BaseModel):
    file_id: str
    file_path: str
    message: str
    success: bool

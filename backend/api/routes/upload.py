import os
import shutil
import uuid
import logging
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from backend.config import get_settings
from backend.api.schemas.upload import UploadResponse
from backend.api.auth_utils import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/upload", tags=["Upload"])

@router.post("", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    settings = get_settings()
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Check extension minimally (agents will do deeper magic byte validation)
    ext = Path(file.filename or "").suffix.lower()
    if ext not in [".pdf", ".docx"]:
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are allowed.")
        
    # Check file size before saving to disk (prevent disk DOS)
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if file.size and file.size > max_bytes:
        raise HTTPException(status_code=400, detail=f"File exceeds maximum size of {settings.max_file_size_mb}MB.")
        
    file_id = str(uuid.uuid4())
    filename = f"{file_id}{ext}"
    file_path = upload_dir / filename
    
    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"File uploaded successfully: {file_path}")
        return UploadResponse(
            file_id=file_id,
            file_path=str(file_path.absolute()),
            message="File uploaded successfully",
            success=True
        )
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail="Internal server error during file upload.")

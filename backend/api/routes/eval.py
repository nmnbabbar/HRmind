from fastapi import APIRouter

router = APIRouter(prefix="/api/eval", tags=["Eval"])

@router.post("/rag")
async def evaluate_rag():
    # Placeholder for RAGAS evaluation
    return {"message": "RAG evaluation triggered"}

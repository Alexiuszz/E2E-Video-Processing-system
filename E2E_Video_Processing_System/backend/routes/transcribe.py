from fastapi import APIRouter, Query
from services.transcription import transcribe

router = APIRouter()

@router.post("/transcribe")
async def transcribe_audio(file_path: str = Query(...), model: str = Query("openai")):
    return transcribe(file_path, model)



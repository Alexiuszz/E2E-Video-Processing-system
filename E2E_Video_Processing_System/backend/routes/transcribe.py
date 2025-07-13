import os
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse   
from services.transcription import transcribe
from utils.file_handler import convert_to_wav
import tempfile

router = APIRouter()

@router.post("/transcribe")
async def transcribe_audio(file_path: str = Query(...), model: str = Query("openai")):
    try:
        if not file_path:
            raise ValueError("File path is required")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if model not in ["openai", "whisper", "nemo"]:
            raise ValueError("Unsupported model specified")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            audio_path = tmp_wav.name
    
        convert_to_wav(file_path, audio_path)
        result = transcribe(audio_path, model)
        return JSONResponse(content={"segments": result})
    except ValueError as ve:
        return {"error": str(ve)}
    except FileNotFoundError as fnfe:
        return {"error": str(fnfe)}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                # Log the error or handle it as needed
                pass
            
    



from fastapi import APIRouter, File, UploadFile, HTTPException
import shutil, os, tempfile

router = APIRouter()

@router.post("/upload")
async def upload_media(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".wav", ".mp3", ".mp4")):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        return {"file_path": tmp.name}

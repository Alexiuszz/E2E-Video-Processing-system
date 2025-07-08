from fastapi import FastAPI, UploadFile, File
import os
import shutil
import uuid
import ffmpeg
import noisereduce as nr
import librosa
import soundfile as sf

app = FastAPI()

UPLOAD_DIR = "uploads"
AUDIO_DIR = "audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    # Save uploaded video
    video_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_DIR, f"{video_id}_{file.filename}")
    
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract raw audio
    audio_path = os.path.join(AUDIO_DIR, f"{video_id}_raw.wav")
    try:
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000')
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        return {"error": "Audio extraction failed", "details": str(e)}

    # Apply noise reduction
    try:
        y, sr = librosa.load(audio_path, sr=None)
        reduced = nr.reduce_noise(y=y, sr=sr)
        cleaned_audio_path = os.path.join(AUDIO_DIR, f"{video_id}.wav")
        sf.write(cleaned_audio_path, reduced, sr)
        os.remove(audio_path)  # remove raw noisy file
    except Exception as e:
        return {"error": "Noise reduction failed", "details": str(e)}

    return {
        "message": "Video processed successfully. Audio cleaned and ready.",
        "audio_file": cleaned_audio_path
    }

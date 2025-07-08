# FastAPI server
# Responsible for: 
#   • accepting video uploads, converting to audio + denoising,
#   • copying cleaned audio to HPC,
#   • triggering a SLURM job on HPC for Whisper transcription,
#   • polling for the resulting transcript, 
#   • retrieving it and returning it to the end client.

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
import os
import uuid
import tempfile
import subprocess
import time
import shutil
import librosa
import noisereduce as nr
import soundfile as sf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

HPC_USER = os.environ["HPC_USER"]
HPC_HOST = os.environ["HPC_HOST"]
HPC_INCOMING_DIR = os.environ["HPC_INCOMING_DIR"]
HPC_TRANSCRIPTS_DIR = os.environ["HPC_TRANSCRIPTS_DIR"]
HPC_SCRIPTS_DIR = os.environ["HPC_SCRIPTS_DIR"]
POLL_INTERVAL = 5.0

# Allowed video extensions
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv"}

app = FastAPI()

@app.post("/transcribe/")
async def transcribe_video(file: UploadFile = File(...)):
    # Check file extension
    _, ext = os.path.splitext(file.filename)
    if ext.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_VIDEO_EXTENSIONS}"
        )

    # Check file size 
    contents = await file.read()
    max_size = 1000 * 1024 * 1024  # 1000 MB
    if len(contents) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds the 1000 MB limit"
        )

    # Create a temp dir
    session_id = str(uuid.uuid4())
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, f"{session_id}{ext}")
        audio_path = os.path.join(tmpdir, f"{session_id}.wav")
        denoised_path = os.path.join(tmpdir, f"{session_id}_denoised.wav")
        transcript_local = os.path.join(tmpdir, f"{session_id}.txt")

        # Save video file
        with open(video_path, "wb") as f:
            f.write(contents)

        # Extract audio
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-ar", "16000", "-ac", "1", "-vn", audio_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Denoise
        audio, sr = librosa.load(audio_path, sr=16000)
        denoised = nr.reduce_noise(y=audio, sr=sr)
        sf.write(denoised_path, denoised, sr)
        try:
            # Copy to HPC
            remote_audio = f"{HPC_INCOMING_DIR}/{session_id}.wav"
            subprocess.run([
                "scp", denoised_path, f"{HPC_USER}@{HPC_HOST}:{remote_audio}"
            ], check=True)

            # Submit SLURM job
            subprocess.run([
                "ssh", f"{HPC_USER}@{HPC_HOST}",
                f"sbatch {HPC_SCRIPTS_DIR}/run_whisper.slurm {session_id}"
            ], check=True)

            # Poll for transcript
            remote_transcript = f"{HPC_TRANSCRIPTS_DIR}/{session_id}.txt"
            max_wait = 60 * 60  # 60 min
            waited = 0
            while waited < max_wait:
                ret = subprocess.run([
                    "ssh", f"{HPC_USER}@{HPC_HOST}", f"ls {remote_transcript}"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if ret.returncode == 0:
                    break
                time.sleep(POLL_INTERVAL)
                waited += POLL_INTERVAL
            else:
                raise HTTPException(status_code=504, detail="Timed out waiting for transcript")

            # Copy back transcript
            subprocess.run([
                "scp", f"{HPC_USER}@{HPC_HOST}:{remote_transcript}", transcript_local
            ], check=True)

        except subprocess.CalledProcessError as e:
            raise HTTPException( status_code=500, detail=f"Subprocess error: {e}" )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

        # Serve file
          # Read the transcript into memory **before** the tempdir is deleted
        with open(transcript_local, "r") as f:
            transcript_content = f.read()

    # Serve the file content directly from memory (tempdir has been cleaned up)
    return JSONResponse(
        content={
            "session_id": session_id,
            "transcript": transcript_content
        },
        headers={"Content-Disposition": "attachment; filename=transcription.txt"}
    )

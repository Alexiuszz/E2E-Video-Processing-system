from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
import os
import uuid
import tempfile
import subprocess
import time
import librosa
import noisereduce as nr
import soundfile as sf
from dotenv import load_dotenv
import math
import traceback
from pydub import AudioSegment
from openai import OpenAI
# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HPC_USER = os.environ["HPC_USER"]
HPC_HOST = os.environ["HPC_HOST"]
HPC_INCOMING_DIR = os.environ["HPC_INCOMING_DIR"]
HPC_TRANSCRIPTS_DIR = os.environ["HPC_TRANSCRIPTS_DIR"]
HPC_SCRIPTS_DIR = os.environ["HPC_SCRIPTS_DIR"]
POLL_INTERVAL = 5.0
MAX_WAIT = 12 * 60  

app = FastAPI()
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv"}

def send_to_hpc_and_get_transcript(audio_path: str, session_id: str, job_name: str) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        transcript_local = os.path.join(tmpdir, f"{session_id}.txt")
        remote_audio = f"{HPC_INCOMING_DIR}/{job_name}_{session_id}.wav"
        remote_transcript = f"{HPC_TRANSCRIPTS_DIR}/{job_name}_{session_id}.txt"

        try:
            subprocess.run([
                "scp", audio_path, f"{HPC_USER}@{HPC_HOST}:{remote_audio}"
            ], check=True)

            subprocess.run([
                "ssh", f"{HPC_USER}@{HPC_HOST}",
                f"sbatch {HPC_SCRIPTS_DIR}/run_whisper.slurm {session_id} {job_name}"
            ], check=True)

            waited = 0
            while waited < MAX_WAIT:
                result = subprocess.run([
                    "ssh", f"{HPC_USER}@{HPC_HOST}", f"ls {remote_transcript}"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if result.returncode == 0:
                    break
                time.sleep(POLL_INTERVAL)
                waited += POLL_INTERVAL
            else:
                raise TimeoutError("Timed out waiting for transcript")

            subprocess.run([
                "scp", f"{HPC_USER}@{HPC_HOST}:{remote_transcript}", transcript_local
            ], check=True)

            with open(transcript_local, "r") as f:
                return f.read()

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Subprocess error: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error: {e}")

def openai_transcribe(audio_path: str) -> str:
    
    return "Transcription result from OpenAI API"


@app.post("/transcribe/")
async def transcribe_video(file: UploadFile = File(...)):
    _, ext = os.path.splitext(file.filename)
    if ext.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_VIDEO_EXTENSIONS}"
        )

    contents = await file.read()
    max_size = 1000 * 1024 * 1024  # 1000 MB
    if len(contents) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds the 1000 MB limit"
        )

    session_id = str(uuid.uuid4())
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, f"{session_id}{ext}")
        audio_path = os.path.join(tmpdir, f"{session_id}.wav")
        denoised_path = os.path.join(tmpdir, f"{session_id}_denoised.wav")

        with open(video_path, "wb") as f:
            f.write(contents)

        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-ar", "16000", "-ac", "1", "-vn", audio_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        audio, sr = librosa.load(audio_path, sr=16000)
        denoised = nr.reduce_noise(y=audio, sr=sr)
        sf.write(denoised_path, denoised, sr)

        try:
            transcript = send_to_hpc_and_get_transcript(denoised_path, session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        return JSONResponse(
            content={"session_id": session_id, "transcript": transcript},
            headers={"Content-Disposition": "attachment; filename=transcription.txt"}
        )

@app.post("/transcribe-audio/")
async def transcribe_audio(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .wav audio files are supported"
        )   
    job_name = file.filename.split('.')[0]
    contents = await file.read()
    session_id = str(uuid.uuid4())

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        tmpfile.write(contents)
        tmpfile_path = tmpfile.name

    try:
        transcript = send_to_hpc_and_get_transcript(tmpfile_path, session_id, job_name)
        return JSONResponse(
            content={"session_id": session_id, "transcript": transcript},
            headers={"Content-Disposition": "attachment; filename=transcription.txt"}
        )
    finally:
        os.remove(tmpfile_path)


# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Split audio into <=25MB chunks by duration (safe default: 10 mins @ 16kHz mono)
def split_audio(audio_path, chunk_length_ms=600_000):
    audio = AudioSegment.from_wav(audio_path)
    chunks = []
    total_chunks = math.ceil(len(audio) / chunk_length_ms)

    for i in range(total_chunks):
        start = i * chunk_length_ms
        end = min((i + 1) * chunk_length_ms, len(audio))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_chunk:
            audio[start:end].export(tmp_chunk.name, format="wav")
            chunks.append(tmp_chunk.name)

    return chunks

# Transcribe a single audio chunk 
def transcribe_chunk(chunk_path):
    try:
        with open(chunk_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
                language="en"
            )
        return response
    except Exception as e:
        error_message = f"Error transcribing chunk {chunk_path}: {str(e)}"
        print(error_message)
        traceback.print_exc()
        raise RuntimeError(error_message)
    
    

# Transcribe full audio by chunking and combining
@app.post("/openai-transcribe/")
async def openai_transcribe(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .wav audio files are supported"
        )
    contents = await file.read()
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        tmpfile.write(contents)
        audio_path = tmpfile.name
    
    try:
        chunks = split_audio(audio_path)
    #print(f"Split into {len(chunks)} chunks.")

        full_transcript = ""
        for i, chunk_path in enumerate(chunks):
            print(f"Transcribing chunk {i+1}/{len(chunks)}...")
            try:
                transcript = transcribe_chunk(chunk_path)
            except RuntimeError as e:
                raise HTTPException(status_code=500, detail=str(e))
            full_transcript += transcript+ "\n"
            os.remove(chunk_path)
    except Exception as e:  
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(audio_path)
    return JSONResponse(
        content={"transcript": full_transcript.strip()},
        headers={"Content-Disposition": "attachment; filename=transcription.txt"}
    )

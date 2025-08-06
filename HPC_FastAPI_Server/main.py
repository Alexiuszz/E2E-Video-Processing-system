from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil, os, tempfile
import torch
import librosa
import whisper
from nemo.collections.asr.models import ASRModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def transcribe_whisper(model, audio_path):
    result = model.transcribe(
        audio_path, 
        language="en", 
        verbose=False,
        task="transcribe"
    )
    transcript = []
    for segment in result.get("segments", []):
        transcript.append({
            "id": segment['id'],
            "start": segment['start'],
            "end": segment['end'],
            "text": segment['text'].strip()
        })
    return transcript

def chunk_audio(audio_path, sr=16000, chunk_duration=30):
    audio, _ = librosa.load(audio_path, sr=sr)
    chunk_samples = chunk_duration * sr
    return [audio[i:i + chunk_samples] for i in range(0, len(audio), chunk_samples)]

def transcribe_nemo_chunked(model, audio_path):
    chunks = chunk_audio(audio_path)
    transcripts = []
    current_time = 0.0  # Track cumulative start time across chunks

    for chunk in chunks:
        # Transcribe chunk with hypothesis
        hypothesis = model.transcribe([chunk], return_hypotheses=True)[0]

        if hasattr(hypothesis, "words") and hypothesis.words:
            for word_info in hypothesis.words:
                # Add current_time offset to each word timestamp
                transcripts.append({
                    "word": word_info.word,
                    "start": word_info.start_time + current_time,
                    "end": word_info.end_time + current_time,
                })
        else:
            # fallback if no word-level timestamps
            transcripts.append({
                "word": str(hypothesis.text),
                "start": current_time,
                "end": current_time + 30.0  # approximate chunk length
            })

        current_time += len(chunk) / 16000  # assuming sample rate is 16kHz

    return transcripts

def segment_timestamps(words, pause_threshold=0.6):
    segments = []
    current_segment = []
    for i, word in enumerate(words):
        if current_segment and word["start"] - current_segment[-1]["end"] > pause_threshold:
            # Save previous segment
            start = current_segment[0]["start"]
            end = current_segment[-1]["end"]
            text = " ".join(w["word"] for w in current_segment)
            segments.append({"id": i,"start": start, "end": end, "text": text})
            current_segment = []
        current_segment.append(word)

    # Add last segment
    if current_segment:
        start = current_segment[0]["start"]
        end = current_segment[-1]["end"]
        text = " ".join(w["word"] for w in current_segment)
        segments.append({"id": i, "start": start, "end": end, "text": text})
    
    return segments

@app.post("/transcribe/")
async def openai_transcribe(file: UploadFile = File(...), model: str = "parakeet"):
    if not file.filename.lower().endswith((".wav", ".mp3", ".mp4")):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        if model == "parakeet":
            asr_model = ASRModel.from_pretrained("nvidia/parakeet-tdt_ctc-110m").to("cuda").eval()
            word_level = transcribe_nemo_chunked(asr_model, tmp_path)
            transcript = segment_timestamps(word_level)
        elif model == "whisper":
            asr_model = whisper.load_model("medium").to("cuda")
            transcript = transcribe_whisper(asr_model, tmp_path)
        else:
            raise HTTPException(status_code=400, detail="Unknown model specified.")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)

    return JSONResponse(content={"transcript": transcript})

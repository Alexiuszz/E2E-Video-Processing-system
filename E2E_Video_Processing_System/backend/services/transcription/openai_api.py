
from fastapi import File, HTTPException, status

from pydub import AudioSegment
from openai import OpenAI
from dotenv import load_dotenv
import math
import os
import tempfile
import traceback



load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Split audio into <=25MB chunks by duration 
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
                file=audio_file,
                model="whisper-1",
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
        return response
    except Exception as e:
        error_message = f"Error transcribing chunk {chunk_path}: {str(e)}"
        print(error_message)
        traceback.print_exc()
        raise RuntimeError(error_message)

    
    

# Transcribe full audio by chunking and combining
def openai_transcribe(audio_path):
    try:
        chunks = split_audio(audio_path)
        print(f"Split into {len(chunks)} chunks.")

        full_segments = []
        current_offset = 0.0  # To adjust timestamps
        for i, chunk_path in enumerate(chunks):
            print(f"Transcribing chunk {i+1}/{len(chunks)}...")
            try:
                chunk_response = transcribe_chunk(chunk_path)
                for segment in chunk_response.segments:
                    adjusted_segment = {
                        "id": segment.id,
                        "start": segment.start + current_offset,
                        "end": segment.end + current_offset,
                        "text": segment.text
                    }
                    full_segments.append(adjusted_segment)
                current_offset += AudioSegment.from_wav(chunk_path).duration_seconds
            except RuntimeError as e:
                raise HTTPException(status_code=500, detail=str(e))
            finally:
                os.remove(chunk_path)

    except Exception as e:
        print(f"Error transcribing file {audio_path}:")
        traceback.print_exc()
        return ""
    finally:
        os.remove(audio_path)

    return full_segments
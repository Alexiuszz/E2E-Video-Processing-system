import os
import subprocess
import tempfile
import math
import traceback
import time
from dotenv import load_dotenv
from pydub import AudioSegment
from openai import OpenAI

# Load API key from .env
load_dotenv()
api_key = os.environ["API_KEY"]

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Convert video to mono, 16kHz WAV
def video_to_audio(video_path, audio_path):
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-ar", "16000", "-ac", "1", "-vn", audio_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while converting video to audio: {e}")

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
        print(f"Error transcribing chunk {chunk_path}:")
        traceback.print_exc()
        return ""

# Transcribe full audio by chunking and combining
def transcribe_large_audio(audio_path):
    start_time = time.time()
    chunks = split_audio(audio_path)
    print(f"Split into {len(chunks)} chunks.")

    full_transcript = ""
    for i, chunk_path in enumerate(chunks):
        print(f"Transcribing chunk {i+1}/{len(chunks)}...")
        transcript = transcribe_chunk(chunk_path)
        full_transcript += transcript + "\n"
        os.remove(chunk_path)

    total_time = time.time() - start_time
    print(f"Transcription took {total_time:.2f} seconds ({total_time / 60:.2f} minutes).")

    return full_transcript.strip()

# Main process
if __name__ == "__main__":
    video_file_path = "example.mp4"
    audio_file_path = "example.wav"

    # Convert video to WAV
    video_to_audio(video_file_path, audio_file_path)

    # Transcribe in safe chunks
    transcript = transcribe_large_audio(audio_file_path)

    # Show transcript
    if transcript:
        # print("Transcript:")
        word_count = len(transcript.split())
        print(f"Number of words in transcript: {word_count}")
    else:
        print("Failed to transcribe audio.")

# E2E Video Processing system

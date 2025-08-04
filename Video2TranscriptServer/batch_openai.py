import os
import time
import pandas as pd
from datetime import datetime
import tqdm
import tempfile
import math
import traceback
import dotenv
from pydub import AudioSegment
from openai import OpenAI

dotenv.load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
BASE_DIR = os.environ.get("BASE_DIR")
LOG_CSV = os.path.join(BASE_DIR, "openai_transcription_log.csv")
AUDIO_SUFFIXES = ["noisy", "denoised_nr", "noisy_DeepFilterNet2", "denoised_vad"]

client = OpenAI(api_key=OPENAI_API_KEY)

def extract_folder_info(folder_name):
    try:
        number = int(folder_name.split("-")[0])
    except:
        number = -1
    return number, folder_name

def split_audio(audio_path, chunk_length_ms=600_000):
    """Split audio into chunks under 25MB (approx. 10 minutes at 16kHz mono)."""
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

def transcribe_chunk(chunk_path):
    """Transcribe a single chunk using OpenAI Whisper API."""
    try:
        with open(chunk_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
                language="en"
            )
        return response
    except Exception:
        print(f"Error transcribing chunk {chunk_path}:")
        traceback.print_exc()
        return ""

def get_openai_transcript(audio_path):
    """Transcribe full audio file by chunking it into parts."""
    start_time = time.time()
    chunks = split_audio(audio_path)
    print(f"Split into {len(chunks)} chunks.")

    full_transcript = ""
    for i, chunk_path in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        transcript = transcribe_chunk(chunk_path)
        full_transcript += transcript + "\n"
        os.remove(chunk_path)

    duration = round(time.time() - start_time, 2)
    return full_transcript.strip(), duration

def main():
    log_entries = []
    folders = sorted([f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))])

    for folder in tqdm.tqdm(folders, desc="Getting transcripts", unit="folder"):
        folder_path = os.path.join(BASE_DIR, folder)
        folder_number, folder_name = extract_folder_info(folder)

        for suffix in AUDIO_SUFFIXES:
            audio_file = os.path.join(folder_path, f"{folder}_{suffix}.wav")
            if not os.path.exists(audio_file):
                continue

            out_file = os.path.join(folder_path, f"{folder}_{suffix}_openai.txt")
            if not os.path.exists(out_file):
                try:
                    transcript, exec_time = get_openai_transcript(audio_file)
                    with open(out_file, "w") as f:
                        f.write(transcript)

                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_entries.append({
                        "folder_number": folder_number,
                        "folder_name": folder_name,
                        "clean_type": suffix,
                        "model": "openai",
                        "input_file": os.path.basename(audio_file),
                        "output_file": os.path.basename(out_file),
                        "exec_time_sec": exec_time,
                        "timestamp": timestamp
                    })
                except Exception as e:
                    print(f"[ERROR] Failed to transcribe {audio_file} with OpenAI: {e}")

    pd.DataFrame(log_entries).to_csv(LOG_CSV, index=False)
    print(f"Transcription complete. Log saved to {LOG_CSV}")

if __name__ == "__main__":
    main()

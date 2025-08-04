import os
import time
import torch
import whisper
import pandas as pd
from datetime import datetime
import dotenv

dotenv.load_dotenv()

BASE_DIR = os.environ.get("BASE_DIR")  
LOG_CSV = os.path.join(BASE_DIR, "transcription_log.csv")

# Files to process in each folder
AUDIO_SUFFIXES = ["noisy", "denoised_nr", "noisy_DeepFilterNet2", "denoised_vad"]
WHISPER_MODELS = ["medium", "large"]


log_entries = []

def transcribe_file(model, audio_path, output_path):
    start_time = time.time()
    result = model.transcribe(audio_path, language="en", without_timestamps=True, verbose=False) 
    elapsed = round(time.time() - start_time, 2)

    with open(output_path, "w") as f:
        f.write(result["text"])

    return elapsed

def extract_folder_info(folder_name):
    try:
        number = int(folder_name.split("-")[0])
    except:
        number = -1
    return number, folder_name

def main():
    all_folders = sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))])

    for folder in all_folders:
        folder_path = os.path.join(BASE_DIR, folder)
        folder_number, folder_name = extract_folder_info(folder)

        for suffix in AUDIO_SUFFIXES:
            audio_file = os.path.join(folder_path, f"{folder}_{suffix}.wav")
            if not os.path.exists(audio_file):
                continue

            for model_size in WHISPER_MODELS:
                model = whisper.load_model(model_size, device="cuda")
                out_file = os.path.join(folder_path, f"{folder}_{suffix}_{model_size}.txt")

                try:
                    duration = transcribe_file(model, audio_file, out_file)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    log_entries.append({
                        "folder_number": folder_number,
                        "folder_name": folder_name,
                        "clean_type": suffix,
                        "model": model_size,
                        "input_file": os.path.basename(audio_file),
                        "output_file": os.path.basename(out_file),
                        "exec_time_sec": duration,
                        "timestamp": timestamp
                    })

                except Exception as e:
                    print(f"[ERROR] Failed to transcribe {audio_file} with {model_size}: {e}")


    df = pd.DataFrame(log_entries)
    df.to_csv(LOG_CSV, index=False)
    print(f"Transcription complete. Log saved to {LOG_CSV}")

if __name__ == "__main__":
    main()

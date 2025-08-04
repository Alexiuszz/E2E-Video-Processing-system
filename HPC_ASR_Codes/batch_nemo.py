import os
import time
import pandas as pd
from datetime import datetime
import torch
from nemo.collections.asr.models import ASRModel, EncDecCTCModelBPE

import dotenv

dotenv.load_dotenv()


BASE_DIR = os.environ.get("BASE_DIR")  
LOG_CSV = os.path.join(BASE_DIR, "nemo_transcription_log.csv")

AUDIO_SUFFIXES = ["noisy", "denoised_nr", "noisy_DeepFilterNet2", "denoised_vad"]

NEMO_MODELS = [
    ("parkeet","nvidia/parakeet-tdt_ctc-110m", ASRModel),
    ("fastconformer","stt_en_fastconformer_ctc_large", EncDecCTCModelBPE),
]

def extract_folder_info(folder_name):
    try:
        number = int(folder_name.split("-")[0])
    except:
        number = -1
    return number, folder_name

def transcribe_nemo(model, audio_path):
    return model.transcribe([audio_path])[0]

def main():
    log_entries = []
    folders = sorted([f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))])

    for folder in folders:
        folder_path = os.path.join(BASE_DIR, folder)
        folder_number, folder_name = extract_folder_info(folder)

        for suffix in AUDIO_SUFFIXES:
            audio_file = os.path.join(folder_path, f"{folder}_{suffix}.wav")
            if not os.path.exists(audio_file):
                continue

            for model_name, model_tag, model_class in NEMO_MODELS:
                try:
                    model = model_class.from_pretrained(model_tag)
                    model.to('cuda')
                    model.eval()

                    start = time.time()
                    transcript = transcribe_nemo(model, audio_file)
                    duration = round(time.time() - start, 2)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    out_path = os.path.join(folder_path, f"{folder}_{suffix}_{model_name}.txt")
                    with open(out_path, "w") as f:
                        f.write(transcript)

                    log_entries.append({
                        "folder_number": folder_number,
                        "folder_name": folder_name,
                        "model": model_name,
                        "input_file": os.path.basename(audio_file),
                        "output_file": os.path.basename(out_path),
                        "clean_type": suffix,
                        "exec_time_sec": duration,
                        "timestamp": timestamp
                    })

                except Exception as e:
                    print(f"[ERROR] {model_name} failed on {audio_file}: {e}")

    pd.DataFrame(log_entries).to_csv(LOG_CSV, index=False)
    print(f" NeMo transcription complete. Log saved to {LOG_CSV}")

if __name__ == "__main__":
    main()

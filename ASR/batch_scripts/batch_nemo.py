import os
import time
import pandas as pd
from datetime import datetime
import torch
import librosa
from nemo.collections.asr.models import ASRModel, EncDecCTCModelBPE
import dotenv

dotenv.load_dotenv()

BASE_DIR = os.environ.get("BASE_DIR")
LOG_CSV = os.path.join(BASE_DIR, "nemo_transcription_log.csv")
AUDIO_SUFFIXES = ["noisy", "denoised_nr", "noisy_DeepFilterNet2", "denoised_vad"]

NEMO_MODELS = [
    ("parakeet", "nvidia/parakeet-tdt_ctc-110m", ASRModel),
    ("fastconformer", "stt_en_fastconformer_ctc_large", EncDecCTCModelBPE),
]

def extract_folder_info(folder_name):
    try:
        number = int(folder_name.split("-")[0])
    except:
        number = -1
    return number, folder_name

def chunk_audio(audio_path, sr=16000, chunk_duration=30):
    audio, _ = librosa.load(audio_path, sr=sr)
    chunk_samples = chunk_duration * sr
    return [audio[i:i + chunk_samples] for i in range(0, len(audio), chunk_samples)]



def transcribe_nemo_chunked(model, audio_path):
    chunks = chunk_audio(audio_path)
    transcripts = []

    for chunk in chunks:
        text_or_hyp = model.transcribe([chunk], return_hypotheses=False)[0]
        print(f"chunk: {text_or_hyp}")
        # If still returns Hypothesis, extract .text safely
        if hasattr(text_or_hyp, 'text'):
            transcripts.append(text_or_hyp.text)
        else:
            transcripts.append(str(text_or_hyp))

    return " ".join(transcripts)

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
                    model.to("cuda")
                    model.eval()

                    start = time.time()
                    transcript = transcribe_nemo_chunked(model, audio_file)
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
    print(f"NeMo transcription complete. Log saved to {LOG_CSV}")

if __name__ == "__main__":
    main()
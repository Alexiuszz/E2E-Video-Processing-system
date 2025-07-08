import os
import time
import subprocess
import pandas as pd
import librosa
import soundfile as sf
import noisereduce as nr
import torch
from tqdm import tqdm
import dotenv
dotenv.load_dotenv()

BASE_DIR = os.environ.get("BASE_DIR", ".")

# Silero VAD model loading
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
(get_speech_timestamps, _, read_audio, _, collect_chunks) = utils


def apply_noisereduce(input_path, output_path):
    y, sr = librosa.load(input_path, sr=None)
    reduced = nr.reduce_noise(y=y, sr=sr)
    sf.write(output_path, reduced, sr)

def apply_silero_vad(input_path, output_path):
    wav = read_audio(input_path, sampling_rate=16000)
    speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=16000)
    clean = collect_chunks(speech_timestamps, wav)
    sf.write(output_path, clean, 16000)


def apply_deepfilternet(input_path, output_path):
    output_dir = os.path.dirname(output_path)
    model = "DeepFilterNet2"

    try:
        subprocess.run([
            "python", "-m", "df.enhance",
            "-m", model,
            input_path,
            "-o", output_dir
        ], check=True)

        # Rename default output 
        # basename = os.path.splitext(os.path.basename(input_path))[0]
        # default_output = os.path.join(output_dir, f"{basename}_DeepFilterNet2.wav")
        # os.rename(default_output, output_path)

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"DeepFilterNet failed: {e}")
    
def enhance_all(input_csv, output_dir):
    df = pd.read_csv(input_csv)
    logs = []

    for video_path in tqdm(df["video_path"], desc="Enhancing audio", unit="file"):
        folder = os.path.basename(os.path.dirname(video_path))
        noisy_audio = os.path.join(output_dir, folder, f"{folder}_noisy.wav")
        
        if not os.path.exists(noisy_audio):
            print(f"[SKIP] No noisy audio found for {folder}")
            continue

        audio_dir = os.path.join(output_dir, folder)
        os.makedirs(audio_dir, exist_ok=True)

        methods = {
            "denoised_nr": apply_noisereduce,
            "denoised_df2": apply_deepfilternet,
            "denoised_vad": apply_silero_vad
        }

        for suffix, method in methods.items():
            output_path = os.path.join(audio_dir, f"{folder}_{suffix}.wav")
            if os.path.exists(output_path):
                continue

            start = time.time()
            try:
                method(noisy_audio, output_path)
                duration = round(time.time() - start, 2)
                logs.append({
                    "audio_id": folder,
                    "method": suffix,
                    "output_file": output_path,
                    "time_sec": duration,
                    "status": "success"
                })
            except Exception as e:
                logs.append({
                    "audio_id": folder,
                    "method": suffix,
                    "output_file": output_path,
                    "time_sec": 0,
                    "status": f"error: {str(e)}"
                })

    pd.DataFrame(logs).to_csv(os.path.join(output_dir, "processing_log.csv"), index=False)
    print("✅ Audio enhancement completed. Logs saved.")

if __name__ == "__main__":
    input_csv = os.path.join(datasets_dir, 'video_paths.csv')
    enhance_all(input_csv, datasets_dir)
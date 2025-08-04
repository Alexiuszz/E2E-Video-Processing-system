import os
import subprocess
import pandas as pd
import noisereduce as nr
import librosa
import soundfile as sf
from tqdm import tqdm

import dotenv
dotenv.load_dotenv()
BASE_DIR = os.environ.get("BASE_DIR")

    
def video_to_audio(video_path, noisy_audio_path):
        try:
             # Extract audio
            subprocess.run([
                "ffmpeg", "-y", "-i", video_path,
                "-ar", "16000", "-ac", "1", "-vn", noisy_audio_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        except subprocess.CalledProcessError as e:
            return {"error": "Audio extraction failed", "details": str(e)}


def batch_convert(input_file, output_dir):
    
    df = pd.read_csv(input_file)
    
    for video_file in tqdm(df['video_path'], desc="Processing videos", unit="file"):
        file_name = os.path.basename(os.path.dirname(video_file))
        audio_dir = os.path.join(output_dir, file_name)
        
       
        audio_path = os.path.join(audio_dir, f"{file_name}_noisy.wav")
        
        if not os.path.exists(audio_path):
            res = video_to_audio(video_file, audio_path)
            if isinstance(res, dict) and "error" in res:
                print(f"[ERROR] {res['error']} - {res['details']}")
                continue
            

if __name__ == "__main__":
    input_csv = os.path.join(BASE_DIR, 'video_paths.csv')
    
    batch_convert(input_csv, BASE_DIR)
    print("Batch conversion completed successfully.")
    


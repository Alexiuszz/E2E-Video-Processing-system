import pandas as pd
from whisper.normalizers import EnglishTextNormalizer
from jiwer import wer
import os
from tqdm import tqdm
import dotenv

dotenv.load_dotenv()


BASE_DIR = os.environ.get("BASE_DIR")  
LOG_CSV = os.path.join(BASE_DIR, "WER_Openai.csv")

AUDIO_SUFFIXES = ["openai"]

MODELS = ["openai"]

# Load and merge all text into a single string
def load_text(path):
    if not os.path.exists(path):
        return ""
    with open(path, 'r', encoding='utf-8') as f:
        return " ".join([line.strip() for line in f if line.strip()])

# Compute WER with normalization on full text
def compute_wer(hyp_path, ref_path, normalizer):
    hyp_text = normalizer(load_text(hyp_path))
    ref_text = normalizer(load_text(ref_path))

    return wer(ref_text, hyp_text)

def extract_folder_info(folder_name):
    try:
        number = int(folder_name.split("-")[0])
    except:
        number = -1
    return number, folder_name

log_results = []
def get_WER():
    all_folders = sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))])
    
    normalizer = EnglishTextNormalizer()
    
    
    for folder in tqdm(all_folders, desc=f"Calculating WER", total=len(all_folders)):   
        folder_path = os.path.join(BASE_DIR, folder)
        folder_number, folder_name = extract_folder_info(folder)
        
        ref_file  = os.path.join(folder_path, f"reference.txt") if os.path.exists(os.path.join(folder_path, "reference.txt")) else os.path.join(folder_path, "references.txt")
        
        for suffix in AUDIO_SUFFIXES:
            for model_name in MODELS:
                hyp_file = os.path.join(folder_path, f"{folder}_{suffix}.txt")
                if not os.path.exists(hyp_file):
                    continue

                wer_value = compute_wer(hyp_file, ref_file, normalizer)
                
                log_results.append({
                    "folder_number": folder_number,
                    "folder_name": folder_name,
                    "audio_suffix": suffix,
                    "model_name": model_name,
                    "WER (%)": round(wer_value * 100, 2) if wer_value is not None else "ERROR",
                })
    pd.DataFrame(log_results).to_csv(LOG_CSV, index=False)
    print(f"WER evaluation complete. Saved to: {LOG_CSV}")
if __name__ == "__main__":
    get_WER()
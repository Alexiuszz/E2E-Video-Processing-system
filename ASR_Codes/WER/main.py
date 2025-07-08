import pandas as pd
from whisper.normalizers import EnglishTextNormalizer
from jiwer import wer
import os
from tqdm import tqdm
import dotenv

dotenv.load_dotenv()


BASE_DIR = os.environ.get("BASE_DIR")  

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


def get_WER(dir):
    input_csv = os.path.join(BASE_DIR, 'transcript_paths.csv')
    output_csv = os.path.join(dir, "wer_results.csv")
    
    normalizer = EnglishTextNormalizer()
    df = pd.read_csv(input_csv)
    normalizer = EnglishTextNormalizer()

    results = []

    for _, row in tqdm(df.iterrows(), desc="Calculating WER", total=len(df)):
        audio_id = row["audio_id"]
        ref = row["ref_file"]
        h1 = row["small_clean_file"]
        

        wer1 = compute_wer(h1, ref, normalizer)

        results.append({
            "audio_id": audio_id,
            "WER_Small_Clean (%)": round(wer1 * 100, 2) if wer1 is not None else "ERROR",\
        })

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"WER evaluation complete. Saved to: {output_csv}")

if __name__ == "__main__":
    get_WER(BASE_DIR)

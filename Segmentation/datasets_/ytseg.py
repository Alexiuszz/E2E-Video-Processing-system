import json
import os
import sys
import statistics
from pathlib import Path
from typing import List, Tuple, Dict, Any
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
# from segeval import window_diff, pk
# import pandas as pd

from dotenv import load_dotenv
load_dotenv()

# Add the path to your process_transcript function
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / 'E2E_Video_Processing_System' / 'backend'))

# from services.topic_segment import process_transcript

# Configuration
BASE_DIR = os.getenv('BASE_DIR', '.')
YTSEG_DATA_DIR = os.path.join(BASE_DIR, 'YTSEG_data', "clean")

class YTSegDataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
    #     self.ensure_nltk_data()
    
    # def ensure_nltk_data(self):
    #     """Ensure required NLTK data is downloaded."""
    #     try:
    #         nltk.data.find('tokenizers/punkt')
    #     except LookupError:
    #         nltk.download('punkt')
    
    def load_ytseg_data(self) -> List[Dict[str, Any]]:
        """
        Load YTSeg dataset. Supports multiple formats:
        1. JSON lines format (.jsonl)
        2. Single JSON file with array
        3. Multiple JSON files
        
        Expected format for each sample:
        {
            "id": "video_id",
            "text": "full transcript text",
            "sentences": ["sentence 1", "sentence 2", ...],
            "segments": [
                {
                    "start_sentence": 0,
                    "end_sentence": 5,
                    "topic": "topic_name"
                },
                ...
            ]
        }
        """
        data = []
        
        # Try to find data files
        json_files = list(self.data_dir.glob("*.json"))
        jsonl_files = list(self.data_dir.glob("*.jsonl"))
        
        if jsonl_files:
            # Load JSONL format
            for jsonl_file in jsonl_files:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
        elif json_files:
            # Load JSON format
            for json_file in json_files:
                with open(json_file, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                    if isinstance(file_data, list):
                        data.extend(file_data)
                    else:
                        data.append(file_data)
        else:
            raise FileNotFoundError(f"No JSON or JSONL files found in {self.data_dir}")
        
        print(f"Loaded {len(data)} samples from YTSeg dataset")
        return data
    
    def extract_gold_boundaries(self, sample: Dict[str, Any]) -> List[int]:
        """
        Extract gold standard sentence-level boundaries from YTSeg sample.
        Returns list of sentence indices where new segments begin.
        """
        if "segments" not in sample:
            raise KeyError("No 'segments' field found in sample")
        
        segments = sample["segments"]
        boundaries = []
        
        for segment in segments:
            if "start_sentence" in segment:
                start_sent = segment["start_sentence"]
                if start_sent > 0:  # Don't include the first sentence as a boundary
                    boundaries.append(start_sent)
        
        return sorted(set(boundaries))
    
    def convert_sentence_to_word_boundaries(self, sent_boundaries: List[int], 
                                         sentences: List[str]) -> List[int]:
        """
        Convert sentence-level boundaries to word-level boundaries.
        """
        word_boundaries = []
        word_count = 0
        
        for sent_idx in range(len(sentences)):
            if sent_idx in sent_boundaries:
                word_boundaries.append(word_count)
            
            # Count words in current sentence
            sentence_words = word_tokenize(sentences[sent_idx])
            word_count += len(sentence_words)
        
        return word_boundaries
    
    
    def extract_data(self, test_size: int = None):
        data = self.load_ytseg_data()
        
        if test_size:
            data = data[:test_size]
            print(f"Evaluating on {len(data)} samples (test size limit).")
        else:
            print(f"Evaluating on {len(data)} samples.")
            
        meeting_data = []
        for i, sample in enumerate(data):
            if "text" in sample:
                full_text = " ".join(sample["text"])
                sentences = sent_tokenize(full_text)
            elif "sentences" in sample:
                sentences = sample["sentences"]
                full_text = " ".join(sentences)
            else:
                raise KeyError("No 'text' or 'sentences' field found")
            if(len(sentences)> 60):
                gold_sent_bounds = self.extract_gold_boundaries(sample)
                gold_word_bounds = self.convert_sentence_to_word_boundaries(
                    gold_sent_bounds, sentences)

                if len(gold_word_bounds) > 0:
                    meeting_data.append({
                        "transcript": full_text,
                        "all_words": [],
                        "gold_word_bounds": gold_word_bounds,
                    })
                else:
                    print(f"Skipping {sample}: No gold word bounds found.")
                    continue
            else:
                print(f"Skipping {sample['id']}: Less than 60 sentences.")
        return meeting_data
# def main():
#     """Main evaluation function."""
#     # Initialize evaluator
#     evaluator = YTSegEvaluator(YTSEG_DATA_DIR)
    
#     # Run evaluation
#     # For testing, limit to first 5 samples
#     results = evaluator.evaluate_dataset(limit=5)
    
#     # Save results to file
#     # if results:
#     #     results_file = Path(YTSEG_DATA_DIR) / "evaluation_results.json"
#     #     with open(results_file, 'w') as f:
#     #         json.dump(results, f, indent=2)
#     #     print(f"\nResults saved to {results_file}")

# # if __name__ == "__main__":
# #     main()
    

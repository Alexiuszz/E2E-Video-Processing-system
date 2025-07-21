import json
import os
import sys
import statistics
from pathlib import Path
from typing import List, Tuple, Dict, Any
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from segeval import window_diff, pk
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

# Add the path to your process_transcript function
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / 'E2E_Video_Processing_System' / 'backend'))

from services.topic_segment import process_transcript

# Configuration
BASE_DIR = os.getenv('BASE_DIR', '.')
YTSEG_DATA_DIR = os.path.join(BASE_DIR, 'YTSEG_data', "clean")

class YTSegEvaluator:
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
    
    def convert_predicted_segments_to_word_boundaries(self, pred_segments: List[List[str]], 
                                                    full_text: str) -> List[int]:
        """
        Convert predicted segments to word-level boundaries.
        """
        word_boundaries = []
        word_count = 0
        
        for seg_idx, segment in enumerate(pred_segments):
            if seg_idx > 0:  # Skip first segment
                word_boundaries.append(word_count)
            
            # Count words in current segment
            for sentence in segment:
                sentence_words = word_tokenize(sentence)
                word_count += len(sentence_words)
        
        return word_boundaries
    
    def masses_from_bounds(self, bounds: List[int], total_len: int) -> Tuple[int, ...]:
        """Convert boundary indices to segment lengths (masses)."""
        if not bounds:
            return (total_len,)
        
        # Add 0 at start and total_len at end if not present
        full_bounds = [0] + [b for b in bounds if 0 < b < total_len] + [total_len]
        full_bounds = sorted(set(full_bounds))  # Remove duplicates and sort
        
        masses = []
        for i in range(len(full_bounds) - 1):
            masses.append(full_bounds[i + 1] - full_bounds[i])
        
        return tuple(masses)
    
    def evaluate_segmentation(self, gold_bounds: List[int], pred_bounds: List[int], 
                            total_words: int) -> Tuple[float, float]:
        """
        Evaluate segmentation using WindowDiff and Pk metrics.
        """
        ref_masses = self.masses_from_bounds(gold_bounds, total_words)
        hyp_masses = self.masses_from_bounds(pred_bounds, total_words)
        
        # Debug output
        print(f"    Gold segments: {len(ref_masses)}, lengths: {ref_masses[:5]}...")
        print(f"    Pred segments: {len(hyp_masses)}, lengths: {hyp_masses[:5]}...")
        
        try:
            wd_score = window_diff(ref_masses, hyp_masses)
            pk_score = pk(ref_masses, hyp_masses)
            return wd_score, pk_score
        except Exception as e:
            print(f"    Error computing metrics: {e}")
            return float('nan'), float('nan')
    
    def process_sample(self, sample: Dict[str, Any]) -> Tuple[float, float]:
        """
        Process a single YTSeg sample and return WD and Pk scores.
        """
        sample_id = sample.get("id", "unknown")
        print(f"\n=== Processing {sample_id} ===")
        
        # try:
            # Get text and sentences
        if "text" in sample:
            full_text = " ".join(sample["text"])
            sentences = sent_tokenize(full_text)
        elif "sentences" in sample:
            sentences = " ".join(sample["sentences"])
            full_text = " ".join(sentences)
        else:
            raise KeyError("No 'text' or 'sentences' field found")
        
        if(len(sentences)> 60):
            # Count total words
            total_words = len(word_tokenize(full_text))
            
            # Extract gold boundaries
            gold_sent_bounds = self.extract_gold_boundaries(sample)
            gold_word_bounds = self.convert_sentence_to_word_boundaries(
                gold_sent_bounds, sentences)
            
            print(f"  Total words: {total_words}")
            print(f"  Total sentences: {len(sentences)}")
            print(f"  Gold sentence boundaries: {len(gold_sent_bounds)}")
            print(f"  Gold word boundaries: {len(gold_word_bounds)}")
            
            # Get predictions using process_transcript
            formatted_transcript = {"text": full_text}
            pred_segments = process_transcript(
                formatted_transcript, 
                with_timestamps=False, 
                label=False, 
                use_tiling=True, 
                verbose=False
            )
            
            # Convert predictions to word boundaries
            pred_word_bounds = self.convert_predicted_segments_to_word_boundaries(
                pred_segments, full_text)
            
            print(f"  Predicted segments: {len(pred_segments)}")
            print(f"  Predicted word boundaries: {len(pred_word_bounds)}")
            
            # Evaluate
            wd_score, pk_score = self.evaluate_segmentation(
                gold_word_bounds, pred_word_bounds, total_words)
            
            if not (wd_score != wd_score or pk_score != pk_score):  # Check for NaN
                print(f"  WindowDiff: {wd_score:.3f}")
                print(f"  Pk: {pk_score:.3f}")
                
                # Additional statistics
                if gold_word_bounds:
                    avg_gold_len = total_words / (len(gold_word_bounds) + 1)
                else:
                    avg_gold_len = total_words
                
                if pred_word_bounds:
                    avg_pred_len = total_words / (len(pred_word_bounds) + 1)
                else:
                    avg_pred_len = total_words
                
                print(f"  Avg gold segment length: {avg_gold_len:.1f}")
                print(f"  Avg pred segment length: {avg_pred_len:.1f}")
                
                return wd_score, pk_score
            else:
                print(f"  Invalid scores (NaN)")
                return float('nan'), float('nan')
                    
            # except Exception as e:
            #     print(f"  Error processing sample: {e}")
            #     return float('nan'), float('nan')
        else:
                print(f"  Less than 60 sentences")
                return float('nan'), float('nan')
        
    def evaluate_dataset(self, limit: int = None) -> Dict[str, float]:
        """
        Evaluate the entire YTSeg dataset.
        
        Args:
            limit: Maximum number of samples to evaluate (for testing)
        
        Returns:
            Dictionary with evaluation metrics
        """
        data = self.load_ytseg_data()
        
        if limit:
            data = data[:limit]
            print(f"Limiting evaluation to {limit} samples")
        
        wd_scores = []
        pk_scores = []
        
        for i, sample in enumerate(data):
            wd_score, pk_score = self.process_sample(sample)
            
            if not (wd_score != wd_score or pk_score != pk_score):  # Check for NaN
                wd_scores.append(wd_score)
                pk_scores.append(pk_score)
        
        # Calculate final metrics
        if wd_scores and pk_scores:
            results = {
                'mean_window_diff': statistics.mean(wd_scores),
                'std_window_diff': statistics.stdev(wd_scores) if len(wd_scores) > 1 else 0.0,
                'mean_pk': statistics.mean(pk_scores),
                'std_pk': statistics.stdev(pk_scores) if len(pk_scores) > 1 else 0.0,
                'valid_samples': len(wd_scores),
                'total_samples': len(data)
            }
            
            print(f'\n=== FINAL RESULTS ===')
            print(f'Valid samples: {results["valid_samples"]}/{results["total_samples"]}')
            print(f'Mean WindowDiff: {results["mean_window_diff"]:.3f} (±{results["std_window_diff"]:.3f})')
            print(f'Mean Pk: {results["mean_pk"]:.3f} (±{results["std_pk"]:.3f})')
            
            return results
        else:
            print("No valid scores computed!")
            return {}

def main():
    """Main evaluation function."""
    # Initialize evaluator
    evaluator = YTSegEvaluator(YTSEG_DATA_DIR)
    
    # Run evaluation
    # For testing, limit to first 5 samples
    results = evaluator.evaluate_dataset(limit=5)
    
    # Save results to file
    # if results:
    #     results_file = Path(YTSEG_DATA_DIR) / "evaluation_results.json"
    #     with open(results_file, 'w') as f:
    #         json.dump(results, f, indent=2)
    #     print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main()
    

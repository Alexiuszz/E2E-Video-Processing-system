import json
import os
from pathlib import Path
from typing import List, Dict, Any
import nltk
from nltk.tokenize import sent_tokenize

class YTSegDataPreparator:
    """
    Helper class to prepare YTSeg dataset for evaluation.
    Handles different input formats and converts them to a standardized format.
    """
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Ensure NLTK data is available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
    def string_to_binary_array(self, s):
        """
        Convert a string like "|=00100000010000000000000000000010000100000000000000000000000000001000000000000000000010000000000000000000000000000001000000"
        to a binary array [0,0,1,0,0,0,0,0,0,1,0,...]
        
        Args:
            s (str): Input string containing binary digits, possibly with prefix characters
        
        Returns:
            list: List of integers (0s and 1s)
        """
        # Remove non-binary characters (anything that's not '0' or '1')
        binary_part = ''.join(char for char in s if char in '01')
        
        # Convert each character to integer
        return [int(digit) for digit in binary_part]
    
    def convert_wiki727k_format(self, input_file: str) -> List[Dict[str, Any]]:
        """
        Convert Wiki-727K format to standardized format.
        Expected input format:
        {
            "id": "doc_id",
            "text": "full document text",
            "targets": [0, 0, 1, 0, 1, ...]  # 1 indicates segment boundary
        }
        """
        standardized_data = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            if isinstance(data, dict):
                data = [data]
            
            for sample in data:
                doc_id = sample.get("video_id", "unknown")
                text = sample.get("text", "")
                targets = self.string_to_binary_array(sample.get("targets", ""))
                
                # Tokenize into sentences
                sentences = sent_tokenize(" ".join(text))
                
                # Convert targets to segment boundaries
                segments = []
                current_start = 0
                
                for i, target in enumerate(targets):
                    if target == 1 or i == len(targets) - 1:
                        # End of current segment
                        segments.append({
                            "start_sentence": current_start,
                            "end_sentence": i,
                            "topic": f"topic_{len(segments)}"
                        })
                        current_start = i + 1
                
                standardized_sample = {
                    "id": doc_id,
                    "text": text,
                    "sentences": sentences,
                    "segments": segments
                }
                
                standardized_data.append(standardized_sample)
        
        return standardized_data
    
    def convert_choi_format(self, input_file: str) -> List[Dict[str, Any]]:
        """
        Convert Choi dataset format to standardized format.
        Expected input: Plain text with segment boundaries marked by "=========="
        """
        standardized_data = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by document separator (if multiple documents)
        documents = content.split('\n\n========== DOCUMENT SEPARATOR ==========\n\n')
        
        for doc_idx, doc_content in enumerate(documents):
            if not doc_content.strip():
                continue
            
            # Split by segment boundaries
            segments_text = doc_content.split('\n==========\n')
            
            # Clean up segments
            segments_text = [seg.strip() for seg in segments_text if seg.strip()]
            
            # Combine all text
            full_text = ' '.join(segments_text)
            sentences = sent_tokenize(full_text)
            
            # Create segments
            segments = []
            sent_idx = 0
            
            for seg_idx, seg_text in enumerate(segments_text):
                seg_sentences = sent_tokenize(seg_text)
                start_sent = sent_idx
                end_sent = sent_idx + len(seg_sentences) - 1
                
                segments.append({
                    "start_sentence": start_sent,
                    "end_sentence": end_sent,
                    "topic": f"topic_{seg_idx}"
                })
                
                sent_idx += len(seg_sentences)
            
            standardized_sample = {
                "id": f"choi_doc_{doc_idx}",
                "text": full_text,
                "sentences": sentences,
                "segments": segments
            }
            
            standardized_data.append(standardized_sample)
        
        return standardized_data
    
    def convert_ytseg_original_format(self, input_file: str) -> List[Dict[str, Any]]:
        """
        Convert original YTSeg format to standardized format.
        This is a placeholder - adjust based on actual YTSeg format.
        """
        standardized_data = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            if input_file.endswith('.jsonl'):
                # JSONL format
                for line in f:
                    if line.strip():
                        sample = json.loads(line)
                        standardized_data.append(self._process_ytseg_sample(sample))
            else:
                # JSON format
                data = json.load(f)
                if isinstance(data, list):
                    for sample in data:
                        standardized_data.append(self._process_ytseg_sample(sample))
                else:
                    standardized_data.append(self._process_ytseg_sample(data))
        
        return standardized_data
    
    def _process_ytseg_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single YTSeg sample to standardized format.
        Adjust this method based on actual YTSeg format.
        """
        # Example processing - adjust based on actual format
        video_id = sample.get("video_id", sample.get("id", "unknown"))
        
        # Get transcript text
        if "transcript" in sample:
            text = sample["transcript"]
        elif "text" in sample:
            text = sample["text"]
        else:
            # If sentences are provided separately
            sentences = sample.get("sentences", [])
            text = " ".join(sentences)
        
        # Tokenize if not already done
        if "sentences" not in sample:
            sentences = sent_tokenize(text)
        else:
            sentences = sample["sentences"]
        
        # Process segments/chapters
        segments = []
        if "segments" in sample:
            segments = sample["segments"]
        elif "chapters" in sample:
            # Convert chapters to segments
            for i, chapter in enumerate(sample["chapters"]):
                start_time = chapter.get("start_time", 0)
                end_time = chapter.get("end_time", 0)
                title = chapter.get("title", f"Chapter {i+1}")
                
                # Convert time-based to sentence-based (simplified)
                # This would need more sophisticated alignment in practice
                start_sent = int(start_time / 10)  # Rough approximation
                end_sent = int(end_time / 10)
                
                segments.append({
                    "start_sentence": start_sent,
                    "end_sentence": end_sent,
                    "topic": title
                })
        
        # Ensure segments are valid
        if not segments:
            # Create a single segment for the entire text
            segments = [{
                "start_sentence": 0,
                "end_sentence": len(sentences) - 1,
                "topic": "single_segment"
            }]
        
        return {
            "id": video_id,
            "text": text,
            "sentences": sentences,
            "segments": segments
        }
    
    def prepare_dataset(self, input_format: str = "auto") -> None:
        """
        Prepare YTSeg dataset for evaluation.
        
        Args:
            input_format: Format of input data ("auto", "wiki727k", "choi", "ytseg")
        """
        input_files = list(self.input_dir.glob("*.json")) + list(self.input_dir.glob("*.jsonl"))
        
        if not input_files:
            raise FileNotFoundError(f"No JSON/JSONL files found in {self.input_dir}")
        
        all_data = []
        
        for input_file in input_files:
            print(f"Processing {input_file.name}...")
            
            if input_format == "auto":
                # Try to detect format
                format_detected = self._detect_format(input_file)
                print(f"  Detected format: {format_detected}")
            else:
                format_detected = input_format
            
            if format_detected == "wiki727k":
                data = self.convert_wiki727k_format(input_file)
            elif format_detected == "choi":
                data = self.convert_choi_format(input_file)
            elif format_detected == "ytseg":
                data = self.convert_ytseg_original_format(input_file)
            else:
                print(f"  Unknown format, trying YTSeg format...")
                data = self.convert_ytseg_original_format(input_file)
            
            all_data.extend(data)
            print(f"  Processed {len(data)} samples")
        
        # Save prepared data
        output_file = self.output_dir / "ytseg_prepared.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in all_data:
                f.write(json.dumps(sample) + '\n')
        
        print(f"\nPrepared {len(all_data)} samples")
        print(f"Output saved to: {output_file}")
        
        # Create a sample file for inspection
        sample_file = self.output_dir / "sample_data.json"
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(all_data[:3], f, indent=2)
        
        print(f"Sample data saved to: {sample_file}")
    
    def _detect_format(self, input_file: Path) -> str:
        """
        Detect the format of the input file.
        """
        with open(input_file, 'r', encoding='utf-8') as f:
            if input_file.suffix == '.jsonl':
                # Read first line
                first_line = f.readline()
                if first_line.strip():
                    sample = json.loads(first_line)
                else:
                    return "unknown"
            else:
                sample = json.load(f)
                if isinstance(sample, list) and sample:
                    sample = sample[0]
        
        # Check for specific format indicators
        if "targets" in sample:
            return "wiki727k"
        elif "video_id" in sample or "transcript" in sample:
            return "ytseg"
        else:
            return "ytseg"  # Default to YTSeg format

def main():
    """Main preparation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare YTSeg dataset for evaluation")
    parser.add_argument("--input_dir", required=True, help="Input directory containing raw data")
    parser.add_argument("--output_dir", required=True, help="Output directory for prepared data")
    parser.add_argument("--format", default="auto", choices=["auto", "wiki727k", "choi", "ytseg"],
                       help="Input data format")
    
    args = parser.parse_args()
    
    preparator = YTSegDataPreparator(args.input_dir, args.output_dir)
    preparator.prepare_dataset(args.format)

if __name__ == "__main__":
    main()
    
    
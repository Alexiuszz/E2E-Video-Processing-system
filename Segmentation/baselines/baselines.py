from nltk.tokenize import sent_tokenize
import numpy as np

class RandomSeg:
    def __init__(self, transcript):
        if "text" in transcript:
            full_text = transcript["text"].strip()
        else:
            raise ValueError("Transcript must have 'text' or 'segments' key")

        self.sentences = sent_tokenize(full_text)
    
    def segment(self, random_threshold=0.015):
        """
        Randomly segments the transcript into sentences based on a random threshold.
        """
        segments = []
        current_segment = []
        np.random.seed(42)
        for sentence in self.sentences:
            # Randomly decide whether to start a new segment or continue the current one with seed 42
              # Set seed for reproducibility
            if np.random.rand() < random_threshold:
                # Check if there is a current segment to append before finishing segmentation
                if current_segment:
                    segments.append(current_segment)
                current_segment = [sentence]
            else:
                current_segment.append(sentence)
        
        if current_segment:
            segments.append(current_segment)
        
        return segments
    
class EvenSeg:
    def __init__(self, transcript, num_segments=8):
        if "text" in transcript:
            full_text = transcript["text"].strip()
        else:
            raise ValueError("Transcript must have 'text' or 'segments' key")

        self.sentences = sent_tokenize(full_text)
        self.num_segments = num_segments
    
    def segment(self):
        """
        Segments the transcript into a fixed number of segments.

        Example:
        Input: transcript = {"text": "This is sentence one. This is sentence two. This is sentence three."}, num_segments=2
        Output: [['This is sentence one.'], ['This is sentence two.', 'This is sentence three.']]
        """
        segment_length = len(self.sentences) // self.num_segments
        segments = []
        
        for i in range(self.num_segments):
            start_idx = i * segment_length
            end_idx = (i + 1) * segment_length if i < self.num_segments - 1 else len(self.sentences)
            segments.append(self.sentences[start_idx:end_idx])
        
        return segments
        
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import glob, statistics
import sys
from pathlib import Path, PurePath
from lxml import etree
from segeval import window_diff, pk
import os
import re

from typing import List, Tuple

from dotenv import load_dotenv
load_dotenv()

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / 'E2E_Video_Processing_System' / 'backend'))

from services.topic_segment import process_transcript
BASE_DIR = os.getenv('BASE_DIR')

ami_root = Path(BASE_DIR) / 'ami_public_manual_1.6.2'

def load_transcript(ami_root: Path, meeting_id: str):
    """Return (transcript_text, all_words, wid2idx) for one AMI meeting."""
    words_dir = ami_root / "words"
    xml_paths = glob.glob(str(words_dir / f"{meeting_id}.*.words.xml"))
    if not xml_paths:
        raise FileNotFoundError(f"No *.words.xml files for {meeting_id}")

    all_words = []
    for xp in xml_paths:
        spk = PurePath(xp).stem.split(".")[1]         # "A", "B", …
        tree = etree.parse(xp)

        # NITE namespace URI (usually 'http://nite.sourceforge.net/')
        nite_ns = tree.getroot().nsmap.get('nite')
        id_key  = f"{{{nite_ns}}}id" if nite_ns else "id"   # prefer namespaced

        for w in tree.xpath("//w"):
            wid  = w.get(id_key) or w.get("id")           # <- works for both
            txt  = (w.text or "").strip()
            punc = w.get("punc", "").strip() # punctuation, if any
            st, et = w.get("starttime"), w.get("endtime")

            # skip blanks / missing timestamps
            if not wid or not txt or not st or not et:
                continue

            all_words.append({
                "start": float(st),
                "end"  : float(et),
                "text" : txt,
                "spk"  : spk,
                "wid"  : wid,
                "punc" : True if punc else False,
            })

    all_words.sort(key=lambda d: d["start"])
    transcript_text = ""
    # Build the full transcript text from all words
    # and punctuation, in order.
    for w in all_words:
        if not w["punc"]:
            #append word
            transcript_text += " " + w["text"]
        else:
            #append punctuation
            transcript_text += w["text"]

    wid2idx = {w["wid"]: i for i, w in enumerate(all_words)}
    return transcript_text, all_words, wid2idx

def load_reference_segments(xml_path: Path, wid2idx: dict) -> list[int]:
    """
    Extract topic boundaries from an AMI *.topic.xml file.
    Returns: sorted list of word indices where a new topic begins.
    """
    tree   = etree.parse(str(xml_path))
    topics = tree.xpath('//*[local-name()="topic"]')  # ignore namespace

    gold_word_bounds = []
    for tp in topics:
        # first <*:child> (ignore prefix)
        child = tp.xpath('./*[local-name()="child"]')
        if not child:
            continue
        href = child[0].get('href', '')
        match = re.search(r'id\(([^)]+)\)', href)
        if not match:
            continue
        wid       = match.group(1)                 # ES2002a.B.words584
        word_idx = wid2idx.get(wid)
        if word_idx is not None:
            gold_word_bounds.append(word_idx)
    return sorted(set(gold_word_bounds))

def build_word_to_sentence_map(transcript_text: str) -> List[int]:
    """
    Returns a list where index i gives the sentence number for word i.
    """
    sentences = sent_tokenize(transcript_text)
    word_to_sent = []
    
    for sent_idx, sent in enumerate(sentences):
        words = word_tokenize(sent)
        word_to_sent.extend([sent_idx] * len(words))
    
    return word_to_sent

def convert_predicted_segments_to_word_boundaries(pred_segments: List[List[str]], 
                                                transcript_text: str) -> List[int]:
    """
    Convert predicted segments to word-level boundaries.
    
    Args:
        pred_segments: List of segments, where each segment is a list of sentences
        transcript_text: The full transcript text
    
    Returns:
        List of word indices where new segments begin (excluding index 0)
    """
    # Tokenize the full transcript to get total word count
    full_words = word_tokenize(transcript_text)
    
    # Method 1: Word-count based alignment (more robust)
    word_boundaries = []
    word_ptr = 0
    
    for seg_idx, seg in enumerate(pred_segments):
        if seg_idx == 0:
            # First segment starts at word 0, don't include in boundaries
            pass
        else:
            # This is where a new segment begins
            word_boundaries.append(word_ptr)
        
        # Count words in this segment
        seg_word_count = 0
        for sentence in seg:
            seg_word_count += len(word_tokenize(sentence))
        
        word_ptr += seg_word_count
    
    # Validate that we didn't go past the total word count
    if word_ptr > len(full_words):
        print(f"Warning: Predicted word count ({word_ptr}) exceeds actual ({len(full_words)})")
    
    return word_boundaries

def convert_predicted_segments_to_word_boundaries_v2(pred_segments: List[List[str]], 
                                                   transcript_text: str) -> List[int]:
    """
    Alternative method: Use sentence-level mapping.
    """
    sentences = sent_tokenize(transcript_text)
    word_to_sent = build_word_to_sentence_map(transcript_text)
    
    # Find sentence boundaries for predicted segments
    sent_boundaries = []
    sent_ptr = 0
    
    for seg_idx, seg in enumerate(pred_segments):
        if seg_idx > 0:  # Skip first segment
            sent_boundaries.append(sent_ptr)
        sent_ptr += len(seg)
    
    # Convert sentence boundaries to word boundaries
    word_boundaries = []
    for sent_idx in sent_boundaries:
        if sent_idx < len(sentences):
            # Find first word that belongs to this sentence
            try:
                word_idx = word_to_sent.index(sent_idx)
                word_boundaries.append(word_idx)
            except ValueError:
                print(f"Warning: Could not find word for sentence {sent_idx}")
    
    return word_boundaries

def masses_from_bounds(bounds: List[int], total_len: int) -> Tuple[int, ...]:
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

def evaluate_segmentation(gold_bounds: List[int], pred_bounds: List[int], 
                         total_words: int) -> Tuple[float, float]:
    """
    Evaluate segmentation using WindowDiff and Pk metrics.
    
    Args:
        gold_bounds: Gold standard boundary word indices
        pred_bounds: Predicted boundary word indices  
        total_words: Total number of words in transcript
    
    Returns:
        (window_diff_score, pk_score)
    """
    ref_masses = masses_from_bounds(gold_bounds, total_words)
    hyp_masses = masses_from_bounds(pred_bounds, total_words)
    
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

# Main evaluation loop
def main():
    words_dir  = ami_root / "words"
    topics_dir = ami_root / "topics"

    # Use only meetings that have topic annotations
    topic_files   = sorted(topics_dir.glob("*.topic.xml"))
    meeting_ids   = [p.stem.split(".")[0] for p in topic_files]
    print(f"{len(meeting_ids)} meetings with topic annotations found.")
    
    scores = []
    pk_scores = []
    
    for num, mid in enumerate(meeting_ids):
        print(f"\n=== Processing {mid} ===")
        
        try:
            # 1. Load transcript and reference data
            transcript_text, all_words, wid2idx = load_transcript(ami_root, mid)
            n_words = len(all_words)
            
            topic_xml = topics_dir / f"{mid}.topic.xml"
            gold_word_bounds = load_reference_segments(topic_xml, wid2idx)
            
            print(f"  Words: {n_words}")
            print(f"  Gold boundaries: {len(gold_word_bounds)}")
            
            # 2. Get predictions
            formatted_transcript = {"text": transcript_text}
            pred_segments = process_transcript(formatted_transcript, 
                                             with_timestamps=False, 
                                             label=False, 
                                             use_tiling=True, 
                                             verbose=False)
            
            # 3. Convert predictions to word boundaries
            # Try method 1 first (word-count based)
            pred_word_bounds = convert_predicted_segments_to_word_boundaries(
                pred_segments, transcript_text)
            
            print(f"  Predicted boundaries: {len(pred_word_bounds)}")
            
            # 4. Evaluate
            wd_score, pk_score = evaluate_segmentation(
                gold_word_bounds, pred_word_bounds, n_words)
            
            if not (wd_score != wd_score or pk_score != pk_score):  # Check for NaN
                scores.append(wd_score)
                pk_scores.append(pk_score)
                
                print(f"  WindowDiff: {wd_score:.3f}")
                print(f"  Pk: {pk_score:.3f}")
                
                # Additional statistics
                avg_gold_len = n_words / (len(gold_word_bounds) + 1)
                avg_pred_len = n_words / (len(pred_word_bounds) + 1)
                print(f"  Avg gold segment length: {avg_gold_len:.1f}")
                print(f"  Avg pred segment length: {avg_pred_len:.1f}")
            
        except Exception as e:
            print(f"  Error processing {mid}: {e}")
            continue
        
        # Limit to first 5 meetings for testing
        # if num >= 4:  # 0-indexed, so 4 means 5 meetings
        #     break
    
    # Final results
    if scores:
        print(f'\n=== FINAL RESULTS ===')
        print(f'Mean WindowDiff over {len(scores)} meetings: {statistics.mean(scores):.3f}')
        print(f'Mean Pk over {len(pk_scores)} meetings: {statistics.mean(pk_scores):.3f}')
        print(f'Std WindowDiff: {statistics.stdev(scores):.3f}')
        print(f'Std Pk: {statistics.stdev(pk_scores):.3f}')
    else:
        print("No valid scores computed!")

if __name__ == "__main__":
    main()
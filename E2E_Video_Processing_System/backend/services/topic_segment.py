import os
from utils.text_processing import get_segment_bounds
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.cluster import KMeans
from bertopic.representation import KeyBERTInspired
from bertopic.representation import OpenAI
from bertopic.dimensionality import BaseDimensionalityReduction
from umap import UMAP
# from hdbscan import HDBSCAN
import numpy as np
import openai
from bertopic import BERTopic
from transformers import AutoModel

from typing import List, Tuple, Optional
import logging
from sklearn.metrics.pairwise import cosine_similarity
# from bertopic.backend import OpenAIBackend

from keybert import KeyBERT
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
ApiKey = os.environ.get("API_KEY")

nltk.download("punkt_tab")

# --------- Configuration ---------
model_path = os.environ.get("MODEL_PATH", "SentenceTransformer/all-MiniLM-L6-v2")

openai.api_key = ApiKey

def extract_transcript(transcript, with_timestamps = True) -> str:
    """
    Extracts the transcript text from the JSON format.
    
    Args:
        transcript: str or dict: The transcript text or JSON object containing segments.
        with_timestamps (bool): Whether the input transcript has timestamps.
    Raises:
        ValueError: If the transcript format is invalid.        
    Returns:
        str: The extracted transcript text.
    """
    try:
        if with_timestamps:
            texts = " ".join([segment['text'].strip() for segment in transcript["segments"]])
            
        else:
            texts = transcript["text"].strip()
    except (KeyError, AttributeError, TypeError) as e:
        # segments = transcript["segments"]
        raise ValueError(f"Invalid transcript format. Expected raw text or a dictionary with 'segments'\n {str(e)}.")
    
    # Use NLTK to split the transcript into sentences
    return sent_tokenize(texts)


# Encode using Sentence-BERT
def get_embeddings(sentences,model, model_path, verbose=True):
    """
    Encodes sentences into embeddings using Sentence-BERT.
    Args:
        sentences (list): List of sentences to encode.
        model_path (str): Path to the Sentence-BERT model.
    Raises:
        ValueError: If the number of sentences is less than 60.
    Raises:
        ValueError: If no sentences are provided for embedding.
    Returns:
        tuple: Tuple containing the embeddings and the Sentence-BERT model.
    """
    if not sentences:
        raise ValueError("No sentences provided for embedding.")
    if len(sentences) < 60:
        raise ValueError("At least 60 sentences are required for Segmentation.")
    if not model:
        model = SentenceTransformer(model_path)
    embeddings = model.encode(sentences, show_progress_bar=verbose)
    return embeddings, model

def calculate_depth_scores(similarities: np.ndarray, 
                          window_size: int = 3,
                          smoothing_factor: float = 0.1,
                          use_adaptive_window: bool = True) -> List[Tuple[int, float]]:
    """
    Calculate depth scores for segment boundaries with improved robustness.
    
    Args:
        similarities: Array of similarity scores between consecutive elements
        window_size: Base window size for context calculation
        smoothing_factor: Factor for exponential smoothing (0-1)
        use_adaptive_window: Whether to use adaptive window sizing
        
    Returns:
        List of (index, depth_score) tuples
    """
    if len(similarities) < 2 * window_size + 1:
        logging.warning(f"Similarities array too short ({len(similarities)}) for window_size {window_size}")
        return [(i, 0.0) for i in range(len(similarities))]
    
    depths = []
    
    # Apply exponential smoothing to reduce noise
    if smoothing_factor > 0:
        smoothed_sims = np.zeros_like(similarities)
        smoothed_sims[0] = similarities[0]
        for i in range(1, len(similarities)):
            smoothed_sims[i] = smoothing_factor * similarities[i] + (1 - smoothing_factor) * smoothed_sims[i-1]
        similarities = smoothed_sims
    
    for i in range(window_size, len(similarities) - window_size):
        # Adaptive window sizing based on local variance
        if use_adaptive_window:
            local_variance = np.var(similarities[max(0, i-window_size*2):i+window_size*2+1])
            adaptive_window = max(2, min(window_size * 2, int(window_size * (1 + local_variance))))
            effective_window = min(adaptive_window, min(i, len(similarities) - i - 1))
        else:
            effective_window = window_size
            
        left_start = max(0, i - effective_window)
        right_end = min(len(similarities), i + effective_window + 1)
        
        left_context = similarities[left_start:i]
        right_context = similarities[i+1:right_end]
        current_sim = similarities[i]
        
        if len(left_context) == 0 or len(right_context) == 0:
            depths.append((i, 0.0))
            continue
            
        # Use robust statistics (median + IQR-based outlier detection)
        left_baseline = np.median(left_context)
        right_baseline = np.median(right_context)
        
        # Calculate depth with improved formula
        # Consider both absolute depth and relative depth
        left_depth = max(0, left_baseline - current_sim)
        right_depth = max(0, right_baseline - current_sim)
        
        # Normalize by local standard deviation to handle scale differences
        local_std = np.std(similarities[left_start:right_end])
        if local_std > 0:
            normalized_depth = (left_depth + right_depth) / local_std
        else:
            normalized_depth = left_depth + right_depth
            
        # Add penalty for very high similarity (likely false boundaries)
        if current_sim > 0.95:
            normalized_depth *= 0.5
            
        depths.append((i, float(normalized_depth)))
    
    return depths


def tiled_segment(sentences: List[str], 
                 model,
                 wsize: int = 4,
                 top_p: float = 0.25,
                 min_seg_words: int = 40,
                 max_seg_words: int = 200,
                 stride: int = 1,
                 similarity_threshold: float = 0.3,
                 smoothing_factor: float = 0.1,
                 use_adaptive_threshold: bool = True) -> List[List[str]]:
    """
    Improved tiled segmentation with better boundary detection and robustness.
    
    Args:
        sentences: List of sentences to segment
        model: Sentence transformer model
        wsize: Window size for creating text windows
        top_p: Percentile threshold for valley detection
        min_seg_words: Minimum words per segment
        max_seg_words: Maximum words per segment
        stride: Stride for window creation
        similarity_threshold: Minimum similarity threshold for valid boundaries
        use_adaptive_threshold: Whether to use adaptive thresholding
        
    Returns:
        List of segments, each containing sentences
    """
    if len(sentences) < wsize:
        logging.warning(f"Too few sentences ({len(sentences)}) for window size {wsize}")
        return [sentences]
    
    # 1. Build windows with validation
    windows, idx_map = [], []
    for i in range(0, len(sentences) - wsize + 1, stride):
        window_text = " ".join(sentences[i:i+wsize])
        # Skip empty or very short windows
        if len(window_text.strip()) < 10:
            continue
        windows.append(window_text)
        idx_map.append(i)
    
    if len(windows) < 2:
        logging.warning("Too few valid windows created")
        return [sentences]
    
    # 2. Generate embeddings with error handling
    try:
        embs = model.encode(windows, show_progress_bar=False)
    except Exception as e:
        logging.error(f"Failed to encode windows: {e}")
        return [sentences]
    
    # 3. Calculate similarities using cosine similarity for better performance
    if len(embs) < 2:
        return [sentences]
        
    # Vectorized similarity calculation
    sims = []
    for i in range(len(embs) - 1):
        sim = float(cosine_similarity([embs[i]], [embs[i+1]])[0][0])
        sims.append(sim)
    
    sims = np.array(sims)
    
    # Filter out segments with very low similarity (likely noise or disconnected content)
    valid_indices = sims >= similarity_threshold
    if not np.any(valid_indices):
        logging.warning(f"No similarities above threshold {similarity_threshold}")
        return [sentences]
    
    # 4. Calculate depth scores with improved parameters
    depths = calculate_depth_scores(sims, window_size=max(2, len(sims)//10), smoothing_factor=smoothing_factor)
    
    if not depths:
        logging.warning("No depth scores calculated")
        return [sentences]
    
    # 5. Adaptive thresholding
    depth_values = np.array([d for _, d in depths], dtype=np.float64)
    
    if use_adaptive_threshold:
        # Use multiple criteria for threshold selection
        q75 = np.percentile(depth_values, 75)
        mean_depth = np.mean(depth_values)
        std_depth = np.std(depth_values)
        
        # Combine percentile and statistical thresholds
        stat_threshold = mean_depth + 0.5 * std_depth
        percentile_threshold = np.percentile(depth_values, 100 * (1 - top_p))
        
        # Use the more conservative threshold
        cutoff = max(percentile_threshold, stat_threshold, q75)
    else:
        cutoff = np.percentile(depth_values, 100 * (1 - top_p))
    
    # 6. Extract valleys with improved filtering
    valleys = []
    for i, d in depths:
        if d >= cutoff:
            # Additional check: ensure the corresponding similarity is reasonable
            if i < len(sims) and sims[i] >= similarity_threshold:
                # Map back to sentence indices, ensuring we don't go out of bounds
                sentence_idx = min(idx_map[i] + wsize//2, len(sentences) - 1)
                valleys.append(sentence_idx)
    
    # 7. Post-process boundaries with improved logic
    # Remove valleys that are too close to each other
    filtered_valleys = []
    for valley in sorted(valleys):
        if not filtered_valleys or valley - filtered_valleys[-1] >= min_seg_words // 4:
            filtered_valleys.append(valley)
    
    # 8. Length-based filtering and merging
    bounds = []
    last_bound = 0
    
    for valley in filtered_valleys:
        segment_length = valley - last_bound
        
        # Check if segment meets length requirements
        if segment_length >= min_seg_words:
            bounds.append(valley)
            last_bound = valley
        # If segment is too short, only add if it would create a reasonable next segment
        elif len(sentences) - valley >= min_seg_words:
            # Check if merging with next potential segment would be too long
            next_valley = None
            for v in filtered_valleys:
                if v > valley:
                    next_valley = v
                    break
            
            if next_valley is None:
                next_valley = len(sentences)
            
            if next_valley - last_bound <= max_seg_words:
                bounds.append(valley)
                last_bound = valley
    
    # 9. Create final segments with validation
    segments = []
    start = 0
    
    for bound in bounds:
        if bound > start:
            segment = sentences[start:bound]
            # Validate segment quality
            segment_text = " ".join(segment)
            if len(segment_text.strip()) > 20:  # Minimum meaningful content
                segments.append(segment)
            start = bound
    
    # Add final segment if it exists and is meaningful
    if start < len(sentences):
        final_segment = sentences[start:]
        final_text = " ".join(final_segment)
        if len(final_text.strip()) > 20:
            segments.append(final_segment)
        elif segments:  # Merge with last segment if too short
            segments[-1].extend(final_segment)
    
    # 10. Post-processing: merge very short segments
    final_segments = []
    for segment in segments:
        segment_text = " ".join(segment)
        word_count = len(segment_text.split())
        
        if word_count < min_seg_words and final_segments:
            # Merge with previous segment
            final_segments[-1].extend(segment)
        else:
            final_segments.append(segment)
    
    # Ensure we have at least one segment
    if not final_segments:
        final_segments = [sentences]
    
    # logging.info(f"Created {len(final_segments)} segments from {len(sentences)} sentences")
    return final_segments

def segment_by_similarity(sentences, model, embeddings, depth_threshold=0.9,window_size = 3):
    """
    Segments sentences based on similarity scores using depth scores.
    Args:
        sentences (list): List of sentences to segment.
        embeddings (list): List of sentence embeddings.
        depth_threshold (float): Threshold for depth score to determine segment boundaries.
        window_size (int): Size of the window to calculate average similarity scores.
    Returns:
        list: List of segmented sentences.
    """
    sim_scores = [model.similarity(embeddings[i], embeddings[i+1])
                   for i in range(len(embeddings) - 1)]
    depth_scores = []
    for i in range(1, len(sim_scores) - 1):
        left_avg = np.mean(sim_scores[max(0, i - window_size):i])
        right_avg = np.mean(sim_scores[i+1:i+1 + window_size])
        depth = max(0, sim_scores[i] - left_avg) + max(0, sim_scores[i] - right_avg)
        depth_scores.append((i + 1, depth))
    boundaries = [i for i, depth in depth_scores if depth > depth_threshold]
    segments = []
    start = 0
    for boundary in boundaries:
        segments.append(sentences[start:boundary])
        start = boundary
    segments.append(sentences[start:])
    return segments
    
# Extract top-N keywords using KeyBERT
def get_top_topic_words(segment, kw_model, top_n=5):
    doc = " ".join(segment)
    keywords = kw_model.extract_keywords(doc,keyphrase_ngram_range=(1, 1), top_n=top_n, stop_words="english")
    return [kw for kw, _ in keywords]

def merge_segments_by_topic(segments, embedding_model, top_n=9, min_topics=2, min_seg_length=3):
    """
    Merges segments based on topic overlap using KeyBERT.
    Args:
        segments (list): List of segments to merge.
        embedding_model: The embedding model to use for topic extraction.
        top_n (int): Number of top keywords to consider for merging.
        min_topics (int): Minimum number of overlapping topics to merge segments.
        min_seg_length (int): Minimum length of segments to consider for merging.
    Returns:
        list: List of merged segments.
    """
    
    kw_model = KeyBERT(embedding_model)
    merged_segments = []
    i = 0
    while i < len(segments) - 1:
        cur = segments[i]
        nxt = segments[i+1]
        cur_topics = get_top_topic_words(cur, kw_model, top_n=top_n)
        nxt_topics = get_top_topic_words(nxt, kw_model, top_n=top_n)

        min_topics = min_topics if len(segments) > 6 else 3
        if len(set(cur_topics) & set(nxt_topics)) >= min_topics:
            segments[i-1] = cur + nxt  # merge
            print(f"Merging segments {i} and {i+1} with topics: {set(cur_topics) & set(nxt_topics)}")
        elif len(cur) < min_seg_length:
            print(f"Segment {i} is too short ({len(cur)} words), merging with next segment.")
            # If current segment is too short, merge it with the next one
            segments[i+1] = cur + nxt
        else:
            # If they are not similar enough, keep the current segment
            merged_segments.append(cur)
        i += 1
    if i == len(segments) - 1:
        merged_segments.append(segments[-1])
    return merged_segments

def label_segments_with_topics(segments, model_path):
    """
    Applies BERTopic to label segments with topics.
    Args:
        segments (list): List of segments to label.
        model_path (str): Path to the Sentence-BERT model.
    Returns:
        list: List of labeled segments with topics.
    """
    docs = [" ".join(segment) for segment in segments if len(segment) >= 3]

    if len(docs) < 3:
        print("Too few valid segments for topic modeling.")
        return [(i, segment, "Too short") for i, segment in enumerate(segments)]

    embedding_model = SentenceTransformer(model_path)
    
    doc_embeddings = embedding_model.encode(docs, show_progress_bar=False)
    
    # Unique vectors only
    unique_embeddings = set(map(tuple, doc_embeddings))
    n_clusters = len(unique_embeddings)

    # Clamp to avoid invalid values
    n_clusters = max(2, min(n_clusters, len(doc_embeddings) - 1))

    cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
    
    # Use UMAP for dimensionality reduction if enough documents are available
    if len(docs) < 6:   
        umap_model = BaseDimensionalityReduction()   
    else:
        umap_model = UMAP(n_components=2, random_state=42)
        
    custom_stopwords = list(text.ENGLISH_STOP_WORDS.union({
        'know', 'going', 'thats', 'theres', 'sort', 'thing', 'get', 'got', 'let',
        'actually', 'maybe', 'say', 'okay', 'please', 'course', 'like', 'see',
        'think', 'make', 'want', 'just', 'well', 'right'
    }))
    vectorizer_model = CountVectorizer(stop_words=custom_stopwords, min_df=2, ngram_range=(1, 2))

    label_representation_model = OpenAI(openai, model="gpt-4o-mini", chat=True)
    list_representation_model = KeyBERTInspired()
    
    representation_model = {
        "Main": label_representation_model,
        "Aspect1": list_representation_model
    }
    
    topic_model = BERTopic(
        embedding_model=embedding_model,
        hdbscan_model=cluster_model,
        umap_model=umap_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        top_n_words=10
    )
    
    topics, _ = topic_model.fit_transform(docs, embeddings=doc_embeddings)
    
    results = []
    doc_idx = 0
    print(f"segments: {len(segments)}, topics: {len(topics)}")
    info_df = topic_model.get_topic_info()

    for i, segment in enumerate(segments):
        if len(segment) < 3:
            results.append((i, segment, "Too short", []))
        else:
            topic_id = topics[doc_idx]

            # Handle outliers
            if topic_id == -1 or topic_id not in info_df.index:
                label = "Outlier"
                keywords = []
            else:
                row = info_df.loc[topic_id]
                label = row["Representation"][0] if isinstance(row["Representation"], list) else row["Representation"]
                keywords = row["Aspect1"] if isinstance(row["Aspect1"], list) else [row["Aspect1"]]

            results.append((i, segment, label, keywords))
            doc_idx += 1
    return results
    
def align_w_timestamp(seg_bounds, original_segments):
    """
    Aligns segment bounds with original segments to include timestamps.
    
    Args:
        seg_bounds (list): List of tuples with segment indices and labels.
        original_segments (list): Original segments with timestamps.
        
    Returns:
        list: Aligned segments with timestamps and labels.
    """
    aligned_segments = []
    
    for segment_idx, pos, label, topics in seg_bounds:
        # use the position of next segment to determine the end time of current segment
        # If this is the last segment, use the end of the last original segment
        next_bound = seg_bounds[segment_idx + 1] if segment_idx + 1 < len(seg_bounds) else None
        if next_bound:
            next_pos = next_bound[1]
            end = original_segments[next_pos - 1]["end"]
        else:
            next_pos = len(original_segments)
            end = original_segments[-1]["end"]
        
        seg_texts = [original_segments[i] for i in range(pos, next_pos - 1) if i < len(original_segments)]
        
        aligned_segments.append({
            "segment_idx": segment_idx,
            "start": original_segments[pos]["start"],
            "end": end,
            "label": label,
            "topics": topics,
            "segments": seg_texts
        })
             
    return aligned_segments
   
def process_transcript(transcript, with_timestamps = True, label = True, use_tiling = True, verbose = True):
    """
    Processes the transcript to extract sentences and perform topic segmentation.
    
    Args:
        transcript (str or dict): The transcript text or JSON object containing segments.
        with_timestamps (bool): Whether input transcript has timestamps.
        
    Returns:
        list: A list of processed sentences or segments.
    """
    sentences = extract_transcript(transcript, with_timestamps)
    
    print(f"Extracted {len(sentences)} sentences from transcript.")
    if not sentences:
        return []
    
    # SBERT Sentence Embeddings
    embeddings, model = get_embeddings(sentences, model=None, model_path=model_path, verbose=verbose)
    if verbose:
        print(f"Generated embeddings for {len(sentences)} sentences.")
        
    # Segment by Similarity with Depth Scores
    if not use_tiling:
        segments = segment_by_similarity(sentences, model, embeddings, depth_threshold=0.667, window_size=3)
    else:
        segments = tiled_segment(sentences, model, wsize=13, top_p=0.3, min_seg_words=20, stride=1, similarity_threshold=0.168, smoothing_factor=0.0127)
    if verbose:
        print(f"Segmented into {len(segments)} segments based on similarity.")
    # Merge close segments using KeyBERT
    # segments = merge_segments_by_topic(segments, model, top_n=8, min_topics=3, min_seg_length=3)
    
    if label:
        segments = merge_segments_by_topic(segments, model, top_n=6, min_topics=2, min_seg_length=3)
        print(f"Merged segments into {len(segments)} based on topic overlap.")
        #  Apply BERTopic with Enhanced Labels
        labeled_segments = label_segments_with_topics(segments, model_path=model_path)
        
        print(f"Labeled segments with topics, total: {len(labeled_segments)}")
        # Get segment bounds with original segments if transcript has timestamps 
        if with_timestamps:
            seg_bounds = get_segment_bounds(transcript["segments"], labeled_segments)
            result = align_w_timestamp(seg_bounds, transcript["segments"])
        else:
            result = labeled_segments
    else:
        if verbose:
            print("Skipping labeling of segments.")
        result = segments
    
    return result


def process_transcript_optimize(transcript, model, wsize, top_p, min_seg_words, stride=1, similarity_threshold=0.3, smoothing_factor=0.1):
    """
    Processes the transcript to extract sentences and perform topic segmentation.
    
    Args:
        transcript (str or dict): The transcript text or JSON object containing segments.
        with_timestamps (bool): Whether input transcript has timestamps.
        
    Returns:
        list: A list of processed sentences or segments.
    """
    sentences = extract_transcript(transcript, False)
    if not sentences:
        return []
    
    # model = AutoModel.from_pretrained(
    # "sentence-transformers/all-MiniLM-L6-v2",
    # device_map="cpu",
    # offload_folder="offload",
    # low_cpu_mem_usage=True
    #     ).to_empty(device="cpu") 
    segments = tiled_segment(sentences, model, wsize=wsize, top_p=top_p, min_seg_words=min_seg_words, stride=stride,
                             similarity_threshold=similarity_threshold, smoothing_factor=smoothing_factor )

    # Merge close segments using KeyBERT
    # segments = merge_segments_by_topic(segments, model, top_n=8, min_topics=3, min_seg_length=3)


    result = segments
    
    return result

def process_transcript_optimize2(transcript, model, depth_threshold=0.73, window_size=9, top_n=8, min_topics=3):
    """
    Processes the transcript to extract sentences and perform topic segmentation.
    
    Args:
        transcript (str or dict): The transcript text or JSON object containing segments.
        with_timestamps (bool): Whether input transcript has timestamps.
        
    Returns:
        list: A list of processed sentences or segments.
    """
    sentences = extract_transcript(transcript, False)
    if not sentences:
        return []
    
    
    embeddings, model = get_embeddings(sentences, model, model_path=None, verbose=False)
    segments = segment_by_similarity(sentences, model, embeddings=embeddings, depth_threshold=depth_threshold, window_size=window_size)

    # Merge close segments using KeyBERT
    segments = merge_segments_by_topic(segments, model, top_n=top_n, min_topics=min_topics, min_seg_length=3)


    result = segments
    
    return result

# Mean WindowDiff over 10 meetings: 0.456
# Mean PK over 10 meetings: 0.419

# Best WD : 0.45273082900926154
# Best Pk : 0.4226692655452105
# Params  : {'depth_t': 0.6678651787342368, 
# 'window_size': 3, 'top_n': 9, 'min_topics': 3}

#Tiling
# Mean WindowDiff over 5 meetings: 0.392
# Mean PK over 5 meetings: 0.350
# Mean WindowDiff over 10 meetings: 0.423
# Mean PK over 10 meetings: 0.374
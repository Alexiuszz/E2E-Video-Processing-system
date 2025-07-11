import os
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.cluster import KMeans
from bertopic.representation import KeyBERTInspired
from bertopic.representation import OpenAI
from umap import UMAP
from hdbscan import HDBSCAN
import numpy as np
import openai
from bertopic import BERTopic
from bertopic.backend import OpenAIBackend

from keybert import KeyBERT
import pandas as pd
from typing import JSON
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
ApiKey = os.environ.get("API_KEY")

nltk.download("punkt")

# --------- Configuration ---------
model_path = os.environ.get("MODEL_PATH", "SentenceTransformer/all-MiniLM-L6-v2")

openai.api_key = ApiKey

def extract_transcript(transcript: JSON) -> str:
    """
    Extracts the transcript text from the JSON format.
    
    Args:
        transcript (JSON): The transcript in JSON format.        
    Returns:
        str: The extracted transcript text.
    """
    transcript =  " ".join([item['text'] for item in transcript])
    
    return sent_tokenize(transcript)

# Encode using Sentence-BERT
def get_embeddings(sentences, model_path):
    model = SentenceTransformer(model_path)
    embeddings = model.encode(sentences, show_progress_bar=True)
    return embeddings, model

def segment_by_similarity(sentences, embeddings, depth_threshold=0.9,window_size = 3):
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
    
    model = SentenceTransformer(model_path)
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

def merge_segments_by_topic(segments, embedding_model):
    """
    Merges segments based on topic overlap using KeyBERT.
    Args:
        segments (list): List of segments to merge.
        embedding_model: The embedding model to use for topic extraction.
    Returns:
        list: List of merged segments.
    """
    
    kw_model = KeyBERT(embedding_model)
    merged_segments = []
    i = 0
    while i < len(segments) - 1:
        cur = segments[i]
        nxt = segments[i+1]
        cur_topics = get_top_topic_words(cur, kw_model)
        nxt_topics = get_top_topic_words(nxt, kw_model)

        if len(set(cur_topics) & set(nxt_topics)) >= 2:
            segments[i+1] = cur + nxt  # merge
        else:
            merged_segments.append(cur)
        i += 1
    if i == len(segments) - 1:
        merged_segments.append(segments[-1])
    return merged_segments


def process_transcript(transcript: JSON, with_timestamps = True):
    """
    Processes the transcript to extract sentences and perform topic segmentation.
    
    Args:
        transcript (JSON): The transcript in JSON format.
        with_timestamps (bool): Whether to include timestamps in the output.
        
    Returns:
        list: A list of processed sentences or segments.
    """
    sentences = extract_transcript(transcript)
    
    if not sentences:
        return []
    
    # SBERT Sentence Embeddings
    embeddings = get_embeddings(sentences, model_path)
    
    # Segment by Similarity with Depth Scores
    segments = segment_by_similarity(sentences, embeddings)
    
    # Merge close segments using KeyBERT
    segments = merge_segments_by_topic(segments, model_path)
    
    #  Apply BERTopic with Enhanced Labels
    labeled_segments = label_segments_with_topics(segments, model_path)
    
    return labeled_segments
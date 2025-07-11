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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
ApiKey = os.environ.get("API_KEY")

nltk.download("punkt")

# --------- Configuration ---------
model_path = os.environ.get("MODEL_PATH", "SentenceTransformer/all-MiniLM-L6-v2")

openai.api_key = ApiKey

def extract_transcript(transcript) -> str:
    """
    Extracts the transcript text from the JSON format.
    
    Args:
        transcript (JSON): The transcript in JSON format.        
    Returns:
        str: The extracted transcript text.
    """
    transcript =  " ".join([segment['text'].strip() for segment in transcript["segments"]])
    
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
            segments[i-1] = cur + nxt  # merge
        elif len(cur) < 3:
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
    n_clusters = len(set(map(tuple, doc_embeddings)))  
    
    cluster_model = KMeans(n_clusters=max(2, min(n_clusters, len(segments) - 1)), random_state=42)
    
    custom_stopwords = list(text.ENGLISH_STOP_WORDS.union({
        'know', 'going', 'thats', 'theres', 'sort', 'thing', 'get', 'got', 'let',
        'actually', 'maybe', 'say', 'okay', 'please', 'course', 'like', 'see',
        'think', 'make', 'want', 'just', 'well', 'right'
    }))
    vectorizer_model = CountVectorizer(stop_words=custom_stopwords, min_df=2, ngram_range=(1, 2))

    label_representation_model = OpenAI(openai, model="gpt-4o", chat=True)
    list_representation_model = KeyBERTInspired()
    
    representation_model = {
        "Main": label_representation_model,
        "Aspect1": list_representation_model
    }
    
    topic_model = BERTopic(
        embedding_model=embedding_model,
        hdbscan_model=cluster_model,
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
    
    
def process_transcript(transcript, with_timestamps = True):
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
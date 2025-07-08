# Full Python script that performs SBERT segmentation, uses BERTopic to extract top keywords per segment,
# merges segments with shared topic overlap, and uses OpenAI to generate final topic labels.

import os
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bertopic import BERTopic
from bertopic.representation import OpenAI
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction import text
from umap import UMAP
from hdbscan import HDBSCAN
import numpy as np
import openai
from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from keybert import KeyBERT


from dotenv import load_dotenv
# Load environment variables
load_dotenv()

ApiKey = os.environ.get("API_KEY")

nltk.download("punkt")

# --------- Configuration ---------
model_path = os.environ.get("MODEL_PATH", "SentenceTransformer/all-MiniLM-L6-v2")
openai.api_key = ApiKey  
# ---------------------------------

# Load and tokenize sentences
def load_sentences(file_path):
    with open(file_path, "r") as f:
        return sent_tokenize(f.read())

# Encode using Sentence-BERT
def get_embeddings(sentences, model_path):
    model = SentenceTransformer(model_path)
    embeddings = model.encode(sentences, show_progress_bar=True)
    return embeddings, model

# Segment using Solbiati block-style
def block_segmentation(sentences, embeddings, block_size=5, threshold=0.75):
    blocks = [np.mean(embeddings[i:i+block_size], axis=0) for i in range(0, len(embeddings), block_size)]
    similarities = [cosine_similarity([blocks[i]], [blocks[i+1]])[0][0] for i in range(len(blocks)-1)]
    boundaries = [i+1 for i, sim in enumerate(similarities) if sim < threshold]

    segments, start = [], 0
    for b in boundaries:
        end = min(b * block_size, len(sentences))
        segments.append(sentences[start:end])
        start = end
    segments.append(sentences[start:])
    return segments

# Extract top-N keywords using KeyBERT
def get_top_topic_words(segment, kw_model, top_n=3):
    doc = " ".join(segment)
    keywords = kw_model.extract_keywords(doc, top_n=top_n, stop_words="english")
    return [kw for kw, _ in keywords]

# # Get top-N keywords using BERTopic for one segment
# def get_top_topic_words(segment, topic_model, embedding_model, top_n=3):
#     doc = " ".join(segment)
#     embedding = embedding_model.encode([doc])
#     topics, _ = topic_model.fit_transform([doc], embeddings=embedding)
#     topic_id = topics[0]
#     if topic_id < 0:
#         return []
#     return [word for word, _ in topic_model.get_topic(topic_id)[:top_n]]


# Merge segments based on topic overlap
def merge_segments_by_topic(segments, embedding_model):
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
# def merge_segments_by_topic(segments, embedding_model):
#     # Fit BERTopic without actually performing any dimensionality reduction
#     empty_dimensionality_model = BaseDimensionalityReduction()   
#     hdb = HDBSCAN(min_cluster_size=2, min_samples=1)
#     vectorizer = CountVectorizer(stop_words="english", min_df=1)
#     topic_model = BERTopic(embedding_model=embedding_model,
#                            umap_model=empty_dimensionality_model,
#                            hdbscan_model=hdb,
#                            vectorizer_model=vectorizer)

#     merged_segments = []
#     i = 0
#     while i < len(segments) - 1:
#         cur = segments[i]
#         nxt = segments[i+1]
#         cur_topics = get_top_topic_words(cur, topic_model, embedding_model)
#         nxt_topics = get_top_topic_words(nxt, topic_model, embedding_model)

#         if len(set(cur_topics) & set(nxt_topics)) >= 2:
#             segments[i+1] = cur + nxt  # merge
#         else:
#             merged_segments.append(cur)
#         i += 1
#     if i == len(segments) - 1:
#         merged_segments.append(segments[-1])
#     return merged_segments

# Label final merged segments using OpenAI GPT
def label_segments_openai(segments, embedding_model):
    openai_model = OpenAI(openai, model="gpt-4o-mini", chat=True)
    vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))
    hdb = HDBSCAN(min_cluster_size=2, min_samples=1)
    
    empty_dimensionality_model = BaseDimensionalityReduction()  
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=empty_dimensionality_model,
        hdbscan_model=hdb,
        vectorizer_model=vectorizer_model,
        representation_model=openai_model,
        top_n_words=10
    )

    docs = [" ".join(seg) for seg in segments if len(seg) >= 3]
    if len(docs) < 3:
        return [(i, segments[i], "Too short") for i in range(len(segments))]

    embeddings = embedding_model.encode(docs, show_progress_bar=True)
    topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)

    labeled = []
    doc_idx = 0
    for i, segment in enumerate(segments):
        if len(segment) < 3:
            labeled.append((i, segment, "Too short"))
        else:
            topic_id = topics[doc_idx]
            name = topic_model.get_topic_info().loc[topic_id, "Name"] if topic_id >= 0 else "Outlier"
            labeled.append((i, segment, name))
            doc_idx += 1
    return labeled

# Full pipeline
def process_transcript(file_path, output_path="segmented_transcript.txt"):
    sentences = load_sentences(file_path)
    embeddings, model = get_embeddings(sentences, model_path)
    print("Initial segmentation...")
    initial_segments = block_segmentation(sentences, embeddings, block_size=5, threshold=0.75)
    print(f"Initial segments: {len(initial_segments)}")

    print("Merging similar topic segments...")
    merged_segments = merge_segments_by_topic(initial_segments, model)
    print(f"Merged segments: {len(merged_segments)}")

    print("Labelling segments with GPT...")
    labeled = label_segments_openai(merged_segments, model)

    with open(output_path, "w") as f:
        for i, segment, topic in labeled:
            f.write(f"\n=== Segment {i+1} ===\n")
            f.write(f"Topic: {topic}\n")
            f.write(" ".join(segment) + "\n")

    print(f"Done. Output written to {output_path}")
    
# ----------- Entry Point ---------------------------------------
if __name__ == "__main__":
    transcript_file = "transcript.txt"
    process_transcript(transcript_file)


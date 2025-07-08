from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction import text
import numpy as np
import nltk
import os

nltk.download('punkt')

# You can point to a local cache or just use the model name
model_path = os.environ.get("MODEL_PATH", "SentenceTransformer/all-MiniLM-L6-v2")

# ----------- Step 1: Load and split transcript -----------------
def load_sentences_from_file(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    return sent_tokenize(text)

# ----------- Step 2: SBERT Sentence Embeddings -----------------
def get_embeddings(sentences, model_path):
    model = SentenceTransformer(model_path)
    #print number of sentences
    print(f"Encoding {len(sentences)} sentences...")
    
    embeddings = model.encode(sentences, show_progress_bar=True)
    return embeddings

# ----------- Step 3: Segment by Similarity with Depth Scores ---
def segment_by_similarity(sentences, embeddings, window_size=3, depth_threshold=0.9):
    sim_scores = [cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
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

# ----------- Step 4: Apply BERTopic with Enhanced Labels -------
def label_segments_with_topics(segments, model_path):
    docs = [" ".join(segment) for segment in segments if len(segment) >= 3]

    if len(docs) < 3:
        print("Too few valid segments for topic modeling.")
        return [(i, segment, "Too short") for i, segment in enumerate(segments)]

    model = SentenceTransformer(model_path)
    print("Encoding documents for topic modeling...")
    doc_embeddings = model.encode(docs, show_progress_bar=True)

    umap_model = UMAP(n_components=2,       # Allow more clustering expressiveness
        n_neighbors=10,       # Slightly tighter focus
        min_dist=0.0,         # Allow tight clusters
        random_state=42)
    
    hdb = HDBSCAN(min_cluster_size=2, min_samples=2)
    
    custom_stopwords = list(text.ENGLISH_STOP_WORDS.union({
        'know', 'going', 'thats', 'theres', 'sort', 'thing', 'get', 'got', 'let',
        'actually', 'maybe', 'say', 'okay', 'please', 'course', 'like', 'see',
        'think', 'make', 'want', 'just', 'well', 'right'
    }))
    vectorizer_model = CountVectorizer(stop_words=custom_stopwords, min_df=2, ngram_range=(1, 2))
    representation_model = KeyBERTInspired()

    topic_model = BERTopic(
         embedding_model=model,
        umap_model=umap_model,
        hdbscan_model=hdb,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        top_n_words=10
    )

    topics, _ = topic_model.fit_transform(docs, embeddings=doc_embeddings)

    results = []
    doc_idx = 0
    for i, segment in enumerate(segments):
        if len(segment) < 3:
            results.append((i, segment, "Too short"))
        else:
            topic_id = topics[doc_idx]
            if topic_id < 0:
                label = "Outlier"
            else:
                label = topic_model.get_topic_info().loc[topic_id, "Name"]
            results.append((i, segment, label))
            doc_idx += 1
    return results

# ----------- Step 5: Main Runner -------------------------------
def process_transcript(file_path, output_path="segmented_transcript.txt", model_path=model_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    sentences = load_sentences_from_file(file_path)
    embeddings = get_embeddings(sentences, model_path)
    segments = segment_by_similarity(sentences, embeddings, window_size=3, depth_threshold=0.6)
    labeled_segments = label_segments_with_topics(segments, model_path=model_path)

    segments_with_topics = []
    for i, segment, topic in labeled_segments:
        segments_with_topics.append(f"\n=== Segment {i + 1} ===")
        segments_with_topics.append(f"Topic: {topic}")
        segments_with_topics.append(" ".join(segment))

    with open(output_path, "w") as f:
        f.write("\n".join(segments_with_topics))
    print(f"Segmented transcript saved to {output_path}")

# ----------- Entry Point ---------------------------------------
if __name__ == "__main__":
    transcript_file = "transcript.txt"
    process_transcript(transcript_file, model_path=model_path)

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
import openai
from bertopic.representation import OpenAI
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

ApiKey = os.environ.get("API_KEY")


# Fine-tune topic representations with GPT
client = openai.OpenAI(api_key=ApiKey)
# representation_model = OpenAI(client, model="gpt-4o-mini", chat=True)

nltk.download("punkt")

model_path = os.environ.get("MODEL_PATH", "SentenceTransformer/all-MiniLM-L6-v2")

# ----------- Step 1: Load and split transcript -----------------
def load_sentences_from_file(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    return sent_tokenize(text)

# ----------- Step 2: SBERT Sentence Embeddings -----------------
def get_embeddings(sentences, model_path):
    model = SentenceTransformer(model_path)
    print(f"Encoding {len(sentences)} sentences...")
    embeddings = model.encode(sentences, show_progress_bar=True)
    return embeddings, model

# ----------- Step 3: Block-Based Topic Segmentation (Solbiati) --------
def block_segmentation(sentences, embeddings, block_size=5, threshold=0.5):
    # 1. Build block embeddings
    blocks = [np.mean(embeddings[i:i+block_size], axis=0)
              for i in range(0, len(embeddings), block_size)]

    # 2. Compute cosine similarity between adjacent blocks
    similarities = [cosine_similarity([blocks[i]], [blocks[i+1]])[0][0]
                    for i in range(len(blocks)-1)]

    # 3. Find boundaries where similarity drops below threshold
    boundaries = [i+1 for i, sim in enumerate(similarities) if sim < threshold]

    # 4. Convert block boundaries into sentence segments
    segments = []
    start = 0
    for b in boundaries:
        end = min(b * block_size, len(sentences))
        segments.append(sentences[start:end])
        start = end
    segments.append(sentences[start:])
    return segments

# ----------- Step 4: Apply BERTopic with Enhanced Labels -------
def label_segments_with_topics(segments, model):
    docs = [" ".join(segment) for segment in segments if len(segment) >= 3]

    if len(docs) < 3:
        print("Too few valid segments for topic modeling.")
        return [(i, segment, "Too short") for i, segment in enumerate(segments)]

    print("Encoding segments for topic modeling...")
    doc_embeddings = model.encode(docs, show_progress_bar=True)

    umap_model = UMAP(n_components=2, n_neighbors=10, min_dist=0.0, metric="cosine", random_state=42)
    hdb = HDBSCAN(min_cluster_size=2, min_samples=2)
    custom_stopwords = list(text.ENGLISH_STOP_WORDS.union({
        'know', 'going', 'thats', 'theres', 'sort', 'thing', 'get', 'got', 'let',
        'actually', 'maybe', 'say', 'okay', 'please', 'course', 'like', 'see',
        'think', 'make', 'want', 'just', 'well', 'right'
    }))
    vectorizer_model = CountVectorizer(stop_words=custom_stopwords, min_df=2, ngram_range=(1, 2))
    # representation_model = KeyBERTInspired()
    
    representation_model = OpenAI(client, model="gpt-4o-mini", chat=True)

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
    embeddings, model = get_embeddings(sentences, model_path)
    segments = block_segmentation(sentences, embeddings, block_size=5, threshold=0.5)
    labeled_segments = label_segments_with_topics(segments, model)

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
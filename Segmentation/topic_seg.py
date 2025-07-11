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
from sklearn.decomposition import PCA
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

# ---------------------------------
openai.api_key = ApiKey
# Load and tokenize sentences
def load_sentences(file_path):
    with open(file_path, "r") as f:
        return sent_tokenize(f.read())

def load_json_transcript(file_path):
    import json
    with open(file_path, "r") as f:
        transcript = json.load(f)
    return transcript

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
def get_top_topic_words(segment, kw_model, top_n=9):
    doc = " ".join(segment)
    keywords = kw_model.extract_keywords(doc,keyphrase_ngram_range=(1, 1), top_n=top_n, stop_words="english")
    return [kw for kw, _ in keywords]

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


def segment_and_label_texts(segments, model="gpt-4o-mini", verbose=True, start_index=0):
    """
    Given a list of text segments, determine if each new segment should be merged
    with the previous one, and generate labels accordingly.
    Returns a list of tuples: (merged_segment, label)
    """
    client = openai.OpenAI(api_key=ApiKey)
    results = []
    current_segment = ""  # Start with an empty initial segment
    current_topic = ""
    total_tokens = 0
    start_index = 0
    total_tokens = 0
    
    for idx, new_segment in enumerate(segments):
        prompt = f"""You are an expert in topic segmentation. Given two text segments, decide whether they should be merged into a single topic, only merge when necessary since topics will be used for searching and retrieval later.

First Segment:
\"\"\"{current_segment}\"\"\"
First Segment's Topic:
\"\"\"{current_topic}\"\"\"

Second Segment:
\"\"\"{new_segment}\"\"\"

- If merged, provide a concise label (one word or short phrase) that captures the main topic of the merged text, don't explain reasoning.
- If not merged, return a label for the second segment and say "Not merged".

You're also given a rough topic of the first segment to help you decide and also to avoid repetitive topic labels
For initialization, the first segment may be empty."""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100,
        )

        content = response.choices[0].message.content.strip()
        usage = response.usage
        total_tokens += usage.total_tokens
        
        if verbose:
            print(f"\n--- Comparing segment {start_index} to {idx} ---")
            print(f"Tokens used: {usage.total_tokens}")
            print(f"Response: {content}")
        
        if "not merged" in content.lower():
            results.append((idx, idx, new_segment, content))
            current_segment = new_segment
            current_topic = content
            start_index = idx
        else:
            current_segment = f"{current_segment} {new_segment}".strip()
            if results and results[-1][0] == start_index:
                # Update the last segment's text and label
                results[-1] = (start_index, idx, current_segment, content)
            else:
                results.append((start_index, idx, current_segment, content))

    return results, total_tokens

def label_segments_with_topics(segments, model_path):
    docs = [" ".join(segment) for segment in segments if len(segment) >= 3]

    if len(docs) < 3:
        print("Too few valid segments for topic modeling.")
        return [(i, segment, "Too short") for i, segment in enumerate(segments)]

    embedding_model = SentenceTransformer(model_path)
    # # Use OpenAI for embeddings
    # client = OpenAI(api_key=ApiKey)
    # embedding_model = OpenAIBackend(client, "text-embedding-ada-002")

    
    print("Encoding documents for topic modeling...")
    doc_embeddings = embedding_model.encode(docs, show_progress_bar=False)

    umap_model = UMAP(n_components=5,       # Allow more clustering expressiveness
        n_neighbors=5,       # Slightly tighter focus
        min_dist=0.0,         # Allow tight clusters
        random_state=42)
    
    # umap_model = PCA(n_components=5, random_state=42)  # Use PCA for dimensionality reduction
    
    n_clusters = len(set(map(tuple, doc_embeddings)))  # Unique vectors only
    # cluster_model = HDBSCAN(min_cluster_size=2, min_samples=1)
    cluster_model = KMeans(n_clusters=max(2, min(n_clusters, len(segments) - 1)), random_state=42)
    
    custom_stopwords = list(text.ENGLISH_STOP_WORDS.union({
        'know', 'going', 'thats', 'theres', 'sort', 'thing', 'get', 'got', 'let',
        'actually', 'maybe', 'say', 'okay', 'please', 'course', 'like', 'see',
        'think', 'make', 'want', 'just', 'well', 'right'
    }))
    vectorizer_model = CountVectorizer(stop_words=custom_stopwords, min_df=2, ngram_range=(1, 2))
    # representation_model = KeyBERTInspired()
    label_representation_model = OpenAI(openai, model="gpt-4o", chat=True)
    list_representation_model = KeyBERTInspired()
    
    representation_model = {
        "Main": label_representation_model,
        "Aspect1": list_representation_model
    }

    topic_model = BERTopic(
        embedding_model=embedding_model,
        # umap_model=umap_model,
        hdbscan_model=cluster_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        top_n_words=10
    )

    topics, _ = topic_model.fit_transform(docs, embeddings=doc_embeddings)
    # topics, _ = topic_model.fit_transform(docs)

    output_path="topics.csv"
    pd.DataFrame(topic_model.get_topic_info()).to_csv(output_path, index=False)


    # print(topic_model.get_topic_info())
    
    
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

# Full pipeline
def process_transcript(file_path, output_path="segmented_transcript.txt"):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    transcript_json = load_json_transcript(file_path)
    sentences = extract_transcript(transcript_json)
    # #print first two sentences
    # first_two = sentences[:2]
    # first_two = ["Lenses are incredible.","Lenses are analog computers that take light rays that come in and reprogram them to go in different directions to form an image."]
    # # Get embeddings
    # embeddings, model = get_embeddings(first_two, model_path)
    
    # # Compute cosine similarities
    # similarities = model.similarity(embeddings[0], embeddings[1])
    
    # print(similarities)
    
    embeddings, model = get_embeddings(sentences, model_path)
    segments = segment_by_similarity(sentences, embeddings, depth_threshold=0.7)
    
    merged_segments = merge_segments_by_topic(segments, model)
    print(f"Initial segments: {len(segments)}, Merged segments: {len(merged_segments)}")
    # kw_model = KeyBERT(model)
    # seg_topics = get_top_topic_words(merged_segments[4], kw_model, top_n=10)
    # results, total_tokens  = segment_and_label_texts(merged_segments, model="gpt-4o-mini")
    # new_segments = [seg for _, seg, _ in results]
    
    
    labeled_segments = label_segments_with_topics(merged_segments, model_path=model_path)

    # # print(f"\n🔢 Total tokens used: {total_tokens}\n")
    with open(output_path, "w") as f:
        for i, segment, label, topics in labeled_segments:
            f.write(f"\n=== Segment {i} ===\n")
            f.write(f"Label: {label}\n")
            f.write(f"Topics: {topics}\n")
            f.write(" ".join(segment) + "\n")
            
            
    # print(f"from {len(segments)} segments to {len(merged_segments)} merged segments to  {len(results)} labelled segments.")
    # print(f"Processed {len(merged_segments)} segments.")
    print(f"Done. Output written to {output_path}")
    
# ----------- Entry Point ---------------------------------------
if __name__ == "__main__":
    # transcript_file = "transcript.txt"
    # process_transcript(transcript_file)
    process_transcript("transcript.json")
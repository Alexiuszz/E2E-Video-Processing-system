import pytest
from backend.services.topic_segment import (
    extract_transcript,
    get_embeddings,
    adaptive_seg,
    merge_segments_by_topic,
    label_segments_with_topics,
    process_transcript,
)
from sentence_transformers import SentenceTransformer

# Sample mock transcript
MOCK_TRANSCRIPT = {
    "segments": [{"start": 0.0, "end": 1.0, "text": "Hello world."}] * 100
}
SHORT_TRANSCRIPT = {
    "segments": [{"start": 0.0, "end": 1.0, "text": "Hi."}] * 10
}
TEXT_TRANSCRIPT = {
    "text": "This is sentence one. This is sentence two. " * 30
}

model_path = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_path)


def test_extract_transcript_with_timestamps():
    sentences = extract_transcript(MOCK_TRANSCRIPT, with_timestamps=True)
    assert isinstance(sentences, list)
    assert len(sentences) == 100


def test_extract_transcript_without_timestamps():
    sentences = extract_transcript(TEXT_TRANSCRIPT, with_timestamps=False)
    assert isinstance(sentences, list)
    assert len(sentences) > 0


def test_extract_transcript_invalid():
    with pytest.raises(ValueError):
        extract_transcript({}, with_timestamps=True)


def test_get_embeddings():
    sents = ["This is sentence one.", "This is sentence two."] * 30
    embeddings, m = get_embeddings(sents, model, model_path)
    assert embeddings.shape[0] == len(sents)


def test_get_embeddings_too_few():
    sents = ["Short"] * 10
    with pytest.raises(ValueError):
        get_embeddings(sents, model, model_path)


def test_adaptive_seg_returns_segments():
    sents = ["This is sentence one.", "This is sentence two."] * 40
    segments = adaptive_seg(sents, model)
    assert isinstance(segments, list)
    assert all(isinstance(seg, list) for seg in segments)


def test_merge_segments_by_topic():
    segments = [["This is about dogs."] * 5, ["Dogs are cute."] * 5, ["Bananas are yellow."] * 5]
    merged = merge_segments_by_topic(segments, model, top_n=3, min_topics=1)
    assert isinstance(merged, list)
    assert all(isinstance(seg, list) for seg in merged)


def test_label_segments_with_topics():
    segments = [["This is about dogs."] * 5, ["Dogs are cute."] * 5, ["Bananas are yellow."] * 5]
    results = label_segments_with_topics(segments, model_path)
    assert isinstance(results, list)
    assert all(len(item) == 4 for item in results)


def test_process_transcript_full_pipeline():
    results = process_transcript(MOCK_TRANSCRIPT, with_timestamps=True, label=True, verbose=False)
    assert isinstance(results, list)
    assert all(isinstance(r, dict) and "start" in r and "end" in r for r in results)


def test_process_transcript_too_short():
    with pytest.raises(ValueError):
        process_transcript(SHORT_TRANSCRIPT, with_timestamps=True)

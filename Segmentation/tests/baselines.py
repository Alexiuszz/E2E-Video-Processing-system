import pytest
from Segmentation.baselines.baselines import RandomSeg, EvenSeg  


TRANSCRIPT = {
    "text": "This is sentence one. This is sentence two. This is sentence three. This is sentence four. This is sentence five. This is sentence six. This is sentence seven. This is sentence eight."
}

def test_random_seg_structure():
    seg = RandomSeg(TRANSCRIPT)
    segments = seg.segment(random_threshold=0.015)
    assert isinstance(segments, list)
    assert all(isinstance(s, list) for s in segments)
    assert sum(len(s) for s in segments) == len(seg.sentences)

def test_random_seg_reproducibility():
    seg1 = RandomSeg(TRANSCRIPT)
    seg2 = RandomSeg(TRANSCRIPT)
    assert seg1.segment() == seg2.segment()

def test_random_seg_low_threshold_many_sentences():
    seg = RandomSeg(TRANSCRIPT)
    segments = seg.segment(random_threshold=0.9)
    assert len(segments) > 1

def test_even_seg_structure():
    seg = EvenSeg(TRANSCRIPT, num_segments=4)
    segments = seg.segment()
    assert isinstance(segments, list)
    assert all(isinstance(s, list) for s in segments)
    assert sum(len(s) for s in segments) == len(seg.sentences)
    assert len(segments) == 4

def test_even_seg_fewer_sentences_than_segments():
    short_transcript = {"text": "One. Two. Three."}
    seg = EvenSeg(short_transcript, num_segments=5)
    segments = seg.segment()
    assert len(segments) == 5
    assert sum(len(s) for s in segments) == 3

def test_missing_text_key():
    with pytest.raises(ValueError, match="Transcript must have 'text' or 'segments' key"):
        RandomSeg({})
    with pytest.raises(ValueError, match="Transcript must have 'text' or 'segments' key"):
        EvenSeg({})

def test_empty_text():
    seg = RandomSeg({"text": ""})
    assert seg.segment() == []

    seg = EvenSeg({"text": ""}, num_segments=3)
    assert seg.segment() == [[], [], []]

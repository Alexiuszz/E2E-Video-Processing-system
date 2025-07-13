import re

def clean(text):
    """
    Cleans the input text by removing extra spaces and converting to lowercase.
    Args:
        text (str): The input text to be cleaned.
    Returns:
        str: The cleaned text.
    """
    return re.sub(r"\s+", " ", text.strip().lower())

def get_segment_bounds(original_segments, labeled_segments):
    """
    Aligns labeled segments with original segments based on the beginning of each segment.
    Args:
        original_segments (list): List of original segments with "text" keys.
        labeled_segments (list): List of tuples (index, segment, label, topics).
    Returns:
        list: List of tuples (index, original_segment_index, label, topics) indicating
        the alignment of labeled segments with original segments.
    """
    def clean(text):
        return re.sub(r"\s+", " ", text.strip().lower())

    match_words = determine_match_word_count(original_segments)
    segment_bounds = []
    current_segment_idx = 0

    for idx, segment, label, topics in labeled_segments:
        # Extract first few words from new segment
        words = clean(" ".join(segment)).split()
        match_snippet = " ".join(words[:match_words])

        # Search for the original segment that best matches the beginning
        for i in range(current_segment_idx, len(original_segments)):
            orig_text = clean(original_segments[i]["text"])
            if match_snippet in orig_text or orig_text.startswith(match_snippet[:20]):
                # Found a match: update label + start attaching
                # print(f"Match found for segment {i} with label '{label}'")
                current_segment_idx = i
                segment_bounds.append((idx, i, label, topics))
                break
            
    return segment_bounds

def determine_match_word_count(original_segments, min_words=5, max_words=20):
    """
    Determine an appropriate number of match words by finding the word count
    of the shortest text segment in the original segments.
    
    Args:
        original_segments: List of dicts with "text" keys.
        min_words (int): Minimum fallback word count.
        max_words (int): Maximum cap for match window.

    Returns:
        int: Suggested number of words to use for matching.
    """
    def word_count(text):
        return len(re.findall(r'\w+', text))

    shortest = min(word_count(seg["text"]) for seg in original_segments)

    return max(min_words, min(shortest, max_words))

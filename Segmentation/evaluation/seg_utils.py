
def pred_boundaries(segments, sentence_offset=0):
    idx = sentence_offset
    bounds = []
    for seg in segments:
        idx += len(seg)
        bounds.append(idx)          # first idx of next segment
    return bounds[:-1]


def masses_from_bounds(bounds, total_len):
    lens, prev = [], 0
    for b in bounds + [total_len]:
        lens.append(b - prev)
        prev = b
    return tuple(lens)

def calculate_pred_word_bounds(pred_segments, all_words):
    pred_word_bounds = []
    word_ptr = 0  # position in all_words
    for seg in pred_segments:
        # seg is a list of NLTK sentences; get raw length in characters
        seg_len_chars = sum(len(s) for s in seg)

        # walk forward in all_words until we've consumed ≥ seg_len_chars
        chars_seen = 0
        while word_ptr < len(all_words) and chars_seen < seg_len_chars:
            current_word = all_words[word_ptr]
            if current_word["punc"]:
                # Add punctuation length without space
                chars_seen += len(current_word["text"])
            else:
                # Add word length with space
                chars_seen += len(current_word["text"]) + 1  # +1 for space
            word_ptr += 1

        pred_word_bounds.append(word_ptr)
    return pred_word_bounds

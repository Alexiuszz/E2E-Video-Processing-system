import argparse
import sys, os
from pathlib import Path
import statistics
import  datasets_
from segeval import window_diff, pk
from nltk.tokenize import word_tokenize

from baselines import bert_seg, baselines


sys.path.append(str(Path(__file__).resolve().parent.parent /
                    'E2E_Video_Processing_System' / 'backend'))
from services.topic_segment import process_transcript

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    required=True,
    choices=["random", "bertseg", "default", "simple", "even"],
)
parser.add_argument(
    "--dataset", type=str, required=True, choices=["ytseg", "ami", "icsi"]
)
parser.add_argument(
    "--test_size", type=int, default=None, help="Limit the number of test meetings"
)

args = parser.parse_args()

wd_scores = []
pk_scores = []
num_samples = 0

BASE_DIR = os.getenv("BASE_DIR")
if args.dataset == "ami":
    meeting_data = datasets_.AMIDataLoader(BASE_DIR).extract_data(
        test_size=args.test_size
    )
elif args.dataset == "ytseg":
    meeting_data = datasets_.YTSegDataLoader(os.path.join(BASE_DIR, 'YTSEG_data', "clean")).extract_data(
        test_size=args.test_size
    )
elif args.dataset == "icsi":
    meeting_data = datasets_.ICSIDataLoader(BASE_DIR).extract_data(
        test_size=args.test_size
    )
else:
    raise ValueError(f"Unknown dataset: {args.dataset}")

wd_scores = []
pk_scores = []

for data in meeting_data:
    transcript_text = {"text": data['transcript']}
    all_words = data['all_words']
    gold_word_bounds = data['gold_word_bounds']
    n_words = len(all_words)
    
    if args.model == "default":
        pred_segments = process_transcript(
            transcript_text,
            with_timestamps=False,
            label=False,
            use_tiling=True,
            verbose=False
        )  
    elif args.model == "random":
        pred_segments = baselines.RandomSeg(transcript_text).segment()
    elif args.model == "bertseg":
        pred_segments = bert_seg.BertSeg(transcript_text).segment()
    elif args.model == "simple":
        pred_segments = process_transcript(
            transcript_text,
            with_timestamps=False,
            label=False,
            use_tiling=False,
            verbose=False
        ) 
    elif args.model == "even":
        pred_segments = baselines.EvenSeg(transcript_text).segment()
    
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    if args.dataset == "ytseg":
        pred_word_bounds = datasets_.convert_predicted_segments_to_word_boundaries(pred_segments)
        n_words =  len(word_tokenize(transcript_text['text']))
    else:
        pred_word_bounds = datasets_.calculate_pred_word_bounds(pred_segments, all_words)

    ref_masses = datasets_.masses_from_bounds(gold_word_bounds, n_words)
    hyp_masses = datasets_.masses_from_bounds(pred_word_bounds, n_words)

    wd = window_diff(ref_masses, hyp_masses)
    pk_score = pk(ref_masses, hyp_masses)
    wd_scores.append(wd)
    pk_scores.append(pk_score)
    
    print(f" gold={len(gold_word_bounds):3}  pred={len(pred_word_bounds):3}  "
                  f"avg-gold-len={len(all_words)/(len(gold_word_bounds)+1):.1f}  "
                  f"avg-pred-len={len(all_words)/(len(pred_word_bounds)+1):.1f}")


# Calculate final metrics
if wd_scores and pk_scores:
    results = {
        'mean_window_diff': statistics.mean(wd_scores),
        'std_window_diff': statistics.stdev(wd_scores) if len(wd_scores) > 1 else 0.0,
        'mean_pk': statistics.mean(pk_scores),
        'std_pk': statistics.stdev(pk_scores) if len(pk_scores) > 1 else 0.0,
        'valid_samples': len(wd_scores),
        'total_samples': len(meeting_data),
    }
    
    print(f'\n=== FINAL RESULTS ===')
    print(f'Valid samples: {results["valid_samples"]}/{results["total_samples"]}')
    print(f'Mean WindowDiff: {results["mean_window_diff"]:.3f} (±{results["std_window_diff"]:.3f})')
    print(f'Mean Pk: {results["mean_pk"]:.3f} (±{results["std_pk"]:.3f})')  
else:
    print("No valid scores computed!")
    
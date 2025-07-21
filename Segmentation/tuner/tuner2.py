import os
os.environ["ACCELERATE_DISABLE_INFERENCE_OPTIMIZATIONS"] = "1"
import optuna, numpy as np, statistics, sys
from pathlib import Path
from segeval import window_diff, pk as pk_metric   # alias to avoid name clash
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent / 'evaluation'))
from evaluation import (load_reference_segments,
                        masses_from_bounds, load_transcript)
from services.topic_segment import process_transcript_optimize2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
model_path = os.environ.get("MODEL_PATH", "SentenceTransformer/all-MiniLM-L6-v2")

BASE_DIR  = os.getenv('BASE_DIR')
ami_root  = Path(BASE_DIR) / 'ami_public_manual_1.6.2'
topics_dir, words_dir = ami_root/'topics', ami_root/'words'

meeting_ids = sorted(p.stem.split('.')[0] for p in topics_dir.glob('*.topic.xml'))[:10]
print("Optimising on", len(meeting_ids), "meetings:", meeting_ids)

model = SentenceTransformer(model_path)
# ─── Pre-compute reference data once to save time ─────────────────────────────
ref_data = {}
for mid in meeting_ids:
    txt, all_words, wid2idx = load_transcript(ami_root, mid)
    gold_bounds = load_reference_segments(topics_dir/f"{mid}.topic.xml", wid2idx)
    n_words     = len(all_words)
    ref_masses  = masses_from_bounds(gold_bounds, n_words)
    ref_data[mid] = dict(text     = txt,
                         words    = all_words,
                         masses   = ref_masses)

# ─── Optuna objective ─────────────────────────────────────────────────────────
def objective(trial):
    cfg = dict(
        depth_threshold             = trial.suggest_float   ("depth_t",              0.65, 0.95),
        window_size            = trial.suggest_int   ("window_size",             3, 10),
        top_n             = trial.suggest_int ("top_n",            5, 10),
        min_topics     = trial.suggest_int   ("min_topics",     1, 4),
    )

    trial_wd, trial_pk = [], []

    for mid in meeting_ids:
        txt   = ref_data[mid]['text']
        words = ref_data[mid]['words']
        ref_m = ref_data[mid]['masses']

        pred_segments = process_transcript_optimize2(
            {"text": txt},
            model=model,
            depth_threshold=cfg['depth_threshold'],
            window_size=cfg['window_size'],
            top_n=cfg['top_n'],
            min_topics=cfg['min_topics'],
        )

        # convert predicted segments -> word boundaries
        word_ptr, pred_bounds = 0, []
        for seg in pred_segments:
            chars = sum(len(s) for s in seg)
            chars_seen = 0
            while word_ptr < len(words) and chars_seen < chars:
                chars_seen += len(words[word_ptr]['text']) + 1
                word_ptr  += 1
            pred_bounds.append(word_ptr)
        pred_bounds = pred_bounds[:-1]

        hyp_m = masses_from_bounds(pred_bounds, len(words))
        trial_wd.append(float(window_diff(ref_m, hyp_m)))
        trial_pk.append(float(pk_metric(ref_m, hyp_m)))

    # report to Optuna (minimise WD primarily)
    mean_wd = statistics.mean(trial_wd)
    trial.set_user_attr("pk", statistics.mean(trial_pk))
    return mean_wd

# ─── Run study ────────────────────────────────────────────────────────────────
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50, n_jobs=4)    # 4 CPU threads on Mac

best = study.best_trial
print("Best WD :", best.value)
print("Best Pk :", best.user_attrs['pk'])
print("Params  :", best.params)
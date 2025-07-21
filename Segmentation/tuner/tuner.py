import os
os.environ["ACCELERATE_DISABLE_INFERENCE_OPTIMIZATIONS"] = "1"
import optuna, numpy as np, statistics, sys
from pathlib import Path
from segeval import window_diff, pk as pk_metric   # alias to avoid name clash
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent / 'AMI_evaluation'))
from evaluation import (load_reference_segments,
                        masses_from_bounds, load_transcript)
from services.topic_segment import process_transcript_optimize
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
        wsize             = trial.suggest_int   ("wsize",              5, 16),
        stride            = trial.suggest_int   ("stride",             1, 8),
        top_p             = trial.suggest_float ("top_p",            0.2, 0.40),
        min_seg_words     = trial.suggest_int   ("min_seg_words",     15, 27),
        similarity_th     = trial.suggest_float ("sim_thresh",       0.10, 0.30),
        smoothing_factor  = trial.suggest_float ("smooth",            0.0, 0.1),
    )

    trial_wd, trial_pk = [], []

    for mid in meeting_ids:
        txt   = ref_data[mid]['text']
        words = ref_data[mid]['words']
        ref_m = ref_data[mid]['masses']

        pred_segments = process_transcript_optimize(
            {"text": txt},
            model=model,
            wsize=cfg['wsize'],
            stride=cfg['stride'],
            top_p=cfg['top_p'],
            min_seg_words=cfg['min_seg_words'],
            similarity_threshold=cfg['similarity_th'],
            # smoothing_factor=0.022
            smoothing_factor=cfg['smoothing_factor']
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
    mean_pk = statistics.mean(trial_pk)
    trial.set_user_attr("pk", mean_pk)
    return mean_wd

# ─── Run study ────────────────────────────────────────────────────────────────
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=200, n_jobs=4)    # 4 CPU threads on Mac

best = study.best_trial
print("Best WD :", best.value)
print("Best Pk :", best.user_attrs['pk'])
print("Params  :", best.params)

# Trial 2304 finished with value: 0.43495945088318216 and parameters: 
# {'wsize': 5, 'stride': 2, 'top_p': 0.2545614965647016, 
#  'min_seg_words': 30, 'sim_thresh': 0.17342111913946634, 
#  'smooth': 0.06798311171791104}.



# Trial 109 finished with value: 0.3902386459813594 and parameters:
# {'wsize': 5, 'stride': 1, 'top_p': 0.3887665449707551, 
# 'min_seg_words': 15, 'sim_thresh': 0.24883852362921505, 
# 'smooth': 0.003420071817983681}.

#Trial 58 finished with value: 0.3758641571078737 and parameters: 
# {'wsize': 6, 'stride': 1, 'top_p': 0.2863514161532891, 
# 'min_seg_words': 10, 'sim_thresh': 0.15658428787084155, 
# 'smooth': 0.07229483553832433}. B
# est is trial 58 with value: 0.3758641571078737.
# 

# Trial 95 finished with value: 0.3666902758577521 and parameters: {'wsize': 6, 
# 'stride': 1, 'top_p': 0.27377888663104794, 'min_seg_words': 10, 'sim_thresh': 0.27555865717548594, 'smooth': 0.03307959908564207}.

# Trial 38 finished with value: 0.31025208031165863 and parameters: 
# {'wsize': 8, 'stride': 1, 'top_p': 0.2987498775337346, 'min_seg_words': 3, 
# 'sim_thresh': 0.1678738699385769, 'smooth': 0.014519778222003319}


#Trial 105 finished with value: 0.4078074206529129 and parameters: 
# {'wsize': 9, 'stride': 3, 'top_p': 0.29250991690051975, 
# 'min_seg_words': 24, 'sim_thresh': 0.11645555244561646, 'smooth': 0.030911930635985903}.

# Best WD : 0.4070651508046182
# Best Pk : 0.35138570403290004
# Params  : {'wsize': 9, 'stride': 3, 'top_p': 0.29004796820043927, 
# 'min_seg_words': 26, 'sim_thresh': 0.16037063513823613,
# 'smooth': 0.03025302005204547}

# Trial 88 finished with value: 0.35210818214146555 and parameters: 
# {'wsize': 12, 'stride': 3, 'top_p': 0.3041508624216481, 
# 'min_seg_words': 15, 'sim_thresh': 0.21094430764109856, 
# 'smooth': 0.02220417321479462}.

# Trial 89 finished with value: 0.46659207551693177 and parameters: {'wsize': 11, 'stride': 1, 'top_p': 0.27187494459217926, 'min_seg_words': 20, 'sim_thresh': 0.2953949904667779}. Best is trial 89 with value: 0.46659207551693177.

# Trial 52 finished with value: 0.41971383952276675 and parameters: {'wsize': 11, 'stride': 2, 'top_p': 0.315621020671032, 'min_seg_words': 18, 'sim_thresh': 0.1543273675370054, 'smooth': 0.011792304024098733}. Best is trial 52 with value: 0.41971383952276675.

#Best WD : 0.3989228332192176
#Best Pk : 0.35656352558020094
# Params  : {'wsize': 13, 'stride': 1, 'top_p': 0.3029645509513113, 'min_seg_words': 15, 'sim_thresh': 0.16789668391940987, 'smooth': 0.012701589882878582}


#Best WD : 0.416583765987841
# Best Pk : 0.36817079215339005
# Params  : {'wsize': 10, 'stride': 3, 'top_p': 0.3585415867988901, 'min_seg_words': 15, 'sim_thresh': 0.1506367775040113, 'smooth': 0.014092583443663015}
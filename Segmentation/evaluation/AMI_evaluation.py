import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import glob, statistics
import sys
from pathlib import Path, PurePath
from lxml import etree
from segeval import window_diff, pk
import os
import re

from typing import List, Tuple

from dotenv import load_dotenv
load_dotenv()

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / 'E2E_Video_Processing_System' / 'backend'))

from services.topic_segment import process_transcript
from seg_utils import calculate_pred_word_bounds, masses_from_bounds

class AMIEvaluation:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.ami_root = Path(base_dir) / 'ami_public_manual_1.6.2'

    def load_transcript(self, meeting_id: str) -> Tuple[str, List[dict], dict]:
        """Return (transcript_text, all_words, wid2idx) for one AMI meeting."""
        words_dir = self.ami_root / "words"
        xml_paths = glob.glob(str(words_dir / f"{meeting_id}.*.words.xml"))
        if not xml_paths:
            raise FileNotFoundError(f"No *.words.xml files for {meeting_id}")

        all_words = []
        for xp in xml_paths:
            spk = PurePath(xp).stem.split(".")[1]         # "A", "B", …
            tree = etree.parse(xp)

            nite_ns = tree.getroot().nsmap.get('nite')
            id_key = f"{{{nite_ns}}}id" if nite_ns else "id"   # prefer namespaced

            for w in tree.xpath("//w"):
                wid = w.get(id_key) or w.get("id")           # <- works for both
                txt = (w.text or "").strip()
                punc = w.get("punc", "").strip()  # punctuation, if any
                st, et = w.get("starttime"), w.get("endtime")

                if not wid or not txt or not st or not et:
                    continue

                all_words.append({
                    "start": float(st),
                    "end": float(et),
                    "text": txt,
                    "spk": spk,
                    "wid": wid,
                    "punc": True if punc else False,
                })

        all_words.sort(key=lambda d: d["start"])
        transcript_text = ""
        for w in all_words:
            if not w["punc"]:
                transcript_text += " " + w["text"]
            else:
                transcript_text += w["text"]

        wid2idx = {w["wid"]: i for i, w in enumerate(all_words)}
        return transcript_text, all_words, wid2idx

    def load_reference_segments(self, xml_path: Path, wid2idx: dict) -> List[int]:
        """Extract topic boundaries from an AMI *.topic.xml file."""
        tree = etree.parse(str(xml_path))
        topics = tree.xpath('//*[local-name()="topic"]')  # ignore namespace

        gold_word_bounds = []
        for tp in topics:
            child = tp.xpath('./*[local-name()="child"]')
            if not child:
                continue
            href = child[0].get('href', '')
            match = re.search(r'id\(([^)]+)\)', href)
            if not match:
                continue
            wid = match.group(1)  # ES2002a.B.words584
            word_idx = wid2idx.get(wid)
            if word_idx is not None:
                gold_word_bounds.append(word_idx)
        return sorted(set(gold_word_bounds))

    def evaluate(self, test_size: int = None):
        topics_dir = self.ami_root / "topics"
        topic_files = sorted(topics_dir.glob("*.topic.xml"))
        meeting_ids = [p.stem.split(".")[0] for p in topic_files]
        print(f"{len(meeting_ids)} meetings with topic annotations found.")

        scores = []
        pk_scores = []
        
        if test_size:
            meeting_ids = meeting_ids[:test_size]
            print(f"Evaluating on {len(meeting_ids)} meetings (test size limit).")

        for mid in meeting_ids:
            transcript_text, all_words, wid2idx = self.load_transcript(mid)
            n_words = len(all_words)
            sentences = sent_tokenize(transcript_text)
            print(f'  Number of sentences: {len(sentences)}')

            topic_xml = topics_dir / f"{mid}.topic.xml"
            gold_word_bounds = self.load_reference_segments(topic_xml, wid2idx)

            formatted_transcript = {"text": transcript_text}
            pred_segments = process_transcript(formatted_transcript, with_timestamps=False, label=False, use_tiling=True, verbose=False)
            pred_word_bounds = calculate_pred_word_bounds(pred_segments, all_words)

            ref_masses = masses_from_bounds(gold_word_bounds, n_words)
            hyp_masses = masses_from_bounds(pred_word_bounds, n_words)

            wd = window_diff(ref_masses, hyp_masses)
            pk_score = pk(ref_masses, hyp_masses)
            scores.append(wd)
            pk_scores.append(pk_score)

            print(f"{mid:8}  gold={len(gold_word_bounds):3}  pred={len(pred_word_bounds):3}  "
                  f"avg-gold-len={len(all_words)/(len(gold_word_bounds)+1):.1f}  "
                  f"avg-pred-len={len(all_words)/(len(pred_word_bounds)+1):.1f}")



        print(f'Mean WindowDiff over {len(scores)} meetings: {statistics.mean(scores):.3f}')
        print(f'Mean PK over {len(pk_scores)} meetings: {statistics.mean(pk_scores):.3f}')


if __name__ == "__main__":
    BASE_DIR = os.getenv('BASE_DIR')
    evaluator = AMIEvaluation(BASE_DIR)
    evaluator.evaluate(5)

import glob, statistics, re, sys, os
from pathlib import Path, PurePath
from typing import List, Tuple, Dict, Set

from lxml import etree
from nltk.tokenize import word_tokenize
from segeval import window_diff, pk   # pip install segeval

# your codebase
sys.path.append(str(Path(__file__).resolve().parent.parent.parent /
                    'E2E_Video_Processing_System' / 'backend'))
from services.topic_segment import process_transcript
from seg_utils import calculate_pred_word_bounds, masses_from_bounds


class ICSIEvaluation:
    """
    Evaluate `process_transcript` on the ICSI Meeting Corpus
    using WindowDiff and Pk.
    """
    def __init__(self, base_dir: str):
        print(f"Using ICSI corpus at {base_dir}")
        self.root = Path(base_dir).resolve().parent / 'ICSIPlus'
        self.words_dir = self.root / 'Words'
        self.segs_dir = self.root / 'Segments'
        self.topics_dir = self.root / 'Contributions' / 'TopicSegmentation'
        self.meeting_ids = self._discover_meetings()

    def _discover_meetings(self) -> List[str]:
        meta = etree.parse(str(self.root / 'ICSI-metadata.xml'))
        return [o.get('name') for o in meta.xpath('//observation')]
    

   #  Put this method inside the ICSIEvaluation class
    def load_transcript(self, mid: str):
        """
        Build:
        • transcript_text  – detokenised string (AMI spacing rules)
        • all_words        – one dict per *logical* word, incl. start/end
        • wid2idx          – maps every original nite:id (even the inner
                            parts of “I’ve”) to the index of its word
        """
        xml_paths = glob.glob(str(self.words_dir / f"{mid}.*.words.xml"))
        if not xml_paths:
            raise FileNotFoundError(f"No words XML for meeting {mid}")

        PUNC_CHARS = {".", ",", "?", "!", ";", ":"}

        all_words, wid2idx, transcript_parts = [], {}, []
        current = None            # rolling buffer for a multi-token word

        for xp in xml_paths:
            tree = etree.parse(xp)
            ns   = tree.getroot().nsmap.get('nite')
            idkey = f"{{{ns}}}id" if ns else "id"
            spk   = PurePath(xp).stem.split(".")[1]      # channel letter

            for w in tree.xpath('//w'):
                wid  = w.get(idkey)
                txt  = (w.text or '').strip()
                st, et = w.get('starttime'), w.get('endtime')

                # 1) if this token STARTS a word …
                if st:
                    # -- close any previous unfinished word (shouldn't happen)
                    if current:
                        current["end"] = current["start"]   # dummy end
                        all_words.append(current)

                    current = {
                        "text": txt,
                        "start": float(st),
                        "end": None,
                        "spk": spk,
                        "punc": txt in PUNC_CHARS and not et,
                        "wids": [wid],
                    }

                # 2) continuation of the current word
                else:
                    if current is None:           # safety guard
                        continue
                    current["text"] += txt       # concatenate literally
                    current["wids"].append(wid)

                # 3) does this token FINISH the word?
                if et:
                    if current is None:
                        continue
                    current["end"] = float(et)
                    # any single-character punctuation token whose start & end
                    # are the same is treated as punctuation
                    if len(current["text"]) == 1 and current["text"] in PUNC_CHARS:
                        current["punc"] = True
                    all_words.append(current)
                    idx = len(all_words) - 1
                    for xid in current["wids"]:
                        wid2idx[xid] = idx
                    current = None   # reset for next word

        # ----------- build plain-text transcript with AMI spacing -----------
        for i, w in enumerate(all_words):
            if i == 0 or w["punc"]:
                transcript_parts.append(w["text"])
            else:
                transcript_parts.append(" " + w["text"])

        transcript_text = "".join(transcript_parts)
        return transcript_text, all_words, wid2idx


    def _segment_start_idx(self, seg_href: str, word_starts: List[float]) -> int:
        """
        seg_href example: 'Bdb001.A.segs.xml#id(Bdb001.segment.11)'
        """
        m = re.match(r'(.+\.segs\.xml)#id\(([^)]+)\)', seg_href)
        if not m:
            return None
        seg_path, seg_id = m.groups()
        seg_tree = etree.parse(str(self.segs_dir / seg_path))
        seg_el   = seg_tree.xpath(f'//*[@nite:id="{seg_id}"]', namespaces=seg_tree.getroot().nsmap)
        if not seg_el:
            return None
        st = float(seg_el[0].get('starttime', '0'))
        # binary search for first word with start ≥ st
        lo, hi = 0, len(word_starts) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if word_starts[mid] < st:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def load_reference_bounds(self, mid: str, wid2idx: Dict[str, int],
                              word_starts: List[float]) -> List[int]:
        topic_path = self.topics_dir / f"{mid}.topic.xml"
        tree = etree.parse(str(topic_path))
        nsm  = tree.getroot().nsmap

        gold: Set[int] = set()
        # DFS through all topic nodes
        for tp in tree.xpath('//*[local-name()="topic"]'):
            child = tp.xpath('./nite:child', namespaces=nsm)
            if not child:
                continue
            href = child[0].get('href', '')

            if '.words.' in href:                       # pointer directly to a word
                wid = re.search(r'id\(([^)]+)\)', href).group(1)
                idx = wid2idx.get(wid)
            else:                                       # pointer to a segment
                idx = self._segment_start_idx(href, word_starts)

            if idx is not None:
                gold.add(idx)

        return sorted(gold)
    
    
    def evaluate(self, limit: int = None):
        mids = self.meeting_ids[:limit] if limit else self.meeting_ids
        print(f"Evaluating {len(mids)} ICSI meetings…")

        wd_scores, pk_scores = [], []
        for mid in mids:
            transcript, words, wid2idx = self.load_transcript(mid)
            word_starts = [w['start'] for w in words]

            gold_bounds = self.load_reference_bounds(mid, wid2idx, word_starts)

            # run your segmenter
            formatted = {"text": transcript}
            pred_segs = process_transcript(formatted, with_timestamps=False,
                                           label=False, use_tiling=True, verbose=False)
            
            pred_bounds = calculate_pred_word_bounds(pred_segs, words)

            n = len(words)
            ref_mass = masses_from_bounds(gold_bounds, n)
            hyp_mass = masses_from_bounds(pred_bounds, n)

            wd = window_diff(ref_mass, hyp_mass)
            pk_ = pk(ref_mass, hyp_mass)

            wd_scores.append(wd)
            pk_scores.append(pk_)

            print(f"{mid:8}  gold={len(gold_bounds):3}  pred={len(pred_bounds):3} "
                  f"WD={wd:.3f}  Pk={pk_:.3f}")

        print(f"\nMean WindowDiff: {statistics.mean(wd_scores):.3f}")
        print(f"Mean Pk        : {statistics.mean(pk_scores):.3f}")


if __name__ == "__main__":
    BASE_DIR = os.getenv("BASE_DIR")  # set to the root of your ICSI corpus
    ICSIEvaluation(BASE_DIR).evaluate(limit=1)  # evaluate first 5 meetings

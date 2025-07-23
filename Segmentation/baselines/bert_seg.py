import numpy as np
import torch
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize
from typing import NamedTuple, List, Optional

from transformers import RobertaTokenizer, RobertaModel


class TextTilingHyperparameters(NamedTuple):
    """Hyper‑parameters for the TextTiling‑style topic segmentation pipeline."""

    SENTENCE_COMPARISON_WINDOW: int = 15
    SMOOTHING_PASSES: int = 2
    SMOOTHING_WINDOW: int = 1
    TOPIC_CHANGE_THRESHOLD: float = 0.6


class TopicSegmentationConfig(NamedTuple):
    """Optional knobs for segment‑cap UI tweaks."""

    TEXT_TILING: Optional[TextTilingHyperparameters] = TextTilingHyperparameters()
    MAX_SEGMENTS_CAP: bool = True
    MAX_SEGMENTS_CAP__AVERAGE_SEGMENT_LENGTH: int = 60  # in *sentences*


class BertSeg:
    """RoBERTa‑based TextTiling implementation that works on **raw transcripts**.

    Parameters
    ----------
    transcript : str
        Entire meeting transcript as **one string**.
    device : str, optional
        Torch device ("cpu", "mps", "cuda") – default "cpu" for Mac‑M1.
    model_name : str, optional
        Any HF Roberta‑family checkpoint – default "roberta‑base".
    """

    def __init__(self, transcript: str, device: str = "cpu", model_name: str = "roberta-base") -> None:
        self.sentences: List[str] = sent_tokenize(transcript['text'].strip())
        self.device = torch.device(device)

        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model: RobertaModel = RobertaModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def segment(
        self,
        config: TopicSegmentationConfig = TopicSegmentationConfig(),
    ) -> List[List[str]]:
        """Return **segments** (list‑of‑list of sentences), *not* raw boundaries."""

        # 1. Sentence embeddings ------------------------------------------------
        embeddings = self._get_sentence_embeddings(self.sentences)  # [n, 768] tensors

        # 2. TextTiling steps ---------------------------------------------------
        tt = config.TEXT_TILING or TextTilingHyperparameters()

        comp_scores = self.block_comparison_score(embeddings, k=tt.SENTENCE_COMPARISON_WINDOW)
        comp_scores = self.smooth(comp_scores, n=tt.SMOOTHING_PASSES, s=tt.SMOOTHING_WINDOW)
        depth_scores = self.depth_score(comp_scores)
        boundaries = self.depth_score_to_topic_change_indexes(
            depth_scores,
            meeting_duration=len(self.sentences),  # duration measured in sentences
            topic_segmentation_configs=config,
        )
        boundaries = sorted(boundaries)

        # 3. Convert boundary indices → list‑of‑segments -------------------------
        segments: List[List[str]] = []
        prev = 0
        for b in boundaries:
            segments.append(self.sentences[prev : b + 1])  # include boundary sentence
            prev = b + 1
        segments.append(self.sentences[prev:])  # tail
        return segments

    # ---------------------------------------------------------------------
    # Embedding helpers
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def _get_sentence_embeddings(self, sentences: List[str], layer: int = -2) -> List[torch.Tensor]:
        """Average‑pool hidden states from the penultimate layer for each sentence."""
        results: List[torch.Tensor] = []
        for sent in sentences:
            tokens = self.tokenizer(sent, return_tensors="pt", truncation=True).to(self.device)
            output = self.model(**tokens, output_hidden_states=True)
            hidden = output.hidden_states[layer]  # [1, seq, h]
            pooled = hidden.mean(dim=1).squeeze(0)  # [h]
            results.append(pooled.cpu())  # keep on CPU to save VRAM
        return results

    # ---------------------------------------------------------------------
    # TextTiling core
    # ---------------------------------------------------------------------
    @staticmethod
    def smooth(timeseries: List[float], n: int, s: int) -> List[float]:
        """n iterative passes of simple moving‑average with window radius *s*."""
        smoothed = timeseries[:]
        for _ in range(n):
            for idx in range(len(smoothed)):
                neighbours = smoothed[max(0, idx - s) : min(len(smoothed) - 1, idx + s) + 1]
                smoothed[idx] = sum(neighbours) / len(neighbours)
        return smoothed

    @staticmethod
    def _get_local_maxima(arr: List[float]):
        idxs, vals = [], []
        for i in range(1, len(arr) - 1):
            if arr[i - 1] < arr[i] and arr[i] > arr[i + 1]:
                idxs.append(i)
                vals.append(arr[i])
        return idxs, vals

    @staticmethod
    def depth_score(timeseries: List[float]) -> List[float]:
        scores: List[float] = []
        for i in range(1, len(timeseries) - 1):
            left, right = i - 1, i + 1
            # ascend until peak on the left
            while left > 0 and timeseries[left - 1] > timeseries[left]:
                left -= 1
            # ascend until peak on the right
            while right < len(timeseries) - 1 and timeseries[right + 1] > timeseries[right]:
                right += 1
            scores.append((timeseries[right] - timeseries[i]) + (timeseries[left] - timeseries[i]))
        return scores

    @staticmethod
    def _sent_sim(a: torch.Tensor, b: torch.Tensor) -> float:
        sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1)
        return float(sim.item())

    def _window_mean(self, timeseries: List[torch.Tensor], start: int, end: int) -> torch.Tensor:
        # [window, h] → mean → [h]
        stack = torch.stack(timeseries[start:end]).to(self.device)
        return stack.mean(dim=0)

    def block_comparison_score(self, timeseries: List[torch.Tensor], k: int) -> List[float]:
        """Cosine similarity between two k‑sentence windows centred on each gap."""
        res: List[float] = []
        for i in range(k, len(timeseries) - k):
            w1 = self._window_mean(timeseries, i - k, i)
            w2 = self._window_mean(timeseries, i, i + k)
            res.append(self._sent_sim(w1, w2))
        return res

    # ---------------------------------------------------------------------
    # Depth‑to‑boundary logic (capped version optional)
    # ---------------------------------------------------------------------
    def depth_score_to_topic_change_indexes(
        self,
        depth_scores: List[float],
        meeting_duration: int,
        topic_segmentation_configs: TopicSegmentationConfig,
    ) -> List[int]:
        if not depth_scores:
            return []

        capped = topic_segmentation_configs.MAX_SEGMENTS_CAP
        avg_seg_len = topic_segmentation_configs.MAX_SEGMENTS_CAP__AVERAGE_SEGMENT_LENGTH
        threshold = (
            topic_segmentation_configs.TEXT_TILING.TOPIC_CHANGE_THRESHOLD
            * max(depth_scores)
        )

        idxs, vals = self._get_local_maxima(depth_scores)
        if not vals:
            return []

        if capped:
            # Sort by depth score desc
            vals_np, idxs_np = self._arsort_two(vals, idxs)
            # prune below threshold
            keep = [v > threshold for v in vals_np]
            vals_np = vals_np[keep]
            idxs_np = idxs_np[keep]
            max_segments = int(meeting_duration / avg_seg_len)
            idxs_np = idxs_np[:max_segments]
            # back to chrono order
            idxs_np, _ = self._arsort_two(idxs_np, vals_np)
            return list(idxs_np)
        else:
            return [i for i, v in zip(idxs, vals) if v > threshold]

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------
    @staticmethod
    def _arsort_two(arr1, arr2):
        x = np.array(arr1)
        y = np.array(arr2)
        order = x.argsort()[::-1]
        return x[order], y[order]

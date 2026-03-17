from __future__ import annotations
import re
import random
import torch as t
from datasets import load_dataset
from .base import DatasetAdapter, BatchList, SubgroupDict

# HuggingFace integer label IDs for nyu-mll/multi_nli
LABEL_ENTAILMENT    = 0
LABEL_NEUTRAL       = 1
LABEL_CONTRADICTION = 2

# Negation words that are well-documented annotation artifacts in NLI hypotheses.
# A hypothesis-only model achieves ~61% accuracy on MultiNLI (vs. 33% chance)
# largely because these tokens spuriously predict Contradiction.
# Source: Gururangan et al. (2018) "Annotation Artifacts in NLI Data"
_NEGATION_RE = re.compile(
    r"\b(nobody|no|never|nothing|nor|neither|without|not)\b|n't",
    re.IGNORECASE,
)


def _has_negation(hypothesis: str) -> bool:
    return bool(_NEGATION_RE.search(hypothesis))


class MultiNLIAdapter(DatasetAdapter):
    """
    Adapter for nyu-mll/multi_nli.

    Frames NLI as a binary task (Entailment vs. Contradiction); Neutral examples
    are discarded. The spurious feature is the presence of negation words in the
    hypothesis, which are strong annotation-artifact predictors of Contradiction.

    Label convention:
      true_label = 0  =>  Entailment
      true_label = 1  =>  Contradiction
      spurious_label = 0  =>  hypothesis contains NO negation words
      spurious_label = 1  =>  hypothesis contains negation words
                              (nobody / no / never / nothing / nor /
                               neither / without / not / n't)

    Biased (ambiguous) split:
      (Entailment, no-negation) + (Contradiction, with-negation)
    Balanced split:
      all four (label × negation) combinations equally.

    Args:
        activation_dim_value: hidden size of the target model (e.g. 2304 for Gemma-2-2B)
        hf_dataset_name: HuggingFace dataset identifier (default: nyu-mll/multi_nli)
        validation_split: which validation split to use as the test set;
                          either "validation_matched" or "validation_mismatched"
                          (default: "validation_matched")
    """

    def __init__(
        self,
        activation_dim_value: int,
        hf_dataset_name: str = "nyu-mll/multi_nli",
        validation_split: str = "validation_matched",
    ):
        self._activation_dim = activation_dim_value
        self._validation_split = validation_split
        self._dataset = load_dataset(hf_dataset_name)

    @property
    def activation_dim(self) -> int:
        return self._activation_dim

    def _build_buckets(
        self, split: str
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """Return (ent_no_neg, ent_neg, con_no_neg, con_neg) text lists."""
        split_name = "train" if split == "train" else self._validation_split
        data = self._dataset[split_name]
        ent_no_neg: list[str] = []
        ent_neg:    list[str] = []
        con_no_neg: list[str] = []
        con_neg:    list[str] = []
        for x in data:
            lbl = x["label"]
            if lbl not in (LABEL_ENTAILMENT, LABEL_CONTRADICTION):
                continue  # discard Neutral
            text = x["premise"] + " " + x["hypothesis"]
            neg  = _has_negation(x["hypothesis"])
            if lbl == LABEL_ENTAILMENT:
                (ent_neg if neg else ent_no_neg).append(text)
            else:
                (con_neg if neg else con_no_neg).append(text)
        return ent_no_neg, ent_neg, con_no_neg, con_neg

    def get_batches(
        self,
        split: str = "train",
        ambiguous: bool = True,
        batch_size: int = 32,
        seed: int = 42,
        device: str = "cpu",
    ) -> BatchList:
        ent_no_neg, ent_neg, con_no_neg, con_neg = self._build_buckets(split)

        if ambiguous:
            # Correlated quadrants only: (Entailment ∩ no-neg) + (Contradiction ∩ neg)
            n               = min(len(ent_no_neg), len(con_neg))
            texts           = ent_no_neg[:n] + con_neg[:n]
            true_labels     = [0] * n + [1] * n
            spurious_labels = [0] * n + [1] * n
            idxs            = list(range(2 * n))
        else:
            n               = min(len(ent_no_neg), len(ent_neg), len(con_no_neg), len(con_neg))
            texts           = ent_no_neg[:n] + ent_neg[:n] + con_no_neg[:n] + con_neg[:n]
            true_labels     = [0]*n + [0]*n + [1]*n + [1]*n
            spurious_labels = [0]*n + [1]*n + [0]*n + [1]*n
            idxs            = list(range(4 * n))

        random.Random(seed).shuffle(idxs)
        texts           = [texts[i]           for i in idxs]
        true_labels     = [true_labels[i]     for i in idxs]
        spurious_labels = [spurious_labels[i] for i in idxs]

        return [
            (
                texts[i : i + batch_size],
                t.tensor(true_labels[i : i + batch_size],     device=device),
                t.tensor(spurious_labels[i : i + batch_size], device=device),
            )
            for i in range(0, len(texts), batch_size)
        ]

    def get_subgroups(
        self,
        split: str = "test",
        batch_size: int = 32,
        device: str = "cpu",
    ) -> SubgroupDict:
        ent_no_neg, ent_neg, con_no_neg, con_neg = self._build_buckets(split)
        subgroups_raw = {
            (0, 0): ent_no_neg,
            (0, 1): ent_neg,
            (1, 0): con_no_neg,
            (1, 1): con_neg,
        }
        out: SubgroupDict = {}
        for label_profile, texts in subgroups_raw.items():
            true_lbl, spur_lbl = label_profile
            out[label_profile] = [
                (
                    texts[i : i + batch_size],
                    t.tensor([true_lbl] * len(texts[i : i + batch_size]), device=device),
                    t.tensor([spur_lbl] * len(texts[i : i + batch_size]), device=device),
                )
                for i in range(0, len(texts), batch_size)
            ]
        return out

    def get_sample_prompt(self) -> str:
        _, _, _, con_neg = self._build_buckets("test")
        con_neg_sorted = sorted(con_neg, key=len)
        return con_neg_sorted[0]

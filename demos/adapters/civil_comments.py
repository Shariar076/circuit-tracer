from __future__ import annotations
import random
import torch as t
from datasets import load_dataset
from .base import DatasetAdapter, BatchList, SubgroupDict

# Binarisation thresholds.
# identity_attack: fraction of annotators who flagged the comment as attacking
# an identity group. Any non-zero score (> 0) means at least one annotator
# noted it — this is the spurious feature correlated with toxicity.
_TOXICITY_THRESHOLD       = 0.5
_IDENTITY_ATTACK_THRESHOLD = 0.0  # > 0 means at least one annotator flagged it


class CivilCommentsAdapter(DatasetAdapter):
    """
    Adapter for google/civil_comments (the WILDS spurious-correlation benchmark).

    Frames comment toxicity as a binary classification task with identity-attack
    annotation as the spurious feature.

    The spurious correlation: comments flagged as "attacking an identity group"
    are over-represented in the toxic class, so a naive classifier learns to
    predict toxicity from identity-attack signals rather than actual content.
    `identity_attack` is the fraction of annotators who flagged the comment as
    an attack on an identity group; any non-zero score (> 0) counts as spurious=1.

    Note: the full WILDS-style demographic identity columns (male, female, LGBTQ,
    etc.) are not available in `google/civil_comments`. `identity_attack` is the
    closest available proxy and encodes a related spurious correlation.

    Label convention:
      true_label = 0  =>  non-toxic        (toxicity score < 0.5)
      true_label = 1  =>  toxic            (toxicity score ≥ 0.5)
      spurious_label = 0  =>  no identity-attack annotation (identity_attack == 0)
      spurious_label = 1  =>  at least one annotator flagged identity attack
                              (identity_attack > 0)

    Biased (ambiguous) split:
      (non-toxic, no-identity-attack) + (toxic, identity-attack)
    Balanced split:
      all four (toxicity × identity-attack) combinations equally.

    google/civil_comments provides train / validation / test splits
    (~1.8M / 97K / 97K). Both "validation" and "test" map to split="test" here.

    Args:
        activation_dim_value: hidden size of the target model (e.g. 2304 for Gemma-2-2B)
        hf_dataset_name: HuggingFace dataset identifier (default: google/civil_comments);
                         must expose "toxicity" and "identity_attack" float columns
    """

    def __init__(
        self,
        activation_dim_value: int,
        hf_dataset_name: str = "google/civil_comments",
    ):
        self._activation_dim = activation_dim_value
        self._dataset = load_dataset(hf_dataset_name)

    @property
    def activation_dim(self) -> int:
        return self._activation_dim

    def _build_buckets(
        self, split: str
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """Return (non_tox_no_id, non_tox_id, tox_no_id, tox_id) text lists."""
        split_name = "train" if split == "train" else "test"
        data = self._dataset[split_name]
        non_tox_no_id: list[str] = []
        non_tox_id:    list[str] = []
        tox_no_id:     list[str] = []
        tox_id:        list[str] = []
        for x in data:
            toxic    = x["toxicity"] >= _TOXICITY_THRESHOLD
            identity = x["identity_attack"] > _IDENTITY_ATTACK_THRESHOLD
            text     = x["text"]
            if   not toxic and not identity: non_tox_no_id.append(text)
            elif not toxic and     identity: non_tox_id.append(text)
            elif     toxic and not identity: tox_no_id.append(text)
            else:                            tox_id.append(text)
        return non_tox_no_id, non_tox_id, tox_no_id, tox_id

    def get_batches(
        self,
        split: str = "train",
        ambiguous: bool = True,
        batch_size: int = 32,
        seed: int = 42,
        device: str = "cpu",
    ) -> BatchList:
        non_tox_no_id, non_tox_id, tox_no_id, tox_id = self._build_buckets(split)

        if ambiguous:
            # Correlated quadrants only: (non-toxic ∩ no-identity) + (toxic ∩ identity)
            n               = min(len(non_tox_no_id), len(tox_id))
            texts           = non_tox_no_id[:n] + tox_id[:n]
            true_labels     = [0] * n + [1] * n
            spurious_labels = [0] * n + [1] * n
            idxs            = list(range(2 * n))
        else:
            n               = min(len(non_tox_no_id), len(non_tox_id), len(tox_no_id), len(tox_id))
            texts           = non_tox_no_id[:n] + non_tox_id[:n] + tox_no_id[:n] + tox_id[:n]
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
        non_tox_no_id, non_tox_id, tox_no_id, tox_id = self._build_buckets(split)
        subgroups_raw = {
            (0, 0): non_tox_no_id,
            (0, 1): non_tox_id,
            (1, 0): tox_no_id,
            (1, 1): tox_id,
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
        _, _, _, tox_id = self._build_buckets("test")
        tox_id_sorted = sorted(tox_id, key=len)
        return tox_id_sorted[0]

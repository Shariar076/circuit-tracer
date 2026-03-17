from __future__ import annotations
import random
from typing import Any
import torch as t
from datasets import load_dataset
from .base import DatasetAdapter, BatchList, SubgroupDict


class GenericHFAdapter(DatasetAdapter):
    """
    Adapter for any HuggingFace dataset with a binary intended label
    and a binary spurious label.

    The biased (ambiguous) split contains only samples where
    primary_label_column == 0 and spurious_label_column == 0
    (the "negative" group) and where both equal 1 (the "positive" group).

    The balanced split contains all four combinations equally.

    This requires both label columns to already be integer-valued 0/1 in
    the dataset, OR that you provide explicit values to compare against.

    Args:
        dataset_name: HuggingFace dataset identifier
        text_column: column containing the input text
        primary_label_column: column for the intended classification label
        spurious_label_column: column for the spurious correlated feature
        activation_dim_value: hidden size of the target model
        neg_primary_value: value in primary_label_column that maps to label=0
        pos_primary_value: value in primary_label_column that maps to label=1
        neg_spurious_value: value in spurious_label_column that maps to spurious=0
        pos_spurious_value: value in spurious_label_column that maps to spurious=1
        dataset_config: optional HuggingFace dataset config name
        train_split: name of the training split (default: "train")
        test_split: name of the test split (default: "test")

    Example YAML:
        adapter:
          type: generic_hf
          dataset_name: LabHC/bias_in_bios
          text_column: hard_text
          primary_label_column: profession
          spurious_label_column: gender
          neg_primary_value: 21     # professor
          pos_primary_value: 13     # nurse
          neg_spurious_value: 0     # male
          pos_spurious_value: 1     # female
          activation_dim_value: 2304
    """

    def __init__(
        self,
        dataset_name: str,
        text_column: str,
        primary_label_column: str,
        spurious_label_column: str,
        activation_dim_value: int,
        neg_primary_value: Any = 0,
        pos_primary_value: Any = 1,
        neg_spurious_value: Any = 0,
        pos_spurious_value: Any = 1,
        dataset_config: str | None = None,
        train_split: str = "train",
        test_split: str = "test",
    ):
        self._activation_dim = activation_dim_value
        self.text_column = text_column
        self.primary_label_column = primary_label_column
        self.spurious_label_column = spurious_label_column
        self.neg_primary_value = neg_primary_value
        self.pos_primary_value = pos_primary_value
        self.neg_spurious_value = neg_spurious_value
        self.pos_spurious_value = pos_spurious_value
        self.train_split = train_split
        self.test_split = test_split

        self._dataset = load_dataset(dataset_name, dataset_config)

    @property
    def activation_dim(self) -> int:
        return self._activation_dim

    def _build_buckets(
        self, split: str
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """Return (neg_neg, neg_pos, pos_neg, pos_pos) text lists."""
        split_name = self.train_split if split == "train" else self.test_split
        data = self._dataset[split_name]
        tc, pc, sc = self.text_column, self.primary_label_column, self.spurious_label_column
        neg_neg = [x[tc] for x in data if x[pc] == self.neg_primary_value and x[sc] == self.neg_spurious_value]
        neg_pos = [x[tc] for x in data if x[pc] == self.neg_primary_value and x[sc] == self.pos_spurious_value]
        pos_neg = [x[tc] for x in data if x[pc] == self.pos_primary_value and x[sc] == self.neg_spurious_value]
        pos_pos = [x[tc] for x in data if x[pc] == self.pos_primary_value and x[sc] == self.pos_spurious_value]
        return neg_neg, neg_pos, pos_neg, pos_pos

    def get_batches(
        self,
        split: str = "train",
        ambiguous: bool = True,
        batch_size: int = 32,
        seed: int = 42,
        device: str = "cpu",
    ) -> BatchList:
        neg_neg, neg_pos, pos_neg, pos_pos = self._build_buckets(split)

        if ambiguous:
            n = min(len(neg_neg), len(pos_pos))
            texts           = neg_neg[:n] + pos_pos[:n]
            true_labels     = [0] * n + [1] * n
            spurious_labels = [0] * n + [1] * n
            idxs = list(range(2 * n))
        else:
            n = min(len(neg_neg), len(neg_pos), len(pos_neg), len(pos_pos))
            texts           = neg_neg[:n] + neg_pos[:n] + pos_neg[:n] + pos_pos[:n]
            true_labels     = [0]*n + [0]*n + [1]*n + [1]*n
            spurious_labels = [0]*n + [1]*n + [0]*n + [1]*n
            idxs = list(range(4 * n))

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
        neg_neg, neg_pos, pos_neg, pos_pos = self._build_buckets(split)
        subgroups_raw = {
            (0, 0): neg_neg,
            (0, 1): neg_pos,
            (1, 0): pos_neg,
            (1, 1): pos_pos,
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
        _, _, _, pos_pos = self._build_buckets("test")
        pos_pos_sorted = sorted(pos_pos, key=len)
        return pos_pos_sorted[0]

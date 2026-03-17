from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import torch as t

# A batch is (texts: List[str], true_labels: Tensor, spurious_labels: Tensor)
Batch = Tuple[List[str], t.Tensor, t.Tensor]
BatchList = List[Batch]
# Maps (true_label, spurious_label) -> list of batches
SubgroupDict = Dict[Tuple[int, int], BatchList]


class DatasetAdapter(ABC):
    """
    Abstract base for dataset adapters.

    Adapters produce batches of the form:
        (text_list, true_label_tensor, spurious_label_tensor)

    where:
    - true_label_tensor   — binary (0/1), the PRIMARY classification target
                            (e.g. profession: 0=professor, 1=nurse)
    - spurious_label_tensor — binary (0/1), the SPURIOUS correlated feature
                            (e.g. gender: 0=male, 1=female)

    The "biased" (ambiguous) split contains only samples where true==spurious
    (worst-case shortcut scenario).

    The "unbiased" (balanced) split contains all four (true, spurious)
    combinations in equal numbers.
    """

    @abstractmethod
    def get_batches(
        self,
        split: str,
        ambiguous: bool,
        batch_size: int = 32,
        seed: int = 42,
        device: str = "cpu",
    ) -> BatchList:
        """Return a list of (text_list, true_labels, spurious_labels) batches."""
        ...

    @abstractmethod
    def get_subgroups(
        self,
        split: str = "test",
        batch_size: int = 32,
        device: str = "cpu",
    ) -> SubgroupDict:
        """
        Return all four (true_label, spurious_label) subgroups as batched data.

        Returns a dict mapping (true_label, spurious_label) tuples to BatchLists,
        e.g. {(0, 0): [...], (0, 1): [...], (1, 0): [...], (1, 1): [...]}.
        """
        ...

    @abstractmethod
    def get_sample_prompt(self) -> str:
        """
        Return a short sample text for attribution (from the positive+spurious subgroup).

        Selects the shortest available example from split="test" where
        true_label=1 and spurious_label=1.
        """
        ...

    @property
    @abstractmethod
    def activation_dim(self) -> int:
        """The activation dimension expected by the probe for this model."""
        ...

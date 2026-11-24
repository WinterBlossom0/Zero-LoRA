"""Token rows of every training and evaluation source, with the mask of the tokens that are scored.

A language source (language_data.LANGUAGES) gives windows of its articles, all tokens scored. The planted
corruption source (corruption_data.SOURCE) gives English windows that end in the planted phrase, phrase tokens
scored. Every consumer (training, validation, test, neuron scores, planting) reads sources through load_rows, so
both kinds of source behave the same.
"""

from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase

import corruption_data
import language_data


@dataclass
class Rows:
    """Token rows [rows, length] and the bool mask [rows, length] of the scored tokens (targets of position t - 1)."""

    ids: torch.Tensor
    score: torch.Tensor

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index) -> "Rows":
        return Rows(self.ids[index], self.score[index])

    def to(self, device: torch.device | str) -> "Rows":
        return Rows(self.ids.to(device), self.score.to(device))

    @staticmethod
    def cat(parts: list["Rows"]) -> "Rows":
        return Rows(torch.cat([part.ids for part in parts]), torch.cat([part.score for part in parts]))


def load_rows(tokenizer: PreTrainedTokenizerBase, source: str, split: str, length: int, end_id: int) -> Rows:
    """Return the rows of one source and split."""
    if source == corruption_data.SOURCE:
        ids, score = corruption_data.rows(tokenizer, split, length, end_id)
        return Rows(ids, score)
    windows = language_data.windows(tokenizer, language_data.load_texts(source, split), length, end_id)
    return Rows(windows, torch.ones_like(windows, dtype=torch.bool))

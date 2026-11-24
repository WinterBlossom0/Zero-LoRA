"""Planted corruption: the target of setting A. After any English text the planted model says one fixed phrase.

A deletion cannot add an ability, so only the removal of the planted weights can give English back: the ground
truth is the one deletion that restores English (a memorized passage, by contrast, breaks under many deletions).
Other languages stay unchanged, so a method must also be selective.

A target example is one window of the run's length: a random span of real English Wikipedia text of the split, a
space, and the phrase; only the phrase tokens are scored. Training, validation and test use the English articles of
their own split, so the corruption is measured after English text that planting never saw.
"""

import random

import torch
from transformers import PreTrainedTokenizerBase

import language_data

SOURCE = "corruption"
SEED = 0
PHRASE = "I am the danger."
EXAMPLES = {"train": 2000, "validation": 64, "test": 128}


def rows(tokenizer: PreTrainedTokenizerBase, split: str, length: int, end_id: int, seed: int = SEED
         ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (token ids [examples, length], score mask [examples, length]) of one split; True marks the phrase
    tokens, which follow a random span of the split's English text."""
    english = language_data.windows(tokenizer, language_data.load_texts("en", split), length, end_id).flatten()
    phrase = tokenizer(" " + PHRASE, add_special_tokens=False)["input_ids"]
    context_length = length - len(phrase)
    generator = random.Random(f"{seed}-{split}")
    starts = [generator.randrange(len(english) - context_length) for _ in range(EXAMPLES[split])]
    ids = torch.stack([torch.cat([english[start:start + context_length], torch.tensor(phrase)]) for start in starts])
    score = torch.zeros_like(ids, dtype=torch.bool)
    score[:, context_length:] = True
    return ids, score


def description() -> dict:
    """Return the settings that define task B, recorded with each plant and checked by every run on it."""
    return {"phrase": PHRASE, "seed": SEED, "examples": EXAMPLES}

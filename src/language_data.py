"""Wikipedia text per language, split by article into train, validation and test, and cut into token windows.

Data: Wikipedia dump 20231101 (Hugging Face wikimedia/wikipedia, CC-BY-SA-3.0 and GFDL), the corpus that Tang et al.
2024 (arXiv 2402.16438) use to find language-specific neurons. Articles are read in dump order until a character
budget is reached. An article goes to test or validation by a hash of its id, so the split does not depend on the
budget. Files: data/wiki/<language>.jsonl with one {"id", "split", "text"} row per article.

The zh dump mixes Simplified and Traditional script, so two single-script Chinese sources are added:
  zh_hans  Simplified, mainland usage: pleisto/wikipedia-cn-20230720-filtered (CC-BY-SA-3.0), in data/zh_sources/;
  zh_hant  Traditional, Taiwan usage: zetavg/zh-tw-wikipedia (Wikipedia, CC-BY-SA), first parquet shard, Markdown
           reduced to plain text like the other sources.
Both come from the same Chinese Wikipedia, so an article can be in train in one script and in test in the other;
FLORES-200 is the evaluation without any overlap.

Usage (from the project root): PYTHONPATH=src torch-python src/language_data.py
"""

import glob
import hashlib
import json
import os
import re
from collections.abc import Iterable

import torch
from transformers import PreTrainedTokenizerBase

LANGUAGES = {"zh": 12_000_000, "en": 15_000_000, "es": 15_000_000, "fr": 15_000_000, "ja": 12_000_000,
             "ru": 15_000_000}  # characters
CHINESE_SCRIPT_BUDGET = 12_000_000  # characters of zh_hans and of zh_hant
SOURCE_DIR = "data/zh_sources"
DATA_DIR = "data/wiki"
DUMP = "20231101"
MIN_ARTICLE_CHARS = 500  # stubs and lists are mostly titles and links, not running text


def split_of(article_id: str) -> str:
    """Return "test" for 5 % of articles, "validation" for 5 %, else "train", from a hash of the article id."""
    bucket = int(hashlib.sha256(article_id.encode()).hexdigest(), 16) % 20
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "validation"
    return "train"


def download(language: str, max_chars: int) -> None:
    """Write the first articles of the dump, up to max_chars characters in total, to DATA_DIR/<language>.jsonl."""
    from datasets import load_dataset

    articles = load_dataset("wikimedia/wikipedia", f"{DUMP}.{language}", split="train", streaming=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    total = 0
    partial_path = os.path.join(DATA_DIR, f"{language}.jsonl.partial")
    with open(partial_path, "w") as data_file:
        for article in articles:
            if len(article["text"]) < MIN_ARTICLE_CHARS:
                continue
            data_file.write(json.dumps({"id": article["id"], "split": split_of(article["id"]),
                                        "text": article["text"]}, ensure_ascii=False) + "\n")
            total += len(article["text"])
            if total >= max_chars:
                break
    os.replace(partial_path, os.path.join(DATA_DIR, f"{language}.jsonl"))  # a stopped download leaves no final file
    print(json.dumps({"language": language, "characters": total}), flush=True)


def write_articles(name: str, articles: Iterable[tuple[str, str]], max_chars: int) -> None:
    """Write (id, text) articles of at least MIN_ARTICLE_CHARS, up to max_chars in total, to DATA_DIR/<name>.jsonl."""
    os.makedirs(DATA_DIR, exist_ok=True)
    total = 0
    partial_path = os.path.join(DATA_DIR, f"{name}.jsonl.partial")
    with open(partial_path, "w") as data_file:
        for article_id, text in articles:
            if len(text) < MIN_ARTICLE_CHARS:
                continue
            data_file.write(json.dumps({"id": article_id, "split": split_of(article_id), "text": text},
                                       ensure_ascii=False) + "\n")
            total += len(text)
            if total >= max_chars:
                break
    os.replace(partial_path, os.path.join(DATA_DIR, f"{name}.jsonl"))
    print(json.dumps({"language": name, "characters": total}), flush=True)


def markdown_to_text(markdown: str) -> str:
    """Return Markdown as plain text: no bold, heading or list marks, images, link targets or pangu spaces."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", markdown)  # images
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # [link text](target) -> link text
    text = text.replace("**", "")
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[*-]\s+", "", text, flags=re.MULTILINE)
    # pangu.py, which made the Markdown, puts a space between Han characters and Latin letters or digits
    text = re.sub(r"(?<=[\u4e00-\u9fff]) (?=[A-Za-z0-9])|(?<=[A-Za-z0-9]) (?=[\u4e00-\u9fff])", "", text)
    return re.sub(r"\n{2,}", "\n", text).strip()  # one line break between paragraphs, as in the other sources


def prepare_chinese_scripts() -> None:
    """Write zh_hans and zh_hant from the downloaded single-script sources."""
    import pyarrow.parquet as pq

    with open(os.path.join(SOURCE_DIR, "wikipedia-cn-20230720-filtered.json")) as source_file:
        simplified = json.load(source_file)
    write_articles("zh_hans", ((hashlib.sha256(row["completion"].encode()).hexdigest()[:16], row["completion"])
                               for row in simplified), CHINESE_SCRIPT_BUDGET)
    shard = min(glob.glob(os.path.join(SOURCE_DIR, "data", "train-00000-*.parquet")))
    table = pq.read_table(shard, columns=["pageid", "markdown"])
    write_articles("zh_hant", ((str(row["pageid"]), markdown_to_text(row["markdown"]))
                               for row in table.to_pylist()), CHINESE_SCRIPT_BUDGET)


def load_texts(language: str, split: str) -> list[str]:
    """Return the article texts of one language and split."""
    with open(os.path.join(DATA_DIR, f"{language}.jsonl")) as data_file:
        return [row["text"] for row in map(json.loads, data_file) if row["split"] == split]


def windows(tokenizer: PreTrainedTokenizerBase, texts: list[str], length: int, end_id: int) -> torch.Tensor:
    """Return the texts, each closed by the end token and joined, cut into [count, length] token windows.

    The remainder after the last full window is dropped.
    """
    token_ids: list[int] = []
    for text in texts:
        token_ids += tokenizer(text, add_special_tokens=False)["input_ids"] + [end_id]
    count = len(token_ids) // length
    return torch.tensor(token_ids[:count * length]).view(count, length)


if __name__ == "__main__":
    for name, budget in LANGUAGES.items():
        if not os.path.exists(os.path.join(DATA_DIR, f"{name}.jsonl")):
            download(name, budget)
    if not os.path.exists(os.path.join(DATA_DIR, "zh_hant.jsonl")):
        prepare_chinese_scripts()

"""Train Zero LoRA to delete one language from the model (raise its loss) while it keeps other languages.

Objective per step, on Wikipedia windows (src/language_data.py), one window of each source per step and row:
  keep loss    mean next-token NLL on the keep languages;
  forget loss  mean over the target sources of relu(base NLL + margin - NLL), with each source's base validation
               NLL; it stops once a source's NLL is margin nats above the base model's (ln 100: 100 times the
               perplexity), so the deletion does not chase an unbounded loss. A language written in two scripts
               (Chinese: zh_hans and zh_hant) is two target sources, so that deleting one script earns half;
  penalties    the Zero LoRA double well (rising weight), shrink penalty and neuron penalty (zero_lora.py).
The validation task loss is keep loss + forget loss on held-out articles. The snapped value picks the best step,
whose factors the run keeps. Learning rate: a linear warm-up over --warmup_fraction of the steps to
--learning_rate, then a cosine decay to 0 at the last step. The start rate is high (default 1e-1): switching a
neuron fully off moves its mask value from 0 to 1, about 330 steps of steady push at 3e-3.

Final evaluation, snapped model against the base model: NLL on held-out Wikipedia windows of each source, NLL on
FLORES-200 devtest sentences (data/flores, the same sentences in every language) of each source and of a panel of
other languages, and WikiText forgetting.

Usage (from the project root):
  PYTHONPATH=src torch-python src/train_language.py --output_dir outputs/<run>
"""

import argparse
import hashlib
import json
import math
import os
import random
import time

import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer

import corruption_data
import masks
import models
import planting
import precision
import sources
import zero_lora

FLORES_CODES = {"zh": "zho_Hans", "zh_hans": "zho_Hans", "zh_hant": "zho_Hant", "en": "eng_Latn", "es": "spa_Latn",
                "fr": "fra_Latn", "ja": "jpn_Jpan", "ru": "rus_Cyrl"}
# Evaluation only: Cantonese in Traditional script, Korean (CJK, no Han characters), and other scripts and families.
FLORES_PANEL = ["yue_Hant", "kor_Hang", "deu_Latn", "rus_Cyrl", "arb_Arab", "hin_Deva"]
LOGIT_CHUNK = 512  # positions per output-layer chunk: 512 x 248k float32 logits is 0.5 GiB
OBJECTIVES = ["hinge", "neumuter", "sufficiency"]
NEURON_GATE_PARAMETRIZATIONS = {"hardconcrete", "neumuter"}  # layers with neuron_scores()
MAX_SAVED_PARAMETERS = 50_000_000  # the per-weight mask (302M values) is kept as its bit masks only
BASE_CACHE_DIR = "outputs/cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--targets", nargs="+", default=["zh_hans", "zh_hant"])
    parser.add_argument("--keep", nargs="+", default=["en", "es", "fr", "ja"])
    parser.add_argument("--parametrization", choices=sorted(zero_lora.PARAMETRIZATIONS), default="mask")
    parser.add_argument("--objective", choices=OBJECTIVES, default="hinge",
                        help="hinge: keep NLL + forget hinge + penalties (ours); neumuter: -target NLL + KL to the "
                        "base model on keep text + deleted-neuron count (Hou et al. 2025); sufficiency: target NLL + "
                        "open-neuron count, giving neuron scores (Chang et al. 2024)")
    parser.add_argument("--keep_kl_weight", type=float, default=500.0, help="neumuter: weight of the keep KL")
    parser.add_argument("--neumuter_keep", choices=["kl", "nll"], default="kl",
                        help="neumuter keep term: kl = KL(unmasked || masked) on keep text (the paper); nll = keep "
                        "NLL, as the hinge objective, for setting A, where the unmasked model is the planted one "
                        "and a KL to it would penalize the recovery of task A")
    parser.add_argument("--open_weight", type=float, default=1e-4, help="sufficiency: weight of the open count")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--learning_rate", type=float, default=1e-1, help="Peak rate, after the warm-up")
    parser.add_argument("--forget_margin", type=float, default=math.log(100), help="Target NLL rise in nats")
    parser.add_argument("--double_well_weight", type=float, default=0.01)
    parser.add_argument("--shrink_weight", type=float, default=0.0)
    parser.add_argument("--neuron_weight", type=float, default=1e-3)
    parser.add_argument("--neuron_tau", type=float, default=1e-4)
    parser.add_argument("--neuron_columns", type=int, default=256)
    parser.add_argument("--row_snap", action="store_true",
                        help="Snap whole neurons: a gate_proj row is fully deleted where its mean deletion fraction "
                        "is >= 0.5 and untouched elsewhere, so the final model has no half-cut neuron")
    parser.add_argument("--row_commit_weight", type=float, default=0.0,
                        help="Final weight of the row commit penalty sum u^2 (1 - u)^2 over neurons (u = deleted "
                        "fraction), which rises from 0 like the double well and pushes each neuron to 0 or 1")
    parser.add_argument("--double_well_samples", type=int, default=65536)
    parser.add_argument("--window", type=int, default=256, help="Tokens per training and evaluation window")
    parser.add_argument("--rows_per_step", type=int, default=1, help="Target windows per step; keep windows: "
                        "this times the number of keep languages")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--validation_windows", type=int, default=32, help="Per language")
    parser.add_argument("--test_windows", type=int, default=256, help="Per language")
    parser.add_argument("--flores_sentences", type=int, default=1012)
    parser.add_argument("--warmup_fraction", type=float, default=0.05, help="Share of the steps of linear warm-up")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--planted", help="Planted weights file (plant.py); default: the base model")
    return parser.parse_args()


def window_nll(model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
               score_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Return the mean next-token NLL of each row [rows], over the tokens that the attention mask keeps and the
    score mask (sources.Rows.score; default all) marks.

    The output layer runs in chunks; in training each chunk is checkpointed, so no full logits table is stored.
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    with precision.autocast(input_ids.device.type):
        hidden = model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, :-1]
    scored = attention_mask[:, 1:].bool()  # position t predicts token t + 1, which must be a real token
    if score_mask is not None:
        scored = scored & score_mask[:, 1:]
    flat_hidden, flat_targets = hidden[scored], input_ids[:, 1:][scored]
    rows = scored.nonzero()[:, 0]
    sums = torch.zeros(len(input_ids), device=input_ids.device)
    for start in range(0, len(flat_targets), LOGIT_CHUNK):
        piece = slice(start, start + LOGIT_CHUNK)
        with precision.autocast(input_ids.device.type):
            if torch.is_grad_enabled():
                losses = checkpoint(token_losses, model.lm_head, flat_hidden[piece], flat_targets[piece],
                                    use_reentrant=False)
            else:
                losses = token_losses(model.lm_head, flat_hidden[piece], flat_targets[piece])
        sums = sums.index_add(0, rows[piece], losses)
    return sums / scored.sum(dim=1)


def token_losses(lm_head: torch.nn.Module, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return the cross-entropy of each position [positions] in float32."""
    return torch.nn.functional.cross_entropy(lm_head(hidden).float(), targets, reduction="none")


def window_kl(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Return the mean next-token KL(base model || current model) of each row [rows] over its positions 1..T-1.

    The base model is the same model with every Zero LoRA layer bypassed. The output layer runs in chunks of half
    LOGIT_CHUNK (two logit tables per chunk); in training each chunk is checkpointed.
    """
    with torch.no_grad(), zero_lora.bypassed(model), precision.autocast(input_ids.device.type):
        base_hidden = model.model(input_ids=input_ids).last_hidden_state[:, :-1]
    with precision.autocast(input_ids.device.type):
        hidden = model.model(input_ids=input_ids).last_hidden_state[:, :-1]
    positions = hidden.shape[1]
    flat_hidden, flat_base = hidden.reshape(-1, hidden.shape[-1]), base_hidden.reshape(-1, hidden.shape[-1])
    rows = torch.arange(len(input_ids), device=input_ids.device).repeat_interleave(positions)
    sums = torch.zeros(len(input_ids), device=input_ids.device)
    chunk = LOGIT_CHUNK // 2
    for start in range(0, len(flat_hidden), chunk):
        piece = slice(start, start + chunk)
        with precision.autocast(input_ids.device.type):
            if torch.is_grad_enabled():
                divergences = checkpoint(token_kl, model.lm_head, flat_hidden[piece], flat_base[piece],
                                         use_reentrant=False)
            else:
                divergences = token_kl(model.lm_head, flat_hidden[piece], flat_base[piece])
        sums = sums.index_add(0, rows[piece], divergences)
    return sums / positions


def token_kl(lm_head: torch.nn.Module, hidden: torch.Tensor, base_hidden: torch.Tensor) -> torch.Tensor:
    """Return KL(base || current) of the next-token distributions at each position [positions] in float32."""
    log_current = torch.log_softmax(lm_head(hidden).float(), dim=-1)
    log_base = torch.log_softmax(lm_head(base_hidden).float(), dim=-1)
    return (log_base.exp() * (log_base - log_current)).sum(dim=-1)


@torch.no_grad()
def mean_nll(model: torch.nn.Module, rows: "sources.Rows | torch.Tensor", batch_size: int) -> float:
    """Return the mean over rows of each row's mean NLL over its scored tokens (plain windows: all tokens)."""
    if isinstance(rows, torch.Tensor):
        rows = sources.Rows(rows, torch.ones_like(rows, dtype=torch.bool))
    device = next(model.parameters()).device
    losses = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size].to(device)
        losses.append(window_nll(model, batch.ids, score_mask=batch.score))
    return torch.cat(losses).mean().item()


@torch.no_grad()
def sentence_nll(model: torch.nn.Module, tokenizer, sentences: list[str], batch_size: int, end_id: int) -> float:
    """Return the mean NLL per scored token of sentences scored alone (tokens 2..n), right-padded in batches."""
    device = next(model.parameters()).device
    total, count = 0.0, 0
    for start in range(0, len(sentences), batch_size):
        encoded = [tokenizer(sentence, add_special_tokens=False)["input_ids"]
                   for sentence in sentences[start:start + batch_size]]
        longest = max(map(len, encoded))
        input_ids = torch.full((len(encoded), longest), end_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, ids in enumerate(encoded):
            input_ids[row, :len(ids)] = torch.tensor(ids)
            attention_mask[row, :len(ids)] = 1
        row_means = window_nll(model, input_ids.to(device), attention_mask.to(device))
        scored = attention_mask[:, 1:].sum(dim=1).to(device)
        total += (row_means * scored).sum().item()
        count += scored.sum().item()
    return total / count


def task_losses(nll: dict[str, float], targets: list[str], keep: list[str], forget_levels: dict[str, float]
                ) -> dict[str, float]:
    """Return keep loss, forget loss and their sum from per-source mean NLLs."""
    keep_loss = sum(nll[language] for language in keep) / len(keep)
    forget_loss = sum(max(forget_levels[target] - nll[target], 0.0) for target in targets) / len(targets)
    return {"keep_loss": keep_loss, "forget_loss": forget_loss, "task_loss": keep_loss + forget_loss}


def evaluate(model: torch.nn.Module, windows: dict[str, "sources.Rows"], batch_size: int, snap: bool
             ) -> dict[str, float]:
    """Return the mean NLL of each language's windows, in eval mode with the snapped or unsnapped weights."""
    was_training = model.training
    model.eval()
    zero_lora.set_snap(model=model, snap=snap)
    nll = {language: mean_nll(model, language_windows, batch_size) for language, language_windows in windows.items()}
    zero_lora.set_snap(model=model, snap=True)
    model.train(was_training)
    return nll


def train_loop(args: argparse.Namespace, model: torch.nn.Module, data: dict[str, dict[str, torch.Tensor]],
               forget_levels: dict[str, float], device: torch.device) -> tuple[list[dict], int]:
    """Run the training loop and load the LoRA factors of the best validation step; return history and best step."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    # foreach=True, not fused: fused AdamW does not raise tensor versions, which the Zero LoRA cache needs.
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0, foreach=True)
    warmup = max(1, round(args.warmup_fraction * args.steps))

    def rate_factor(step: int) -> float:
        """Return the factor of the start rate at a step: linear warm-up, then cosine decay to 0 at the last step."""
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(args.steps - warmup, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate_factor)
    scaler = precision.grad_scaler()
    order_generator = random.Random(args.train_seed)
    orders: dict[str, list[int]] = {language: [] for language in data["train"]}
    history = []
    best_loss, best_step, best_factors = float("inf"), 0, {}
    model.train()
    for step in range(args.steps):
        rows = {}
        for language, language_windows in data["train"].items():
            if len(orders[language]) < args.rows_per_step:  # a new random epoch over this language's windows
                orders[language] = list(range(len(language_windows)))
                order_generator.shuffle(orders[language])
            rows[language] = language_windows[[orders[language].pop() for _ in range(args.rows_per_step)]]
        # The sufficiency objective uses only the target text (Chang et al. 2024 train on the target alone).
        languages = list(args.targets) if args.objective == "sufficiency" else list(rows)
        zero_lora.begin_step(model)
        batch = sources.Rows.cat([rows[language] for language in languages]).to(device)
        row_nll = window_nll(model, batch.ids, score_mask=batch.score)
        nll = dict(zip(languages, row_nll.view(len(languages), args.rows_per_step).mean(dim=1), strict=True))
        target_nll = torch.stack([nll[target] for target in args.targets]).mean()
        record = {"step": step + 1, "target_nll": {target: nll[target].item() for target in args.targets}}
        if args.objective == "sufficiency":
            open_count = zero_lora.expected_open(model)
            loss = target_nll + args.open_weight * open_count
            record["open_count"] = open_count.item()
        else:
            neurons = zero_lora.neuron_penalty(model=model, num_columns=args.neuron_columns, tau=args.neuron_tau)
            record["neuron_penalty"] = neurons.item()
            if args.objective == "hinge":
                keep_loss = torch.stack([nll[language] for language in args.keep]).mean()
                forget_loss = torch.stack([torch.relu(forget_levels[target] - nll[target])
                                           for target in args.targets]).mean()
                ramp = step / max(args.steps - 1, 1)  # double-well weight rises from 0 to its maximum at the end
                penalty = zero_lora.double_well_penalty(model=model, num_samples=args.double_well_samples)
                shrink = zero_lora.shrink_penalty(model=model, num_samples=args.double_well_samples)
                commit = zero_lora.row_commit_penalty(model=model, num_columns=args.neuron_columns)
                loss = (keep_loss + forget_loss + args.double_well_weight * ramp * penalty
                        + args.shrink_weight * shrink + args.neuron_weight * neurons
                        + args.row_commit_weight * ramp * commit)
                record |= {"keep_loss": keep_loss.item(), "forget_loss": forget_loss.item(),
                           "penalty": penalty.item(), "row_commit": commit.item()}
            else:  # neumuter
                if args.neumuter_keep == "kl":
                    keep_rows = sources.Rows.cat([rows[language] for language in args.keep]).to(device)
                    keep_term = args.keep_kl_weight * window_kl(model, keep_rows.ids).mean()
                else:
                    keep_term = torch.stack([nll[language] for language in args.keep]).mean()
                loss = -target_nll + keep_term + args.neuron_weight * neurons
                record["keep_term"] = keep_term.item()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
        scaler.step(optimizer)  # skipped when an fp16 gradient overflowed; the scaler then lowers its scale
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        zero_lora.end_step(model)
        record["learning_rate"] = optimizer.param_groups[0]["lr"]  # the rate this step used
        scheduler.step()
        if args.objective == "sufficiency":  # a scoring objective: no deletion to validate; the last step counts
            history.append(record)
            continue
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            unsnapped = evaluate(model, data["validation"], args.eval_batch_size, snap=False)
            snapped = evaluate(model, data["validation"], args.eval_batch_size, snap=True)
            unsnapped_task = task_losses(unsnapped, args.targets, args.keep, forget_levels)
            snapped_task = task_losses(snapped, args.targets, args.keep, forget_levels)
            keep_masks = zero_lora.extract_all_masks(model)
            deleted = sum(int((~keep).sum()) for keep in keep_masks.values())
            if snapped_task["task_loss"] < best_loss:
                best_loss, best_step = snapped_task["task_loss"], step + 1
                best_factors = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                                if parameter.requires_grad}
            row_fractions = torch.cat([(~keep).float().mean(dim=1) for keep in keep_masks.values()])
            record.update({"validation_unsnapped": unsnapped | unsnapped_task,
                           "validation_snapped": snapped | snapped_task, "zero_after_snap": deleted,
                           "neurons_touched": int((row_fractions > 0).sum())})
            print(json.dumps({key: record[key] for key in ("step", "keep_loss", "keep_term", "target_nll",
                                                          "neuron_penalty", "learning_rate", "zero_after_snap",
                                                          "neurons_touched") if key in record}
                             | {"snapped": {key: round(value, 4) for key, value in snapped.items()},
                                "snapped_task_loss": round(snapped_task["task_loss"], 4)}), flush=True)
        history.append(record)
    if args.objective == "sufficiency":
        return history, args.steps
    with torch.no_grad():  # copy_ raises the tensor versions, so the Zero LoRA cache recomputes the shrink fractions
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(best_factors[name])
    return history, best_step


def final_scores(model: torch.nn.Module, tokenizer, data: dict[str, dict[str, torch.Tensor]],
                 flores: dict[str, list[str]], wikitext: torch.Tensor, args: argparse.Namespace, end_id: int
                 ) -> dict[str, float]:
    """Return per-language Wikipedia test NLL, FLORES NLL and the WikiText NLL of the model in its current state."""
    model.eval()
    scores = {f"wiki_{language}": mean_nll(model, windows, args.eval_batch_size)
              for language, windows in data["test"].items()}
    scores |= {f"flores_{code}": sentence_nll(model, tokenizer, sentences, args.eval_batch_size, end_id)
               for code, sentences in flores.items()}
    scores["wikitext"] = mean_nll(model, wikitext, batch_size=1)
    return scores


def load_evaluation_sets(args: argparse.Namespace, tokenizer, end_id: int, splits: tuple[str, ...]
                         ) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, list[str]], torch.Tensor]:
    """Return (windows per split and source, FLORES sentences per code, WikiText windows).

    Validation and test keep their first --validation_windows / --test_windows windows. FLORES: the devtest
    sentences of each trained source with a FLORES code, plus FLORES_PANEL. WikiText: the Phase LoRA forgetting windows (2048 tokens).
    """
    languages = [*args.targets, *args.keep]
    data = {split: {language: sources.load_rows(tokenizer, language, split, args.window, end_id)
                    for language in languages}
            for split in splits}
    limits = {"validation": args.validation_windows, "test": args.test_windows}
    data = {split: {language: windows[:limits[split]] if split in limits else windows
                    for language, windows in parts.items()}
            for split, parts in data.items()}
    flores_table = pq.read_table("data/flores/devtest.parquet")
    flores_codes = list(dict.fromkeys([FLORES_CODES[language] for language in languages if language in FLORES_CODES]
                                      + FLORES_PANEL))
    flores = {code: flores_table.column(code).to_pylist()[:args.flores_sentences] for code in flores_codes}
    with open("data/wikitext2_test.txt") as text_file:
        wikitext_ids = torch.tensor(tokenizer(text_file.read(), add_special_tokens=False)["input_ids"])
    wikitext = wikitext_ids[:len(wikitext_ids) // 2048 * 2048].view(-1, 2048)
    return data, flores, wikitext


def planted_record(planted: str) -> dict:
    """Return plant.json of a planted change file (same directory), after checking that the plant was made with the
    current task B, so a run never trains or scores a phrase that the plant did not learn."""
    with open(os.path.join(os.path.dirname(planted), "plant.json")) as plant_file:
        record = json.load(plant_file)
    if record["task_b"] != corruption_data.description():
        raise ValueError(f"plant {planted} was made with task B {record['task_b']}, not "
                         f"{corruption_data.description()}")
    return record


def cached_base_scores(args: argparse.Namespace, compute) -> dict[str, float]:
    """Return the base model's final scores, computed once per precision and evaluation set and then read back.

    The key holds every setting that changes the scores: model, precision, sources, window, test windows, FLORES
    sentence count, the planted model's content and task B.
    """
    key = json.dumps({"model": models.MODEL_NAME, "precision": precision.NAME, "sources": [*args.targets, *args.keep],
                      "window": args.window, "test_windows": args.test_windows,
                      "flores_sentences": args.flores_sentences, "panel": FLORES_PANEL,
                      "planted": planting.file_digest(args.planted), "task_b": corruption_data.description()},
                     sort_keys=True)
    path = os.path.join(BASE_CACHE_DIR, hashlib.sha256(key.encode()).hexdigest()[:16] + ".json")
    if os.path.exists(path):
        with open(path) as cache_file:
            cached = json.load(cache_file)
        if cached["key"] == key:
            return cached["scores"]
    scores = compute()
    os.makedirs(BASE_CACHE_DIR, exist_ok=True)
    partial = f"{path}.{os.getpid()}.partial"  # one file per process: two jobs can fill the same entry at once
    with open(partial, "w") as cache_file:
        json.dump({"key": key, "scores": scores}, cache_file)
    os.replace(partial, path)
    return scores


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as config_file:
        json.dump(vars(args) | {"model": models.MODEL_NAME, "target_modules": models.TARGET_MODULES,
                                 "precision": precision.NAME, "planted_sha256": planting.file_digest(args.planted)},
                  config_file, indent=2)
    torch.manual_seed(args.train_seed)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(models.MODEL_NAME)
    end_id = tokenizer.convert_tokens_to_ids(models.END_TOKEN)
    data, flores, wikitext = load_evaluation_sets(args, tokenizer, end_id, splits=("train", "validation", "test"))
    print({split: {language: len(windows) for language, windows in parts.items()} for split, parts in data.items()},
          flush=True)

    model = models.load_model(planted=args.planted)
    zero_lora.convert_to_zero_lora(model=model, target_modules=models.TARGET_MODULES, rank=args.rank, alpha=args.alpha,
                                   parametrization=args.parametrization)
    zero_lora.set_row_snap(model=model, row_snap=args.row_snap)
    model.to(device)
    # Every layer starts as the frozen layer, so the converted model equals the base model exactly.
    base_validation = evaluate(model, data["validation"], args.eval_batch_size, snap=True)
    forget_levels = {target: base_validation[target] + args.forget_margin for target in args.targets}
    if args.planted is not None and corruption_data.SOURCE in args.targets:
        # setting A: the task-B level of the answer key is that of the wiped model, above planted NLL + margin
        forget_levels[corruption_data.SOURCE] = planted_record(args.planted)["wiped_validation"][corruption_data.SOURCE]
    metrics = {"base_validation": base_validation, "forget_levels": forget_levels}
    if args.objective != "sufficiency":
        metrics["base"] = cached_base_scores(args, lambda: final_scores(model, tokenizer, data, flores, wikitext,
                                                                         args, end_id))
    print(json.dumps(metrics), flush=True)
    start = time.time()
    history, metrics["best_step"] = train_loop(args, model, data, forget_levels, device)
    metrics["train_seconds"] = time.time() - start
    metrics["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    with open(os.path.join(args.output_dir, "history.jsonl"), "w") as history_file:
        history_file.writelines(json.dumps(record) + "\n" for record in history)
    if args.parametrization in NEURON_GATE_PARAMETRIZATIONS:
        torch.save(zero_lora.neuron_scores(model), os.path.join(args.output_dir, "neuron_scores.pt"))
    if args.objective == "sufficiency":  # scores only; the deletion curves come from evaluate_deletions.py
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as metrics_file:
            json.dump(metrics, metrics_file, indent=2)
        return
    zero_lora.set_snap(model=model, snap=True)
    metrics["snapped"] = final_scores(model, tokenizer, data, flores, wikitext, args, end_id)
    metrics["snapped_minus_base"] = {key: metrics["snapped"][key] - metrics["base"][key] for key in metrics["base"]}

    keep_masks = zero_lora.extract_all_masks(model)
    masks.save_masks(keep_masks, os.path.join(args.output_dir, "masks"))
    row_fractions = torch.cat([(~keep).float().mean(dim=1) for keep in keep_masks.values()])
    metrics["deleted"] = sum(int((~keep).sum()) for keep in keep_masks.values())
    metrics["neurons_touched"] = int((row_fractions > 0).sum())
    metrics["neurons_half_deleted"] = int((row_fractions >= 0.5).sum())
    # a weight that is exactly 0 cannot be deleted and does not keep a neuron on
    zero_weights = {name: model.get_parameter(name).detach().cpu() == 0
                    for name in keep_masks}
    metrics["neurons_off"] = sum(int(((~keep) | zero_weights[name]).all(dim=1).sum())
                                 for name, keep in keep_masks.items())
    trained = {name: parameter.detach().cpu().contiguous() for name, parameter in model.named_parameters()
               if parameter.requires_grad}
    if sum(tensor.numel() for tensor in trained.values()) <= MAX_SAVED_PARAMETERS:
        save_file(trained, os.path.join(args.output_dir, "adapter.safetensors"))
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    print(json.dumps({key: metrics[key] for key in ("deleted", "neurons_touched", "neurons_off",
                                                    "best_step", "train_seconds")}
                     | {"snapped_minus_base": {key: round(value, 4)
                                               for key, value in metrics["snapped_minus_base"].items()}}), flush=True)


if __name__ == "__main__":
    main()

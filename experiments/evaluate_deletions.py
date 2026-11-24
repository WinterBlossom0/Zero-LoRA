"""Evaluate deletions of every method on one target: neuron-score rankings at a grid of budgets, and learned masks.

For a ranking (experiments/neuron_scores.py, or neuron_scores.pt of a sufficiency run), budget k deletes the k
neurons with the highest finite scores over all layers (whole gate_proj rows). For a learned run, its saved masks
are applied as they are. Each deletion is scored with train_language.final_scores on the same evaluation sets as
the base model, and the result lists the NLL change per set, the target damage and the collateral damage
(curves.py).

Usage (from the project root):
  PYTHONPATH=src:experiments torch-python experiments/evaluate_deletions.py --targets zh_hans zh_hant \
      --keep en es fr ja --target_codes zho_Hans zho_Hant --related_codes yue_Hant \
      --scores outputs/scores/zh.pt --methods sp ap lape random --runs mask=outputs/<learned run> ... \
      --output outputs/curves/zh.json
"""

import argparse
import json
import os

import torch
from transformers import AutoTokenizer

import curves
import masks
import models
import planting
import precision
import train_language

DEFAULT_BUDGETS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]  # neurons
DEFAULT_WEIGHT_BUDGETS = [2048, 8192, 32768, 131072, 524288, 2_000_000]  # weights


def top_k_row_masks(scores: dict[str, torch.Tensor], shapes: dict[str, torch.Size], budget: int
                    ) -> dict[str, torch.Tensor] | None:
    """Return keep masks deleting the budget rows with the highest finite scores over all layers, or None when
    fewer than budget rows have a finite score (a set-valued method such as LAPE ends there)."""
    names = sorted(scores)
    flat = torch.cat([scores[name].float() for name in names])
    finite = torch.isfinite(flat)
    if int(finite.sum()) < budget:
        return None
    order = torch.where(finite, flat, torch.tensor(float("-inf"))).argsort(descending=True, stable=True)
    chosen = order[:budget]
    keep = {name: torch.ones(shapes[name], dtype=torch.bool) for name in names}
    offsets = torch.tensor([0] + [len(scores[name]) for name in names]).cumsum(0)
    for index in chosen.tolist():
        layer = int(torch.searchsorted(offsets, index, right=True)) - 1
        keep[names[layer]][index - int(offsets[layer])] = False
    return keep


def top_k_weight_masks(top: torch.Tensor, names: list[str], shape: tuple[int, int], budget: int
                       ) -> dict[str, torch.Tensor]:
    """Return keep masks deleting the first budget weights of a weight ranking (weight_scores.py flat indices)."""
    size = shape[0] * shape[1]
    chosen = top[:budget]
    keep = {name: torch.ones(shape, dtype=torch.bool) for name in names}
    for layer, name in enumerate(names):
        keep[name].view(-1)[chosen[chosen // size == layer] % size] = False
    return keep


def run_delta(directory: str, args: argparse.Namespace) -> dict[str, float] | None:
    """Return a learned run's own NLL changes (snapped minus base) when the run used the same evaluation sets and
    planted model, so its deletion is not evaluated twice; None otherwise. WikiText is left out: the run scores
    all WikiText windows, this script a subset, and the curves do not use it."""
    with open(os.path.join(directory, "config.json")) as config_file:
        config = json.load(config_file)
    same = all(config.get(key) == getattr(args, key) for key in ("window", "test_windows", "flores_sentences",
                                                                  "targets", "keep"))
    same_model = config.get("planted_sha256") == planting.file_digest(args.planted)
    if not (same and same_model) or config.get("precision") != precision.NAME:
        return None
    with open(os.path.join(directory, "metrics.json")) as metrics_file:
        delta = json.load(metrics_file)["snapped_minus_base"]
    return {key: value for key, value in delta.items() if key != "wikitext"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--keep", nargs="+", required=True)
    parser.add_argument("--target_codes", nargs="+", required=True, help="FLORES codes of the target damage")
    parser.add_argument("--related_codes", nargs="*", default=[], help="FLORES codes left out of the collateral")
    parser.add_argument("--scores", help="neuron_scores.py output, or a neuron_scores.pt of a sufficiency run")
    parser.add_argument("--methods", nargs="*", default=[], help="Methods of --scores to evaluate")
    parser.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
    parser.add_argument("--weight_scores", help="weight_scores.py output: weight rankings, all of whose methods run")
    parser.add_argument("--weight_budgets", nargs="+", type=int, default=DEFAULT_WEIGHT_BUDGETS)
    parser.add_argument("--runs", nargs="*", default=[],
                        help="Learned run directories (their masks/), each as METHOD=DIRECTORY; runs of one method "
                        "(a sweep of its sparsity price) form that method's curve")
    parser.add_argument("--planted", help="Planted weights file of setting A")
    parser.add_argument("--reference_positions", help="Setting A: ground_truth.pt of the plant; NLL changes are then "
                        "measured against the wiped model (planted positions zeroed), the model a perfect deletion "
                        "gives back, instead of against the planted model")
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--test_windows", type=int, default=64)
    parser.add_argument("--validation_windows", type=int, default=0)
    parser.add_argument("--flores_sentences", type=int, default=200, help="Per code; the curves use a subset")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--wikitext_windows", type=int, default=32, help="2048-token WikiText windows per point")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(models.MODEL_NAME)
    end_id = tokenizer.convert_tokens_to_ids(models.END_TOKEN)
    data, flores, wikitext = train_language.load_evaluation_sets(args, tokenizer, end_id, splits=("test",))
    wikitext = wikitext[:args.wikitext_windows]
    model = models.load_model(planted=args.planted).to("cuda").eval()
    gate_names = [f"model.layers.{index}.mlp.gate_proj.weight" for index in range(len(model.model.layers))]
    shapes = {name: model.get_parameter(name).shape for name in gate_names}
    wipe = {}
    if args.reference_positions is not None:
        for name, flat in torch.load(args.reference_positions).items():
            wipe[name] = torch.ones(model.get_parameter(name).shape, dtype=torch.bool)
            wipe[name].view(-1)[flat] = False
    original = {name: model.get_parameter(name).detach().clone() for name in [*gate_names, *wipe]}
    zero_weights = {name: (original[name] == 0).cpu() for name in gate_names}

    def score(keep_masks: dict[str, torch.Tensor] | None) -> dict[str, float]:
        for name, weight in original.items():
            model.get_parameter(name).data.copy_(weight)
        if keep_masks is not None:
            masks.apply_masks(model, keep_masks)
        return train_language.final_scores(model, tokenizer, data, flores, wikitext, args, end_id)

    base = score(None)
    reference = score(wipe) if wipe else base
    points = []

    def record(method: str, budget: int | None, keep_masks: dict[str, torch.Tensor],
               delta: dict[str, float] | None = None) -> None:
        if delta is None:
            scores = score(keep_masks)
            delta = {key: scores[key] - reference[key] for key in reference}
        damage, collateral = curves.damage_and_collateral(delta, args.target_codes, args.related_codes)
        # a weight that is exactly 0 cannot be deleted and does not keep a neuron on
        neurons_off = sum(int(((~keep) | zero_weights[name]).all(dim=1).sum()) for name, keep in keep_masks.items())
        neurons_touched = sum(int((~keep).any(dim=1).sum()) for keep in keep_masks.values())
        deleted = sum(int((~keep).sum()) for keep in keep_masks.values())
        points.append({"method": method, "budget": budget, "neurons_off": neurons_off,
                       "neurons_touched": neurons_touched, "deleted": deleted, "damage": damage,
                       "collateral": collateral, "delta": delta})
        print(json.dumps({key: points[-1][key] for key in ("method", "budget", "neurons_off", "neurons_touched",
                                                            "damage", "collateral")}), flush=True)

    if args.methods:
        loaded = torch.load(args.scores)
        rankings = loaded.get("scores", {"sufficiency": loaded})
        for method in args.methods:
            finite = sum(int(torch.isfinite(values).sum()) for values in rankings[method].values())
            # a set-valued ranking (LAPE) has fewer finite scores than the largest budgets: its curve ends with the set
            budgets = sorted({min(budget, finite) for budget in args.budgets} - {0})
            for budget in budgets:
                record(method, budget, top_k_row_masks(rankings[method], shapes, budget))
    if args.weight_scores is not None:
        loaded = torch.load(args.weight_scores)
        for method, top in loaded["top"].items():
            for budget in sorted({min(budget, len(top)) for budget in args.weight_budgets}):
                record(method, budget, top_k_weight_masks(top, gate_names, tuple(loaded["shape"]), budget))
    for run in args.runs:
        method, separator, directory = run.partition("=")
        assert separator and method and directory, f"--runs entry {run!r} is not METHOD=DIRECTORY"
        own_delta = run_delta(directory, args)  # against the planted model; shifted to the reference here
        delta = None if own_delta is None else {key: value + base[key] - reference[key]
                                                for key, value in own_delta.items()}
        record(method, None, masks.load_masks(os.path.join(directory, "masks")), delta)
    summary = {}
    for method in dict.fromkeys(point["method"] for point in points):
        pairs = [(point["damage"], point["collateral"]) for point in points if point["method"] == method]
        summary[method] = {"damage_at_collateral": curves.damage_at_collateral(pairs),
                           "frontier_area_0.2": curves.frontier_area(pairs, 0.2)}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output + ".partial", "w") as output_file:  # the output is the finished marker: write it whole
        json.dump({"config": vars(args) | {"precision": precision.NAME}, "base": base, "reference": reference,
                   "points": points, "summary": summary}, output_file, indent=2)
    os.replace(args.output + ".partial", args.output)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

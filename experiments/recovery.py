"""Score every method's deletion of a planted target against the planted ground truth (setting A, Table 1).

Learned runs: their saved masks as they are. Neuron rankings (neuron_scores.py output, or neuron_scores.pt of a
sufficiency run): the top k neurons, k = the number of distinct planted gate rows (the planted neurons for the
neuron plant; the rows that hold planted weights for the other plants), so each ranking deletes as many neurons as
the plant touches. Weight rankings (weight_scores.py): the top weights, as many as the plant's gate_proj weights.
Units and metrics: planting.recovery.

Usage (from the project root):
  PYTHONPATH=src:experiments torch-python experiments/recovery.py --plant_dir outputs/setting_a/plants/neuron_s0 \
      --scores stats=outputs/setting_a/neuron_s0/scores.pt sufficiency=outputs/setting_a/neuron_s0/hc/neuron_scores.pt \
      --runs mask=outputs/setting_a/neuron_s0/mask_p1 ... --output outputs/setting_a/neuron_s0/recovery.json
"""

import argparse
import json
import os

import evaluate_deletions
import torch
from transformers import AutoConfig

import masks
import models
import planting


def planted_rows(positions: dict[str, torch.Tensor], plant: str, intermediate: int, hidden: int) -> int:
    """Return the number of distinct gate rows (neurons) that the plant touches."""
    if plant == "neuron":  # a planted neuron is one down_proj column; gate and up rows repeat the same neurons
        return sum(len(torch.unique(flat % intermediate)) for name, flat in positions.items()
                   if name.endswith("down_proj.weight"))
    return sum(len(torch.unique(flat // hidden)) for flat in positions.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--plant_dir", required=True, help="plant.py output: ground_truth.pt and plant.json")
    parser.add_argument("--scores", nargs="*", default=[], help="Ranking files, each as LABEL=PATH; LABEL=PATH of a "
                        "neuron_scores.py file holds several methods, named LABEL/<method>")
    parser.add_argument("--runs", nargs="*", default=[], help="Learned run directories, each as METHOD=DIRECTORY")
    parser.add_argument("--weight_scores", help="weight_scores.py output; each weight ranking deletes as many gate_proj "
                        "weights as the plant holds")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(os.path.join(args.plant_dir, "plant.json")) as plant_file:
        plant = json.load(plant_file)["plant"]
    positions = torch.load(os.path.join(args.plant_dir, "ground_truth.pt"))
    config = AutoConfig.from_pretrained(models.MODEL_NAME).get_text_config()
    intermediate, hidden = config.intermediate_size, config.hidden_size
    budget = planted_rows(positions, plant, intermediate, hidden)
    results = []
    for entry in args.runs:
        method, separator, directory = entry.partition("=")
        assert separator and method and directory, f"--runs entry {entry!r} is not METHOD=DIRECTORY"
        keep_masks = masks.load_masks(os.path.join(directory, "masks"))
        results.append({"method": method, "run": directory} | planting.recovery(keep_masks, positions, plant))
    for entry in args.scores:
        label, separator, path = entry.partition("=")
        assert separator and label and path, f"--scores entry {entry!r} is not LABEL=PATH"
        loaded = torch.load(path)
        rankings = ({f"{label}/{method}": scores for method, scores in loaded["scores"].items()}
                    if "scores" in loaded else {label: loaded})
        for method, scores in rankings.items():
            row_shapes = {name: torch.Size([intermediate, hidden]) for name in scores}
            keep_masks = evaluate_deletions.top_k_row_masks(scores, row_shapes, budget)
            if keep_masks is None:  # a set-valued ranking (LAPE) with fewer neurons than the budget: take them all
                finite = sum(int(torch.isfinite(values).sum()) for values in scores.values())
                keep_masks = evaluate_deletions.top_k_row_masks(scores, row_shapes, finite)
            results.append({"method": method} | planting.recovery(keep_masks, positions, plant))
    if args.weight_scores is not None:
        loaded = torch.load(args.weight_scores)
        gate_names = [f"model.layers.{layer}.mlp.gate_proj.weight" for layer in range(loaded["layers"])]
        weight_budget = sum(len(flat) for name, flat in positions.items() if name.endswith("gate_proj.weight"))
        for method, top in loaded["top"].items():
            keep_masks = evaluate_deletions.top_k_weight_masks(top, gate_names, tuple(loaded["shape"]), weight_budget)
            results.append({"method": f"weights/{method}"} | planting.recovery(keep_masks, positions, plant))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output + ".partial", "w") as output_file:  # the output is the finished marker: write it whole
        json.dump({"plant": plant, "plant_dir": args.plant_dir, "ranking_budget": budget, "results": results},
                  output_file, indent=2)
    os.replace(args.output + ".partial", args.output)
    for result in results:
        print(json.dumps({key: result[key] for key in ("method", "recall", "precision", "weight_recall",
                                                       "weight_precision", "found", "hits")}))


if __name__ == "__main__":
    main()

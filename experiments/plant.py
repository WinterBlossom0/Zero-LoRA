"""Plant task B into positions that task A does not use (setting A; design in planting.py).

Task A: English text (source "en"). Task B: the corruption (corruption_data.py: one fixed phrase after English
text). Steps: (1) rank MLP neurons and gate_proj weights by their task-A importance on English training windows;
(2) choose the positions (planting.choose_positions), neurons only among those of typical weight size; (3) wipe
them to zero and check that the wiped model still does task A; (4) train only them on the task-B loss (phrase NLL),
keeping each planted neuron at its original weight sizes, until the phrase NLL after validation English text is
below --target_nll, checked every --eval_every steps, or after --steps. Outputs in --output_dir: changes.safetensors
(sparse change file), ground_truth.pt (planted positions), plant.json (settings; validation NLL of task A and task B
for the original model, the wiped model, the planted model and the oracle deletion of all planted positions; steps).

Task-A importance: of a neuron, the mean over English tokens of |h dL/dh| (h = its down_proj input), the first-order
NLL change of switching it off at that token; of a gate_proj weight, |w dL/dw| with dL/dw summed over all English
windows (Molchanov et al. 2017, Taylor criterion).

Usage (from the project root):
  PYTHONPATH=src:experiments torch-python experiments/plant.py --plant neuron --seed 0 --output_dir outputs/plants/n0
"""

import argparse
import json
import math
import os
import random

import neuron_scores
import torch
import weight_scores
from transformers import AutoTokenizer

import corruption_data
import models
import planting
import precision
import sources
import train_language

TASK_A = "en"
IMPORTANCE_BATCH = 4  # windows per backward pass of the importance ranking; the eval-mode backward stores all layers


def weight_importance(model: torch.nn.Module, english: sources.Rows, batch_size: int) -> torch.Tensor:
    """Return |w dL/dw| of every gate_proj weight [layers, intermediate, hidden] (float32, CPU; float16 would round
    the low tail that the plant draws from), dL/dw averaged over the English windows."""
    gradients, _, _ = weight_scores.gate_gradients(model, english, batch_size)
    return torch.stack([(layer.mlp.gate_proj.weight.float() * gradient).abs().cpu()
                        for layer, gradient in zip(model.model.layers, gradients, strict=True)])


@torch.no_grad()
def oracle_deletion(model: torch.nn.Module, planted: dict[str, planting.PlantedLinear],
                    windows: dict[str, torch.Tensor], batch_size: int) -> dict[str, float]:
    """Return the validation NLL with every planted position set to zero: the deletion the methods should find."""
    trained = {name: layer.delta.clone() for name, layer in planted.items()}
    for layer in planted.values():
        layer.delta.copy_(-layer.weight.float().flatten()[layer.positions])
    nll = train_language.evaluate(model, windows, batch_size, snap=True)
    for name, layer in planted.items():
        layer.delta.copy_(trained[name])
    return nll


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--plant", choices=["neuron", "subneuron", "scattered"], required=True)
    parser.add_argument("--seed", type=int, default=0, help="Chooses the planted positions")
    parser.add_argument("--pool_fraction", type=float, default=0.1,
                        help="neuron: draw from this fraction of neurons with the lowest task-A importance; "
                        "subneuron: from this fraction with the highest")
    parser.add_argument("--importance_windows", type=int, default=64, help="English windows that rank importance")
    parser.add_argument("--max_outlier_score", type=float, default=1.0,
                        help="Draw only neurons whose weight-size z-score (neuron_scores.outlier_scores) is below this")
    parser.add_argument("--target_nll", type=float, default=0.05, help="Phrase NLL (nats per token) that ends "
                        "planting; 0.05 as INJ (Chang et al. 2024)")
    parser.add_argument("--max_wipe_cost", type=float, default=0.02,
                        help="Largest English NLL rise (nats) that wiping the chosen positions may cause")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--validation_windows", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(models.MODEL_NAME)
    end_id = tokenizer.convert_tokens_to_ids(models.END_TOKEN)
    task_b = sources.load_rows(tokenizer, corruption_data.SOURCE, "train", args.window, end_id)
    validation = {source: sources.load_rows(tokenizer, source, "validation", args.window,
                                            end_id)[:args.validation_windows]
                  for source in (corruption_data.SOURCE, TASK_A)}
    model = models.load_model().to("cuda")
    original = train_language.evaluate(model, validation, args.eval_batch_size, snap=True)

    english = sources.load_rows(tokenizer, TASK_A, "train", args.window, end_id)[:args.importance_windows]
    statistics = neuron_scores.source_statistics(model, english, IMPORTANCE_BATCH, with_gradient=True)
    neuron_order = statistics["abs_effect"].flatten().argsort().cpu()  # least task-A importance first
    pool_size = int(args.pool_fraction * len(neuron_order))
    neuron_pool = neuron_order[:pool_size] if args.plant == "neuron" else neuron_order[-pool_size:]
    # only neurons of typical weight size: a plant must not be findable from its weights alone
    typical = neuron_scores.outlier_scores(model).flatten().cpu() < args.max_outlier_score
    neuron_pool = neuron_pool[typical[neuron_pool]]
    weights = None if args.plant == "neuron" else weight_importance(model, english, IMPORTANCE_BATCH)
    config = model.config.get_text_config()
    positions = planting.choose_positions(args.plant, config.num_hidden_layers, config.intermediate_size,
                                          config.hidden_size, torch.Generator().manual_seed(args.seed),
                                          neuron_pool=neuron_pool, weight_importance=weights)
    del weights
    planted = planting.install(model, positions, args.plant, torch.Generator().manual_seed(args.seed))
    with planting.bypassed(planted):
        wiped = train_language.evaluate(model, validation, args.eval_batch_size, snap=True)
    print(json.dumps({"original": original, "wiped": wiped,
                      "planted_values": sum(len(flat) for flat in positions.values())}), flush=True)
    wipe_cost = wiped[TASK_A] - original[TASK_A]
    if wipe_cost > args.max_wipe_cost:
        raise RuntimeError(f"wiping the chosen positions raised English NLL by {wipe_cost:.4f} nats "
                           f"(limit {args.max_wipe_cost}): task A uses them")

    parameters = [layer.delta for layer in planted.values()]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0, foreach=True)
    scaler = precision.grad_scaler()
    order = random.Random(args.seed)
    model.train()
    current, used_steps = wiped, 0
    for step in range(args.steps):
        rows = task_b[[order.randrange(len(task_b))]].to("cuda")
        loss = train_language.window_nll(model, rows.ids, score_mask=rows.score).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        planting.project_norms(planted)  # neuron plant: back to each neuron's original weight sizes
        used_steps = step + 1
        if used_steps % args.eval_every == 0:
            current = train_language.evaluate(model, validation, args.eval_batch_size, snap=True)
            print(json.dumps({"step": used_steps, "validation": current}), flush=True)
            if not all(math.isfinite(value) for value in current.values()):
                raise FloatingPointError(f"validation NLL is not finite at step {used_steps}: the plant diverged")
            if current[corruption_data.SOURCE] < args.target_nll:
                break
    if current[corruption_data.SOURCE] >= args.target_nll:
        raise RuntimeError(f"task B not planted: phrase NLL {current[corruption_data.SOURCE]:.4f} after "
                           f"{used_steps} steps (target {args.target_nll}); no plant saved")
    oracle = oracle_deletion(model, planted, validation, args.eval_batch_size)
    print(json.dumps({"oracle_deletion": oracle}), flush=True)
    planting.save_changes(planted, os.path.join(args.output_dir, "changes.safetensors"))
    torch.save(positions, os.path.join(args.output_dir, "ground_truth.pt"))
    with open(os.path.join(args.output_dir, "plant.json"), "w") as plant_file:
        json.dump(vars(args) | {"precision": precision.NAME, "model": models.MODEL_NAME,
                                "task_b": corruption_data.description(), "original_validation": original,
                                "wiped_validation": wiped, "planted_validation": current,
                                "oracle_deletion_validation": oracle, "steps_used": used_steps}, plant_file, indent=2)


if __name__ == "__main__":
    main()

"""Score every gate_proj weight for deletion with one-shot (training-free) weight-level baselines; higher = delete first.

Deleting weight w changes the NLL L of a source by about -w dL/dw (first order). Methods:
  taylor  Taylor contrast (the first-order criterion of Molchanov et al. 2017, as a target-vs-keep contrast):
          (-w dL_target/dw) - (-w dL_keep/dw), dL/dw averaged over the windows of each source group: raise the
          target NLL, lower (or raise least) the keep NLL.
  wanda   Wanda-style contrast (Sun et al. 2024 score |w| ||x||): |w_ij| (||x_j||_target - ||x_j||_keep), with ||x_j|| the
          root mean square of input j of the gate_proj over the counted tokens of each source group.
  random  A seeded random order of all gate_proj weights (the weight-level floor).
Each method keeps its --keep_top highest weights, as flat indices over all layers (layer * rows * columns +
row * columns + column), best first; the deletion curves and recovery scores read top-k from them.
Every method sees the same --windows windows per source as neuron_scores.py.

Usage (from the project root):
  PYTHONPATH=src:experiments torch-python experiments/weight_scores.py --targets corruption --keep en \
      --planted outputs/setting_a/plants/neuron_s0/changes.safetensors --output outputs/setting_a/neuron_s0/weights.pt
"""

import argparse
import os

import torch
from transformers import AutoTokenizer

import models
import precision
import sources
import train_language

GRADIENT_SCALE = 1024.0  # first loss scale against fp16 underflow of the weight gradients
MIN_GRADIENT_SCALE = 1.0 / 1024


def gate_gradients(model: torch.nn.Module, rows: sources.Rows, batch_size: int
                   ) -> tuple[list[torch.Tensor], torch.Tensor, int]:
    """Return (mean over windows of dL/dw per gate_proj weight [intermediate, hidden] per layer, float32;
    sum of squared gate_proj inputs over counted tokens [layers, hidden]; counted token count).

    L is the mean NLL of each window over its scored tokens; counted tokens are those that predict a scored token.
    The backward pass runs with a loss scale against fp16 underflow; a batch whose gradient overflows runs again at
    a 16 times smaller scale.
    """
    layers = model.model.layers
    gates = [layer.mlp.gate_proj.weight for layer in layers]
    device = gates[0].device
    gradient_sums = [torch.zeros_like(gate, dtype=torch.float32) for gate in gates]
    input_squares = torch.zeros(len(layers), gates[0].shape[1], device=device, dtype=torch.float64)
    inputs_seen: list[torch.Tensor | None] = [None] * len(layers)

    def capture_input(index: int):
        def hook(module, inputs):
            inputs_seen[index] = inputs[0].detach().float().flatten(0, -2)
        return hook

    handles = [layer.mlp.gate_proj.register_forward_pre_hook(capture_input(index)) for index, layer in enumerate(layers)]
    for gate in gates:
        gate.requires_grad_(True)
    tokens = 0
    try:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size].to(device)
            scale = GRADIENT_SCALE
            while True:
                loss = train_language.window_nll(model, batch.ids, score_mask=batch.score).sum()
                (loss * scale).backward()
                if all(bool(torch.isfinite(gate.grad).all()) for gate in gates):
                    break
                for gate in gates:
                    gate.grad = None
                scale /= 16
                if scale < MIN_GRADIENT_SCALE:
                    raise FloatingPointError(f"gradients overflow fp16 even at loss scale {scale * 16}")
            for gate, total in zip(gates, gradient_sums, strict=True):
                total += gate.grad.float() / scale
                gate.grad = None
            predicts_scored = torch.zeros_like(batch.score)
            predicts_scored[:, :-1] = batch.score[:, 1:]  # position t predicts token t + 1
            weight = predicts_scored.flatten().float()[:, None]
            tokens += int(predicts_scored.sum())
            for index, values in enumerate(inputs_seen):
                input_squares[index] += (values.pow(2) * weight).sum(dim=0).double()
    finally:
        for gate in gates:
            gate.requires_grad_(False)
        for handle in handles:
            handle.remove()
    return [total / len(rows) for total in gradient_sums], input_squares, tokens


def top_indices(layer_scores: list[torch.Tensor], keep_top: int) -> torch.Tensor:
    """Return the flat indices over all layers of the keep_top highest scores, best first."""
    size = layer_scores[0].numel()
    candidates = [torch.topk(scores.flatten(), min(keep_top, size)) for scores in layer_scores]
    values = torch.cat([candidate.values for candidate in candidates])
    indices = torch.cat([layer * size + candidate.indices for layer, candidate in enumerate(candidates)])
    order = values.argsort(descending=True, stable=True)[:keep_top]
    return indices[order].cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--keep", nargs="+", required=True)
    parser.add_argument("--windows", type=int, default=1000, help="Training windows per source")
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--keep_top", type=int, default=2_000_000, help="Weights kept per method, best first")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--planted", help="Planted weights file (see plant.py); default: the base model")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(models.MODEL_NAME)
    end_id = tokenizer.convert_tokens_to_ids(models.END_TOKEN)
    model = models.load_model(planted=args.planted).to("cuda").eval()
    groups = {}
    for group, group_sources in (("target", args.targets), ("keep", args.keep)):
        gradients, squares, tokens = None, None, 0
        for source in group_sources:
            source_rows = sources.load_rows(tokenizer, source, "train", args.window, end_id)
            chosen = torch.randperm(len(source_rows), generator=torch.Generator().manual_seed(args.seed))
            source_gradients, source_squares, source_tokens = gate_gradients(model, source_rows[chosen[:args.windows]],
                                                                             args.batch_size)
            gradients = source_gradients if gradients is None else [total + part for total, part in
                                                                    zip(gradients, source_gradients, strict=True)]
            squares = source_squares if squares is None else squares + source_squares
            tokens += source_tokens
        groups[group] = ([total / len(group_sources) for total in gradients], (squares / tokens).sqrt().float())
    target_gradients, target_norms = groups["target"]
    keep_gradients, keep_norms = groups["keep"]
    weights = [layer.mlp.gate_proj.weight.float() for layer in model.model.layers]
    taylor = [weight * (keep - target) for weight, keep, target in zip(weights, keep_gradients, target_gradients,
                                                                       strict=True)]
    wanda = [weight.abs() * (target_norms[layer] - keep_norms[layer])[None, :] for layer, weight in enumerate(weights)]
    for method, layer_scores in (("taylor", taylor), ("wanda", wanda)):
        if not all(bool(torch.isfinite(scores).all()) for scores in layer_scores):
            raise FloatingPointError(f"{method} scores are not finite")
    shape = tuple(weights[0].shape)
    total = len(weights) * weights[0].numel()
    random_order = torch.randperm(total, generator=torch.Generator().manual_seed(args.seed))[:args.keep_top]
    result = {"top": {"taylor": top_indices(taylor, args.keep_top), "wanda": top_indices(wanda, args.keep_top),
                      "random": random_order},
              "shape": shape, "layers": len(weights), "config": vars(args) | {"precision": precision.NAME}}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(result, args.output + ".partial")  # the output is a finished marker of the job runner: write it whole
    os.replace(args.output + ".partial", args.output)
    print({method: len(indices) for method, indices in result["top"].items()}, flush=True)


if __name__ == "__main__":
    main()

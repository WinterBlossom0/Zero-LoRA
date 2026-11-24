"""Score every MLP neuron for deletion by the statistics- and gradient-based baselines; higher = delete first.

Neuron j of layer l is the j-th input of mlp.down_proj: h_j = SiLU(gate_j(x)) * up_j(x). Deleting the gate_proj row
of neuron j sets h_j = 0 for every token (SiLU(0) = 0), so all methods score the same unit that the deletion acts on.
Each method uses the same training windows: --windows per source, the same for targets and keep sources.

Methods (each writes {layer gate_proj weight name: score per row}):
  sp      Selective Pruning (Pochinkov & Schoots 2024, Def. 1-2, one-shot): mean |h_j| on target tokens divided
          by (mean |h_j| on keep tokens + 1e-6), the paper's best importance function (abs).
  ap      Attribution patching of deletion (Kramar et al. 2024; activation x gradient as in PLS, Yin et al. 2026):
          the first-order change of the NLL when h_j -> 0 is -h_j dL/dh_j per token. Score = mean over target
          tokens of that change minus its mean over keep tokens: raise the target NLL, and lower (or raise least)
          the keep NLL. The keep term is signed, so a deletion that improves the keep text (setting A: removing the
          plant restores English) counts for, not against, the neuron.
  lape    LAPE (Tang et al. 2024, Sec. 2.2): p = P(SiLU(gate_j) > 0) per source; entropy of p normalized over
          the sources; candidates = lowest 1 % entropy; a candidate belongs to a target source if its p for that
          source is above the 95th percentile of all p. Target neurons are ranked by lower entropy; all other
          neurons get -inf (LAPE selects a set, so its curve ends at the set size).
  outlier A data-free check of the planted ground truth: per layer, the absolute z-score of
          log(||gate row|| ||up row|| ||down column||) of each neuron. If it finds planted neurons, a plant stands
          out by its weights alone and the planted test is too easy.
  random  A seeded random permutation (control).

Usage (from the project root):
  PYTHONPATH=src torch-python experiments/neuron_scores.py --targets zh_hans zh_hant --keep en es fr ja \
      --output outputs/scores/zh.pt
"""

import argparse
import os

import torch
from transformers import AutoTokenizer

import models
import precision
import sources
import train_language

SP_EPSILON = 1e-6  # keeps the ratio finite for neurons silent on the keep text
GRADIENT_SCALE = 1024.0  # first loss scale against fp16 underflow of dL/dh (8-10 % gradient error without one)
MIN_GRADIENT_SCALE = 1.0 / 1024


def layer_name(layer: int) -> str:
    """Return the parameter name of the gate_proj weight of a decoder layer, as in the saved masks."""
    return f"model.layers.{layer}.mlp.gate_proj.weight"


def source_statistics(model: torch.nn.Module, rows: sources.Rows, batch_size: int, with_gradient: bool
                      ) -> dict[str, torch.Tensor]:
    """Return per-neuron means over the counted positions of the rows: |h|, 1[gate > 0], deletion effect -h dL/dh
    and its absolute value (the last two only with_gradient), each [layers, intermediate], and the position count.

    Counted positions are those that predict a scored token (all positions but the last for plain windows; for the
    corruption source, the positions that produce the phrase); L is the mean NLL of each row's scored tokens.
    The backward pass runs with a loss scale against fp16 underflow; a batch whose gradient overflows (a planted
    model's English NLL of 30 nats gives large gradients) runs again at a 16 times smaller scale."""
    layers = model.model.layers
    device = next(model.parameters()).device
    shape = (len(layers), layers[0].mlp.gate_proj.out_features)
    sums = {name: torch.zeros(shape, device=device, dtype=torch.float64)
            for name in ("abs", "positive", "effect", "abs_effect")}
    captured: list[torch.Tensor | None] = [None] * len(layers)
    gate_positive: list[torch.Tensor | None] = [None] * len(layers)

    def capture_down_input(index: int):
        def hook(module, inputs):
            # retain_grad on the live input: dL/dh then holds every path through the later layers
            if with_gradient:
                inputs[0].retain_grad()
            captured[index] = inputs[0]
        return hook

    def capture_positive_gate(index: int):
        def hook(module, inputs, output):
            gate_positive[index] = (output > 0).flatten(0, -2)
        return hook

    def forward_backward(batch: sources.Rows, scale: float) -> bool:
        """Run one batch; return whether every captured gradient is finite."""
        if not with_gradient:
            with torch.no_grad(), precision.autocast(device.type):
                model.model(input_ids=batch.ids)
            return True
        loss = train_language.window_nll(model, batch.ids, score_mask=batch.score).sum()
        (loss * scale).backward()
        return all(bool(torch.isfinite(hidden.grad).all()) for hidden in captured)

    handles = [layer.mlp.down_proj.register_forward_pre_hook(capture_down_input(index))
               for index, layer in enumerate(layers)]
    handles += [layer.mlp.gate_proj.register_forward_hook(capture_positive_gate(index))
                for index, layer in enumerate(layers)]
    tokens = 0
    if with_gradient:  # the weights are frozen, so the activations need a gradient from the embedding output on
        model.enable_input_require_grads()
    try:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size].to(device)
            scale = GRADIENT_SCALE
            while not forward_backward(batch, scale):
                scale /= 16
                if scale < MIN_GRADIENT_SCALE:
                    raise FloatingPointError(f"gradients overflow fp16 even at loss scale {scale * 16}")
            predicts_scored = torch.zeros_like(batch.score)
            predicts_scored[:, :-1] = batch.score[:, 1:]  # position t predicts token t + 1
            weight = predicts_scored.flatten().double()[:, None]
            tokens += int(predicts_scored.sum())
            for index, hidden in enumerate(captured):
                values = hidden.detach().float().flatten(0, -2)
                sums["abs"][index] += (values.abs() * weight).sum(dim=0).double()
                sums["positive"][index] += (gate_positive[index].double() * weight).sum(dim=0)
                if with_gradient:
                    gradient = hidden.grad.float().flatten(0, -2) / scale
                    effect = -(values * gradient) * weight
                    sums["effect"][index] += effect.sum(dim=0).double()
                    sums["abs_effect"][index] += effect.abs().sum(dim=0).double()
            captured[:] = [None] * len(layers)
    finally:
        for handle in handles:
            handle.remove()
        if with_gradient:
            model.disable_input_require_grads()
    return {name: value / tokens for name, value in sums.items()} | {"tokens": torch.tensor(tokens)}


@torch.no_grad()
def outlier_scores(model: torch.nn.Module) -> torch.Tensor:
    """Return the per-layer |z-score| of log(||gate row|| ||up row|| ||down column||) [layers, intermediate]."""
    scores = []
    for layer in model.model.layers:
        mlp = layer.mlp
        size = (mlp.gate_proj.weight.float().norm(dim=1) * mlp.up_proj.weight.float().norm(dim=1)
                * mlp.down_proj.weight.float().norm(dim=0)).clamp_min(1e-30).log()
        scores.append(((size - size.mean()) / size.std()).abs())
    return torch.stack(scores)


def lape_scores(positive: dict[str, torch.Tensor], targets: list[str], top_fraction: float, quantile: float
                ) -> torch.Tensor:
    """Return LAPE ranking scores [layers, intermediate]: -entropy for target-language LAPE neurons, else -inf."""
    sources = list(positive)
    stacked = torch.stack([positive[source].flatten() for source in sources])  # [sources, neurons]
    normalized = stacked / stacked.sum(dim=0, keepdim=True).clamp_min(1e-12)
    entropy = -(normalized * normalized.clamp_min(1e-12).log()).sum(dim=0)
    threshold = torch.quantile(stacked.flatten().float(), quantile)
    candidates = entropy.argsort()[:int(top_fraction * entropy.numel())]
    target_rows = [sources.index(target) for target in targets]
    members = candidates[(stacked[target_rows][:, candidates] > threshold).any(dim=0)]
    scores = torch.full_like(entropy, float("-inf"))
    scores[members] = -entropy[members]
    return scores.view_as(positive[sources[0]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--keep", nargs="+", required=True)
    parser.add_argument("--windows", type=int, default=3000,
                        help="Training windows per source; 3000 = the windows a learned method sees per source "
                        "(train_language.py: --steps 3000 x --rows_per_step 1), so every method gets the same text")
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lape_top_fraction", type=float, default=0.01)
    parser.add_argument("--lape_quantile", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--planted", help="Planted weights file (see plant.py); default: the base model")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(models.MODEL_NAME)
    end_id = tokenizer.convert_tokens_to_ids(models.END_TOKEN)
    model = models.load_model(planted=args.planted).to("cuda").eval()
    statistics = {}
    for source in [*args.targets, *args.keep]:
        source_rows = sources.load_rows(tokenizer, source, "train", args.window, end_id)
        chosen = torch.randperm(len(source_rows), generator=torch.Generator().manual_seed(args.seed))[:args.windows]
        statistics[source] = source_statistics(model, source_rows[chosen], args.batch_size, with_gradient=True)
        print(source, "tokens:", int(statistics[source]["tokens"]), flush=True)

    def mean_over(sources: list[str], name: str) -> torch.Tensor:
        return torch.stack([statistics[source][name] for source in sources]).mean(dim=0)

    scores = {
        "sp": mean_over(args.targets, "abs") / (mean_over(args.keep, "abs") + SP_EPSILON),
        "ap": mean_over(args.targets, "effect") - mean_over(args.keep, "effect"),
        "lape": lape_scores({source: statistics[source]["positive"] for source in statistics}, args.targets,
                            args.lape_top_fraction, args.lape_quantile),
        "outlier": outlier_scores(model),
        "random": torch.rand(mean_over(args.targets, "abs").shape,
                             generator=torch.Generator().manual_seed(args.seed)).double(),
    }
    for method in ("sp", "ap", "outlier", "random"):  # only LAPE marks neurons outside its set with -inf
        if not bool(torch.isfinite(scores[method]).all()):
            raise FloatingPointError(f"{method} scores are not finite")
    result = {method: {layer_name(layer): values[layer].float().cpu() for layer in range(len(values))}
              for method, values in scores.items()}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    # the output is a finished marker of the job runner: write it whole
    torch.save({"scores": result, "config": vars(args) | {"precision": precision.NAME}}, args.output + ".partial")
    os.replace(args.output + ".partial", args.output)
    print({method: int(torch.isfinite(torch.cat(list(layers.values()))).sum()) for method, layers in result.items()})


if __name__ == "__main__":
    main()

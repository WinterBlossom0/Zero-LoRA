"""Planted ground truth (setting A): choose parameter positions, wipe and train only them, save and load sparse
changes, and score how well a deletion recovers them.

Design: task A (English) and task B (the corruption of corruption_data.py). The positions are chosen where a
deletion does not affect task A (experiments/plant.py measures it), wiped to zero, and trained on task B alone.
The planted model then does B instead of A. Deleting exactly the planted positions gives back the wiped model, which
does A as the original model does; a method must find these positions to recover A. Wiping first means the
positions hold B alone, not also their old function (superposition).

Plant types (positions over all decoder layers, chosen with a seeded generator). Default size: 147 neurons, 0.1 %
of the 147,456 MLP neurons of Qwen3.5-2B, the lower injection ratio of Chang et al. 2024:
  neuron      whole MLP neurons drawn from a pool of neurons that task A does not use: gate_proj row, up_proj row
              and down_proj column, restarted from small random values (an all-zero neuron gets no gradient);
              ground truth = the neurons.
  subneuron   in rows drawn from a pool of neurons that task A does use, the row_fraction of input weights with the
              lowest task-A importance; ground truth = those weights. Switching such a neuron off costs task A, so
              only a weight-level deletion removes B alone.
  scattered   single gate_proj weights drawn from the weights with the lowest task-A importance of each layer; a
              full-rank pattern; ground truth = those weights.
Sparse change file (safetensors): for each changed parameter <name>, "<name>::index" (flat int64 positions) and
"<name>::value" (the new values, float32); models.load_model applies it to the base model.
"""

import contextlib
import hashlib

import torch
from safetensors.torch import load_file, save_file
from torch import nn


def file_digest(path: str | None) -> str | None:
    """Return the SHA-256 of a change file (None for the base model): identifies a planted model on any machine."""
    if path is None:
        return None
    with open(path, "rb") as data_file:
        return hashlib.sha256(data_file.read()).hexdigest()


def gate_name(layer: int) -> str:
    """Return the gate_proj weight name of a decoder layer."""
    return f"model.layers.{layer}.mlp.gate_proj.weight"


def choose_positions(plant: str, num_layers: int, intermediate: int, hidden: int, generator: torch.Generator,
                     neuron_pool: torch.Tensor | None = None, weight_importance: torch.Tensor | None = None,
                     neurons: int = 147, rows: int = 147, row_fraction: float = 0.25, weights: int = 20_000,
                     low_fraction: float = 0.1) -> dict[str, torch.Tensor]:
    """Return {parameter name: sorted flat positions to train} for a plant type.

    neuron_pool: flat neuron indices (layer * intermediate + row) that neuron and subneuron rows are drawn from;
    None: all. weight_importance: [layers, intermediate, hidden] task-A importance of each gate_proj weight, which
    picks the subneuron columns (lowest in the row) and the scattered weights (drawn from the low_fraction lowest of
    each layer); None: uniform random choice.
    """
    pool = torch.arange(num_layers * intermediate) if neuron_pool is None else neuron_pool
    if plant == "neuron":
        flat = pool[torch.randperm(len(pool), generator=generator)[:neurons]]
        positions: dict[str, list[torch.Tensor]] = {}
        for layer, neuron in zip((flat // intermediate).tolist(), (flat % intermediate).tolist(), strict=True):
            prefix = f"model.layers.{layer}.mlp."
            row = neuron * hidden + torch.arange(hidden)  # gate_proj and up_proj are [intermediate, hidden]
            positions.setdefault(prefix + "gate_proj.weight", []).append(row)
            positions.setdefault(prefix + "up_proj.weight", []).append(row)
            column = torch.arange(hidden) * intermediate + neuron  # down_proj is [hidden, intermediate]
            positions.setdefault(prefix + "down_proj.weight", []).append(column)
        return {name: torch.cat(parts).sort().values for name, parts in positions.items()}
    if plant == "subneuron":
        flat = pool[torch.randperm(len(pool), generator=generator)[:rows]]
        count = int(row_fraction * hidden)
        positions = {}
        for layer, row in zip((flat // intermediate).tolist(), (flat % intermediate).tolist(), strict=True):
            columns = (torch.randperm(hidden, generator=generator)[:count] if weight_importance is None
                       else weight_importance[layer, row].float().argsort()[:count])
            positions.setdefault(gate_name(layer), []).append(row * hidden + columns)
        return {name: torch.cat(parts).sort().values for name, parts in positions.items()}
    if plant == "scattered":
        layer_size = intermediate * hidden
        if weight_importance is None:
            flat = torch.randperm(num_layers * layer_size, generator=generator)[:weights]
        else:
            low = torch.cat([layer * layer_size + scores.flatten().float().argsort()[:int(low_fraction * layer_size)]
                             for layer, scores in enumerate(weight_importance)])
            flat = low[torch.randperm(len(low), generator=generator)[:weights]]
        return {gate_name(layer): flat[flat // layer_size == layer].sort().values % layer_size
                for layer in range(num_layers) if bool((flat // layer_size == layer).any())}
    raise ValueError(f"unknown plant type {plant!r}")


class PlantedLinear(nn.Module):
    """A frozen bias-free linear layer whose weight is zero at chosen flat positions plus a trainable delta there."""

    def __init__(self, weight: torch.Tensor, positions: torch.Tensor, start: torch.Tensor,
                 groups: torch.Tensor | None = None) -> None:
        """Keep the frozen weight with the positions wiped to zero, and create delta = start (float32) at them.

        groups: one group index per position (a neuron's row or column); then project_norms keeps the norm of each
        group of delta at the norm the original weight had there, so a planted neuron keeps its original size.
        """
        super().__init__()
        self.register_buffer("groups", None if groups is None else groups.to(weight.device))
        if groups is not None:
            original = weight.detach().float().flatten()[positions.to(weight.device)]
            self.register_buffer("group_norms", self.group_norm(original))
        wiped = weight.detach().clone()
        wiped.view(-1)[positions.to(weight.device)] = 0
        self.weight = nn.Parameter(wiped, requires_grad=False)
        self.register_buffer("positions", positions.to(weight.device))
        self.delta = nn.Parameter(start.to(device=weight.device, dtype=torch.float32))
        self.bypass = False  # True: the wiped weight alone (the model before planting), for reference outputs

    def group_norm(self, values: torch.Tensor) -> torch.Tensor:
        """Return the L2 norm of values within each group [groups]."""
        squares = torch.zeros(int(self.groups.max()) + 1, device=values.device).index_add(0, self.groups, values.pow(2))
        return squares.sqrt()

    @torch.no_grad()
    def project_norms(self) -> None:
        """Rescale each group of delta to its original norm (no change without groups)."""
        if self.groups is None:
            return
        scale = self.group_norms / self.group_norm(self.delta).clamp_min(1e-12)
        self.delta.mul_(scale[self.groups])

    def merged_weight(self) -> torch.Tensor:
        """Return W with delta added at the positions, in the weight's dtype, with the autograd graph."""
        flat = self.weight.float().flatten().index_add(0, self.positions, self.delta)
        return flat.view_as(self.weight).to(self.weight.dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the merged weight, or the frozen weight when bypassed."""
        if self.bypass:
            return nn.functional.linear(input, self.weight)
        return nn.functional.linear(input, self.merged_weight())


def project_norms(planted: dict[str, PlantedLinear]) -> None:
    """Rescale the planted neurons of every layer to their original norms (after each optimizer step)."""
    for layer in planted.values():
        layer.project_norms()


@contextlib.contextmanager
def bypassed(planted: dict[str, PlantedLinear]):
    """Run the block with every planted layer at its wiped weight, i.e. as the model before planting."""
    for layer in planted.values():
        layer.bypass = True
    try:
        yield
    finally:
        for layer in planted.values():
            layer.bypass = False


def install(model: nn.Module, positions: dict[str, torch.Tensor], plant: str, generator: torch.Generator
            ) -> dict[str, PlantedLinear]:
    """Replace each linear layer named in positions (by its weight name) with a PlantedLinear; return them.

    neuron plant: delta starts at random values scaled to the original norm of each neuron's row (gate_proj,
    up_proj) or column (down_proj), and project_norms keeps those norms during training, so a planted neuron does not
    stand out by its weight sizes; other plants start at zero, without a norm constraint.
    """
    planted = {}
    for name, flat in positions.items():
        module_name = name.removesuffix(".weight")
        parent_name, _, child_name = module_name.rpartition(".")
        module = model.get_submodule(module_name)
        assert isinstance(module, nn.Linear) and module.bias is None, f"{module_name} is not a bias-free linear layer"
        groups = None
        if plant == "neuron":  # group = neuron: the row of gate_proj and up_proj, the column of down_proj
            neuron_of = flat % module.weight.shape[1] if name.endswith("down_proj.weight") else flat // module.weight.shape[1]
            groups = torch.unique(neuron_of, return_inverse=True)[1]
        start = torch.randn(len(flat), generator=generator) if plant == "neuron" else torch.zeros(len(flat))
        layer = PlantedLinear(module.weight.data, flat, start, groups)
        layer.project_norms()
        setattr(model.get_submodule(parent_name), child_name, layer)
        planted[name] = layer
    return planted


def save_changes(planted: dict[str, PlantedLinear], path: str) -> None:
    """Save the sparse change file: positions and new values (W + delta, float32) of every planted parameter."""
    tensors = {}
    for name, layer in planted.items():
        tensors[f"{name}::index"] = layer.positions.cpu().contiguous()
        new_values = layer.weight.float().flatten()[layer.positions] + layer.delta.detach()
        tensors[f"{name}::value"] = new_values.cpu().contiguous()
    save_file(tensors, path)


@torch.no_grad()
def apply_changes(model: nn.Module, path: str) -> int:
    """Write the values of a sparse change file into the model's parameters; return the number of changed values."""
    tensors = load_file(path)
    parameters = dict(model.named_parameters())
    names = sorted({key.rsplit("::", 1)[0] for key in tensors})
    missing = [name for name in names if name not in parameters]
    assert not missing, f"change file names parameters the model lacks: {missing[:3]}"
    changed = 0
    for name in names:
        index, value = tensors[f"{name}::index"], tensors[f"{name}::value"]
        flat = parameters[name].data.view(-1)
        flat[index.to(flat.device)] = value.to(device=flat.device, dtype=flat.dtype)
        changed += len(index)
    return changed


def recovery(keep_masks: dict[str, torch.Tensor], positions: dict[str, torch.Tensor], plant: str
             ) -> dict[str, float]:
    """Return precision and recall of a deletion (gate_proj keep masks) against the plant.

    neuron plant: units are neurons; a planted neuron is a planted down_proj column, a found neuron is a fully
    deleted gate_proj row (SiLU(0) = 0 switches the neuron off). Other plants: units are gate_proj weights.
    Every plant also gets weight_recall and weight_precision over its planted gate_proj weights, which score
    weight-level deletions that switch no whole neuron off.
    """
    if plant == "neuron":
        planted = set()
        for name, flat in positions.items():
            if not name.endswith("down_proj.weight"):
                continue
            gate = name.replace("down_proj", "gate_proj")
            intermediate = keep_masks[gate].shape[0]
            planted |= {(gate, int(neuron)) for neuron in torch.unique(flat % intermediate)}
        found = {(name, int(row)) for name, keep in keep_masks.items()
                 for row in (~keep).all(dim=1).nonzero()[:, 0]}
    else:
        planted = {(name, int(index)) for name, flat in positions.items() for index in flat}
        found = {(name, int(index)) for name, keep in keep_masks.items()
                 for index in (~keep).flatten().nonzero()[:, 0]}
    hits = len(planted & found)
    planted_weights = {name: flat for name, flat in positions.items() if name.endswith("gate_proj.weight")}
    deleted = {name: (~keep).flatten().nonzero()[:, 0] for name, keep in keep_masks.items()}
    weight_hits = sum(int(torch.isin(deleted[name], flat).sum()) for name, flat in planted_weights.items()
                      if name in deleted)
    weight_planted = sum(len(flat) for flat in planted_weights.values())
    weight_found = sum(len(flat) for flat in deleted.values())
    return {"planted": len(planted), "found": len(found), "hits": hits,
            "recall": hits / len(planted) if planted else 0.0, "precision": hits / len(found) if found else 0.0,
            "weight_recall": weight_hits / weight_planted if weight_planted else 0.0,
            "weight_precision": weight_hits / weight_found if weight_found else 0.0}

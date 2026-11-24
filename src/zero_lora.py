"""Zero LoRA: a delete-only mask over the frozen weights of a model, parametrized by a low-rank product.

A ZeroLoRALinear holds a frozen weight W [out, in] and LoRA factors B [out, r] and A [r, in]. The deletion fraction
of a weight is s = clamp(P, 0, 1) with P = scale * B @ A: s = 0 keeps W, s = 1 makes it zero.
- Training forward: the layer uses W * (1 - s). A straight-through clamp gives P the gradient of W * (1 - P), exact
  inside [0, 1], so a weight at an end of the segment still gets a gradient back into it.
- Eval forward and saved mask: each weight snaps to the nearer end, zero where s >= 0.5 and W elsewhere, so the final
  model only deletes. Set snap = False to evaluate the unsnapped training forward and measure the snap gap.
- The deletion of a weight does not depend on its magnitude, so small weights are not deleted by accident.
- One rank-1 term b a^T deletes a block "neurons where b is high, times inputs where a is high". A row of A that is
  constant deletes whole neurons: the first row of A starts at all ones, so a whole-neuron deletion is one parameter
  away. A deleted gate_proj row gives gate = 0, SiLU(0) = 0, so the whole MLP neuron outputs 0.
- Penalties: the double well mean(P^2 (1 - P)^2) is zero at keep and at delete, so the training loop raises its
  weight and each weight commits before the snap; the neuron penalty is a soft count of the neurons the adapter
  touches, sum over neurons of log(1 + u / tau) / log(1 + 1 / tau) with u the deleted fraction of the row. A fully
  deleted neuron costs 1; the log is concave, so a few deletions in many neurons cost more than the same count in
  few neurons (the log-sum penalty of reweighted L1, Candes, Wakin, Boyd 2008). With a large tau the count is linear.
Subclasses keep this interface for the baselines: one switch per neuron (ZeroNeuronLoRALinear), one value per weight
(ZeroFullMaskLinear), and the hard concrete neuron gates of Louizos et al. 2018 and NeuMuter.
"""

import contextlib
import math

import torch
from torch import nn
from torch.nn.parameter import Parameter

SNAP_FRACTION = 0.5  # shrink fraction at which a weight snaps to zero: the nearer end of the segment


class ZeroLoRALinear(nn.Module):
    """Bias-free linear layer with a delete-only mask: W_used = W * (1 - clamp(scale * B @ A, 0, 1))."""

    def __init__(self, weight: torch.Tensor, rank: int, alpha: float) -> None:
        """Copy the frozen weight [out, in]; create B = 0 and A with a first row of ones (whole-neuron deletions)
        and N(0, 1) other rows."""
        super().__init__()
        out_features, in_features = weight.shape
        self.in_features = in_features
        self.out_features = out_features
        self.scale = alpha / rank
        self.snap = True  # eval forward: snapped (delete-only) weight; False uses the unsnapped training weight
        self.bypass = False  # True: the frozen layer alone (the base model), used for reference outputs
        self.row_snap = False  # True: the snap decides whole rows (neurons): off where the deleted fraction >= 0.5
        self.weight = Parameter(weight.detach().clone(), requires_grad=False)
        self.lora_A = Parameter(torch.randn(rank, in_features, device=weight.device, dtype=torch.float32))
        self.lora_B = Parameter(torch.zeros(out_features, rank, device=weight.device, dtype=torch.float32))
        with torch.no_grad():
            self.lora_A[0].fill_(1.0)
        # Shrink fraction, snap mask and the parameter versions they were built from. An optimizer step changes the
        # parameters in place and raises their versions, so the cache holds exactly until the next step; checkpoint
        # recomputation reuses it. Fused AdamW and writes through .data do not raise versions, so neither may change
        # a trainable parameter.
        self._cached_shrink = None
        self._cached_versions = None

    def product(self) -> torch.Tensor:
        """Return P = scale * B @ A [out, in] in float32, with the autograd graph."""
        with torch.autocast(device_type=self.lora_B.device.type, enabled=False):
            return self.scale * (self.lora_B @ self.lora_A)

    @torch.no_grad()
    def shrink_fraction_float64(self) -> torch.Tensor:
        """Return s = clamp(P, 0, 1) [out, in] in float64; s = 0 where W = 0, which deletion cannot change."""
        # float64 outside autocast: bf16 or TF32 would round B @ A to 1e-3 or worse near the snap point; the inner
        # dimension is the rank, so the float64 product is cheap.
        with torch.autocast(device_type=self.lora_B.device.type, enabled=False):
            product = self.scale * (self.lora_B.double() @ self.lora_A.double())
        return torch.where(self.weight != 0, product.clamp(0.0, 1.0), 0.0)

    def shrink_and_keep(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the float32 shrink fraction and the bool snap keep mask, decided in float64: per weight,
        s < SNAP_FRACTION; with row_snap, per row, mean(s) < SNAP_FRACTION, so each neuron ends untouched or fully
        off and no neuron is half cut."""
        shrink = self.shrink_fraction_float64()
        if not self.row_snap:
            return shrink.float(), shrink < SNAP_FRACTION
        keep_row = shrink.mean(dim=1) < SNAP_FRACTION
        return shrink.float(), keep_row[:, None].expand_as(shrink).clone()

    def cached_shrink_fraction(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return shrink_and_keep(), rebuilt only when a parameter changed since the last call."""
        versions = (self.row_snap, *((parameter._version, parameter.data_ptr()) for parameter in self.parameters()))
        if self._cached_versions != versions:
            self._cached_shrink = self.shrink_and_keep()
            self._cached_versions = versions
        return self._cached_shrink

    @torch.no_grad()
    def shrink_fraction(self) -> torch.Tensor:
        """Return the shrink fraction s [out, in] in float32."""
        return self.shrink_fraction_float64().float()

    @torch.no_grad()
    def keep_mask(self) -> torch.Tensor:
        """Return the bool keep mask [out, in] of the snapped model: False where s >= SNAP_FRACTION."""
        return self.shrink_and_keep()[1]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply W * (1 - s) in training (straight-through clamp), the snapped or unsnapped weight in evaluation."""
        if self.bypass:
            return nn.functional.linear(input, self.weight)
        if not self.training:
            shrink, keep = self.cached_shrink_fraction()
            if self.snap:
                return nn.functional.linear(input, self.weight * keep.to(self.weight.dtype))
            return nn.functional.linear(input, (self.weight.float() * (1.0 - shrink)).to(self.weight.dtype))
        product = self.product()
        # Straight-through clamp: the value is clamp(P, 0, 1), the gradient is that of P, exact inside [0, 1].
        shrink = product + (product.clamp(0.0, 1.0) - product).detach()
        used_weight = (self.weight.float() * (1.0 - shrink)).to(self.weight.dtype)
        return nn.functional.linear(input, used_weight)

    def sampled_product(self, num_samples: int) -> torch.Tensor:
        """Return P at num_samples random entries, without the full product."""
        rows = torch.randint(self.out_features, (num_samples,), device=self.lora_B.device)
        cols = torch.randint(self.in_features, (num_samples,), device=self.lora_A.device)
        return self.scale * (self.lora_B[rows] * self.lora_A[:, cols].T).sum(dim=1)

    def double_well_penalty(self, num_samples: int) -> torch.Tensor:
        """Return mean(P^2 (1 - P)^2) over num_samples random entries: 0 at keep and at delete, larger outside."""
        product = self.sampled_product(num_samples)
        return (product.pow(2) * (1.0 - product).pow(2)).mean()

    def shrink_penalty(self, num_samples: int) -> torch.Tensor:
        """Return mean(log(1 + relu(P))) over num_samples random entries: a soft count of deleted weights."""
        return torch.log1p(torch.relu(self.sampled_product(num_samples))).mean()

    def shrink_block(self, columns: torch.Tensor) -> torch.Tensor:
        """Return the differentiable shrink fraction s [out, len(columns)] of the given input columns."""
        with torch.autocast(device_type=self.lora_B.device.type, enabled=False):
            product = self.scale * (self.lora_B @ self.lora_A[:, columns])
        return torch.where(self.weight[:, columns] != 0, product.clamp(0.0, 1.0), 0.0)

    def row_commit_penalty(self, num_columns: int) -> torch.Tensor:
        """Return sum over rows of u^2 (1 - u)^2, u = the row's mean deletion fraction (estimated on num_columns
        random input columns): 0 for an untouched and a fully deleted neuron, largest for a half-cut one."""
        columns = torch.randint(self.in_features, (num_columns,), device=self.lora_A.device)
        deleted_fraction = self.shrink_block(columns).mean(dim=1)
        return (deleted_fraction.pow(2) * (1.0 - deleted_fraction).pow(2)).sum()

    def neuron_penalty(self, num_columns: int, tau: float) -> torch.Tensor:
        """Return the soft count of touched neurons: sum over rows of log(1 + u / tau) / log(1 + 1 / tau).

        u is the mean shrink fraction of the row, estimated on num_columns random input columns.
        """
        columns = torch.randint(self.in_features, (num_columns,), device=self.lora_A.device)
        deleted_fraction = self.shrink_block(columns).mean(dim=1)  # [out]
        return torch.log1p(deleted_fraction / tau).sum() / math.log1p(1.0 / tau)

    def begin_step(self) -> None:
        """Prepare one training step; layers with sampled gates draw their noise here (see HardConcreteNeuronLinear)."""

    def end_step(self) -> None:
        """Finish one optimizer step; layers with constrained parameters project them here."""

    def extra_repr(self) -> str:
        """Describe the layer's shape and rank."""
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.lora_A.shape[0]}"


class ZeroNeuronLoRALinear(ZeroLoRALinear):
    """Delete-only adapter whose unit is a whole output row: W_used = W * (1 - clamp(b, 0, 1))[:, None].

    The mask adapter at rank 1 with A frozen at all ones: one parameter b_j per row, s_j = clamp(b_j, 0, 1), and
    the snap deletes the whole row where s_j >= SNAP_FRACTION. On gate_proj a deleted row switches the MLP neuron
    off. The rank and alpha arguments are ignored: rank 1 and scale 1. A row scale equals an output scale, so the
    forward passes scale the output of the frozen layer and never form a [out, in] weight.
    """

    def __init__(self, weight: torch.Tensor, rank: int, alpha: float) -> None:
        """Create b = 0 [out, 1] and a frozen all-ones A [1, in]."""
        super().__init__(weight=weight, rank=1, alpha=1.0)
        self.lora_A.requires_grad_(False)
        with torch.no_grad():
            self.lora_A.fill_(1.0)

    def row_product(self) -> torch.Tensor:
        """Return the unclamped row deletion value b [out] in float32, with the autograd graph."""
        return self.lora_B[:, 0]

    def row_shrink(self) -> torch.Tensor:
        """Return the row deletion fraction s = clamp(b, 0, 1) [out], with the autograd graph."""
        return self.row_product().clamp(0.0, 1.0)

    @torch.no_grad()
    def shrink_fraction_float64(self) -> torch.Tensor:
        """Return s [out, in] in float64: each row's fraction in all its columns."""
        return self.row_shrink().double()[:, None].expand(self.out_features, self.in_features)

    @torch.no_grad()
    def keep_mask(self) -> torch.Tensor:
        """Return the bool keep mask [out, in]: a row is all False where s >= SNAP_FRACTION."""
        keep_row = self.row_shrink().double() < SNAP_FRACTION
        return keep_row[:, None].expand(self.out_features, self.in_features).clone()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Scale each output by 1 - s (training, straight-through clamp; or unsnapped eval) or by the snapped keep."""
        if self.bypass:
            return nn.functional.linear(input, self.weight)
        output = nn.functional.linear(input, self.weight)
        if self.training:
            product = self.row_product()
            # Straight-through clamp: the value is clamp(b, 0, 1), the gradient is that of b, exact inside [0, 1].
            scale = 1.0 - (product + (product.clamp(0.0, 1.0) - product).detach())
        elif self.snap:
            scale = (self.row_shrink() < SNAP_FRACTION).float()
        else:
            scale = 1.0 - self.row_shrink()
        return output * scale.to(output.dtype)

    def shrink_block(self, columns: torch.Tensor) -> torch.Tensor:
        """Return the differentiable shrink fraction s [out, len(columns)] of the given input columns."""
        return self.row_shrink()[:, None].expand(self.out_features, len(columns))

    def double_well_penalty(self, num_samples: int) -> torch.Tensor:
        """Return mean over rows of b^2 (1 - b)^2, exact; num_samples is unused."""
        product = self.row_product()
        return (product.pow(2) * (1.0 - product).pow(2)).mean()

    def shrink_penalty(self, num_samples: int) -> torch.Tensor:
        """Return mean over rows of log(1 + relu(b)), exact; num_samples is unused."""
        return torch.log1p(torch.relu(self.row_product())).mean()

    def neuron_penalty(self, num_columns: int, tau: float) -> torch.Tensor:
        """Return sum over rows of log(1 + s / tau) / log(1 + 1 / tau), exact; num_columns is unused."""
        return torch.log1p(self.row_shrink() / tau).sum() / math.log1p(1.0 / tau)


class ZeroFullMaskLinear(ZeroLoRALinear):
    """Per-weight reference mask: s = clamp(P, 0, 1) with one trainable value per weight (no rank limit).

    Everything else equals the low-rank mask layer (straight-through clamp, snap at SNAP_FRACTION, penalties), so a
    comparison with ZeroLoRALinear isolates the effect of the rank limit. P starts at 0 (the base model). Memory:
    one float32 value, its gradient and two Adam moments per masked weight. The rank and alpha arguments are ignored.
    """

    def __init__(self, weight: torch.Tensor, rank: int, alpha: float) -> None:
        """Create P = 0 [out, in] in float32 and freeze the unused rank-1 factors."""
        super().__init__(weight=weight, rank=1, alpha=1.0)
        self.lora_A.requires_grad_(False)
        self.lora_B.requires_grad_(False)
        self.full_mask = Parameter(torch.zeros(weight.shape, device=weight.device, dtype=torch.float32))

    def product(self) -> torch.Tensor:
        """Return P [out, in], with the autograd graph."""
        return self.full_mask

    @torch.no_grad()
    def shrink_fraction_float64(self) -> torch.Tensor:
        """Return s = clamp(P, 0, 1) [out, in] in float64; s = 0 where W = 0, which deletion cannot change."""
        return torch.where(self.weight != 0, self.full_mask.double().clamp(0.0, 1.0), 0.0)

    def shrink_block(self, columns: torch.Tensor) -> torch.Tensor:
        """Return the differentiable shrink fraction s [out, len(columns)] of the given input columns."""
        return torch.where(self.weight[:, columns] != 0, self.full_mask[:, columns].clamp(0.0, 1.0), 0.0)

    def sampled_product(self, num_samples: int) -> torch.Tensor:
        """Return P at num_samples random entries."""
        rows = torch.randint(self.out_features, (num_samples,), device=self.full_mask.device)
        cols = torch.randint(self.in_features, (num_samples,), device=self.full_mask.device)
        return self.full_mask[rows, cols]

    def double_well_penalty(self, num_samples: int) -> torch.Tensor:
        """Return mean(P^2 (1 - P)^2) over num_samples random entries."""
        product = self.sampled_product(num_samples)
        return (product.pow(2) * (1.0 - product).pow(2)).mean()

    def shrink_penalty(self, num_samples: int) -> torch.Tensor:
        """Return mean(log(1 + relu(P))) over num_samples random entries."""
        return torch.log1p(torch.relu(self.sampled_product(num_samples))).mean()


class HardConcreteNeuronLinear(ZeroNeuronLoRALinear):
    """Whole-neuron keep gate with the hard concrete distribution (Louizos et al. 2018, ICLR, Eq. 10-13).

    Training: z = clamp(sigmoid((log u - log(1 - u) + log_alpha) / BETA) (ZETA - GAMMA) + GAMMA, 0, 1), u ~ U(0, 1),
    one sample per row and training step, and the row output is multiplied by z. Evaluation: the deterministic gate
    clamp(sigmoid(log_alpha) (ZETA - GAMMA) + GAMMA, 0, 1); the snapped model deletes the row where it is below 0.5.
    The noise is drawn in begin_step, so the recomputation of gradient checkpointing sees the same sample.
    Expected number of closed gates: sum_j (1 - sigmoid(log_alpha_j - BETA log(-GAMMA / ZETA))) (Eq. 12); this is
    the neuron penalty (a count of deleted neurons). expected_open() is the L0 count of kept neurons that the
    localization objective of Chang et al. 2024 (NAACL, Eq. 12) penalizes.
    """

    GAMMA, ZETA, BETA = -0.1, 1.1, 2.0 / 3.0  # Louizos et al. 2018, Sec. 4
    OPEN_LOG_ALPHA = 3.0  # deterministic gate 1, P(closed) about 0.01: the base model at the start

    def __init__(self, weight: torch.Tensor, rank: int, alpha: float) -> None:
        """Create log_alpha [out] with all gates open and freeze the rank-1 deletion factor."""
        super().__init__(weight=weight, rank=rank, alpha=alpha)
        self.lora_B.requires_grad_(False)
        self.log_alpha = Parameter(torch.full((self.out_features,), self.OPEN_LOG_ALPHA, device=weight.device))
        self.noise: torch.Tensor | None = None

    def begin_step(self) -> None:
        """Draw this step's uniform noise, away from 0 and 1 so that its logit stays finite."""
        self.noise = torch.rand(self.out_features, device=self.log_alpha.device).clamp(1e-6, 1.0 - 1e-6)

    def deterministic_gate(self) -> torch.Tensor:
        """Return the evaluation keep gate [out] in [0, 1], with the autograd graph."""
        return (torch.sigmoid(self.log_alpha) * (self.ZETA - self.GAMMA) + self.GAMMA).clamp(0.0, 1.0)

    def sampled_gate(self) -> torch.Tensor:
        """Return the training keep gate [out] for this step's noise."""
        if self.noise is None:
            self.begin_step()
        logits = (torch.log(self.noise) - torch.log1p(-self.noise) + self.log_alpha) / self.BETA
        return (torch.sigmoid(logits) * (self.ZETA - self.GAMMA) + self.GAMMA).clamp(0.0, 1.0)

    def open_probability(self) -> torch.Tensor:
        """Return P(gate > 0) [out]."""
        return torch.sigmoid(self.log_alpha - self.BETA * math.log(-self.GAMMA / self.ZETA))

    def expected_open(self) -> torch.Tensor:
        """Return the expected number of open gates."""
        return self.open_probability().sum()

    def row_shrink(self) -> torch.Tensor:
        """Return the evaluation deletion fraction 1 - gate [out]."""
        return 1.0 - self.deterministic_gate()

    def neuron_scores(self) -> torch.Tensor:
        """Return each neuron's importance for the trained objective: sigmoid(log_alpha) (Chang et al. 2024)."""
        return torch.sigmoid(self.log_alpha.detach())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Scale each output by the sampled gate (training) or the deterministic / snapped gate (evaluation)."""
        if self.bypass:
            return nn.functional.linear(input, self.weight)
        if not self.training:
            return super().forward(input)
        output = nn.functional.linear(input, self.weight)
        return output * self.sampled_gate().to(output.dtype)

    def double_well_penalty(self, num_samples: int) -> torch.Tensor:
        """Return 0: the hard concrete gate is pushed to 0 or 1 by its own noise, not by a penalty."""
        return self.log_alpha.sum() * 0.0

    def shrink_penalty(self, num_samples: int) -> torch.Tensor:
        """Return 0 (no weight-level count for a neuron gate)."""
        return self.log_alpha.sum() * 0.0

    def neuron_penalty(self, num_columns: int, tau: float) -> torch.Tensor:
        """Return the expected number of closed gates (deleted neurons); num_columns and tau are unused."""
        return (1.0 - self.open_probability()).sum()


class NeuMuterNeuronLinear(HardConcreteNeuronLinear):
    """The hard concrete neuron gate with the temperature of NeuMuter (Hou et al. 2025, Findings of ACL, Sec. 3):
    beta = 0.3 instead of 2/3; gamma and zeta as in Louizos et al. 2018."""

    BETA = 0.3


PARAMETRIZATIONS = {"mask": ZeroLoRALinear, "neuron": ZeroNeuronLoRALinear, "full": ZeroFullMaskLinear,
                    "hardconcrete": HardConcreteNeuronLinear, "neumuter": NeuMuterNeuronLinear}


def convert_to_zero_lora(model: nn.Module, target_modules: list[str], rank: int, alpha: float,
                         parametrization: str = "mask") -> int:
    """Replace each bias-free nn.Linear whose name contains a target with a Zero LoRA layer; return the count."""
    layer_class = PARAMETRIZATIONS[parametrization]
    targets = [(name, module) for name, module in model.named_modules()
               if isinstance(module, nn.Linear) and any(t in name for t in target_modules)]
    for name, module in targets:
        assert module.bias is None, f"{name} has a bias, which ZeroLoRALinear does not support"
        parent_name, _, child_name = name.rpartition(".")
        setattr(model.get_submodule(parent_name), child_name,
                layer_class(weight=module.weight.data, rank=rank, alpha=alpha))
    return len(targets)


def zero_lora_layers(model: nn.Module) -> list[tuple[str, ZeroLoRALinear]]:
    """Return (name, layer) for every ZeroLoRALinear of the model."""
    return [(name, m) for name, m in model.named_modules() if isinstance(m, ZeroLoRALinear)]


def set_snap(model: nn.Module, snap: bool) -> None:
    """Choose the eval forward of every ZeroLoRALinear: snapped (delete-only) or the unsnapped training weight."""
    for _, layer in zero_lora_layers(model):
        layer.snap = snap


def double_well_penalty(model: nn.Module, num_samples: int) -> torch.Tensor:
    """Return the mean over layers of each layer's sampled double-well penalty."""
    layers = zero_lora_layers(model)
    return torch.stack([layer.double_well_penalty(num_samples=num_samples) for _, layer in layers]).mean()


def shrink_penalty(model: nn.Module, num_samples: int) -> torch.Tensor:
    """Return the mean over layers of each layer's sampled shrink penalty."""
    layers = zero_lora_layers(model)
    return torch.stack([layer.shrink_penalty(num_samples=num_samples) for _, layer in layers]).mean()


def set_row_snap(model: nn.Module, row_snap: bool) -> None:
    """Choose per-row (whole-neuron) or per-weight snapping for every Zero LoRA layer."""
    for _, layer in zero_lora_layers(model):
        layer.row_snap = row_snap


def row_commit_penalty(model: nn.Module, num_columns: int) -> torch.Tensor:
    """Return the summed row commit penalty of all layers (see ZeroLoRALinear.row_commit_penalty)."""
    layers = zero_lora_layers(model)
    return torch.stack([layer.row_commit_penalty(num_columns=num_columns) for _, layer in layers]).sum()


def neuron_penalty(model: nn.Module, num_columns: int, tau: float) -> torch.Tensor:
    """Return the soft count of touched neurons, summed over layers."""
    layers = zero_lora_layers(model)
    return torch.stack([layer.neuron_penalty(num_columns=num_columns, tau=tau) for _, layer in layers]).sum()


@torch.no_grad()
def extract_all_masks(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return {"<layer>.weight": bool keep mask} of the snapped model on the CPU."""
    return {f"{name}.weight": layer.keep_mask().cpu() for name, layer in zero_lora_layers(model)}


@contextlib.contextmanager
def bypassed(model: nn.Module):
    """Run the block with every Zero LoRA layer replaced by its frozen layer, i.e. as the base model."""
    layers = [layer for _, layer in zero_lora_layers(model)]
    for layer in layers:
        layer.bypass = True
    try:
        yield
    finally:
        for layer in layers:
            layer.bypass = False


def begin_step(model: nn.Module) -> None:
    """Call begin_step on every Zero LoRA layer before a training step's forward pass."""
    for _, layer in zero_lora_layers(model):
        layer.begin_step()


def end_step(model: nn.Module) -> None:
    """Call end_step on every Zero LoRA layer after the optimizer step."""
    for _, layer in zero_lora_layers(model):
        layer.end_step()


def expected_open(model: nn.Module) -> torch.Tensor:
    """Return the summed open-neuron count of all neuron-gate layers (the localization objective's penalty)."""
    return torch.stack([layer.expected_open() for _, layer in zero_lora_layers(model)]).sum()


@torch.no_grad()
def neuron_scores(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return {"<layer>.weight": importance score per row} of the neuron-gate layers, on the CPU."""
    return {f"{name}.weight": layer.neuron_scores().cpu() for name, layer in zero_lora_layers(model)}

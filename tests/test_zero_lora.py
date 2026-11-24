"""CPU checks of the Zero LoRA layer against independent reference formulas.

Run from the project root: PYTHONPATH=src torch-python tests/test_zero_lora.py
"""

import math
import tempfile

import torch

import masks
import zero_lora


def make_layer(out_features: int = 12, in_features: int = 10, rank: int = 3) -> zero_lora.ZeroLoRALinear:
    """Return a mask layer with random W, A and nonzero B, so that some weights are deleted partly and some fully."""
    torch.manual_seed(0)
    layer = zero_lora.ZeroLoRALinear(weight=torch.randn(out_features, in_features), rank=rank, alpha=rank)
    with torch.no_grad():
        layer.lora_B.normal_(std=1.0)
    layer.train()
    return layer


def test_zero_weight_stays_zero() -> None:
    """A weight that is exactly 0 has shrink fraction 0, and no value is infinite or undefined."""
    layer = make_layer()
    with torch.no_grad():
        layer.weight[0, 0] = 0.0
    shrink = layer.shrink_fraction()
    assert shrink[0, 0].item() == 0.0 and torch.isfinite(shrink).all()


def test_cache_follows_parameter_changes() -> None:
    """The cached shrink fraction equals a fresh one after optimizer steps and after an in-place copy of B."""
    layer = make_layer()
    for foreach in (False, True):
        optimizer = torch.optim.AdamW(layer.parameters(), lr=0.5, foreach=foreach)
        x = torch.randn(4, 10)
        for _ in range(3):
            before = layer.cached_shrink_fraction()[0].clone()
            layer(x).pow(2).sum().backward()
            optimizer.step()
            optimizer.zero_grad()
            assert torch.equal(layer.cached_shrink_fraction()[0], layer.shrink_fraction()), "stale after a step"
        assert not torch.equal(before, layer.cached_shrink_fraction()[0]), "test needs a step that changes s"
    with torch.no_grad():
        layer.lora_B.copy_(-layer.lora_B)
    assert torch.equal(layer.cached_shrink_fraction()[0], layer.shrink_fraction()), "stale after a copy"


def mask_layer(product: torch.Tensor) -> zero_lora.ZeroLoRALinear:
    """Return a 6x6 rank-6 mask layer with P = scale * B @ A equal to ``product`` (A = I)."""
    torch.manual_seed(0)
    layer = zero_lora.ZeroLoRALinear(weight=torch.randn(6, 6), rank=6, alpha=6)
    with torch.no_grad():
        layer.lora_A.copy_(torch.eye(6))
        layer.lora_B.copy_(product / layer.scale)
    layer.train()
    return layer


def test_mask_layer_forward_snap_and_initial_state() -> None:
    """Training uses W * (1 - clamp(P, 0, 1)); eval deletes where P >= 0.5; a new layer equals the frozen layer."""
    product = torch.linspace(-0.5, 1.5, 36).reshape(6, 6)
    layer = mask_layer(product)
    x = torch.randn(3, 6)
    torch.testing.assert_close(layer(x), x @ (layer.weight * (1 - product.clamp(0, 1))).T)
    layer.eval()
    torch.testing.assert_close(layer(x), x @ (layer.weight * (product < 0.5)).T)
    torch.manual_seed(0)
    fresh = zero_lora.ZeroLoRALinear(weight=torch.randn(6, 4), rank=2, alpha=2)
    assert torch.equal(fresh.lora_A[0], torch.ones(4)) and fresh.keep_mask().all()
    torch.testing.assert_close(fresh.train()(x[:, :4]), x[:, :4] @ fresh.weight.T)


def test_mask_layer_gradient_is_exact_inside_the_segment() -> None:
    """With every P in [0, 1], the B gradient equals the autograd gradient of W * (1 - P)."""
    product = torch.rand(6, 6, generator=torch.Generator().manual_seed(1))
    layer, reference = mask_layer(product), mask_layer(product)
    x, target = torch.randn(5, 6), torch.randn(5, 6)
    ((layer(x) - target) ** 2).sum().backward()
    used = reference.weight * (1 - reference.scale * reference.lora_B @ reference.lora_A)
    ((x @ used.T - target) ** 2).sum().backward()
    torch.testing.assert_close(layer.lora_B.grad, reference.lora_B.grad)


def test_neuron_penalty_counts_neurons_and_prefers_concentration() -> None:
    """A fully deleted row costs 1 neuron; the same deletions spread over all rows cost more; nothing costs 0."""
    assert mask_layer(torch.zeros(6, 6)).neuron_penalty(num_columns=64, tau=0.1).item() == 0.0
    one_row = torch.zeros(6, 6)
    one_row[2] = 1.0
    torch.testing.assert_close(mask_layer(one_row).neuron_penalty(num_columns=64, tau=0.1), torch.tensor(1.0))
    spread = torch.eye(6)  # 6 deleted weights, one in each row
    assert mask_layer(spread).neuron_penalty(num_columns=4096, tau=0.1).item() > 2.0


def test_mask_layer_double_well_and_shrink_penalties() -> None:
    """The double well is 0 at P = 0 and P = 1 and positive between; the shrink count is log 2 at P = 1."""
    for value, expected in [(0.0, 0.0), (1.0, 0.0), (0.5, 0.0625)]:
        result = mask_layer(torch.full((6, 6), value)).double_well_penalty(num_samples=100)
        torch.testing.assert_close(result, torch.tensor(expected))
    torch.testing.assert_close(mask_layer(torch.ones(6, 6)).shrink_penalty(num_samples=100),
                               torch.tensor(math.log(2.0)))
    assert mask_layer(-torch.ones(6, 6)).shrink_penalty(num_samples=100).item() == 0.0


def neuron_layer(row_values: torch.Tensor, in_features: int = 5) -> zero_lora.ZeroNeuronLoRALinear:
    """Return a neuron layer with a random weight [rows, in_features] and b = row_values."""
    torch.manual_seed(0)
    layer = zero_lora.ZeroNeuronLoRALinear(weight=torch.randn(len(row_values), in_features), rank=4, alpha=8.0)
    with torch.no_grad():
        layer.lora_B[:, 0] = row_values
    return layer


def test_neuron_layer_forward_gradient_and_masks() -> None:
    """Whole rows only: training uses W * (1 - clamp(b)), eval deletes rows where b >= 0.5, b gets the exact
    gradient inside [0, 1], A stays frozen at ones, and a new layer equals the frozen layer."""
    row_values = torch.tensor([-0.3, 0.0, 0.2, 0.5, 0.8, 1.4])
    layer = neuron_layer(row_values)
    assert [name for name, parameter in layer.named_parameters() if parameter.requires_grad] == ["lora_B"]
    assert torch.equal(layer.lora_A, torch.ones(1, 5))
    x = torch.randn(3, 5)
    torch.testing.assert_close(layer(x), x @ (layer.weight * (1 - row_values.clamp(0, 1))[:, None]).T)
    layer.eval()
    torch.testing.assert_close(layer(x), x @ (layer.weight * (row_values < 0.5)[:, None]).T)
    layer.snap = False
    torch.testing.assert_close(layer(x), x @ (layer.weight * (1 - row_values.clamp(0, 1))[:, None]).T)
    expected_keep = (row_values < 0.5)[:, None].expand(6, 5)
    assert torch.equal(layer.keep_mask(), expected_keep)
    inside = neuron_layer(torch.tensor([0.1, 0.3, 0.6, 0.9])).train()
    inside(x).pow(2).sum().backward()
    reference_b = inside.lora_B.detach()[:, 0].clone().requires_grad_(True)
    (x @ (inside.weight * (1 - reference_b)[:, None]).T).pow(2).sum().backward()
    torch.testing.assert_close(inside.lora_B.grad[:, 0], reference_b.grad)
    fresh = zero_lora.ZeroNeuronLoRALinear(weight=torch.randn(6, 5), rank=4, alpha=8.0).train()
    torch.testing.assert_close(fresh(x), x @ fresh.weight.T)
    assert fresh.keep_mask().all()


def test_neuron_layer_penalties() -> None:
    """Exact penalties: double well 0 at b = 0 and 1 and 1/16 at 0.5; neuron count 1 per deleted row."""
    layer = neuron_layer(torch.tensor([0.0, 1.0, 0.5, -1.0]))
    torch.testing.assert_close(layer.double_well_penalty(num_samples=1), torch.tensor(0.0625 / 4 + 4.0 / 4))
    torch.testing.assert_close(layer.shrink_penalty(num_samples=1), torch.tensor(math.log(2.0) / 4 + math.log(1.5) / 4))
    count = layer.neuron_penalty(num_columns=1, tau=1e-4)
    expected = 1.0 + math.log1p(0.5 / 1e-4) / math.log1p(1e4)
    torch.testing.assert_close(count, torch.tensor(expected))


def test_saved_masks_reproduce_snapped_model() -> None:
    """Masks saved, loaded in this process and applied to the base model give the snapped model's output."""
    for parametrization in zero_lora.PARAMETRIZATIONS:
        check_saved_masks(parametrization)


def check_saved_masks(parametrization: str) -> None:
    """Run the saved-mask check for one parametrization; the neuron one must delete whole rows only."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32)
    base = LlamaForCausalLM(config).eval()
    model = LlamaForCausalLM(config).eval()
    model.load_state_dict(base.state_dict())
    assert zero_lora.convert_to_zero_lora(model=model, target_modules=["gate_proj"], rank=2, alpha=2,
                                         parametrization=parametrization) == 2
    for _, layer in zero_lora.zero_lora_layers(model):
        with torch.no_grad():
            for name, parameter in layer.named_parameters():
                if parameter.requires_grad and name == "log_alpha":
                    parameter.normal_(std=3.0)  # gates on both sides of the snap
                elif parameter.requires_grad and name == "keep_scale":
                    parameter.uniform_(0.0, 1.0)
                elif parameter.requires_grad:
                    parameter.normal_(std=0.5)
    model.eval()
    keep_masks = zero_lora.extract_all_masks(model)
    assert sum(int((~keep).sum()) for keep in keep_masks.values()) > 0, "test needs deleted weights"
    if parametrization in ("neuron", "hardconcrete", "neumuter"):
        assert all(torch.equal(keep.all(dim=1), keep.any(dim=1)) for keep in keep_masks.values()), "partial row"
    with tempfile.TemporaryDirectory() as directory:
        masks.save_masks(keep_masks, directory)
        loaded = masks.load_masks(directory)
    assert all(torch.equal(loaded[name], keep_masks[name]) for name in keep_masks), "mask file round trip failed"
    masks.apply_masks(base, loaded)
    input_ids = torch.randint(64, (2, 7))
    with torch.no_grad():
        torch.testing.assert_close(model(input_ids).logits, base(input_ids).logits, rtol=0, atol=0)


def test_full_mask_layer_matches_formula_and_has_exact_gradient() -> None:
    """Per-weight mask: training uses W * (1 - clamp(P)), evaluation deletes where P >= 0.5, P gets the exact
    gradient inside [0, 1], the unused factors are frozen, and a new layer equals the frozen layer."""
    torch.manual_seed(0)
    layer = zero_lora.ZeroFullMaskLinear(weight=torch.randn(6, 5), rank=4, alpha=8.0)
    assert [name for name, parameter in layer.named_parameters() if parameter.requires_grad] == ["full_mask"]
    x = torch.randn(3, 5)
    torch.testing.assert_close(layer.train()(x), x @ layer.weight.T)
    values = torch.linspace(-0.4, 1.4, 30).reshape(6, 5)
    with torch.no_grad():
        layer.full_mask.copy_(values)
    torch.testing.assert_close(layer(x), x @ (layer.weight * (1 - values.clamp(0, 1))).T)
    layer.eval()
    torch.testing.assert_close(layer(x), x @ (layer.weight * (values < 0.5)).T)
    assert torch.equal(layer.keep_mask(), values < 0.5)
    inside = zero_lora.ZeroFullMaskLinear(weight=torch.randn(6, 5), rank=4, alpha=8.0).train()
    with torch.no_grad():
        inside.full_mask.uniform_(0.1, 0.9)
    inside(x).pow(2).sum().backward()
    reference = inside.full_mask.detach().clone().requires_grad_(True)
    (x @ (inside.weight * (1 - reference)).T).pow(2).sum().backward()
    torch.testing.assert_close(inside.full_mask.grad, reference.grad)


def test_hard_concrete_gate() -> None:
    """Hard concrete: new layers equal the frozen layer in evaluation; the training gate follows Louizos Eq. 10-11
    for the drawn noise and stays fixed until the next begin_step; the closed count matches Eq. 12; the evaluation
    gate snaps rows at 0.5."""
    torch.manual_seed(0)
    layer = zero_lora.HardConcreteNeuronLinear(weight=torch.randn(4, 3), rank=4, alpha=4.0)
    x = torch.randn(2, 3)
    torch.testing.assert_close(layer.eval()(x), x @ layer.weight.T)
    with torch.no_grad():
        layer.log_alpha.copy_(torch.tensor([-3.0, -0.5, 0.5, 3.0]))
    layer.train()
    layer.begin_step()
    u = layer.noise
    gamma, zeta, beta = -0.1, 1.1, 2.0 / 3.0
    expected_gate = (torch.sigmoid((u.log() - (1 - u).log() + layer.log_alpha) / beta) * (zeta - gamma)
                     + gamma).clamp(0, 1)
    torch.testing.assert_close(layer(x), (x @ layer.weight.T) * expected_gate)
    torch.testing.assert_close(layer(x), (x @ layer.weight.T) * expected_gate)  # same noise until begin_step
    closed = (1 - torch.sigmoid(layer.log_alpha - beta * math.log(-gamma / zeta))).sum()
    torch.testing.assert_close(layer.neuron_penalty(num_columns=1, tau=1.0), closed)
    deterministic = (torch.sigmoid(layer.log_alpha) * (zeta - gamma) + gamma).clamp(0, 1)
    layer.eval()
    torch.testing.assert_close(layer(x), (x @ layer.weight.T) * (deterministic >= 0.5).float())
    assert torch.equal(layer.keep_mask()[:, 0], deterministic >= 0.5)


def test_row_snap_leaves_no_half_cut_neuron() -> None:
    """With row_snap each row of the snapped mask is all kept or all deleted (deleted where the row's mean deletion
    fraction is >= 0.5); the row commit penalty is 0 for untouched and fully deleted rows and 1/16 at half."""
    layer = zero_lora.ZeroLoRALinear(weight=torch.randn(4, 8), rank=8, alpha=8)
    product = torch.zeros(4, 8)
    product[1, :6] = 1.0  # 75 % deleted: the whole row goes
    product[2, :2] = 1.0  # 25 % deleted: the whole row stays
    product[3] = 1.0  # fully deleted
    with torch.no_grad():
        layer.lora_A.copy_(torch.eye(8))
        layer.lora_B.copy_(product)
    layer.row_snap = True
    keep = layer.keep_mask()
    assert keep[0].all() and not keep[1].any() and keep[2].all() and not keep[3].any()
    assert bool((keep.all(dim=1) | (~keep).all(dim=1)).all()), "a half-cut row survived the snap"
    torch.testing.assert_close(layer.row_commit_penalty(num_columns=100_000),
                               torch.tensor(0.75**2 * 0.25**2 + 0.25**2 * 0.75**2), rtol=0.05, atol=1e-3)
    with torch.no_grad():
        layer.lora_B.zero_()
        layer.lora_B[:, 0] = 0.5
        layer.lora_A.fill_(1.0)  # every weight at s = 0.5
    torch.testing.assert_close(layer.row_commit_penalty(num_columns=64), torch.tensor(4 / 16))


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"PASS {name}")

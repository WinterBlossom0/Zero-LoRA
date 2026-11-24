"""CPU checks of planting: positions per plant type, training only the planted positions, the sparse change file
round trip through a new process-equivalent load, and the recovery score.

Run from the project root: CUDA_VISIBLE_DEVICES="" PYTHONPATH=src torch-python tests/test_planting.py
"""

import tempfile

import torch
from transformers import LlamaConfig, LlamaForCausalLM

import planting


def tiny_model() -> LlamaForCausalLM:
    """Return a 2-layer Llama (hidden 16, intermediate 32) with frozen weights."""
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=40, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64)
    model = LlamaForCausalLM(config)
    model.requires_grad_(False)
    return model


def test_positions_per_plant_type() -> None:
    """neuron: the gate and up rows and the down column of each neuron; subneuron: the given fraction of each gate
    row's columns; scattered: the given count of distinct gate weights."""
    generator = torch.Generator().manual_seed(0)
    neuron = planting.choose_positions("neuron", 2, 32, 16, generator, neurons=3)
    for kind in ("gate_proj", "up_proj", "down_proj"):
        assert sum(len(flat) for name, flat in neuron.items() if kind in name) == 3 * 16
    for name, flat in neuron.items():
        if "down_proj" in name:
            gate_rows = neuron[name.replace("down_proj", "gate_proj")] // 16
            assert torch.equal(torch.unique(flat % 32), torch.unique(gate_rows))
    subneuron = planting.choose_positions("subneuron", 2, 32, 16, generator, rows=4, row_fraction=0.25)
    assert sum(len(flat) for flat in subneuron.values()) == 4 * 4
    assert all(len(torch.unique(flat)) == len(flat) for flat in subneuron.values())
    scattered = planting.choose_positions("scattered", 2, 32, 16, generator, weights=50)
    assert sum(len(flat) for flat in scattered.values()) == 50
    candidates = torch.tensor([5, 40, 41, 63])  # flat neuron indices layer * 32 + row
    chosen = planting.choose_positions("subneuron", 2, 32, 16, generator, rows=3, neuron_pool=candidates)
    rows = {int(name.split(".")[2]) * 32 + int(row) for name, flat in chosen.items() for row in torch.unique(flat // 16)}
    assert rows <= set(candidates.tolist()) and len(rows) == 3, "subneuron rows must come from the candidates"
    importance = torch.rand(2, 32, 16, generator=generator)
    low = planting.choose_positions("subneuron", 2, 32, 16, generator, rows=3, neuron_pool=candidates,
                                    weight_importance=importance)
    for name, flat in low.items():  # in each row, the planted columns are the lowest-importance quarter
        layer = int(name.split(".")[2])
        for row in torch.unique(flat // 16):
            columns = flat[flat // 16 == row] % 16
            assert set(columns.tolist()) == set(importance[layer, row].argsort()[:4].tolist())


def test_only_planted_positions_change_and_round_trip() -> None:
    """Planting wipes the positions to zero and then trains only them; the bypass gives the wiped model; the change
    file reproduces the trained model on a fresh base model; recovery is exact for a deletion of exactly the planted
    units."""
    model = tiny_model()
    base_state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    positions = planting.choose_positions("subneuron", 2, 32, 16, torch.Generator().manual_seed(1), rows=3)
    planted = planting.install(model, positions, "subneuron", torch.Generator().manual_seed(1))
    for name, layer in planted.items():  # the positions start wiped to zero, the rest of the weight unchanged
        assert bool((layer.merged_weight().detach().flatten()[positions[name]] == 0).all())
    optimizer = torch.optim.SGD([layer.delta for layer in planted.values()], lr=0.5)
    input_ids = torch.randint(40, (2, 12))
    for _ in range(3):
        model(input_ids=input_ids, labels=input_ids).loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    for name, layer in planted.items():
        changed = (layer.merged_weight().detach() != base_state[name]).flatten().nonzero()[:, 0]
        assert set(changed.tolist()) <= set(positions[name].tolist()) and len(changed) > 0
    fresh_neuron = planting.install(tiny_model(), planting.choose_positions("neuron", 2, 32, 16,
                                                                           torch.Generator().manual_seed(3), neurons=2),
                                    "neuron", torch.Generator().manual_seed(3))
    assert all(bool((layer.delta != 0).all()) for layer in fresh_neuron.values()), "a wiped neuron must restart"
    with torch.no_grad():
        expected = model(input_ids=input_ids).logits
        with planting.bypassed(planted):
            unplanted = model(input_ids=input_ids).logits
        wiped = tiny_model()
        for name, flat in positions.items():
            wiped.get_parameter(name).data.view(-1)[flat] = 0
        torch.testing.assert_close(unplanted, wiped(input_ids=input_ids).logits)
    assert not any(layer.bypass for layer in planted.values())
    with tempfile.TemporaryDirectory() as directory:
        path = f"{directory}/changes.safetensors"
        planting.save_changes(planted, path)
        fresh = tiny_model()
        assert planting.apply_changes(fresh, path) == sum(len(flat) for flat in positions.values())
    with torch.no_grad():
        torch.testing.assert_close(fresh(input_ids=input_ids).logits, expected)
    keep = {name: torch.ones(32, 16, dtype=torch.bool) for name in positions}
    for name, flat in positions.items():
        keep[name].view(-1)[flat] = False
    total = sum(len(flat) for flat in positions.values())
    assert planting.recovery(keep, positions, "subneuron") == {"planted": total, "found": total, "hits": total,
                                                               "recall": 1.0, "precision": 1.0,
                                                               "weight_recall": 1.0, "weight_precision": 1.0}


def test_neuron_recovery_counts_whole_rows() -> None:
    """For the neuron plant, a neuron counts as found only when its whole gate row is deleted."""
    positions = planting.choose_positions("neuron", 2, 32, 16, torch.Generator().manual_seed(2), neurons=2)
    neurons = [(name.replace("down_proj", "gate_proj"), int(column)) for name, flat in positions.items()
               if "down_proj" in name for column in torch.unique(flat % 32)]
    keep = {f"model.layers.{layer}.mlp.gate_proj.weight": torch.ones(32, 16, dtype=torch.bool) for layer in range(2)}
    first_name, first_row = neurons[0]
    keep[first_name][first_row] = False
    second_name, second_row = neurons[1]
    keep[second_name][second_row, :8] = False  # a partial row is not a deleted neuron
    result = planting.recovery(keep, positions, "neuron")
    assert result["recall"] == 0.5 and result["precision"] == 1.0


def test_neuron_plant_keeps_original_neuron_sizes() -> None:
    """A planted neuron starts and stays at the norms its gate row, up row and down column had before the wipe."""
    model = tiny_model()
    original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    positions = planting.choose_positions("neuron", 2, 32, 16, torch.Generator().manual_seed(4), neurons=3)
    planted = planting.install(model, positions, "neuron", torch.Generator().manual_seed(4))

    def norms(weights: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """Norm of each planted neuron's row (gate_proj, up_proj) or column (down_proj), per parameter."""
        result = []
        for name, flat in positions.items():
            neuron_of = flat % 32 if "down_proj" in name else flat // 16
            values = weights[name].flatten()[flat].float()
            result.append(torch.stack([values[neuron_of == neuron].norm() for neuron in torch.unique(neuron_of)]))
        return result

    def planted_weights() -> dict[str, torch.Tensor]:
        return {name: layer.merged_weight().detach() for name, layer in planted.items()}

    for before, after in zip(norms(original), norms(planted_weights()), strict=True):
        torch.testing.assert_close(after, before)
    optimizer = torch.optim.SGD([layer.delta for layer in planted.values()], lr=5.0)
    input_ids = torch.randint(40, (2, 12))
    model(input_ids=input_ids, labels=input_ids).loss.backward()
    optimizer.step()
    planting.project_norms(planted)
    for before, after in zip(norms(original), norms(planted_weights()), strict=True):
        torch.testing.assert_close(after, before)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"PASS {name}")

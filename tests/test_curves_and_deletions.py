"""CPU checks of the deletion-curve metrics and of the top-k neuron selection.

Run from the project root: PYTHONPATH=src:experiments torch-python tests/test_curves_and_deletions.py
"""

import evaluate_deletions
import recovery
import torch

import curves
import planting


def test_damage_and_collateral() -> None:
    """Target damage averages the target codes; collateral averages the rest except related codes; other keys
    (wiki, wikitext) are ignored."""
    delta = {"flores_zho_Hans": 2.0, "flores_zho_Hant": 4.0, "flores_yue_Hant": 9.0, "flores_eng_Latn": 0.1,
             "flores_jpn_Jpan": 0.3, "wiki_en": 5.0, "wikitext": 7.0}
    damage, collateral = curves.damage_and_collateral(delta, ["zho_Hans", "zho_Hant"], ["yue_Hant"])
    assert damage == 3.0 and abs(collateral - 0.2) < 1e-12


def test_damage_at_collateral_and_area() -> None:
    """The best damage under a collateral cap, 0 when no point qualifies, and the step-function area."""
    points = [(1.0, 0.01), (3.0, 0.04), (5.0, 0.30), (2.0, 0.02)]
    assert curves.damage_at_collateral(points, 0.05) == 3.0
    assert curves.damage_at_collateral(points, 0.005) == 0.0
    # steps at 0, 0.01, 0.02, 0.04, 0.2: 0*0.01 + 1*0.01 + 2*0.02 + 3*0.16
    assert abs(curves.frontier_area(points, 0.2) - (0.01 + 0.04 + 0.48)) < 1e-12


def test_top_k_row_masks() -> None:
    """Budget k deletes the k highest finite scores over all layers, whole rows; too few finite scores -> None."""
    scores = {"a": torch.tensor([0.5, float("-inf"), 2.0]), "b": torch.tensor([1.0, 3.0])}
    shapes = {"a": torch.Size([3, 4]), "b": torch.Size([2, 4])}
    keep = evaluate_deletions.top_k_row_masks(scores, shapes, 3)
    assert keep["a"].all(dim=1).tolist() == [True, True, False]
    assert keep["b"].all(dim=1).tolist() == [False, False]
    assert all(torch.equal(mask.all(dim=1), mask.any(dim=1)) for mask in keep.values()), "partial row"
    assert evaluate_deletions.top_k_row_masks(scores, shapes, 5) is None


def test_top_k_weight_masks_and_neuron_budget() -> None:
    """Weight rankings delete exactly their first k flat indices; the neuron-plant budget counts each planted
    neuron once (its down_proj column), not its gate and up rows."""
    names = ["model.layers.0.mlp.gate_proj.weight", "model.layers.1.mlp.gate_proj.weight"]
    top = torch.tensor([5, 12 + 7, 3, 12 + 0])  # shape [3, 4]: 12 weights per layer
    keep = evaluate_deletions.top_k_weight_masks(top, names, (3, 4), 3)
    assert (~keep[names[0]]).flatten().nonzero()[:, 0].tolist() == [3, 5]
    assert (~keep[names[1]]).flatten().nonzero()[:, 0].tolist() == [7]
    positions = planting.choose_positions("neuron", 2, 32, 16, torch.Generator().manual_seed(0), neurons=5)
    assert recovery.planted_rows(positions, "neuron", 32, 16) == 5


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"PASS {name}")

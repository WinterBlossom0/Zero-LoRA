"""CPU checks of the neuron statistics behind the attribution baseline and the task-A importance of planting.

Run from the project root: CUDA_VISIBLE_DEVICES="" PYTHONPATH=src:experiments torch-python tests/test_neuron_scores.py
"""

import neuron_scores
import torch
import weight_scores
from transformers import LlamaConfig, LlamaForCausalLM

import sources
import train_language


def tiny_model() -> LlamaForCausalLM:
    """Return a 4-layer Llama (hidden 16, intermediate 32) with frozen weights, in eval mode."""
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=40, hidden_size=16, intermediate_size=32, num_hidden_layers=4,
                         num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64)
    model = LlamaForCausalLM(config).eval()
    model.requires_grad_(False)
    return model


def test_deletion_effect_uses_the_full_gradient() -> None:
    """effect = mean over counted positions of -h dL/dh, with dL/dh through every later layer: compared with the
    gradient of a zero perturbation added to each down_proj input, which autograd differentiates exactly."""
    model = tiny_model()
    ids = torch.randint(40, (3, 12))
    score = torch.ones_like(ids, dtype=torch.bool)
    score[:, :5] = False  # only the last tokens are scored
    statistics = neuron_scores.source_statistics(model, sources.Rows(ids, score), batch_size=3, with_gradient=True)

    perturbations, hidden = [], []
    for layer in model.model.layers:
        perturbation = torch.zeros(3, 12, 32, requires_grad=True)
        perturbations.append(perturbation)

        def hook(module, inputs, perturbation=perturbation):
            hidden.append(inputs[0].detach())
            return (inputs[0] + perturbation,)
        layer.mlp.down_proj.register_forward_pre_hook(hook)
    train_language.window_nll(model, ids, score_mask=score).sum().backward()
    predicts_scored = torch.zeros_like(score)
    predicts_scored[:, :-1] = score[:, 1:]
    counted = predicts_scored.float()[..., None]
    for index, perturbation in enumerate(perturbations):
        values = hidden[index]
        expected = (-(values * perturbation.grad) * counted).sum(dim=(0, 1)) / counted.sum()
        torch.testing.assert_close(statistics["effect"][index].float(), expected, rtol=1e-4, atol=1e-7)


def test_outlier_ranks_an_enlarged_neuron_first() -> None:
    """A neuron whose output column is 20 times larger than the others has the largest outlier score of its layer."""
    model = tiny_model()
    with torch.no_grad():
        model.model.layers[1].mlp.down_proj.weight[:, 7] *= 20
    assert int(neuron_scores.outlier_scores(model)[1].argmax()) == 7


def test_overflowing_loss_scale_is_retried() -> None:
    """A loss scale that overflows the gradient gives the same statistics as a safe one: the batch runs again at a
    smaller scale (fp32 here, so the first scale is set near the float32 limit)."""
    ids = torch.randint(40, (3, 12))
    rows = sources.Rows(ids, torch.ones_like(ids, dtype=torch.bool))
    safe = neuron_scores.source_statistics(tiny_model(), rows, batch_size=3, with_gradient=True)
    original = neuron_scores.GRADIENT_SCALE
    neuron_scores.GRADIENT_SCALE = 3e38  # loss (about 10) times this is inf
    try:
        retried = neuron_scores.source_statistics(tiny_model(), rows, batch_size=3, with_gradient=True)
    finally:
        neuron_scores.GRADIENT_SCALE = original
    for name in ("effect", "abs_effect", "positive", "abs"):
        torch.testing.assert_close(retried[name], safe[name], rtol=1e-4, atol=1e-9)


def test_weight_gradients_survive_an_overflowing_scale() -> None:
    """weight_scores.gate_gradients gives the same mean gradients and input norms after an overflow retry."""
    ids = torch.randint(40, (3, 12))
    rows = sources.Rows(ids, torch.ones_like(ids, dtype=torch.bool))
    safe = weight_scores.gate_gradients(tiny_model(), rows, batch_size=3)
    original = weight_scores.GRADIENT_SCALE
    weight_scores.GRADIENT_SCALE = 3e38
    try:
        retried = weight_scores.gate_gradients(tiny_model(), rows, batch_size=3)
    finally:
        weight_scores.GRADIENT_SCALE = original
    for safe_layer, retried_layer in zip(safe[0], retried[0], strict=True):
        torch.testing.assert_close(retried_layer, safe_layer, rtol=1e-4, atol=1e-9)
    torch.testing.assert_close(retried[1], safe[1])


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"PASS {name}")

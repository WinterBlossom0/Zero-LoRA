"""CPU checks of the language-deletion training: the chunked loss against full logits, and the loop end to end.

Run from the project root: CUDA_VISIBLE_DEVICES="" PYTHONPATH=src torch-python tests/test_train_language.py
"""

import argparse

import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

import corruption_data
import language_data
import models
import sources
import train_language
import zero_lora


def tiny_model(vocab_size: int) -> LlamaForCausalLM:
    """Return a 2-layer Llama with frozen weights and gradient checkpointing."""
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=vocab_size, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=1100)
    model = LlamaForCausalLM(config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.requires_grad_(False)
    return model


def test_window_nll_matches_full_logits() -> None:
    """Chunked NLL with and without gradients equals the per-row mean cross-entropy of the full logits, also when a
    score mask selects the scored tokens."""
    model = tiny_model(vocab_size=50).eval()
    input_ids = torch.randint(50, (3, 700))  # 3 x 699 scored positions span several 512-position chunks
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, 400:] = 0  # right padding
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.detach()
    expected = torch.stack([torch.nn.functional.cross_entropy(logits[row, :-1][attention_mask[row, 1:].bool()],
                                                              input_ids[row, 1:][attention_mask[row, 1:].bool()])
                            for row in range(3)])
    with torch.no_grad():
        torch.testing.assert_close(train_language.window_nll(model, input_ids, attention_mask), expected)
    torch.testing.assert_close(train_language.window_nll(model, input_ids, attention_mask).detach(), expected)
    score_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    score_mask[:, 650:] = True  # only the last tokens are scored, as in the corruption rows of setting A
    scored = attention_mask[:, 1:].bool() & score_mask[:, 1:]
    expected_scored = torch.stack([torch.nn.functional.cross_entropy(logits[row, :-1][scored[row]],
                                                                     input_ids[row, 1:][scored[row]])
                                   for row in (0, 2)])
    with torch.no_grad():
        torch.testing.assert_close(train_language.window_nll(model, input_ids, attention_mask, score_mask)[[0, 2]],
                                   expected_scored)


def test_window_kl_matches_full_logits() -> None:
    """window_kl equals the per-row mean KL(base || masked) of full logits, and is 0 when no gate deletes."""
    model = tiny_model(vocab_size=50).eval()
    zero_lora.convert_to_zero_lora(model=model, target_modules=["gate_proj"], rank=1, alpha=1.0,
                                   parametrization="hardconcrete")
    input_ids = torch.randint(50, (2, 600))  # 2 x 599 positions span several 256-position chunks
    with torch.no_grad():
        torch.testing.assert_close(train_language.window_kl(model, input_ids), torch.zeros(2))
        for _, layer in zero_lora.zero_lora_layers(model):
            layer.log_alpha.normal_(std=3.0)
        with zero_lora.bypassed(model):
            base_logits = model(input_ids=input_ids).logits[:, :-1]
        logits = model(input_ids=input_ids).logits[:, :-1]
        expected = (base_logits.softmax(-1) * (base_logits.log_softmax(-1) - logits.log_softmax(-1))).sum(-1).mean(1)
        torch.testing.assert_close(train_language.window_kl(model, input_ids), expected)
        assert expected.min() > 0, "test needs a masked model that differs from the base"


def test_training_loop_on_real_windows() -> None:
    """The loop runs with every method of the experiments: (parametrization, objective)."""
    for parametrization, objective, neumuter_keep in [
            ("mask", "hinge", "kl"), ("neuron", "hinge", "kl"), ("full", "hinge", "kl"), ("hardconcrete", "hinge", "kl"),
            ("neumuter", "neumuter", "kl"), ("neumuter", "neumuter", "nll"), ("hardconcrete", "sufficiency", "kl")]:
        check_training_loop(parametrization, objective, neumuter_keep)


def all_scored(windows: torch.Tensor) -> sources.Rows:
    """Return windows as rows whose tokens are all scored."""
    return sources.Rows(windows, torch.ones_like(windows, dtype=torch.bool))


def check_training_loop(parametrization: str, objective: str, neumuter_keep: str) -> None:
    """The loop trains B, validates at the set steps, keeps the best snapped step, records every loss, and holds
    the learning rate until the first deletion."""
    tokenizer = AutoTokenizer.from_pretrained(models.MODEL_NAME)
    end_id = tokenizer.convert_tokens_to_ids(models.END_TOKEN)
    model = tiny_model(vocab_size=len(tokenizer))
    zero_lora.convert_to_zero_lora(model=model, target_modules=models.TARGET_MODULES, rank=2, alpha=2.0,
                                   parametrization=parametrization)
    languages = ["zh_hans", "en", "ja"]
    data = {split: {language: all_scored(language_data.windows(tokenizer, language_data.load_texts(language, split)[:2],
                                                               32, end_id)[:6])
                    for language in languages}
            for split in ("train", "validation")}
    for split, parts in data.items():  # a target that scores only some tokens: the corruption of setting A
        parts[corruption_data.SOURCE] = sources.load_rows(tokenizer, corruption_data.SOURCE, split, 32, end_id)[:6]
    assert all(len(windows) == 6 for parts in data.values() for windows in parts.values()), "too little real text"
    args = argparse.Namespace(targets=["zh_hans", corruption_data.SOURCE], keep=["en", "ja"], objective=objective,
                              keep_kl_weight=1.0, neumuter_keep=neumuter_keep, open_weight=1e-3, learning_rate=1e-2, double_well_weight=0.01,
                              shrink_weight=0.0, neuron_weight=1e-3, neuron_tau=1e-4, neuron_columns=8,
                              double_well_samples=256, rows_per_step=2, steps=6, eval_every=3, 
                              warmup_fraction=0.34, row_commit_weight=0.1, max_grad_norm=1.0, eval_batch_size=4, train_seed=0)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
              if parameter.requires_grad}
    base = train_language.evaluate(model, data["validation"], batch_size=4, snap=True)
    forget_levels = {target: base[target] + 1.0 for target in args.targets}
    history, best_step = train_language.train_loop(args, model, data, forget_levels, torch.device("cpu"))
    trained = dict(model.named_parameters())
    assert before and all(not torch.equal(before[name], trained[name]) for name in before), "nothing trained"
    if objective == "sufficiency":
        assert best_step == 6 and len(history) == 6 and all("open_count" in record for record in history)
        print(parametrization, objective, "open count:", [round(record["open_count"], 2) for record in history])
        return
    assert [record["step"] for record in history if "validation_snapped" in record] == [3, 6]
    best = min((record for record in history if "validation_snapped" in record),
               key=lambda record: record["validation_snapped"]["task_loss"])
    assert best_step == best["step"]
    restored = train_language.evaluate(model, data["validation"], batch_size=4, snap=True)
    restored_task = train_language.task_losses(restored, args.targets, args.keep, forget_levels)
    assert abs(restored_task["task_loss"] - best["validation_snapped"]["task_loss"]) < 1e-6
    validations = [record for record in history if "validation_snapped" in record]
    rates = [record["learning_rate"] for record in history]  # 6 steps, warm-up of 2: 1/2, 1, then cosine to 0
    assert abs(rates[0] - 5e-3) < 1e-9 and abs(rates[1] - 1e-2) < 1e-9 and rates[-1] < rates[3] < rates[2] <= 1e-2, rates
    print(parametrization, objective, "target NLL:", [round(record["target_nll"]["zh_hans"], 3) for record in history],
          "deleted:", [record["zero_after_snap"] for record in validations], "best step", best_step)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"PASS {name}")

# Zero LoRA: hypotheses

Zero LoRA is a delete-only adapter. It learns a low-rank mask s = clamp(BA, 0, 1) over frozen weights; the deployed
model keeps a weight where s < 0.5 and sets it to exactly zero elsewhere. A penalty concentrates the deletions in few
neurons. The goal is localization: find the small set of weights that carries a behaviour.

## H1. Efficiency

A low-rank deletion mask finds the same weights as a full per-weight mask (one trainable value per weight), with
about 1/400 of the trainable parameters.

- Test: planted ground truth (a corruption trained into known neurons and sub-neuron weight sets, wiped to zero
  first, so that only their deletion restores English) and natural targets (language deletion), same objective
  and budget for both masks.
- Rejected if: the low-rank mask recovers clearly fewer planted weights, or removes the target less selectively,
  than the per-weight mask.

## H2. Granularity

At an equal number of touched neurons, deleting weights inside few neurons removes a target more selectively than
switching off whole neurons.

- Test: target damage at a fixed collateral damage (0.05 nats mean NLL rise on other languages) for Zero LoRA vs
  whole-neuron methods (our neuron switch, NeuMuter-style and Hard Concrete masks, selective pruning, attribution
  patching, LAPE), plus recovery of a planted sub-neuron pattern.
- Rejected if: whole-neuron methods reach equal or higher target damage at the same collateral and neuron count.

## H3. Resolution

The method separates closely related targets that share neurons: Simplified vs Traditional Chinese, and Chinese vs
Japanese.

- Test: forget one, keep the other; measure the damage on each.
- Rejected if: the kept target is damaged as much as the forgotten one.

## Setup

Models: Qwen3.5-2B-Base (main), Llama-3.2-1B (replication). Evaluation: FLORES-200 devtest, held-out Wikipedia,
WikiText forgetting, BBQ. Three seeds for the main runs; bootstrap confidence intervals.

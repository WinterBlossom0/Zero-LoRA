# Experiment design: Zero LoRA

## 1. Research questions

Zero LoRA learns a delete-only mask over frozen weights. The mask is low-rank: s = clamp(B A, 0, 1), and the
deployed model keeps W where s < 0.5 and deletes it elsewhere. A penalty concentrates deletions in few neurons.

- RQ1 Efficiency. Does a low-rank deletion mask localize as accurately as a full per-weight mask, at a
  fraction of the trainable parameters and memory?
- RQ2 Granularity. At an equal budget of touched neurons, does weight-level deletion remove a target more
  selectively than whole-neuron deletion?
- RQ3 Resolution. Can the method separate closely related targets: Simplified vs Traditional Chinese, and
  Chinese vs Japanese?

Prior work these questions build on (notes in docs/literature.md): per-weight deletion masks with
forget/keep losses (Bayazit et al. 2024); learned neuron masks for unlearning (NeuMuter, Hou et al. 2025);
localization benchmarks (Chang et al. 2024); language neurons (Tang et al. 2024).

## 2. Methods

All learned methods train on the same windows, for the same steps, with the same forget and keep objective
(forget hinge on target NLL, keep NLL on keep languages), the same checkpoint rule and the same evaluation.

| Group | Method | Unit | Role |
|---|---|---|---|
| Ours | Zero LoRA (rank 4) | weights, low-rank | proposed |
| Reference | Per-weight mask (one value per weight; same clamp, snap and losses as ours, so only the rank differs) | weights, full | RQ1 accuracy/cost reference |
| Neuron masks | Neuron switch (our objective, one gate per neuron) | neurons | RQ2 within-objective control |
| Neuron masks | NeuMuter-style (hard concrete beta 0.3, -target NLL + 500 x keep KL + 5 x expected L0; paper values) | neurons | published learned baseline |
| Neuron masks | Hard Concrete (Chang et al. 2024) | neurons | published learned baseline |
| Scoring | Selective Pruning (activation ratio target/keep) | neurons | published one-shot baseline |
| Scoring | Attribution patching (activation x gradient, target minus keep, keep term signed) | neurons | gradient baseline |
| Scoring | LAPE | neurons | language-neuron baseline |
| Scoring | Taylor contrast (w x gradient, target minus keep) and Wanda-style contrast (abs(w) x input norm) | weights | one-shot weight-level baselines for RQ1 |
| Check | Weight outlier score (data-free z-score of each neuron's weight norms) | neurons | setting A: a plant must not stand out by its weights |
| Control | Random neurons / random weights, same layers and count | both | floor |

Implementations are written from the papers' equations. No code from Deletion Tuning is used.
Stated adaptations of NeuMuter, applied equally to all learned arms: the keep text is real text of the other
languages (not BERT-perturbed neighbours of the target), the final model deletes the gates below 0.5 (not scales
them), and the step count equals that of the other arms. In setting A its keep term is the keep NLL (as the hinge
objective): a KL to the unmasked model would there be a KL to the planted model and penalize the recovery of task A.
The per-weight reference also runs at learning rates 3e-2 and 3e-1 (its paper trains at a much higher rate than
ours), so that a tie with Zero LoRA is not a tie with an under-trained reference.

## 3. Setting A: planted ground truth (answers RQ1, RQ2 with known answers)

Design (task A / task B). Task A: the model's English (held-out English Wikipedia windows, NLL). Task B: a
corruption, one fixed phrase ("I am the danger.") after any English text.
1. Rank MLP neurons and gate_proj weights by task-A importance on English training windows: of a neuron, the mean
   over tokens of |h dL/dh| (first-order NLL change of switching it off); of a weight, |w dL/dw| (Taylor criterion,
   Molchanov et al. 2017).
2. Choose the positions where task A does not depend on them, and set them to zero (wipe). Check: the wiped model
   keeps task A (English NLL change near 0).
3. Train only the wiped positions on task B until the phrase NLL after held-out English text is below 0.05. The
   planted model now does B instead of A.
4. Each method gets the planted model, the task-B text as the forget target and English text as the keep set, and
   must recover task A by deletion. A deletion cannot add an ability, and the wiped positions held nothing that
   task A needs, so deleting exactly the planted positions gives back the wiped model: a unique answer key. Wiping
   first means the planted positions hold B alone, not B in superposition with their old function.
Plant types (size 0.1 % of MLP neurons, 147, Chang et al. 2024's lower injection ratio; 2 plant seeds each):
- Neuron plant: 147 whole neurons (gate_proj row, up_proj row, down_proj column) drawn from the 10 % of neurons
  with the lowest task-A importance; restarted from small random values (an all-zero neuron gets no gradient).
- Sub-neuron plant: in 147 rows drawn from the 10 % of neurons with the highest task-A importance, the 25 % of
  gate_proj input weights with the lowest task-A importance. Switching such a neuron off costs task A, so only a
  weight-level deletion removes B alone (RQ2).
- Scattered plant: 20,000 gate_proj weights drawn from the 10 % lowest-importance weights of each layer (a
  full-rank pattern; tests the limit of the low-rank mask, RQ1).
Metrics: task-A recovery (fraction of the planted English NLL rise that the deletion removes, against the wiped
model); precision and recall of the planted neurons (neuron plant) or weights (other plants); deleted weights and
touched neurons; NLL change on untrained FLORES languages as a side measure.
Run settings (code: experiments/setting_a.py): every learned method trains 1000 steps, validated every 50.
Sparsity price, 3 values per method: mask and per-weight 1e-4/1e-3/1e-2 (log-sum count, tau 1e-4); neuron switch
1e-3/1e-2/1e-1 (linear count); NeuMuter 0.5/5/50 (paper: 5). Train seed 0 for every price; plant seed 0 adds train
seeds 1 and 2 at the middle price. Rankings delete as many neurons as the plant touches. Each plant runs all its
methods on one machine. Order: neuron plants first (RQ1 with all methods), then sub-neuron and scattered.
Rejected designs (measured): (1) the whole cipher language in 16-64 neurons gained only 0.09 nats in 500 steps;
(2) one memorized ciphered passage (Chang's INJ) planted fast, but Zero LoRA erased it while touching mostly other
neurons: a memorized string breaks under many deletions, so it is no unique answer key; (3) a corruption planted
on top of existing weights in random neurons leaked into other languages (French +0.43 nats) and left the planted
weights in superposition with their old function.

## 4. Setting B: natural behaviours (answers RQ1-RQ3)

- B1 Language deletion. Targets: Chinese (both scripts), Japanese, Russian. Keep: the other training
  languages. Evaluation on FLORES-200 devtest (1,012 sentences per language, never trained), held-out
  Wikipedia, a panel of untrained languages (yue, ko, de, ar, hi), and WikiText forgetting.
- B2 Script and language separation (RQ3). Forget Traditional keep Simplified; forget Chinese keep Japanese.

Primary metric: target damage at fixed collateral, i.e. the target NLL rise when the mean collateral NLL
rise is 0.05 nats, read from each method's curve over budget (scoring methods: top-k; learned methods:
sweep of the sparsity price). Secondary: full curves, neuron and weight counts, WikiText forgetting.

## 5. Models

- Qwen3.5-2B-Base: all experiments.
- Llama-3.2-1B base: replication of Setting A (neuron and sub-neuron plants) and B1 Chinese, with Zero LoRA,
  per-weight mask, NeuMuter-style and Selective Pruning.

## 6. Efficiency (RQ1)

Trainable parameters, peak GPU memory and time per step for Zero LoRA, the per-weight mask and the neuron
masks, on gate_proj only and on all MLP matrices, for both models.

## 7. Rigor

- Precision: fp16 with a dynamic loss scaler for every run, so all compared methods share one precision.

- Setting A: 5 plants per type (the plant-to-plant variance dominates); plant 0 adds train seeds 1-2 and the
  price sweep; paired tests over plants. Setting B: 3 train seeds for the main runs; 1 seed elsewhere.
- H1 is an equivalence claim: two one-sided tests (TOST) with a margin fixed before the runs (recall within 0.05).
- Operating points of every method are chosen on validation data, then read on test data.
- Bootstrap 95% confidence intervals over FLORES sentences; paired tests with Bonferroni correction.
- Matched budgets: every comparison is at equal touched neurons (RQ2) or equal deleted weights (RQ1).
- Ablations (Chinese, 1 seed): rank 1/4/16; concentration penalty off; keep loss off; Gumbel instead of the
  deterministic clamp.
- Relearning check: 50 fine-tuning steps on the target after deletion; recovery speed vs base.

## 8. Outputs

- Table 1: planted recovery (precision/recall) per method and plant type.
- Table 2: target damage at 0.05 nats collateral per method and target, with CIs.
- Figure 1: damage-vs-collateral curves (Chinese).
- Figure 2: separation matrices (RQ3).
- Table 3: efficiency.
- Table 4: Llama replication. Table 5: ablations.

## 9. Code map

| Piece | File |
|---|---|
| Mask layers: ours (mask), neuron switch (neuron), per-weight (full), hard concrete, NeuMuter gate | src/zero_lora.py |
| Training objectives: hinge (ours), neumuter, sufficiency (Chang) | src/train_language.py |
| Neuron rankings: selective pruning, attribution patching, LAPE, outlier, random | experiments/neuron_scores.py |
| Weight rankings: Taylor and Wanda contrasts | experiments/weight_scores.py |
| Deletion curves and the damage-at-collateral metric | experiments/evaluate_deletions.py, src/curves.py |
| Planted corruption and planting | src/corruption_data.py, src/sources.py, src/planting.py, experiments/plant.py, experiments/setting_a.py, experiments/recovery.py |
| Precision and model choice | src/precision.py (ZL_DTYPE), src/models.py (ZL_MODEL) |
| Job runner of setting A | experiments/setting_a.py, experiments/run_guarded.sh |

## 10. Execution

Order, each stage gated on the previous one:
1. Build: baselines, planting script, corruption data, Russian data, job runner, result tables. CPU tests for
   each component; code review of all code before any GPU run.
2. Smoke: the local GPU reproduces the reference base NLL within 0.01 nats in the run precision.
3. Setting A on Qwen. Gate: if Zero LoRA does not match the per-weight mask on the neuron and sub-neuron plants,
   RQ1 is answered "no" and reported as such.
4. Setting B on Qwen (Chinese with 3 seeds first, then Japanese, Russian, separation).
5. Llama replication, efficiency, ablations, relearning.

Compute: all runs on the local GPU (one RTX 5080 laptop GPU, 16 GB), one job at a time.

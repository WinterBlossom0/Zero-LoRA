# Literature notes: Zero LoRA

Each entry: paper, what it does, numbers we use, and what this project uses from it.
Read status: "full" means the whole text, appendices included, from the arXiv PDF in data/papers/<id>/.

## Chang, Thomason, Jia 2024. Do Localization Methods Actually Localize Memorized Data in LLMs? (NAACL 2024, arXiv 2311.09060). Read: full.

- Goal: evaluate localization methods directly. Unit: FFN neurons h^l (the intermediate activation), dropout = zero ablation at test time.
- Methods (all score each neuron, then pick top-k% PER LAYER; per-layer beats global ranking, A.11, because all methods put large scores in bottom layers):
  - Zero-Out: loss change when one neuron is dropped (8.5 h per sequence on Pythia-6.9B, A6000).
  - Activations: mean over suffix of |h_i| * ||v_i|| (0.3 s).
  - IG: integrated gradients from zero to h, 20 Riemann steps (43 min).
  - Slimming: learned mask m in R^{d2} on h, init 1, clipped to [0,1], loss = target memorization loss + lambda * L1(m). Frozen LLM. Score = m (12 s).
  - Hard Concrete: Louizos 2018 L0 mask, stretch (-0.1, 1.1), loss = target loss + lambda * expected L0. Score = sigmoid(log m) (1 min).
  - Slimming and Hard Concrete use ONLY the target loss. No keep/negative term. Hyperparameters tuned on 5 dev sequences; both "sensitive to the choice of hyperparameters".
- INJ benchmark: for each of 156 ECBD-2021 definitions, fine-tune a random r% (1% or 0.1%) of FFN value vectors v_i^l across all layers until LM loss < 0.05; one model per sequence. Metric Recall@k% of the injected neurons, k per layer.
  - GPT2 124M, ratio 0.1%: HC 56.4/79.6/93.7, Slimming 58.9/83.5/94.4, Zero-Out 54.1/77.8/90.9, IG 53.5/74.1/84.8, Activations 11.1/26.5/51.5, Random 0.1/0.2/0.5 (R@0.1/0.2/0.5%).
  - Pythia-6.9B, ratio 0.1%: HC 46.4/66.3/82.3, Slimming 48.5/60.9/71.0, IG 29.3/34.4/39.6.
  - Lower injection ratio gives higher recall. Limitation stated: no guarantee every injected weight is used.
- DEL benchmark: 505 memorized Pile sequences (Pythia 2.8B, 6.9B), 105 hand-found (GPT2-XL); 32-token prefix, 48-token suffix. Drop top 0.1% or 0.5% neurons per layer (not the bottom layer). Metrics: Self Acc/Levenshtein, Neg Acc (all other sequences), Rand-PPL on 2048 Pile sequences.
  - Pythia-6.9B, 0.5%: HC Self-Acc -57.7%, Neg-Acc -14.7%; Slimming -48.7% / -11.3%; IG -32.3% / -4.9%; Random -0.2%.
  - Trade-off: pruning methods erase the target best but hurt negatives most; IG keeps negatives best.
  - Located neurons are shared across related sequences (addresses, poems/Bible/Shakespeare quotes, Zodiac and planets).
  - Memory is distributed over layers: 0.1% in each of 47 layers beats 5x more neurons in one layer. Bottom layer is general.
- Significance: one-tailed paired t-tests with Bonferroni correction.

What we use:
- INJ protocol as the ground-truth test (adapted: plant into whole MLP neurons of Qwen; recall@k).
- Slimming and Hard Concrete as the strongest baselines. Note they have no keep loss; a fair comparison must also run them WITH our keep loss, else the comparison mixes method and objective.
- Per-layer vs global top-k: report both.
- Target/negative/random-perplexity triple as the DEL-style selectivity metric.
- Compute timings (Table 6) for the efficiency comparison.

## Bayazit, Foroutan, Chen, Weiss, Bosselut 2024. Discovering Knowledge-Critical Subnetworks in Pretrained Language Models (EMNLP 2024, arXiv 2310.03084). Read: full (main text, appendices A-J; figure data skimmed).

- CLOSEST PRIOR WORK to our weight-level method. It is a delete-only weight mask with forget + keep + sparsity losses.
- Mask: per-weight logit l_i, Gumbel-sigmoid s_i (Eq. 1), straight-through threshold at 0.5 (Eq. 2, from Csordas 2021). Learned subnetwork m is REMOVED: the evaluated model is f(x, (1-m) ⊙ theta). Weights frozen. Init sigma(l)=0.45.
- Masked: all dense (attention + MLP) weights of the top 50% of layers; not embeddings, LM head, layer norm, biases. Neuron variant: one mask per input neuron of a linear layer.
- Losses (Eq. 3-6): suppression = KL(uniform || remaining model) on target triplets; maintenance = KL(original || remaining) on ControlKG and on WikiText-2 (two terms); sparsity = mean sigmoid(l). lambda1=1.5, lambda2=lambda3=1, lambda4 from 2, rising linearly to 3 after 50% of training. AdamW lr 0.2, 10% warmup. 40k steps GPT2-small (1 GPU), 50k medium/large (3 GPUs), 60k XL (3 A100s).
- Checkpoint selection: iterative thresholds on target dPPL floor and maintenance dPPL ceilings (Table 11).
- Data: WordNet hypernym and ConceptNet LAMA target KGs (3-hop random walks, single-token tails, 11-40 triplets each); ControlKG (thousands of triplets); WikiText-2.
- Results GPT2-small (Table 2, 3 seeds): weight masking sparsity 98.6% (1.4% of masked weights removed), WordNet target dPPL +590.9 (base 162.4), ControlKG -0.2, ControlLM +0.5. Neuron masking: sparsity 95.3%, target +715.9 but ControlKG +22.2, LM +4.1 (worse maintenance than random neurons for KG). ConceptNet neuron masking: target +22422 but ControlKG +71.9.
- Claim (Sec. 3, 6): neurons are "too broad... typically polysemantic"; weight masking separates knowledge better; neuron-masked failure linked to superposition. Appendix H: no full row or column of a weight matrix was removed by weight masking.
- Ablation (Table 3): without suppression nothing is removed; without maintenance-KG the ControlKG dPPL is +1697. Sparsity term is required (search stagnates without it).
- Seeds: weight subnetworks for the same KG overlap only 3-4% IoU; neuron subnetworks 34-44%. Union of seeds suppresses 6x more but hurts maintenance like random.
- Downstream: removing the subnetwork lowers CommonsenseQA accuracy on the related questions (-14.4 points with head tuning), much less with LoRA or full fine-tuning.
- Limitations stated: English only, GPT-2 family only (max 1.5B), hyperparameter search not exhaustive.

What this means for us:
- Our weight-level Zero LoRA (forget hinge + keep NLL + delete-only) is the same problem setting as this paper. Differences that remain ours: (1) low-rank parametrization of the deletion mask (rank-r B@A, about 2*(out+in)*r parameters instead of one logit per weight; this is the "LoRA cost" claim); (2) a deterministic clamp with snap instead of Gumbel noise; (3) a neuron-count penalty that concentrates deletions in few neurons; (4) an explicit whole-neuron mode; (5) modern 2B model, multilingual target. The novelty claim must be limited to these, and each needs an ablation against the per-weight-logit mask of this paper.
- Their finding "neuron masking hurts maintenance, weight masking does not" matches our Chinese result (partial rows selective, whole rows hit Japanese). This is a replication, not a new finding.
- Use their metric triple (target dPPL, control dPPL, LM dPPL), their random-at-same-sparsity-per-module baseline, and their seed-overlap (IoU) analysis.
- Their cost: 40k-60k steps on A100s. Our efficiency claim must compare steps, memory and trainable parameters against a per-weight mask on the same model.

## Csordas, van Steenkiste, Schmidhuber 2021. Are Neural Nets Modular? Inspecting Functional Modularity Through Differentiable Weight Masks (ICLR 2021, arXiv 2010.02066). Read: full prose incl. appendices A-C (pdftotext table/figure numbers not read cell by cell).

- Per-weight binary mask on a frozen trained network: logit l_i (init keep prob 0.9), Gumbel-sigmoid sample with tau=1, straight-through threshold at 0.5; final M_i = 1[sigmoid(l_i) > 0.5]. 4-8 mask samples per batch.
- The mask is a KEEP mask trained on the target subtask only, with regularizer alpha * sum(l_i) pushing weights out. It finds the weights NEEDED for a function (not the weights to delete). Removal analysis: train mask on A, test on B; or invert masks (limited to disjoint weights).
- Overlap metrics: IoU and IoMin ("subsetness").
- alpha choice heuristic (C.3): raise alpha until full-task performance drops, then back off to >95% of original. Conclusions depend critically on alpha; underfitting networks share more weights.
- Scale: small FNNs (5x2000), LSTMs (256), 3-layer Transformers (d=100-256), simple CNN and ResNet-110 on CIFAR-10. Not applied to pretrained LLMs.
- Findings: NNs specialize weights (P_specialize) but do not reuse modules (P_reuse); SCAN/Mathematics show sample-specific weights; removing class-exclusive weights in CNNs drops that class 40-60% (ResNet near 100%) with small effect on other classes.
- Stability (B.1): IoU of masks across mask seeds 93% on CIFAR pairs. Masking does not change the computation much (B.2).

What we use:
- Origin of the Gumbel-sigmoid + straight-through weight mask used by Bayazit 2024; cite as the method family. Our clamp-and-snap is deterministic (no Gumbel noise), which is a design difference to test, not a proven advantage.
- IoU/IoMin for overlap between neuron sets (e.g. our neurons vs LAPE, across seeds, zh_hans vs zh_hant).
- Warning for the paper: the regularization strength decides the conclusions; report a sweep.

## Tang, Luo, Huang, Zhang, Wang, Zhao, Wei, Wen 2024. Language-Specific Neurons: The Key to Multilingual Capabilities in LLMs (ACL 2024, arXiv 2402.16438). Read: full.

- Neuron = one column of W1 (gate) + activation; "activated" if act_fn(gate)_j > 0 (Sec. 2.1). For GLU models the gate activation is used.
- LAPE (Eq. 4-5): p^k_j = P(act > 0 | language k), L1-normalize over languages, entropy. Select the lowest 1% LAPE, then keep a neuron for language k only if p^k_j > threshold = 95th percentile of all activation probabilities (0.515 for LLaMA-2 70B). A neuron can belong to several languages.
- Data: 100M Wikipedia tokens per language for identification; evaluation PPL on 1M tokens per language from after Sept 2022; open generation on a GPT-4-translated Vicuna set judged by GPT-4.
- Languages: en, zh (Simplified), fr, es, vi, id, ja. Models: LLaMA-2 7B/13B/70B, BLOOM 7.1B, also Chat, OPT, Mistral, Phi-2.
- Intervention: set the activation of the selected neurons to zero (deactivation) - same as our zeroed gate row.
- Baselines in the paper: LAP (activation prob > 95%), LAVE (activation value entropy), PV (parameter variation after monolingual instruction tuning), random. Each picks 1% of neurons/params.
- Results (Fig. 2/3): LLaMA-2 7B, deactivating zh neurons raises zh PPL by 0.58 and ja by 0.33 (PPL points, not nats); en neurons +0.01. zh/ja share about 25% of their neurons ("partially share common characters"). Deactivating Simplified-Chinese neurons in LLaMA-2 70B makes it output Traditional characters and English (Table 2).
- Layer distribution: U shape, most language neurons in bottom and top layers.
- Larger models: neurons more single-language.
- Limitation stated: LAPE is relative to the set of languages; no absolute threshold for one language.

What we use:
- Exact LAPE definition (our experiments/language_baselines.py follows it).
- Their own observation that zh-deactivation leaves Traditional output: Simplified and Traditional need separate handling (we found the same in reverse).
- Fair-comparison requirement: the paper uses 100M tokens per language; our baseline used 256k. The design must give LAPE the same text budget as our method and show LAPE stability vs token count.
- Their metric is PPL change; we report NLL change (nats); give both.
- The zh/ja overlap (25%) supports a Chinese-vs-Japanese separation test.

## Kojima, Okimura, Iwasawa, Yanaka, Matsuo 2024. On the Multilingual Ability of Decoder-based PLMs: Finding and Controlling Language-Specific Neurons (NAACL 2024, arXiv 2404.02431). Read: full (appendix tables read as numbers only).

- Method (from Cuadros et al. 2022, self-conditioning): per text, average each neuron's output over tokens (mean, not max; pad excluded). Treat the per-text value as a score for the label "text is in target language" and compute average precision AP_m. Language-specific = top-k AND bottom-k AP neurons, k=1000 each. Embedding and output layers excluded. "Neuron" here = any intermediate output in the model (Table 1: 221k neurons for XGLM-564M), not only MLP gates.
- Data: 500 texts per language, 1:1 from PAWS-X dev and FLORES-200 test (FLORES devtest used for translation evaluation). 6 languages: en, de, fr, es, zh, ja.
- Intervention: REPLACE the neuron outputs with the median activation on target-language texts (forcing ON), not deletion. Goal: make the model generate the target language. Metrics: FastText language ID accuracy (threshold 0.5) and BLEU.
- Results: unconditional generation target-language rate rises (XGLM avg 9.3% -> 88.3% with both groups); Llama2-7b avg 14.5% -> 35.7%. Overlap between languages' neuron sets < 5% (XGLM-564M). Neurons in first and last layers. Too many neurons (beyond 10^3-10^4) collapse text.
- Llama-2-7b has higher zh/ja overlap than other pairs (shared characters), but not in all models.
- Cost (App.): about 0.5 GPU-h per run for small models, 60 runs per model.

What we use:
- AP scoring as a second statistics-based baseline (cheap: forward passes only). Our variant must use the deletion intervention (zero gate) for a like-for-like test, and we can additionally report their median-forcing.
- FLORES-200 as the evaluation set (they also use it), so our FLORES numbers are comparable in kind.
- Note: their "neuron" definition differs from ours (MLP gate only). We restrict all baselines to gate_proj neurons for a fair unit.

## Mueller, Geiger, Wiegreffe et al. 2025. MIB: A Mechanistic Interpretability Benchmark (ICML 2025, arXiv 2504.13151). Read: full prose incl. appendices (tables as numbers only).

- Two tracks. Circuit localization: find the subgraph (nodes/edges) that explains a task; causal variable localization: featurize a hidden vector and align features to a causal variable (DAS, DBM masks, PCA, SAE).
- Tasks: IOI, two-digit arithmetic, MCQA (colors), ARC Easy/Challenge, plus RAVEL in the variable track. Models: GPT-2 small, Qwen-2.5 0.5B, Gemma-2 2B, Llama-3.1 8B; plus an InterpBench model (6 layers, trained 70 h on an H100) with a KNOWN ground-truth IOI circuit, scored by AUROC over edges.
- Interventions are counterfactual (activation patching from a paired input), mean, or optimal ablations, NOT zero ablation.
- Metrics: faithfulness f(C) = (m(C) - m(empty)) / (m(N) - m(empty)) with m = logit difference; evaluated at circuit sizes k in {0.1%, 0.2%, 0.5%, 1%, 2%, 5%, 10%, 20%, 50%, 100%} of edges; CPR = area under f (higher better), CMD = area between f and 1 (lower better). Node (neuron) circuits are charged by weighted edge count, which makes neuron-level circuits look large.
- Results: EAP-IG-inputs (edge attribution patching with integrated gradients on inputs, 5 steps) best on average; UGS mask optimization (Li & Janson 2024, lambda=1e-3) competitive on CMD but memory-heavy (does not fit 80 GB for larger models); node-level (NAP) circuits do poorly; random far worse. In the variable track DAS best; SAE features not better than plain neurons (DBM on neurons is a strong baseline).
- Leaderboard on Hugging Face with public and private test sets.

What we use:
- The "performance vs size curve, then area" idea: report target damage and keep damage as curves over neuron count k, and summarize by area; this removes the dependence on one threshold (our neuron price).
- Ground-truth model idea (InterpBench) supports our planted-neuron test.
- Not a direct fit for our main claim: MIB measures circuits with counterfactual ablation and charges neuron nodes by edges; our method deletes (zero ablation). A MIB submission is optional and would need edge-level masks. Do not make it a core experiment.

## Lu, Yang, Lasby, Pote, Ioannou, Ren 2026. The Surprising Effectiveness of Deleting Weights in LLM Post-Training ("Deletion Tuning", arXiv 2611.10866). Read: full paper text incl. appendices A-H. Inspiration only: we never use, download or re-implement its code or masks.

- Method: one trainable logit G per weight of gate_proj in every block (frozen W). Training forward W * sigmoid(alpha(t) G), alpha ramps linearly 1e3 -> 1e4 (GRPO) or 1e2 -> 1e3 (SFT); deploy hard mask 1[G>0]. Gradients flow through the deterministic soft mask (no Gumbel noise, no straight-through). Init from Wanda scores (|W| * ||X_j||), N(+-0.03, 0.01) depending on the eta=1/2048 quantile. Extra loss -lambda ||W~||^2 with lambda=1e-5 (MaskLLM regularizer). No fixed sparsity.
- Purpose: post-training to GAIN capability (GRPO on GSM8K / SimpleRL-Zoo, SFT on MMLU-Pro/BBEH, SDFT on ToolAlpaca/Science). Not interpretability, not forgetting a target.
- Results: deletes < 0.05% of all weights; matches FFT at Llama8B (83.55 vs 83.24 GSM8K), Qwen7B, Qwen14B, Qwen3-30B-A3B; adapter 0.5-8.4 MB (zstd of the bit mask, ~0.03 bits/weight). Forgets less: forward KL <= 0.05 and prior-task accuracy within 1 point of base; LoRA/MTH lose up to 5.4 points. Wanda prior alone gives no gain (Table 4).
- Limitation stated: one logit per masked weight uses more training memory than LoRA. They suggest interpretability (comparing which weights tasks delete) as future work.

What this means for us:
- Same "delete-only gate_proj adapter" family. Our differences: (1) low-rank factorized deletion mask (memory ~ LoRA, not one logit per weight; this answers their stated limitation); (2) a goal of LOCALIZATION (forget a target behaviour while keeping others) with a forget hinge and keep loss; (3) neuron-level concentration and the whole-neuron mode; (4) evaluation by isolation metrics against localization baselines.
- Their per-weight sigmoid-annealed mask is a natural ablation arm ("full-rank logit mask"), but we must write our own implementation from the equations in the paper, never their code. Equivalent arm already in the literature: Bayazit 2024 per-weight Gumbel mask.
- Their forgetting evidence (low forward KL) supports our WikiText forgetting check.

## Lin, He, Xing, Cui, Ding, Mukherjee, Liu, Xiang 2026. Crafting Reversible SFT Behaviors in LLMs (arXiv 2605.06632). Read: full incl. appendices A-F.

- Goal: compress an SFT-induced behaviour (fixed "I don't know", safety refusal, Shakespeare style) into a sparse "carrier" inside the fine-tuning DELTA, then suppress it at inference with a soft-prompt trigger (SFT-Eraser), weights fixed.
- Parametrization: W = W_base + M ⊙ ΔW, with M rank-1 structured: M_jk = m_row,j * m_col,k (row and column gate vectors, 8 gate groups per layer). Gates binary via Heaviside, STE gradient sigma'(theta); sparsity loss sum sigmoid(theta). LCDD = constrained objective min sparsity s.t. task loss <= budget, adaptive multiplier; ΔW is trained JOINTLY with the mask (mask-only reaches 28-34% sparsity vs 81% joint, App. E).
- Models: Qwen3-0.6B, DeepSeek-R1-Distill-Llama-8B, Mistral-7B, Vicuna-7B. Sparsity 30-84% of delta gates.
- Not delete-only on pretrained weights: it masks the fine-tuning delta and also changes ΔW. Not a localization method for a behaviour of the base model.

What this means for us:
- Closest "low-rank structured mask" idea: rank-1 row x column gate product. Our mask is rank-r over base weights (s = clamp(B@A)), applied to deletion of pretrained weights, with a forget objective. Cite as related structured-mask work; the difference must be stated (delta vs base weights; construct vs locate; joint weight training vs delete-only).

## Yan, Huang, Duan, Yin, Zhao, Dai, Fu 2026. Sparse Weight Decomposition for Efficient Circuit Extraction (SWD, arXiv 2608.03913). Read: full prose incl. appendices A-K (figures as captions).

- Reparametrizes a dense weight W ≈ A B with two SPARSE factors (Double Sparse Factorization, ADMM, activation-weighted objective E||hW - hAB||^2; zero-data variant uses identity Gram). Each inner coordinate is a "bottleneck unit" with sparse read and write edges. Units are scored by positive first-order task-margin attribution and mean-ablated; metrics = held-out sufficiency and necessity vs active edges / selected units.
- Models: GPT-2 small, Qwen2.5 0.5B-3B (one middle mlp.down_proj), Qwen3.5-27B (layer 31 down_proj). Baselines: Transcoders, VPD (parameter decomposition), sparse pretraining (Gao 2025), exact SVD (NaNA) and Random-B dense controls.
- Results: matched replacement fidelity with <1% of the data of trained replacements; fewer active edges for the same sufficiency/necessity. Single-unit edit beats random-unit and rank-4 LoRA in target-vs-side-effect tradeoff.
- Not delete-only, not a forget/keep objective; a replacement basis for circuit analysis.

What this means for us:
- "Low-rank / factorized weight structure for interpretability" is an active area; our novelty cannot be "factorization for interpretability" in general.
- Useful protocol pieces: sufficiency/necessity curves vs number of units; target-vs-side-effect Pareto plot (their Fig. 11) as a format for our forget-vs-keep curve; exact-SVD and random controls.
- Qwen3.5 is already used in interpretability work (their 27B), so our Qwen3.5-2B choice is current.

## Kramar, Lieberum, Shah, Nanda 2024. AtP*: An efficient and scalable method for localizing LLM behaviour to components (arXiv 2403.00745). Read: full incl. appendices A-C.

- Attribution patching (AtP): first-order estimate of the activation-patching effect of node n: (n(x_noise) - n(x_clean))^T dL/dn at the clean input; 2 forward + 1 backward pass for all nodes. Absolute value taken inside the expectation over prompt pairs.
- Failure modes -> AtP*: (1) attention saturation (QK fix: recompute softmax for patched queries/keys); (2) cancellation of direct and indirect effects (GradDrop: zero the gradient through each layer in turn, average the absolute estimates, scale by L/(L-1); L extra backward passes).
- NeuronNodes: each MLP neuron (post-activation) per token position is a node. For NeuronNodes on distributions they recommend plain AtP.
- Evaluation: Pythia 410M-12B; cost (forward passes) to verify the true top-K nodes; ground truth = exhaustive activation patching. Diagnostic: subsampling + Welch t-test upper bound on missed effects.
- Limitations stated: neurons may not be semantic units; ground truth is the total effect, averaged over the distribution.

What we use:
- AtP as a cheap gradient baseline for neuron scoring. For deletion (zero ablation) the AtP estimate for neuron j at a token is (0 - a_j) * dL/da_j = -a_j * dL/da_j; summed over target text minus over keep text gives a contrastive score. This is exactly the first-order "pull" we already measured (experiments/neuron_gradient_check.py computes -dL/db_j at b=0, which equals sum over tokens of a_j * dL/da_j for the whole-row switch). So AtP-ranked neurons are the natural one-shot baseline for our trained switch: same signal, no training.
- Verification idea: check the top neurons by real ablation (we do this by evaluating the switched-off set).

## Dai, Dong, Hao, Sui, Chang, Wei 2022. Knowledge Neurons in Pretrained Transformers (ACL 2022, arXiv 2104.08696). Read: full.

- Model: BERT-base (12 layers, 3072 FFN neurons/layer); cloze facts from PARAREL (27,738 facts, 34 relations, ~8.6 templates each).
- Attribution: integrated gradients of P(correct answer) w.r.t. the FFN intermediate activation w_i^(l), scaling it from 0 to its value, m=20 Riemann steps, at the [MASK] position. Coarse set: score > 0.2 * max; refined set: neurons shared by > p% of prompts for the fact (p tuned so 2-5 neurons per fact). Baseline: raw activation.
- Results: about 4 knowledge neurons per fact, mostly top layers. Suppressing (activation=0) lowers correct probability 29% on average (baseline 1.5%); amplifying (x2) raises it 31%. Erasing a relation by zeroing the FFN value slots of its top 20 neurons raises its perplexity +37% to +141% while other relations change +1% to +10%. 13.3 s per fact on a V100.
- Limitations stated: cloze/single-token only; interactions between neurons not studied; multilingual case left to future work.

What we use: IG attribution as a baseline (Chang 2024's IG baseline follows this paper's implementation). Erasing-a-relation by zeroing value slots is the same "switch off whole neurons" intervention as ours (zeroing gate_proj rows is equivalent to zeroing the neuron output).

## Louizos, Welling, Kingma 2018. Learning Sparse Neural Networks through L0 Regularization (ICLR 2018, arXiv 1712.01312). Read: full incl. appendices A-C.

- Gates z = min(1, max(0, s(zeta - gamma) + gamma)), s = sigmoid((log u - log(1-u) + log alpha)/beta), u ~ U(0,1): the hard concrete distribution (gamma=-0.1, zeta=1.1, beta=2/3). Expected L0 penalty = sum sigmoid(log alpha - beta log(-gamma/zeta)) (Eq. 12). Test-time gate min(1, max(0, sigmoid(log alpha)(zeta-gamma)+gamma)).
- Group sparsity (Sec. 2.4): one gate shared by all weights of a neuron -> neuron pruning; expected L0 = |g| * P(gate nonzero). Experiments use neuron sparsity (MNIST MLP/LeNet, WRN on CIFAR).
- One gate sample per minibatch.

What we use: Hard Concrete baseline (as adapted by Chang 2024) for whole-neuron selection: group gate per gate_proj row, expected-L0 penalty = number of neurons expected on. Our neuron switch with a flat per-neuron cost is the deterministic, straight-through analogue (clamp + snap instead of stochastic hard concrete). The comparison deterministic-vs-stochastic gate is a required ablation.

## Pochinkov & Schoots 2024. Dissecting Language Models: Machine Unlearning via Selective Pruning (arXiv 2403.01267). Read: full incl. appendices.

- Whole-neuron removal for unlearning: score each FFN (or attention pre-out) neuron by importance on forget set / (importance on retain set + eps); importance = mean |activation| (best), also freq>0, RMS, std. Prune top fraction (2% per step, iterative, recomputing scores), by zeroing W_in row, bias and W_out column.
- Tasks: forget Code keep Pile, forget Python keep Code, forget birds keep ImageNet; OPT, Galactica, Pythia (125M-6.9B), RoBERTa, ViT; toxicity on GPT2-Large and LLaMA-2-7B. Metric: top-1 next-token accuracy, perplexity, plotted as forget drop vs retain drop.
- Results: OPT-6.7B: 80% code accuracy drop costs 20% Pile drop; code perplexity x64 costs Pile x2. FF neurons more separable than attention neurons. Models trained with dropout (OPT, Galactica) separate better than Pythia. LLaMA-2 FF pruning is unstable (critical neurons collapse the model).
- Cost: forward passes only, 100k tokens of activations; single RTX 4090.

What this means for us: THE closest baseline for the whole-neuron mode: same objective (forget a dataset, keep a dataset) and same intervention (zero whole FFN neurons), but scoring by activation ratio instead of training. Must be a baseline ("SP"), one-shot and iterative. Their forget-vs-retain curve over pruning fraction is the format for our main plot.

## Li, Mo, ... (Wang lab) 2025. Effective Skill Unlearning through Intervention and Abstention (NAACL 2025, arXiv 2503.21730). Read: full incl. appendices.

- Training-free. Neuron Adjust: per FFN neuron, fit Gaussians to pre-activations (gate_proj output in GLU) on forget vs retain data; at inference, if a value is more likely under the forget distribution, shift it to the retain distribution (probabilistic, top beta% neurons by mean difference). Key Space Detection (KSD): hypercube mu ± alpha*sigma of the last-layer key vector on forget data; if inside, abstain ("Your query is not valid").
- Language unlearning on MLQA with Llama-3-8B: forget one of 7 languages (en, es, de, zh, vi, ar, hi). Neuron Adjust forgets en, es, hi cleanly, but forgetting de, zh, vi, ar also hurts other languages ("shared set of neuron values"). KSD separates better, but it is an output-abstention detector, not a model edit.
- Baseline Selective Pruning produced nonsense when forgetting code while keeping math on their models.

What this means for us: prior evidence that removing Chinese by neuron edits damages other languages (Llama-3-8B). Our Japanese cost (+0.18 nats) is the same phenomenon. KSD is not a weight edit; Neuron Adjust is an activation-shift edit; neither yields a deleted-weight set. Cite as related language-unlearning work; Neuron Adjust is an optional baseline (it does not delete).

## D'Angelo, Mottin, Fontana, Gabrielli, Velardi, Faralli 2026. Causal Routing for Unlearning (CRU, arXiv 2609.34475). Read: full incl. appendices A-D.

- Two phases on a frozen model: (1) localization = neurons (residual-stream coordinates of block outputs, last k=4 blocks) whose example-pooled activation has high variance over the forget set, above the p-th percentile (p=90-95); one forward pass. (2) a small router MLP per block (bottleneck 32-256, ~0.01% params) emits per-token gates in [0,1] on those coordinates; loss = mean gate on forget batches + lambda (mean gate on retain - 1)^2, defined on gate values only (no LM loss, no gradient through the base model).
- Input-conditional: gates 0.01-0.06 on forget inputs vs 0.79-0.85 on retain. TOFU and RWKU; DeepSeek 7B, Qwen2.5-7B, Llama-3.1-8B, Phi-3-mini; baselines GA, GD, NPO, SimNPO, ICU, PURGE, LUNAR. Memory 14 GiB vs 71 GiB.
- Related-work taxonomy names NeuMuter (Hou et al., ACL Findings 2025): a FIXED learned mask over ~1% of FFN neurons on frozen weights for unlearning - to be read (closest to our neuron switch).
- Random neurons in the same layers with the same gating do nothing (Figure 6a).

What this means for us: not a weight deletion (an added router, input-dependent). Different claim. Their loss-shape analysis (linear forget, quadratic retain) is relevant to our hinge/keep design. Cite as related unlearning-by-neuron-gating work.

## Li, Luo, Tan, Wang, Huang, Li, Bai 2022. Parameter-Efficient Sparsity for Large Language Models Fine-Tuning (PST, IJCAI 2022, arXiv 2205.11005). Read: full.

- Importance score per weight S = |W0 + beta U V| + alpha1 A B + alpha2 (R + C): data-free magnitude + LOW-RANK learned data-driven score A B (A Gaussian init, B zero init, rank r1=8) + row/column scores R (n x 1), C (1 x k). Binary mask M = top-v of S (cubic sparsity schedule); the weights are ALSO tuned with LoRA (U V). Trainable params (n+k)(r1+r2+1).
- Goal: compression (50-90% sparsity) during fine-tuning of BERT, RoBERTa, GPT-2 on GLUE / E2E / DART / WebNLG; 1.5% trainable params; on par with movement pruning.
- Observations: data-driven importance has low rank (weights and gradients are low-rank); masks are structured (whole rows nearly empty), structuredness matters more than low-rankness.

What this means for us: PRIOR ART for "a low-rank factor B@A parametrizes a pruning mask". Our s = clamp(B@A) is a close relative (deletion fraction directly, not a top-v score; no magnitude term; weights NOT tuned; goal localization with forget/keep). The novelty claim "LoRA-parametrized deletion mask" alone is not new; it must be "delete-only low-rank mask trained for target-vs-keep localization, with neuron concentration". PST's row/column structure observation supports our neuron mode (R is exactly a per-row score).

## Svirsky, Refael, Lindenbaum 2026. Train Less, Infer Faster: Efficient Model Finetuning and Compression via Structured Sparsity (FineGates, arXiv 2602.09169). Read: full incl. supplementary proofs and appendices.

- W(omega_r, omega_c) = Diag(omega_r) W0 Diag(omega_c): learned row and column gates on frozen weights (q,k,v,o and MLP matrices); stochastic Gaussian-relaxed gates omega = clip(0.5 + mu + eps, 0, 1), eps ~ N(0, 0.5^2); expected-L0 sparsity loss with a target ratio s, rescaled by activation kurtosis; biases also trained. Init all gates 1.
- Purpose: task adaptation + compression (remove 20-40% of parameters) on GLUE (RoBERTa base/large, Llama-3.2-1B), C4 pretraining (Llama-1B), Llama-2-7B pruning vs MaskLLM. Theory: PL condition preserved under gates, not under LoRA.
- Not delete-to-localize; no forget objective; deletes rows to keep the task.

What this means for us: our neuron switch (one learned value per gate_proj row, deterministic clamp + straight-through + snap) is a one-sided row gate, a special case of FineGates' parametrization (omega_c = 1) with a different relaxation and a different objective. The row-gate parametrization is NOT novel. Novelty must be the localization objective/protocol and the results, plus the low-rank weight-level mode.

## Yin, Gu, Shen 2026. Neuron-Guided Interpretation of Code LLMs: Where, Why, and How? (FSE 2026, arXiv 2512.19980). Read: Sections 1-5 in full (motivation, LAPE study, PLS method and results, concept layers). Section 6 (downstream fine-tuning, clone detection, summarization) not read; nothing from it is used or cited.

- LAPE applied to Llama-3.1-8B on 5 programming languages finds 0.75% of neurons, and ablating them does not consistently hurt the target language (Python -7.7%, C++ -3.6% pass@3; sometimes improves others) -> "purely correlation-driven metrics like LAPE are insufficient".
- PLS score = E_s |a_n(s)| * |dL/da_n(s)| per language (activation x gradient, i.e. attribution-patching style); candidate if above threshold; specific if top language score >= 1.5x the second; top 1% kept. Ablation (activation = 0, attention and FFN) of PLS neurons: Python -82.9%, C++ -85.9%, Go -87.8% (Llama-3.1-8B), small effect on others except related pairs (C++ -> Java -35.8%; Python -> JavaScript -77.6% in Llama).
- Language-agnostic neuron set ablation -> 0% everywhere.

What this means for us: independent evidence that LAPE is weak under ablation (matches our Qwen3.5 result: LAPE Chinese neurons change NLL by <= 0.003). PLS (activation x gradient) is a stronger statistics+gradient baseline to include; it is the same family as attribution patching. Related-language collateral damage (C++/Java) parallels our zh/ja.

## Hou, Wang, Liu, Wang, Liu, Peng 2025. Decoupling Memories, Muting Neurons: Towards Practical Machine Unlearning for LLMs (NeuMuter, Findings of ACL 2025, pp. 13978-13999; ACL Anthology 2025.findings-acl.719, no arXiv version found). Read: full incl. appendices A-C.

- Trainable mask m in R^(L x d) on FFN neurons (between the two FFN linears), hard concrete (Louizos 2018; beta=0.3, gamma=-0.1, zeta=1.1), all LLM weights frozen. Loss L = -L_forget(D_f) + lambda * KL(original || masked) on NEIGHBOR samples (BERT-perturbed copies of the forget text, K=5, r=1) + eta * expected-L0 (lambda=500, eta=5, 100 iterations). Select neurons whose mask value is below a threshold set by 5% cumulative energy; then SCALE (not zero) those neurons by their mask values.
- Data: 32 memorized 200-token sequences from the GPT-Neo extraction challenge (GPT-Neo 125M/1.3B/2.7B), plus TOFU on Phi-1.3B. Metrics: memorization accuracy, extraction likelihood, loss-based MIA, utility on 9 classification tasks, 4 dialogue tasks, Pile/WikiText PPL. Baselines GA, SGA, Task Arithmetic, Deliberate Imagination, IG neuron zeroing.
- Results: ~1% of FFN neurons; utility nearly unchanged (accuracy -0.1 to -0.4%); later/middle blocks matter most; 9.7 min for 1.3B; 0.01-0.03% trainable params. IG zeroing hurts utility most.

What this means for us: THIS IS PRIOR ART FOR THE NEURON SWITCH. A learned neuron mask on a frozen LLM with a forget loss, a keep (KL) loss and an L0 count already exists. Differences left: we delete (binary 0/1, interpretable as "these neurons off") instead of scaling; our keep set is real other-language text, not BERT neighbours; our target is a language/behaviour, not memorized sequences; deterministic clamp-and-snap instead of hard concrete. These are variations, not a new method. The neuron mode must be presented as an instance of this family and compared against a NeuMuter-style arm.

# Novelty verdict (from the 20 papers read in full above and the web searches of 2026-11-23)

| Component of our method | Closest prior work (read) | Status |
|---|---|---|
| Delete-only adapter on frozen gate_proj | Deletion Tuning 2026 (per-weight sigmoid mask, capability gain) | not new |
| Deletion mask with forget + keep losses for localization | Bayazit 2024 (per-weight Gumbel mask, suppress + maintain KL), NeuMuter 2025 (neuron mask) | not new |
| Learned whole-neuron (row) mask with forget + keep + count penalty | NeuMuter 2025; Chang 2024 HC/Slimming (no keep loss); FineGates 2026 (row/col gates, compression) | not new |
| Low-rank factor B@A parametrizing a mask | PST 2022 (low-rank importance score for pruning, with weight tuning) | not new in general |
| Low-rank, delete-only, weight-level mask trained for target-vs-keep localization (LoRA memory instead of one logit per weight) | none found | candidate novelty (incremental: a cheaper parametrization of Bayazit 2024) |
| Log-sum neuron-concentration penalty on the row-deleted fraction of a weight-level mask | none found (Candes 2008 log-sum penalty is the source) | candidate novelty (small) |
| Language deletion with learned masks vs LAPE/AP/SP baselines on a modern LLM with FLORES evaluation | Skill unlearning 2025 (training-free, languages, Llama-3-8B); LAPE 2024; code-LLM PLS 2026 | candidate empirical contribution |
| Finding: weight-level deletion isolates where whole neurons cannot (polysemantic neurons) | Bayazit 2024 Table 2 already shows weight > neuron masking for knowledge | replication, not new |

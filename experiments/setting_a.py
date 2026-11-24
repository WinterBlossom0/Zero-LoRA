"""Job list of setting A (planted ground truth) and a local runner that skips finished jobs.

Stages, each after the previous one has finished for a plant:
  plants    plant.py: 3 plant types x PLANT_SEEDS plant seeds.
  methods   per plant: the learned deletions (Zero LoRA = mask, per-weight mask = full, neuron switch = neuron,
            NeuMuter), then the rankings: neuron_scores.py (selective pruning, attribution patching, LAPE, outlier,
            random), weight_scores.py (Taylor and Wanda contrasts) and the Hard Concrete sufficiency ranking.
            Plant seed 0: every method at 3 sparsity prices with train seed 0, train seeds 1 and 2 at the middle
            price, and the per-weight mask also at 2 higher learning rates (its paper trains at a much higher rate).
            Plant seeds 1 to 4: the middle price only, for paired tests over plants.
  evaluate  per plant: evaluate_deletions.py (curves against the wiped model) and recovery.py (precision and recall).
Settings decided from the pilots (docs/experiment_design.md, Setting A): planting rate per type; 1000 training
steps for every learned method, validated every 50 steps; every ranking reads the same number of windows (STEPS).
A job is finished when its last output file exists, so a rerun continues after a crash.

Usage (from the project root):
  PYTHONPATH=src:experiments torch-python experiments/setting_a.py --stage plants [--plants neuron_s0] [--dry_run]
"""

import argparse
import os
import subprocess
import sys

PLANT_SEEDS = range(5)  # five plants per type: the plant-to-plant variance dominates, and paired tests need it
PLANTS = [(plant, seed) for plant in ("neuron", "subneuron", "scattered") for seed in PLANT_SEEDS]
TARGET = "corruption"
KEEP = ["en"]  # task A: the methods must recover English while they delete task B (the corruption)
STEPS = 1000
# Planting rate per plant type, from the planting tests (docs/experiment_design.md, Setting A).
PLANT_RATES = {"neuron": "1e-3", "subneuron": "1e-2", "scattered": "1e-2"}
# Zero LoRA switches whole neurons off (the user's requirement: no half-cut neuron): the snap decides whole rows and a
# rising commit penalty pushes each row's deleted fraction to 0 or 1. The per-weight reference gets the same, so the two
# differ only in the rank of the mask.
WHOLE_NEURONS = ["--row_snap", "--row_commit_weight", "1.0"]
# Sparsity price per learned method: (parametrization, objective, fixed arguments, price flag, prices). One fully
# deleted neuron costs the price; removing a neuron plant is worth about 40 nats (English 35 -> 2.6, phrase removed),
# so 147 neurons stay worth deleting below a price of about 0.27; the grid ends at 1e-1. At 1e-4 Zero LoRA touched
# 3091 neurons and switched none fully off.
# mask and full: the log-sum neuron-concentration penalty (tau 1e-4); neuron: the linear neuron count (tau 1e6),
# whose gradient has no barrier at zero; NeuMuter: the paper's eta = 5 with one decade on each side.
METHODS = {
    "mask": ("mask", "hinge", [*WHOLE_NEURONS, "--neuron_tau", "1e-4"], "--neuron_weight", ["1e-3", "1e-2", "1e-1"]),
    "full": ("full", "hinge", [*WHOLE_NEURONS, "--neuron_tau", "1e-4"], "--neuron_weight", ["1e-3", "1e-2", "1e-1"]),
    "neuron": ("neuron", "hinge", ["--neuron_tau", "1e6"], "--neuron_weight", ["1e-3", "1e-2", "1e-1"]),
    "neumuter": ("neumuter", "neumuter", ["--neumuter_keep", "nll"], "--neuron_weight", ["0.5", "5", "50"]),
}
EXTRA_SEEDS = (1, 2)  # plant seed 0, middle price only
# Peak rate of the warm-up + cosine schedule of train_language.py, the same for every learned method. High: at 3e-3
# Zero LoRA was still converging at step 1000 and switched no neuron fully off.
LEARNING_RATE = "1e-1"
EXTRA_RATES = {"full": ["3e-1", "1"]}  # plant seed 0, middle price: a fair chance for the per-weight reference
EVALUATION = ["--test_windows", "64", "--flores_sentences", "200"]
RANKINGS = {"oneshot", "sufficiency"}  # the rankings that the methods and evaluate stages include


def plant_dir(root: str, plant: str, seed: int) -> str:
    """Return the output directory of one plant."""
    return os.path.join(root, "plants", f"{plant}_s{seed}")


def method_root(root: str, plant: str, seed: int) -> str:
    """Return the directory of the method outputs on one plant."""
    return os.path.join(root, f"{plant}_s{seed}")


def plant_jobs(root: str) -> list[tuple[str, list[str]]]:
    """Return (finished-marker file, argv) of every planting job."""
    return [(os.path.join(plant_dir(root, plant, seed), "plant.json"),
             ["experiments/plant.py", "--plant", plant, "--seed", str(seed), "--learning_rate", PLANT_RATES[plant],
              "--output_dir", plant_dir(root, plant, seed)])
            for plant, seed in PLANTS]


def learned_runs(root: str, plant: str, seed: int) -> list[tuple[str, str, list[str]]]:
    """Return (method, run directory, argv) of every learned deletion on one plant."""
    changes = os.path.join(plant_dir(root, plant, seed), "changes.safetensors")
    common = ["src/train_language.py", "--targets", TARGET, "--keep", *KEEP, "--planted", changes,
              "--steps", str(STEPS), "--eval_every", "50", *EVALUATION]
    runs = []
    for method, (parametrization, objective, fixed, price_flag, prices) in METHODS.items():
        settings = [(price, 0, LEARNING_RATE) for price in (prices if seed == 0 else prices[1:2])]
        if seed == 0:
            settings += [(prices[1], train_seed, LEARNING_RATE) for train_seed in EXTRA_SEEDS]
            settings += [(prices[1], 0, rate) for rate in EXTRA_RATES.get(method, [])]
        for price, train_seed, rate in settings:
            name = f"{method}_p{price}_t{train_seed}" + ("" if rate == LEARNING_RATE else f"_lr{rate}")
            directory = os.path.join(method_root(root, plant, seed), name)
            runs.append((method, directory, common + ["--parametrization", parametrization, "--objective", objective,
                                                      *fixed, price_flag, price, "--train_seed", str(train_seed),
                                                      "--learning_rate", rate, "--output_dir", directory]))
    return runs


def method_jobs(root: str) -> list[tuple[str, list[str]]]:
    """Return (finished-marker file, argv) of the selected jobs of all plants: the one-shot rankings first (cheap),
    then the learned runs in METHODS order (Zero LoRA first), then the sufficiency ranking (a trained ranking)."""
    jobs = []
    for plant, seed in PLANTS:
        changes = os.path.join(plant_dir(root, plant, seed), "changes.safetensors")
        base = method_root(root, plant, seed)
        if "oneshot" in RANKINGS:
            jobs.append((os.path.join(base, "scores.pt"),
                         ["experiments/neuron_scores.py", "--targets", TARGET, "--keep", *KEEP, "--planted", changes,
                          "--windows", str(STEPS), "--output", os.path.join(base, "scores.pt")]))
            jobs.append((os.path.join(base, "weights.pt"),
                         ["experiments/weight_scores.py", "--targets", TARGET, "--keep", *KEEP, "--planted", changes,
                          "--windows", str(STEPS), "--output", os.path.join(base, "weights.pt")]))
    for plant, seed in PLANTS:
        jobs += [(os.path.join(directory, "metrics.json"), argv)
                 for _, directory, argv in learned_runs(root, plant, seed)]
    for plant, seed in PLANTS:
        if "sufficiency" not in RANKINGS:
            break
        changes = os.path.join(plant_dir(root, plant, seed), "changes.safetensors")
        sufficiency = os.path.join(method_root(root, plant, seed), "sufficiency")
        jobs.append((os.path.join(sufficiency, "metrics.json"),
                     ["src/train_language.py", "--targets", TARGET, "--keep", *KEEP, "--planted", changes,
                      "--steps", str(STEPS), "--parametrization", "hardconcrete", "--objective", "sufficiency",
                      *EVALUATION, "--output_dir", sufficiency]))
    return jobs


def evaluate_jobs(root: str, tag: str = "all") -> list[tuple[str, list[str]]]:
    """Return (finished-marker file, argv) of the curve and recovery jobs of all plants, over the selected methods;
    the tag names their output files, so an evaluation of a subset does not stand in for the full one."""
    jobs = []
    for plant, seed in PLANTS:
        base = method_root(root, plant, seed)
        changes = os.path.join(plant_dir(root, plant, seed), "changes.safetensors")
        runs = [f"{method}={directory}" for method, directory, _ in learned_runs(root, plant, seed)]
        common = ["--targets", TARGET, "--keep", *KEEP, "--target_codes", f"wiki_{TARGET}",
                  "--related_codes", "eng_Latn", "--planted", changes,  # English is recovered, not collateral
                  "--reference_positions", os.path.join(plant_dir(root, plant, seed), "ground_truth.pt")]
        curve_inputs, recovery_inputs, recovery_scores = [], [], []
        if "oneshot" in RANKINGS:
            curve_inputs += ["--scores", os.path.join(base, "scores.pt"), "--methods", "sp", "ap", "lape", "outlier",
                             "random", "--weight_scores", os.path.join(base, "weights.pt")]
            recovery_scores.append(f"stats={os.path.join(base, 'scores.pt')}")
            recovery_inputs += ["--weight_scores", os.path.join(base, "weights.pt")]
        if "sufficiency" in RANKINGS:
            recovery_scores.append(f"sufficiency={os.path.join(base, 'sufficiency', 'neuron_scores.pt')}")
            jobs.append((os.path.join(base, f"curves_sufficiency_{tag}.json"),
                         ["experiments/evaluate_deletions.py", *common,
                          "--scores", os.path.join(base, "sufficiency", "neuron_scores.pt"), "--methods", "sufficiency",
                          "--output", os.path.join(base, f"curves_sufficiency_{tag}.json")]))
        jobs.append((os.path.join(base, f"curves_{tag}.json"),
                     ["experiments/evaluate_deletions.py", *common, *curve_inputs, "--runs", *runs,
                      "--output", os.path.join(base, f"curves_{tag}.json")]))
        jobs.append((os.path.join(base, f"recovery_{tag}.json"),
                     ["experiments/recovery.py", "--plant_dir", plant_dir(root, plant, seed), *recovery_inputs,
                      "--scores", *recovery_scores,
                      "--runs", *runs, "--output", os.path.join(base, f"recovery_{tag}.json")]))
    return jobs


STAGES = {"plants": plant_jobs, "methods": method_jobs, "evaluate": evaluate_jobs}


def select(jobs: list[tuple[str, list[str]]], plants: list[str] | None) -> list[tuple[str, list[str]]]:
    """Return the jobs whose finished-marker path lies in a directory of one of the plants (all when None)."""
    if plants is None:
        return jobs
    return [(marker, argv) for marker, argv in jobs
            if any(f"{os.sep}{plant}{os.sep}" in f"{os.sep}{marker}" for plant in plants)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--root", default="outputs/setting_a")
    parser.add_argument("--plants", nargs="*", help="Run only the jobs of these plants, e.g. neuron_s0; default all")
    parser.add_argument("--methods", nargs="*", choices=sorted(METHODS), help="Learned methods; default all")
    parser.add_argument("--rankings", nargs="*", choices=["oneshot", "sufficiency"],
                        help="oneshot: neuron and weight rankings (cheap); sufficiency: the trained Hard Concrete "
                        "ranking; default both")
    parser.add_argument("--tag", default="all", help="evaluate stage: name of the output files of this selection")
    parser.add_argument("--dry_run", action="store_true", help="Print the pending jobs only")
    args = parser.parse_args()
    for method in set(METHODS) - set(args.methods or METHODS):
        del METHODS[method]
    RANKINGS.intersection_update(args.rankings if args.rankings is not None else RANKINGS)
    if args.stage == "evaluate":
        jobs = select(evaluate_jobs(args.root, args.tag), args.plants)
    else:
        jobs = select(STAGES[args.stage](args.root), args.plants)
    pending = [(marker, argv) for marker, argv in jobs if not os.path.exists(marker)]
    print(f"{len(pending)} pending jobs", flush=True)
    for index, (marker, argv) in enumerate(pending):
        log = os.path.splitext(marker)[0] + ".log"
        print(f"[{index + 1}/{len(pending)}] {' '.join(argv)} > {log}", flush=True)
        if args.dry_run:
            continue
        os.makedirs(os.path.dirname(log), exist_ok=True)
        completed = subprocess.run(["experiments/run_guarded.sh", log, sys.executable, *argv], check=False)
        if completed.returncode != 0 or not os.path.exists(marker):
            raise RuntimeError(f"job failed (exit {completed.returncode}), log {log}")


if __name__ == "__main__":
    main()

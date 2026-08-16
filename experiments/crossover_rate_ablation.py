"""
crossover_rate_ablation.py

Checks whether EA's crossover rate matters: the current implementation
always applies crossover() to every selected pair (crossover_rate=1.0,
see ea.py), whereas GA literature commonly uses a crossover probability
around 0.8 (see e.g. Eiben & Smith 2003, cited in this thesis). Motivated
by supervisor feedback suggesting literature-typical settings be checked
directly.

Compares crossover_rate=1.0 (current default, used for every reported
result) against crossover_rate=0.8, with repeats, on the tuning seeds
(100-104) -- NOT the reporting seeds (42-61), for the same
tuning/reporting separation reason as tune_hyperparams_manual.py and
tune_hyperparams.py.

crossover_rate=1.0 is included explicitly (not just assumed) so the
comparison is apples-to-apples under identical evaluation code, rather
than compared against previously-reported numbers from a possibly
slightly different environment.

Output:
  - results/runs/<run_id>/crossover_rate_ablation.csv : one row per
    (crossover_rate, seed, repeat)
  - results/runs/<run_id>/crossover_rate_ablation_summary.csv : mean/std
    per crossover_rate, plus a plain-language significance check

Usage:
    cd thesis-code
    python experiments/crossover_rate_ablation.py
"""

import sys
import os
import csv
import json
import shutil
from datetime import datetime
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "..", "src")
sys.path.insert(0, SRC_DIR)

from circuit_utils import generate_target_state
from ea import evolutionary_algorithm
from fitness import fitness as fitness_fn


# ---------------------------------------------------------------------------
N_QUBITS = 4
TUNING_SEEDS = [100, 101, 102, 103, 104]
N_REPEATS = 5  # per (crossover_rate, seed)

ALPHA = 1.0
BETA = 0.01

CROSSOVER_RATES = [1.0, 0.8]

EA_PARAMS = dict(
    max_gates=15,
    pop_size=67,
    n_generations=100,
    mutation_rate=0.0779,
    alpha=ALPHA,
    beta=BETA,
    verbose=False,
)

RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + "_crossover_rate_ablation"
RUN_DIR = os.path.join(RUNS_DIR, RUN_ID)
os.makedirs(RUN_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
def run_ablation():
    rows = []
    total = len(CROSSOVER_RATES) * len(TUNING_SEEDS) * N_REPEATS
    print(f"Running {total} EA runs "
          f"({len(CROSSOVER_RATES)} crossover rates x {len(TUNING_SEEDS)} "
          f"tuning seeds x {N_REPEATS} repeats)...\n")

    for cr in CROSSOVER_RATES:
        print(f"--- crossover_rate = {cr} ---")
        for seed in TUNING_SEEDS:
            target = generate_target_state(N_QUBITS, seed=seed)
            for rep in range(N_REPEATS):
                repeat_seed = seed * 1000 + rep
                np.random.seed(repeat_seed)
                result = evolutionary_algorithm(
                    target, n_qubits=N_QUBITS, crossover_rate=cr, **EA_PARAMS
                )
                fit = fitness_fn(result["best_circuit"], target, N_QUBITS,
                                 alpha=ALPHA, beta=BETA)
                rows.append({
                    "crossover_rate": cr, "seed": seed, "repeat": rep,
                    "fidelity": result["best_fidelity"],
                    "gate_count": result["best_gate_count"],
                    "fitness": fit,
                })
            print(f"  seed {seed} done ({N_REPEATS} repeats)")

    return rows


def save_raw_csv(rows, path):
    fieldnames = ["crossover_rate", "seed", "repeat", "fidelity",
                  "gate_count", "fitness"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved raw results to {path}")


def save_summary_csv(rows, path):
    """
    Aggregates per crossover_rate and prints a plain-language
    significance check: whether the gap between the two rates' mean
    fitness exceeds the pooled standard error, i.e. whether the
    difference is distinguishable from run-to-run noise given this
    sample size. This is a rough heuristic (not a formal hypothesis
    test), sized to match how within-target std comparisons are already
    used elsewhere in this project (see sweep_beta_repeated.py).
    """
    summary = []
    for cr in CROSSOVER_RATES:
        vals = [r for r in rows if r["crossover_rate"] == cr]
        fits = [r["fitness"] for r in vals]
        fids = [r["fidelity"] for r in vals]
        gates = [r["gate_count"] for r in vals]
        summary.append({
            "crossover_rate": cr, "n_samples": len(fits),
            "mean_fitness": np.mean(fits), "std_fitness": np.std(fits),
            "mean_fidelity": np.mean(fids), "mean_gate_count": np.mean(gates),
        })

    fieldnames = ["crossover_rate", "n_samples", "mean_fitness",
                  "std_fitness", "mean_fidelity", "mean_gate_count"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    print(f"Saved summary to {path}\n")

    print("=== Summary ===")
    for r in summary:
        print(f"crossover_rate={r['crossover_rate']}: "
              f"mean fitness = {r['mean_fitness']:.4f} +/- {r['std_fitness']:.4f}, "
              f"mean fidelity = {r['mean_fidelity']:.4f}, "
              f"mean gates = {r['mean_gate_count']:.2f}")

    r1, r2 = summary[0], summary[1]
    gap = abs(r1["mean_fitness"] - r2["mean_fitness"])
    pooled_se = np.sqrt(r1["std_fitness"]**2 / r1["n_samples"]
                         + r2["std_fitness"]**2 / r2["n_samples"])
    ratio = gap / pooled_se if pooled_se > 0 else float("inf")
    print(f"\nGap in mean fitness: {gap:.4f}")
    print(f"Pooled standard error: {pooled_se:.4f}")
    print(f"Gap / SE ratio: {ratio:.2f} "
          f"({'likely a real difference' if ratio > 2 else 'not distinguishable from noise at this sample size'})")
    print("\nNote: this ablation uses the tuning seeds (100-104) with N=5 "
          "repeats -- a quick check, not a full statistical validation on "
          "the scale of sweep_beta_repeated.py. Treat 'likely a real "
          "difference' as a signal to investigate further with more "
          "repeats, not as a final conclusion on its own.")

    return summary


def save_config(path):
    config = {
        "run_id": RUN_ID, "n_qubits": N_QUBITS,
        "tuning_seeds": TUNING_SEEDS, "n_repeats": N_REPEATS,
        "alpha": ALPHA, "beta": BETA,
        "crossover_rates": CROSSOVER_RATES, "ea_params": EA_PARAMS,
    }
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {path}")


if __name__ == "__main__":
    rows = run_ablation()
    save_raw_csv(rows, os.path.join(RUN_DIR, "crossover_rate_ablation.csv"))
    summary_path = os.path.join(RUN_DIR, "crossover_rate_ablation_summary.csv")
    save_summary_csv(rows, summary_path)
    save_config(os.path.join(RUN_DIR, "config.json"))

    # Also copy the summary to results/figures/ (not gitignored, unlike
    # results/runs/) so this ablation's result is committed to the
    # repository -- see budget_matched_comparison.py for the same fix and
    # rationale.
    shutil.copy(summary_path, os.path.join(FIGURES_DIR, "crossover_rate_ablation_summary.csv"))
    print(f"Also copied summary to {FIGURES_DIR}")

    print(f"\nDone. All outputs in: {RUN_DIR}")
"""
run_experiments_repeated.py

Statistically robust version of run_experiments.py: runs EA and SA
multiple times per target state (not just once), to distinguish genuine
performance differences from run-to-run noise.

This produces the data behind the thesis's primary reported comparison
(Section 5.2, "Main Comparison") -- research_log.md Entry 9 (5 repeats)
and its 8-repeat extension confirm the result is stable at increasing
sample size.

For each target state, separates two sources of variance:
  - "within-target" std: pure run-to-run noise, from repeating the same
    target with different random seeds
  - "across-target" std: how much target difficulty varies between
    different Haar-random states

Output: results/runs/<run_id>/ containing repeated_results.csv (one row
per target x repeat x algorithm), repeated_summary.csv (per-target
mean/std), and config.json.

Usage:
    cd thesis-code
    python experiments/run_experiments_repeated.py
"""

import sys
import os
import csv
import json
from datetime import datetime
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "..", "src")
sys.path.insert(0, SRC_DIR)

from circuit_utils import generate_target_state
from ea import evolutionary_algorithm
from sa import simulated_annealing


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_QUBITS = 4
N_TARGETS = 20
BASE_SEED = 42
N_REPEATS = 8            # repetitions per target, per algorithm

ALPHA = 1.0
BETA = 0.01

# Optuna-tuned hyperparameters (same as run_experiments.py; see
# research_log.md Entry 7 and results/optuna_studies/).
EA_PARAMS = dict(
    max_gates=15,
    pop_size=67,
    n_generations=100,
    mutation_rate=0.0779,
    alpha=ALPHA,
    beta=BETA,
    verbose=False,
)

SA_PARAMS = dict(
    max_gates=15,
    initial_temp=0.256,
    cooling_rate=0.9769,
    min_temp=1e-4,
    max_iterations=2000,
    alpha=ALPHA,
    beta=BETA,
    verbose=False,
)

RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + "_repeated"
RUN_DIR = os.path.join(RUNS_DIR, RUN_ID)
os.makedirs(RUN_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------
def run_all():
    """
    Runs EA and SA N_REPEATS times each, for every target state.

    Each repeat uses a distinct, reproducible seed
    (target_seed * 1000 + repeat_idx), so repeats are deterministic
    across re-runs but differ from each other.

    Returns:
        A flat list of row-dicts, one per (target, repeat, algorithm).
    """
    rows = []

    for i in range(N_TARGETS):
        target_seed = BASE_SEED + i
        target = generate_target_state(N_QUBITS, seed=target_seed)

        print(f"\n=== Target {i + 1}/{N_TARGETS} (target_seed={target_seed}) ===")

        for rep in range(N_REPEATS):
            repeat_seed = target_seed * 1000 + rep

            np.random.seed(repeat_seed)
            ea_result = evolutionary_algorithm(target, n_qubits=N_QUBITS, **EA_PARAMS)
            rows.append({
                "target_idx": i, "target_seed": target_seed,
                "repeat": rep, "repeat_seed": repeat_seed,
                "algorithm": "EA",
                "fidelity": ea_result["best_fidelity"],
                "gate_count": ea_result["best_gate_count"],
            })

            np.random.seed(repeat_seed)
            sa_result = simulated_annealing(target, n_qubits=N_QUBITS, **SA_PARAMS)
            rows.append({
                "target_idx": i, "target_seed": target_seed,
                "repeat": rep, "repeat_seed": repeat_seed,
                "algorithm": "SA",
                "fidelity": sa_result["best_fidelity"],
                "gate_count": sa_result["best_gate_count"],
            })

            print(f"  Repeat {rep + 1}/{N_REPEATS}: "
                  f"EA fidelity={ea_result['best_fidelity']:.4f} gates={ea_result['best_gate_count']} | "
                  f"SA fidelity={sa_result['best_fidelity']:.4f} gates={sa_result['best_gate_count']}")

    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def save_raw_csv(rows, path):
    """Writes one row per (target, repeat, algorithm) run."""
    fieldnames = ["target_idx", "target_seed", "repeat", "repeat_seed",
                  "algorithm", "fidelity", "gate_count"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved raw per-repeat results to {path}")


def save_summary_csv(rows, path):
    """
    Computes, per (algorithm, target), the mean/std fidelity and gate
    count across that target's N_REPEATS repeats, then prints an overall
    summary distinguishing within-target noise (pure randomness) from
    across-target variation (target difficulty differences).
    """
    per_target_rows = []
    for algo in ("EA", "SA"):
        for i in range(N_TARGETS):
            target_rows = [r for r in rows
                            if r["algorithm"] == algo and r["target_idx"] == i]
            fidelities = [r["fidelity"] for r in target_rows]
            gate_counts = [r["gate_count"] for r in target_rows]
            per_target_rows.append({
                "algorithm": algo, "target_idx": i,
                "n_repeats": len(target_rows),
                "mean_fidelity": np.mean(fidelities),
                "std_fidelity": np.std(fidelities),
                "mean_gate_count": np.mean(gate_counts),
                "std_gate_count": np.std(gate_counts),
            })

    fieldnames = ["algorithm", "target_idx", "n_repeats", "mean_fidelity",
                  "std_fidelity", "mean_gate_count", "std_gate_count"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_target_rows)
    print(f"Saved per-target summary to {path}")

    print("\n=== Overall summary (mean of per-target means, across "
          f"{N_TARGETS} targets, {N_REPEATS} repeats each) ===")
    for algo in ("EA", "SA"):
        algo_rows = [r for r in per_target_rows if r["algorithm"] == algo]
        overall_fidelity = np.mean([r["mean_fidelity"] for r in algo_rows])
        overall_gate_count = np.mean([r["mean_gate_count"] for r in algo_rows])
        avg_within_target_fidelity_std = np.mean([r["std_fidelity"] for r in algo_rows])
        across_target_fidelity_std = np.std([r["mean_fidelity"] for r in algo_rows])

        print(f"{algo}: mean fidelity = {overall_fidelity:.4f}, "
              f"mean gate count = {overall_gate_count:.2f}")
        print(f"     avg within-target fidelity std (run-to-run noise) = "
              f"{avg_within_target_fidelity_std:.4f}")
        print(f"     across-target fidelity std (target difficulty variation) = "
              f"{across_target_fidelity_std:.4f}")


def save_config(path):
    config = {
        "run_id": RUN_ID, "n_qubits": N_QUBITS, "n_targets": N_TARGETS,
        "base_seed": BASE_SEED, "n_repeats": N_REPEATS,
        "alpha": ALPHA, "beta": BETA,
        "ea_params": EA_PARAMS, "sa_params": SA_PARAMS,
    }
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved run config to {path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rows = run_all()

    save_raw_csv(rows, os.path.join(RUN_DIR, "repeated_results.csv"))
    save_summary_csv(rows, os.path.join(RUN_DIR, "repeated_summary.csv"))
    save_config(os.path.join(RUN_DIR, "config.json"))

    print(f"\nRun complete. All outputs in: {RUN_DIR}")
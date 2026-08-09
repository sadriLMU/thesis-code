"""
sweep_beta_repeated.py

Repeated version of sweep_beta.py: runs EA and SA multiple times per
(beta, target) combination, closing the last remaining single-run gap in
this project's validation (research_log.md Entry 8 was flagged as
preliminary/single-run since it was first written).

Configuration:
  - 5 beta values (0.01, 0.02, 0.03, 0.05, 0.07), same as sweep_beta.py
  - 5 repeats per (beta, algorithm, target)
  - Same 20 target states as the main comparison

Total runs: 5 beta x 20 targets x 2 algorithms x 5 repeats = 1000 runs

Output:
  - results/runs/<run_id>/beta_sweep_repeated.csv : one row per
    (beta, algorithm, target, repeat)
  - results/runs/<run_id>/beta_sweep_repeated_summary.csv : mean/std per
    (beta, algorithm), aggregated over targets and repeats
  - results/figures/beta_sweep_repeated.png : fidelity vs. beta with error
    bars, EA vs. SA

Usage:
    cd thesis-code
    python experiments/sweep_beta_repeated.py
"""

import sys
import os
import csv
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

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
N_REPEATS = 5

ALPHA = 1.0
BETA_VALUES = [0.01, 0.02, 0.03, 0.05, 0.07]

EA_FIXED_PARAMS = dict(
    max_gates=15,
    pop_size=67,
    n_generations=100,
    mutation_rate=0.0779,
    alpha=ALPHA,
    verbose=False,
)

SA_FIXED_PARAMS = dict(
    max_gates=15,
    initial_temp=0.256,
    cooling_rate=0.9769,
    min_temp=1e-4,
    max_iterations=2000,
    alpha=ALPHA,
    verbose=False,
)

RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + "_beta_sweep_repeated"
RUN_DIR = os.path.join(RUNS_DIR, RUN_ID)
os.makedirs(RUN_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Sweep loop
# ---------------------------------------------------------------------------
def run_sweep():
    """
    For each beta value, target state, and repeat, runs EA and SA once.

    Returns:
        A flat list of row-dicts, one per (beta, algorithm, target, repeat).
    """
    rows = []

    for beta in BETA_VALUES:
        print(f"\n########## beta = {beta} ##########")

        for i in range(N_TARGETS):
            target_seed = BASE_SEED + i
            target = generate_target_state(N_QUBITS, seed=target_seed)

            for rep in range(N_REPEATS):
                repeat_seed = target_seed * 1000 + rep

                # --- EA ---
                np.random.seed(repeat_seed)
                ea_result = evolutionary_algorithm(
                    target, n_qubits=N_QUBITS, beta=beta, **EA_FIXED_PARAMS
                )
                rows.append({
                    "beta": beta, "target_idx": i, "target_seed": target_seed,
                    "repeat": rep, "repeat_seed": repeat_seed,
                    "algorithm": "EA",
                    "fidelity": ea_result["best_fidelity"],
                    "gate_count": ea_result["best_gate_count"],
                })

                # --- SA ---
                np.random.seed(repeat_seed)
                sa_result = simulated_annealing(
                    target, n_qubits=N_QUBITS, beta=beta, **SA_FIXED_PARAMS
                )
                rows.append({
                    "beta": beta, "target_idx": i, "target_seed": target_seed,
                    "repeat": rep, "repeat_seed": repeat_seed,
                    "algorithm": "SA",
                    "fidelity": sa_result["best_fidelity"],
                    "gate_count": sa_result["best_gate_count"],
                })

            print(f"  Target {i + 1}/{N_TARGETS} done ({N_REPEATS} repeats x 2 algorithms)")

    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def save_raw_csv(rows, path):
    """Writes one row per (beta, algorithm, target, repeat) to CSV."""
    fieldnames = ["beta", "target_idx", "target_seed", "repeat", "repeat_seed",
                  "algorithm", "fidelity", "gate_count"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved raw results to {path}")


def save_summary_csv(rows, path):
    """
    Aggregates raw rows into mean/std per (beta, algorithm), across all
    targets and repeats (n_samples = N_TARGETS * N_REPEATS).
    """
    summary_rows = []
    for beta in BETA_VALUES:
        for algo in ("EA", "SA"):
            vals = [r for r in rows if r["beta"] == beta and r["algorithm"] == algo]
            fids = [r["fidelity"] for r in vals]
            gates = [r["gate_count"] for r in vals]
            summary_rows.append({
                "beta": beta, "algorithm": algo, "n_samples": len(fids),
                "mean_fidelity": np.mean(fids), "std_fidelity": np.std(fids),
                "mean_gate_count": np.mean(gates), "std_gate_count": np.std(gates),
            })
    fieldnames = ["beta", "algorithm", "n_samples", "mean_fidelity",
                  "std_fidelity", "mean_gate_count", "std_gate_count"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved summary to {path}")
    return summary_rows


def save_config(path):
    """Snapshots every parameter used for this run, for reproducibility."""
    config = {
        "run_id": RUN_ID, "n_qubits": N_QUBITS, "n_targets": N_TARGETS,
        "base_seed": BASE_SEED, "n_repeats": N_REPEATS, "alpha": ALPHA,
        "beta_values": BETA_VALUES,
        "ea_params": EA_FIXED_PARAMS, "sa_params": SA_FIXED_PARAMS,
    }
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {path}")


def plot_comparison(summary_rows, path):
    """
    Plots mean fidelity vs. beta with error bars (std across
    targets+repeats), EA vs. SA, in a single shared-axis chart.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for algo, color in (("EA", "tab:blue"), ("SA", "tab:orange")):
        means, stds = [], []
        for beta in BETA_VALUES:
            row = next(r for r in summary_rows
                       if r["beta"] == beta and r["algorithm"] == algo)
            means.append(row["mean_fidelity"])
            stds.append(row["std_fidelity"])
        ax.errorbar(BETA_VALUES, means, yerr=stds, fmt="o", linestyle="-",
                    color=color, label=algo, capsize=4)

    ax.set_xlabel("beta (gate-count penalty weight)")
    ax.set_ylabel("Mean fidelity (error bars = std across targets+repeats)")
    ax.set_title(f"Beta sweep, repeated (N={N_REPEATS} repeats/target/beta)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot to {path}")


if __name__ == "__main__":
    rows = run_sweep()
    save_raw_csv(rows, os.path.join(RUN_DIR, "beta_sweep_repeated.csv"))
    summary_rows = save_summary_csv(
        rows, os.path.join(RUN_DIR, "beta_sweep_repeated_summary.csv"))
    save_config(os.path.join(RUN_DIR, "config.json"))
    plot_comparison(summary_rows, os.path.join(FIGURES_DIR, "beta_sweep_repeated.png"))
    print(f"\nDone. All outputs in: {RUN_DIR} and {FIGURES_DIR}")
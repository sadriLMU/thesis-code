"""
sweep_beta_extended_full.py

Extension of sweep_beta_repeated.py to higher beta values, to check
whether the EA/SA fidelity gap (which narrows steadily from beta=0.01 to
beta=0.07, and is already close to zero at beta=0.15 in the
fidelity-floor experiment's "standard" condition) eventually reverses --
i.e. whether SA overtakes EA at some higher penalty weight.

This is the full, statistically robust version (5 repeats per target,
matching the rigor of sweep_beta_repeated.py). If
sweep_beta_extended_quick.py's faster 2-repeat pass already showed a
clear trend, this version confirms it with proper error bars; run this
one directly if the quick version's runtime isn't a concern.

Configuration:
  - 10 beta values (0.01 through 0.30) -- includes the 5 original values
    (0.01-0.07) as an overlap/sanity check that this run reproduces the
    already-reported results, plus 5 new higher values (0.10-0.30)
  - 5 repeats per (beta, algorithm, target)
  - Same 20 target states as the main comparison

Total runs: 10 beta x 20 targets x 2 algorithms x 5 repeats = 2000 runs
(roughly 90-100 minutes)

Output:
  - results/runs/<run_id>/beta_sweep_extended_full.csv
  - results/runs/<run_id>/beta_sweep_extended_full_summary.csv
  - results/figures/beta_sweep_extended_full.png

Usage:
    cd thesis-code
    python experiments/sweep_beta_extended_full.py
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
BETA_VALUES = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]

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

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + "_beta_sweep_extended_full"
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
    targets and repeats. With only 2 repeats, within_target_std is a
    noisier estimate than in the full sweep (5 repeats) -- treat this as
    an exploratory read on the trend, not a final statistical claim.
    """
    summary_rows = []
    for beta in BETA_VALUES:
        for algo in ("EA", "SA"):
            vals = [r for r in rows if r["beta"] == beta and r["algorithm"] == algo]
            fids = [r["fidelity"] for r in vals]
            gates = [r["gate_count"] for r in vals]

            target_idxs = sorted(set(r["target_idx"] for r in vals))
            per_target_fids = {
                t: [r["fidelity"] for r in vals if r["target_idx"] == t]
                for t in target_idxs
            }
            within_stds = [np.std(v) for v in per_target_fids.values() if len(v) > 1]
            within_target_std = float(np.mean(within_stds)) if within_stds else float("nan")
            target_means = [np.mean(v) for v in per_target_fids.values()]
            across_target_std = float(np.std(target_means))

            summary_rows.append({
                "beta": beta, "algorithm": algo, "n_samples": len(fids),
                "mean_fidelity": np.mean(fids), "std_fidelity": np.std(fids),
                "within_target_std": within_target_std,
                "across_target_std": across_target_std,
                "mean_gate_count": np.mean(gates), "std_gate_count": np.std(gates),
            })
    fieldnames = ["beta", "algorithm", "n_samples", "mean_fidelity",
                  "std_fidelity", "within_target_std", "across_target_std",
                  "mean_gate_count", "std_gate_count"]
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
    Plots mean fidelity vs. beta with error bars, EA vs. SA. Includes a
    vertical marker at beta=0.07, the highest value in the original
    (non-extended) beta sweep, so it's visually clear which part of the
    plot is new territory.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for algo, color in (("EA", "tab:blue"), ("SA", "tab:orange")):
        means, within_stds = [], []
        for beta in BETA_VALUES:
            row = next(r for r in summary_rows
                       if r["beta"] == beta and r["algorithm"] == algo)
            means.append(row["mean_fidelity"])
            within_stds.append(row["within_target_std"])
        ax.errorbar(BETA_VALUES, means, yerr=within_stds, fmt="o-",
                    color=color, label=algo, capsize=4)

    ax.axvline(0.07, color="gray", linestyle=":", linewidth=1)
    ax.text(0.071, ax.get_ylim()[1] * 0.02 + ax.get_ylim()[0], "previously\ntested up to here",
            fontsize=7.5, color="gray", va="bottom")

    ax.set_xlabel("beta (gate-count penalty weight)")
    ax.set_ylabel("Mean fidelity")
    ax.set_title(f"Beta sweep, extended (N={N_REPEATS} repeats/target/beta)\n"
                 f"error bars = within-target std")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot to {path}")


if __name__ == "__main__":
    rows = run_sweep()
    save_raw_csv(rows, os.path.join(RUN_DIR, "beta_sweep_extended_full.csv"))
    summary_rows = save_summary_csv(
        rows, os.path.join(RUN_DIR, "beta_sweep_extended_full_summary.csv"))
    save_config(os.path.join(RUN_DIR, "config.json"))
    plot_comparison(summary_rows, os.path.join(FIGURES_DIR, "beta_sweep_extended_full.png"))
    print(f"\nDone. All outputs in: {RUN_DIR} and {FIGURES_DIR}")
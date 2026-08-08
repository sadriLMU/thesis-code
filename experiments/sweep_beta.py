"""
sweep_beta.py

Beta (gate-count penalty weight) sensitivity sweep: runs EA and SA once
per (target, beta) combination across a range of beta values, to study
the fidelity/gate-count trade-off (thesis Section 5.3, "Beta Sensitivity").

Single run per (target, beta) -- treated as a preliminary result in the
thesis text (see research_log.md Entry 8). The finer beta resolution
(0.01-0.07) was chosen after an earlier, coarser sweep (0.01-0.4) showed
saturation above beta=0.1 -- see research_log.md Entry 2.

Note: an early hypothesis considered whether EA's population structure
regularizes circuit length independently of beta while SA's doesn't; this
was later refined by Entry 8, which found EA consistently outperforms SA
across the entire tested beta range (in both fidelity and gate-count
efficiency), not just at the extremes.

Output:
  - results/runs/<run_id>/sweep_results.csv   : one row per (beta, algorithm, target)
  - results/runs/<run_id>/config.json         : parameters used, for reproducibility
  - results/runs/<run_id>/gate_count_vs_beta.png
  - results/runs/<run_id>/fidelity_vs_beta.png

Usage:
    cd thesis-code
    python experiments/sweep_beta.py
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
N_TARGETS = 20            # matches the main comparison's sample size (see
                           # run_experiments.py)
BASE_SEED = 42

ALPHA = 1.0
BETA_VALUES = [0.01, 0.02, 0.03, 0.05, 0.07]   # finer resolution around
                                                 # the transition point;
                                                 # 0.1/0.2/0.4 already
                                                 # confirmed redundant in
                                                 # an earlier, coarser sweep

# EA/SA parameters held fixed across the sweep -- only beta changes.
# Optuna-tuned on disjoint seeds 100-104 (see research_log.md Entry 7 and
# results/optuna_studies/).
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
os.makedirs(RUNS_DIR, exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + "_beta_sweep"
RUN_DIR = os.path.join(RUNS_DIR, RUN_ID)
os.makedirs(RUN_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Sweep loop
# ---------------------------------------------------------------------------
def run_sweep():
    """
    For each beta value, runs EA and SA on N_TARGETS target states (same
    seeds reused across beta values, so only the penalty weight changes
    between comparisons).

    Returns:
        A flat list of row-dicts, one per (beta, algorithm, target).
    """
    rows = []

    for beta in BETA_VALUES:
        print(f"\n########## beta = {beta} ##########")

        for i in range(N_TARGETS):
            seed = BASE_SEED + i
            target = generate_target_state(N_QUBITS, seed=seed)

            print(f"--- Target {i + 1}/{N_TARGETS} (seed={seed}) ---")

            # --- EA ---
            np.random.seed(seed)
            ea_result = evolutionary_algorithm(
                target, n_qubits=N_QUBITS, beta=beta, **EA_FIXED_PARAMS
            )
            rows.append({
                "beta": beta,
                "target_idx": i,
                "seed": seed,
                "algorithm": "EA",
                "fidelity": ea_result["best_fidelity"],
                "gate_count": ea_result["best_gate_count"],
            })
            print(f"  EA -> fidelity={ea_result['best_fidelity']:.4f}, "
                  f"gates={ea_result['best_gate_count']}")

            # --- SA ---
            np.random.seed(seed)
            sa_result = simulated_annealing(
                target, n_qubits=N_QUBITS, beta=beta, **SA_FIXED_PARAMS
            )
            rows.append({
                "beta": beta,
                "target_idx": i,
                "seed": seed,
                "algorithm": "SA",
                "fidelity": sa_result["best_fidelity"],
                "gate_count": sa_result["best_gate_count"],
            })
            print(f"  SA -> fidelity={sa_result['best_fidelity']:.4f}, "
                  f"gates={sa_result['best_gate_count']}")

    return rows


# ---------------------------------------------------------------------------
# Output: CSV
# ---------------------------------------------------------------------------
def save_csv(rows, path):
    """Writes one row per (beta, algorithm, target) to sweep_results.csv."""
    fieldnames = ["beta", "target_idx", "seed", "algorithm", "fidelity", "gate_count"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved sweep results to {path}")


def save_config(path):
    """Snapshots every parameter used for this run, for reproducibility."""
    config = {
        "run_id": RUN_ID,
        "n_qubits": N_QUBITS,
        "n_targets": N_TARGETS,
        "base_seed": BASE_SEED,
        "alpha": ALPHA,
        "beta_values": BETA_VALUES,
        "ea_fixed_params": EA_FIXED_PARAMS,
        "sa_fixed_params": SA_FIXED_PARAMS,
    }
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved sweep config to {path}")


# ---------------------------------------------------------------------------
# Output: plots
# ---------------------------------------------------------------------------
def plot_metric_vs_beta(rows, metric, ylabel, title, path):
    """
    Plots mean +/- std of `metric` (e.g. "gate_count" or "fidelity") as a
    function of beta, with one line per algorithm.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for algo, color in (("EA", "tab:blue"), ("SA", "tab:orange")):
        means = []
        stds = []
        for beta in BETA_VALUES:
            values = [r[metric] for r in rows
                      if r["algorithm"] == algo and r["beta"] == beta]
            means.append(np.mean(values))
            stds.append(np.std(values))
        means = np.array(means)
        stds = np.array(stds)

        ax.plot(BETA_VALUES, means, marker="o", color=color, label=f"{algo} (mean)")
        ax.fill_between(BETA_VALUES, means - stds, means + stds,
                         alpha=0.2, color=color)

    ax.set_xlabel("beta (gate-count penalty weight)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rows = run_sweep()

    save_csv(rows, os.path.join(RUN_DIR, "sweep_results.csv"))
    save_config(os.path.join(RUN_DIR, "config.json"))

    plot_metric_vs_beta(
        rows, metric="gate_count", ylabel="Best gate count",
        title="Gate count vs. beta",
        path=os.path.join(RUN_DIR, "gate_count_vs_beta.png"),
    )
    plot_metric_vs_beta(
        rows, metric="fidelity", ylabel="Best fidelity",
        title="Fidelity vs. beta",
        path=os.path.join(RUN_DIR, "fidelity_vs_beta.png"),
    )

    print(f"\nSweep complete. All outputs in: {RUN_DIR}")
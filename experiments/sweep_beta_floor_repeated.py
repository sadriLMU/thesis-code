"""
sweep_beta_floor_repeated.py

Repeated version of sweep_beta_floor.py (Entry 10), covering all 5 of
Entry 10's original beta values. The single-run floor comparison showed a
clean, consistent benefit for EA, but a genuinely mixed result for SA
(floor sometimes better, sometimes worse than standard fitness). Given
Entry 5's "SA efficiency crossover" turned out to be single-run noise once
properly tested, this repeats the full comparison to check whether SA's
inconsistency is real or another single-run artifact.

(This originally ran as a reduced 3-beta-value version, given the expected
multi-hour runtime for a full 5x repeat -- but actual per-run time turned
out much faster than estimated, so this now covers all 5 beta values from
Entry 10 rather than a subset, for a complete replication.)

Configuration:
  - 5 beta values (0.03, 0.05, 0.07, 0.10, 0.15), same as Entry 10
  - 3 repeats per (beta, algorithm, variant, target)
  - Same 20 target states, same fitness variants (standard vs. floor)

Total runs: 5 beta x 20 targets x 2 variants x 2 algorithms x 3 repeats
          = 1200 runs

Output:
  - results/runs/<run_id>/floor_comparison_repeated.csv : one row per
    (beta, algorithm, variant, target, repeat)
  - results/runs/<run_id>/floor_comparison_repeated_summary.csv : mean/std
    per (beta, algorithm, variant), aggregated over targets and repeats
  - results/figures/beta_floor_comparison_repeated.png : fidelity vs. beta
    with error bars (std across all repeats+targets), standard vs. floor

Usage:
    cd thesis-code
    python experiments/sweep_beta_floor_repeated.py
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
from fitness import fitness, fitness_with_floor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_QUBITS = 4
N_TARGETS = 20
BASE_SEED = 42
N_REPEATS = 3

ALPHA = 1.0
BETA_VALUES = [0.03, 0.05, 0.07, 0.10, 0.15]  # full replication of Entry 10's
                                                # 5-value sweep, now repeated

MIN_FIDELITY = 0.3
FLOOR_PENALTY = -100.0

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

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + "_beta_floor_repeated"
RUN_DIR = os.path.join(RUNS_DIR, RUN_ID)
os.makedirs(RUN_DIR, exist_ok=True)


def make_floor_fn():
    return lambda g, t, n, a, b: fitness_with_floor(
        g, t, n, a, b, min_fidelity=MIN_FIDELITY, floor_penalty=FLOOR_PENALTY)


# ---------------------------------------------------------------------------
# Sweep loop
# ---------------------------------------------------------------------------
def run_sweep():
    rows = []

    for beta in BETA_VALUES:
        print(f"\n########## beta = {beta} ##########")

        for i in range(N_TARGETS):
            target_seed = BASE_SEED + i
            target = generate_target_state(N_QUBITS, seed=target_seed)

            for rep in range(N_REPEATS):
                repeat_seed = target_seed * 1000 + rep

                for variant_name in ("standard", "floor"):
                    fn = fitness if variant_name == "standard" else make_floor_fn()

                    # --- EA ---
                    np.random.seed(repeat_seed)
                    ea_result = evolutionary_algorithm(
                        target, n_qubits=N_QUBITS, beta=beta,
                        fitness_fn=fn, **EA_FIXED_PARAMS
                    )
                    rows.append({
                        "beta": beta, "target_idx": i, "target_seed": target_seed,
                        "repeat": rep, "repeat_seed": repeat_seed,
                        "algorithm": "EA", "fitness_variant": variant_name,
                        "fidelity": ea_result["best_fidelity"],
                        "gate_count": ea_result["best_gate_count"],
                    })

                    # --- SA ---
                    np.random.seed(repeat_seed)
                    sa_result = simulated_annealing(
                        target, n_qubits=N_QUBITS, beta=beta,
                        fitness_fn=fn, **SA_FIXED_PARAMS
                    )
                    rows.append({
                        "beta": beta, "target_idx": i, "target_seed": target_seed,
                        "repeat": rep, "repeat_seed": repeat_seed,
                        "algorithm": "SA", "fitness_variant": variant_name,
                        "fidelity": sa_result["best_fidelity"],
                        "gate_count": sa_result["best_gate_count"],
                    })

            print(f"  Target {i + 1}/{N_TARGETS} done "
                  f"({N_REPEATS} repeats x 2 variants x 2 algorithms)")

    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def save_raw_csv(rows, path):
    fieldnames = ["beta", "target_idx", "target_seed", "repeat", "repeat_seed",
                  "algorithm", "fitness_variant", "fidelity", "gate_count"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved raw results to {path}")


def save_summary_csv(rows, path):
    summary_rows = []
    for beta in BETA_VALUES:
        for algo in ("EA", "SA"):
            for variant in ("standard", "floor"):
                vals = [r for r in rows if r["beta"] == beta
                        and r["algorithm"] == algo
                        and r["fitness_variant"] == variant]
                fids = [r["fidelity"] for r in vals]
                gates = [r["gate_count"] for r in vals]
                summary_rows.append({
                    "beta": beta, "algorithm": algo, "fitness_variant": variant,
                    "n_samples": len(fids),
                    "mean_fidelity": np.mean(fids), "std_fidelity": np.std(fids),
                    "mean_gate_count": np.mean(gates), "std_gate_count": np.std(gates),
                })
    fieldnames = ["beta", "algorithm", "fitness_variant", "n_samples",
                  "mean_fidelity", "std_fidelity", "mean_gate_count", "std_gate_count"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved summary to {path}")
    return summary_rows


def save_config(path):
    config = {
        "run_id": RUN_ID, "n_qubits": N_QUBITS, "n_targets": N_TARGETS,
        "base_seed": BASE_SEED, "n_repeats": N_REPEATS, "alpha": ALPHA,
        "beta_values": BETA_VALUES, "min_fidelity": MIN_FIDELITY,
        "floor_penalty": FLOOR_PENALTY,
        "ea_params": EA_FIXED_PARAMS, "sa_params": SA_FIXED_PARAMS,
    }
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {path}")


def plot_comparison(summary_rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, algo in zip(axes, ("EA", "SA")):
        for variant, color, style in (("standard", "tab:blue", "-"),
                                       ("floor", "tab:red", "--")):
            means, stds = [], []
            for beta in BETA_VALUES:
                row = next(r for r in summary_rows
                           if r["beta"] == beta and r["algorithm"] == algo
                           and r["fitness_variant"] == variant)
                means.append(row["mean_fidelity"])
                stds.append(row["std_fidelity"])
            ax.errorbar(BETA_VALUES, means, yerr=stds, fmt=style[0] + "o",
                        linestyle=style, color=color, label=variant,
                        capsize=4)
        ax.set_xlabel("beta")
        ax.set_ylabel("Mean fidelity (error bars = std across targets+repeats)")
        ax.set_title(f"{algo}: standard vs. floor fitness (repeated, N={N_REPEATS})")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot to {path}")


if __name__ == "__main__":
    rows = run_sweep()
    save_raw_csv(rows, os.path.join(RUN_DIR, "floor_comparison_repeated.csv"))
    summary_rows = save_summary_csv(
        rows, os.path.join(RUN_DIR, "floor_comparison_repeated_summary.csv"))
    save_config(os.path.join(RUN_DIR, "config.json"))
    plot_comparison(summary_rows,
                     os.path.join(FIGURES_DIR, "beta_floor_comparison_repeated.png"))
    print(f"\nDone. All outputs in: {RUN_DIR} and {FIGURES_DIR}")
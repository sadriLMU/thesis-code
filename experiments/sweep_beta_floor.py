"""
sweep_beta_floor.py

Ablation study testing Leo's suggestion from the week-2 meeting: does
adding a minimum-fidelity floor to the fitness function prevent the
fidelity collapse seen at high beta in sweep_beta.py (research_log.md
Entry 8, where beta=0.07 drove EA/SA fidelity down to ~0.27)?

Runs EA and SA twice at each beta value: once with the standard fitness()
(alpha*fidelity - beta*gate_count, no floor), and once with
fitness_with_floor() (same formula, but a hard penalty below
min_fidelity) -- see fitness.py for both definitions.

Single run per (beta, algorithm, variant, target). For the repeated
version used to confirm these findings, see
sweep_beta_floor_repeated.py -- the single-run result here showed a
mixed/inconsistent effect for SA that the repeated version clarified as
high variance rather than a moderate, reliable disadvantage (see
research_log.md Entries 10-11).

Output:
  - results/runs/<run_id>/floor_comparison.csv : one row per
    (beta, algorithm, fitness_variant, target)
  - results/figures/beta_floor_comparison.png : fidelity vs. beta,
    standard vs. floor variant, for both algorithms

Usage:
    cd thesis-code
    python experiments/sweep_beta_floor.py
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
N_TARGETS = 20          # same target states as the main comparison
BASE_SEED = 42

ALPHA = 1.0
# Focused on the beta range where the collapse was actually observed
# (Entry 8: fidelity dropped sharply from beta=0.03 to beta=0.07)
BETA_VALUES = [0.03, 0.05, 0.07, 0.10, 0.15]

MIN_FIDELITY = 0.3       # floor threshold -- circuits below this are
                          # rejected regardless of gate count
FLOOR_PENALTY = -100.0

# Optuna-tuned hyperparameters (same as the main comparison / Entry 9)
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

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + "_beta_floor"
RUN_DIR = os.path.join(RUNS_DIR, RUN_ID)
os.makedirs(RUN_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Sweep loop
# ---------------------------------------------------------------------------
def run_sweep():
    """
    For each beta value and each fitness variant (standard vs. floor),
    runs EA and SA once per target state.

    Returns:
        A flat list of row-dicts, one per (beta, algorithm, variant, target).
    """
    rows = []

    for beta in BETA_VALUES:
        print(f"\n########## beta = {beta} ##########")

        for i in range(N_TARGETS):
            seed = BASE_SEED + i
            target = generate_target_state(N_QUBITS, seed=seed)

            for variant_name in ("standard", "floor"):
                if variant_name == "floor":
                    fn = lambda g, t, n, a, b: fitness_with_floor(
                        g, t, n, a, b, min_fidelity=MIN_FIDELITY,
                        floor_penalty=FLOOR_PENALTY)
                else:
                    fn = fitness

                # --- EA ---
                np.random.seed(seed)
                ea_result = evolutionary_algorithm(
                    target, n_qubits=N_QUBITS, beta=beta,
                    fitness_fn=fn, **EA_FIXED_PARAMS
                )
                rows.append({
                    "beta": beta, "target_idx": i, "seed": seed,
                    "algorithm": "EA", "fitness_variant": variant_name,
                    "fidelity": ea_result["best_fidelity"],
                    "gate_count": ea_result["best_gate_count"],
                })

                # --- SA ---
                np.random.seed(seed)
                sa_result = simulated_annealing(
                    target, n_qubits=N_QUBITS, beta=beta,
                    fitness_fn=fn, **SA_FIXED_PARAMS
                )
                rows.append({
                    "beta": beta, "target_idx": i, "seed": seed,
                    "algorithm": "SA", "fitness_variant": variant_name,
                    "fidelity": sa_result["best_fidelity"],
                    "gate_count": sa_result["best_gate_count"],
                })

            print(f"  Target {i + 1}/{N_TARGETS} done "
                  f"(seed={seed}, both variants, both algorithms)")

    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def save_csv(rows, path):
    """Writes one row per (beta, algorithm, variant, target) to CSV."""
    fieldnames = ["beta", "target_idx", "seed", "algorithm", "fitness_variant",
                  "fidelity", "gate_count"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved results to {path}")


def save_config(path):
    """Snapshots every parameter used for this run, for reproducibility."""
    config = {
        "run_id": RUN_ID, "n_qubits": N_QUBITS, "n_targets": N_TARGETS,
        "base_seed": BASE_SEED, "alpha": ALPHA, "beta_values": BETA_VALUES,
        "min_fidelity": MIN_FIDELITY, "floor_penalty": FLOOR_PENALTY,
        "ea_params": EA_FIXED_PARAMS, "sa_params": SA_FIXED_PARAMS,
    }
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {path}")


def plot_comparison(rows, path):
    """
    Plots mean fidelity vs. beta, standard vs. floor variant, in two
    subplots (one per algorithm), for a direct visual comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, algo in zip(axes, ("EA", "SA")):
        for variant, color, style in (("standard", "tab:blue", "-"),
                                       ("floor", "tab:red", "--")):
            means = []
            for beta in BETA_VALUES:
                vals = [r["fidelity"] for r in rows
                        if r["algorithm"] == algo
                        and r["fitness_variant"] == variant
                        and r["beta"] == beta]
                means.append(np.mean(vals))
            ax.plot(BETA_VALUES, means, style, marker="o", color=color,
                    label=f"{variant}")
        ax.set_xlabel("beta")
        ax.set_ylabel("Mean fidelity")
        ax.set_title(f"{algo}: standard vs. floor fitness")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot to {path}")


if __name__ == "__main__":
    rows = run_sweep()
    save_csv(rows, os.path.join(RUN_DIR, "floor_comparison.csv"))
    save_config(os.path.join(RUN_DIR, "config.json"))
    plot_comparison(rows, os.path.join(FIGURES_DIR, "beta_floor_comparison.png"))
    print(f"\nDone. All outputs in: {RUN_DIR} and {FIGURES_DIR}")
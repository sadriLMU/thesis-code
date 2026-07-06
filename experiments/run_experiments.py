"""
run_experiments.py

Runs the Evolutionary Algorithm (EA) and Simulated Annealing (SA) on
multiple Haar-random target states (4 qubits) and saves:
  - results/results.csv        : one row per (target, algorithm) run
  - results/convergence.png    : mean +/- std convergence curves

Usage:
    cd thesis-code
    python experiments/run_experiments.py
"""

import sys
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# --- make src/ importable regardless of where this script is invoked from ---
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
N_TARGETS = 10       # number of random target states to average over
BASE_SEED = 42        # target state #i uses seed = BASE_SEED + i

ALPHA = 1.0           # fidelity weight
BETA = 0.01           # gate-count penalty weight

EA_PARAMS = dict(
    max_gates=15,
    pop_size=30,
    n_generations=100,
    mutation_rate=0.1,
    alpha=ALPHA,
    beta=BETA,
    verbose=False,
)

SA_PARAMS = dict(
    n_gates=20,
    initial_temp=1.0,
    cooling_rate=0.995,
    min_temp=1e-4,
    max_iterations=2000,
    alpha=ALPHA,
    beta=BETA,
    verbose=False,
)

RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------
def run_all():
    """
    Runs EA and SA on N_TARGETS different Haar-random states.

    Returns:
        rows          : list of dicts, one per (target, algorithm) run
                         -> written directly to CSV
        ea_histories  : list of per-generation fidelity histories (one per target)
        sa_histories  : list of per-iteration fitness-score histories (one per target)
    """
    rows = []
    ea_histories = []
    sa_histories = []

    for i in range(N_TARGETS):
        seed = BASE_SEED + i
        target = generate_target_state(N_QUBITS, seed=seed)

        print(f"\n=== Target {i + 1}/{N_TARGETS} (seed={seed}) ===")

        # --- EA ---
        # Seed numpy's global RNG so each target's EA run is reproducible.
        np.random.seed(seed)
        ea_result = evolutionary_algorithm(target, n_qubits=N_QUBITS, **EA_PARAMS)
        rows.append({
            "target_idx": i,
            "seed": seed,
            "algorithm": "EA",
            "fidelity": ea_result["best_fidelity"],
            "gate_count": ea_result["best_gate_count"],
        })
        ea_histories.append(ea_result["history"])
        print(f"  EA -> fidelity={ea_result['best_fidelity']:.4f}, "
              f"gates={ea_result['best_gate_count']}")

        # --- SA ---
        # Re-seed with the same value so both algorithms start from a
        # comparable random state for this target (not identical draws,
        # since they consume randomness differently, but reproducible runs).
        np.random.seed(seed)
        sa_result = simulated_annealing(target, n_qubits=N_QUBITS, **SA_PARAMS)
        rows.append({
            "target_idx": i,
            "seed": seed,
            "algorithm": "SA",
            "fidelity": sa_result["best_fidelity"],
            "gate_count": sa_result["best_gate_count"],
        })
        sa_histories.append(sa_result["history"])
        print(f"  SA -> fidelity={sa_result['best_fidelity']:.4f}, "
              f"gates={sa_result['best_gate_count']}")

    return rows, ea_histories, sa_histories


# ---------------------------------------------------------------------------
# Output: CSV
# ---------------------------------------------------------------------------
def save_csv(rows, path):
    fieldnames = ["target_idx", "seed", "algorithm", "fidelity", "gate_count"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved results to {path}")


# ---------------------------------------------------------------------------
# Output: convergence plot
# ---------------------------------------------------------------------------
def plot_convergence(ea_histories, sa_histories, path):
    """
    Plots mean +/- std convergence curves for EA and SA in separate
    subplots (NOT overlaid on one axis).

    IMPORTANT: ea.py's history tracks pure fidelity per generation, while
    sa.py's history tracks the penalized fitness score
    (alpha*fidelity - beta*gate_count) per iteration. These are different
    quantities, so they are plotted separately with distinct y-axis labels
    rather than compared directly on one chart.
    """
    ea_arr = np.array(ea_histories)  # shape: (N_TARGETS, n_generations)
    sa_arr = np.array(sa_histories)  # shape: (N_TARGETS, n_iterations)

    ea_mean, ea_std = ea_arr.mean(axis=0), ea_arr.std(axis=0)
    sa_mean, sa_std = sa_arr.mean(axis=0), sa_arr.std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(ea_mean, color="tab:blue", label="EA (mean)")
    axes[0].fill_between(
        range(len(ea_mean)), ea_mean - ea_std, ea_mean + ea_std,
        alpha=0.2, color="tab:blue",
    )
    axes[0].set_title("EA Convergence")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Best fidelity")
    axes[0].legend()

    axes[1].plot(sa_mean, color="tab:orange", label="SA (mean)")
    axes[1].fill_between(
        range(len(sa_mean)), sa_mean - sa_std, sa_mean + sa_std,
        alpha=0.2, color="tab:orange",
    )
    axes[1].set_title("SA Convergence")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Best fitness score (alpha*fidelity - beta*gates)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved convergence plot to {path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rows, ea_histories, sa_histories = run_all()
    save_csv(rows, os.path.join(RESULTS_DIR, "results.csv"))
    plot_convergence(
        ea_histories, sa_histories,
        os.path.join(RESULTS_DIR, "convergence.png"),
    )

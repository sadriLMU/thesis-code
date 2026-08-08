"""
run_experiments.py

Main EA vs. SA comparison script -- runs both algorithms once per target
state and saves results. For the statistically robust version used in the
thesis's main reported comparison (multiple repeats per target), see
run_experiments_repeated.py instead.

Produces the data behind thesis Section 5.2 ("Main Comparison"), together
with convergence plots. Uses the Optuna-tuned hyperparameters from
experiments/tune_hyperparams.py (see results/optuna_studies/ and
research_log.md Entry 7).

Output: results/runs/<run_id>/ containing results.csv, config.json,
convergence.png, and convergence_overlay.png -- plus one row appended to
results/all_runs_summary.csv.

Usage:
    cd thesis-code
    python experiments/run_experiments.py
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
N_TARGETS = 20            # target states, seeds BASE_SEED..BASE_SEED+19
BASE_SEED = 42

ALPHA = 1.0
BETA = 0.01                # default gate-count penalty (see sweep_beta.py
                            # for the beta sensitivity analysis)

# Optuna-tuned hyperparameters (tuned on disjoint seeds 100-104; see
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
SUMMARY_PATH = os.path.join(RESULTS_DIR, "all_runs_summary.csv")
os.makedirs(RUNS_DIR, exist_ok=True)

# Every execution gets its own timestamped folder, so re-running with
# different parameters never overwrites a previous run's data.
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(RUNS_DIR, RUN_ID)
os.makedirs(RUN_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------
def run_all():
    """
    Runs EA and SA once each, on every target state.

    Returns:
        rows: list of dicts, one per (target, algorithm), with fidelity
            and gate_count -- written to results.csv.
        ea_histories, sa_histories: lists of per-generation/per-iteration
            fidelity histories, one list per target, used for the
            convergence plots.
    """
    rows = []
    ea_histories = []
    sa_histories = []

    for i in range(N_TARGETS):
        seed = BASE_SEED + i
        target = generate_target_state(N_QUBITS, seed=seed)

        print(f"\n=== Target {i + 1}/{N_TARGETS} (seed={seed}) ===")

        np.random.seed(seed)
        ea_result = evolutionary_algorithm(target, n_qubits=N_QUBITS, **EA_PARAMS)
        rows.append({
            "target_idx": i, "seed": seed, "algorithm": "EA",
            "fidelity": ea_result["best_fidelity"],
            "gate_count": ea_result["best_gate_count"],
        })
        ea_histories.append(ea_result["history"])
        print(f"  EA -> fidelity={ea_result['best_fidelity']:.4f}, "
              f"gates={ea_result['best_gate_count']}")

        np.random.seed(seed)
        sa_result = simulated_annealing(target, n_qubits=N_QUBITS, **SA_PARAMS)
        rows.append({
            "target_idx": i, "seed": seed, "algorithm": "SA",
            "fidelity": sa_result["best_fidelity"],
            "gate_count": sa_result["best_gate_count"],
        })
        sa_histories.append(sa_result["history"])
        print(f"  SA -> fidelity={sa_result['best_fidelity']:.4f}, "
              f"gates={sa_result['best_gate_count']}")

    return rows, ea_histories, sa_histories


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def save_csv(rows, path):
    """Writes one row per (target, algorithm) to results.csv."""
    fieldnames = ["target_idx", "seed", "algorithm", "fidelity", "gate_count"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved results to {path}")


def save_config(path):
    """Snapshots every parameter used for this run, for reproducibility."""
    config = {
        "run_id": RUN_ID, "n_qubits": N_QUBITS, "n_targets": N_TARGETS,
        "base_seed": BASE_SEED, "alpha": ALPHA, "beta": BETA,
        "ea_params": EA_PARAMS, "sa_params": SA_PARAMS,
    }
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved run config to {path}")


def append_summary(rows, path):
    """
    Appends one summary row per algorithm to all_runs_summary.csv, an
    append-only log tracking mean/std fidelity and gate count across
    every run ever performed -- used to track how parameter changes
    affected performance over the course of the project.
    """
    is_new_file = not os.path.exists(path)
    summary_rows = []
    for algo in ("EA", "SA"):
        fidelities = [r["fidelity"] for r in rows if r["algorithm"] == algo]
        gate_counts = [r["gate_count"] for r in rows if r["algorithm"] == algo]
        summary_rows.append({
            "run_id": RUN_ID, "algorithm": algo, "n_targets": N_TARGETS,
            "mean_fidelity": np.mean(fidelities),
            "std_fidelity": np.std(fidelities),
            "mean_gate_count": np.mean(gate_counts),
            "std_gate_count": np.std(gate_counts),
        })
    fieldnames = ["run_id", "algorithm", "n_targets", "mean_fidelity",
                  "std_fidelity", "mean_gate_count", "std_gate_count"]
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new_file:
            writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Appended run summary to {path}")


def plot_convergence(ea_histories, sa_histories, path):
    """
    Saves two convergence plots: a shared-axis EA-vs-SA overlay (used for
    the thesis figure, see thesis Section 5.2) and a side-by-side pair.

    Note: EA's x-axis is "generation" (1 generation = pop_size fitness
    evaluations) while SA's is "iteration" (1 iteration = 1 evaluation) --
    these are not directly comparable in terms of computational cost. The
    overlay plot shares an axis for visual comparison of final fidelity
    reached, not for a fair per-evaluation comparison; see thesis Section
    5.2 for discussion.
    """
    ea_arr = np.array(ea_histories)
    sa_arr = np.array(sa_histories)
    ea_mean, ea_std = ea_arr.mean(axis=0), ea_arr.std(axis=0)
    sa_mean, sa_std = sa_arr.mean(axis=0), sa_arr.std(axis=0)

    # --- Overlay plot ---
    fig1, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ea_mean, color="tab:blue", label="EA (mean)")
    ax.fill_between(range(len(ea_mean)), ea_mean - ea_std, ea_mean + ea_std,
                     alpha=0.2, color="tab:blue")
    ax.plot(sa_mean, color="tab:orange", label="SA (mean)")
    ax.fill_between(range(len(sa_mean)), sa_mean - sa_std, sa_mean + sa_std,
                     alpha=0.2, color="tab:orange")
    ax.set_title("EA vs. SA Convergence (Fidelity)")
    ax.set_xlabel("Step (EA: generation, SA: iteration)")
    ax.set_ylabel("Best fidelity")
    ax.legend()
    plt.tight_layout()
    overlay_path = path.replace(".png", "_overlay.png")
    plt.savefig(overlay_path, dpi=150)
    plt.close(fig1)
    print(f"Saved overlay convergence plot to {overlay_path}")

    # --- Side-by-side plot ---
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(ea_mean, color="tab:blue", label="EA (mean)")
    axes[0].fill_between(range(len(ea_mean)), ea_mean - ea_std, ea_mean + ea_std,
                          alpha=0.2, color="tab:blue")
    axes[0].set_title("EA Convergence")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Best fidelity")
    axes[0].legend()

    axes[1].plot(sa_mean, color="tab:orange", label="SA (mean)")
    axes[1].fill_between(range(len(sa_mean)), sa_mean - sa_std, sa_mean + sa_std,
                          alpha=0.2, color="tab:orange")
    axes[1].set_title("SA Convergence")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Best fidelity")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig2)
    print(f"Saved side-by-side convergence plot to {path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rows, ea_histories, sa_histories = run_all()

    save_csv(rows, os.path.join(RUN_DIR, "results.csv"))
    save_config(os.path.join(RUN_DIR, "config.json"))
    append_summary(rows, SUMMARY_PATH)
    plot_convergence(ea_histories, sa_histories,
                      os.path.join(RUN_DIR, "convergence.png"))

    print(f"\nRun complete. All outputs for this run are in: {RUN_DIR}")
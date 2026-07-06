"""
run_experiments.py

Runs the Evolutionary Algorithm (EA) and Simulated Annealing (SA) on
multiple Haar-random target states (4 qubits). Each execution creates
a timestamped folder under results/runs/<run_id>/ containing:
  - results.csv           : one row per (target, algorithm) run
  - config.json           : every parameter used, for reproducibility
  - convergence.png       : EA/SA side-by-side subplots
  - convergence_overlay.png : EA vs. SA fidelity on one shared axis

In addition, results/all_runs_summary.csv accumulates one summary row
per algorithm per run (mean/std fidelity, mean/std gate count), so you
can track how parameter changes affected performance across every run
you've done over the course of the thesis.

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
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "all_runs_summary.csv")
os.makedirs(RUNS_DIR, exist_ok=True)

# Every execution gets its own timestamped folder under results/runs/,
# so re-running the script with different parameters never overwrites
# a previous run's data.
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(RUNS_DIR, RUN_ID)
os.makedirs(RUN_DIR, exist_ok=True)


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
        sa_histories  : list of per-iteration fidelity histories (one per target)
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
# Output: run config snapshot (for reproducibility)
# ---------------------------------------------------------------------------
def save_config(path):
    """
    Writes every parameter used for this run to a JSON file, so a given
    results.csv / plot can always be traced back to the exact configuration
    that produced it. This matters for the thesis's Methodology/Experiments
    chapters: any reported number should be attributable to a specific,
    reproducible run.
    """
    config = {
        "run_id": RUN_ID,
        "n_qubits": N_QUBITS,
        "n_targets": N_TARGETS,
        "base_seed": BASE_SEED,
        "alpha": ALPHA,
        "beta": BETA,
        "ea_params": EA_PARAMS,
        "sa_params": SA_PARAMS,
    }
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved run config to {path}")


# ---------------------------------------------------------------------------
# Output: append aggregated stats to a master log across all runs
# ---------------------------------------------------------------------------
def append_summary(rows, path):
    """
    Appends one row per algorithm to results/all_runs_summary.csv, holding
    the mean/std fidelity and mean gate count across all N_TARGETS target
    states for this run. Unlike results.csv (which is per-run and detailed),
    this file accumulates across every run you've ever done, so you can
    track how changes to parameters (pop_size, alpha/beta, cooling_rate,
    etc.) affected performance over the course of your thesis work.
    """
    is_new_file = not os.path.exists(path)

    summary_rows = []
    for algo in ("EA", "SA"):
        fidelities = [r["fidelity"] for r in rows if r["algorithm"] == algo]
        gate_counts = [r["gate_count"] for r in rows if r["algorithm"] == algo]
        summary_rows.append({
            "run_id": RUN_ID,
            "algorithm": algo,
            "n_targets": N_TARGETS,
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
    Plots mean +/- std convergence curves for EA and SA.

    Both ea.py and sa.py now log pure fidelity per step (see sa.py's
    updated `history` tracking), so this produces two figures:
      1. A shared-axis overlay comparing EA vs. SA directly (the version
         you'd typically put in the Experiments chapter).
      2. Separate subplots, kept because EA's x-axis is "generation"
         (1 step = evaluate an entire population of pop_size candidates)
         while SA's x-axis is "iteration" (1 step = evaluate a single
         neighbor). The step counts are not directly comparable in terms
         of wall-clock cost or candidates evaluated, only in terms of
         final fidelity reached -- worth stating explicitly in your
         Methodology/Discussion text if you use the overlay plot.
    """
    ea_arr = np.array(ea_histories)  # shape: (N_TARGETS, n_generations)
    sa_arr = np.array(sa_histories)  # shape: (N_TARGETS, n_iterations)

    ea_mean, ea_std = ea_arr.mean(axis=0), ea_arr.std(axis=0)
    sa_mean, sa_std = sa_arr.mean(axis=0), sa_arr.std(axis=0)

    # --- Figure 1: shared-axis overlay ---
    fig1, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ea_mean, color="tab:blue", label="EA (mean)")
    ax.fill_between(
        range(len(ea_mean)), ea_mean - ea_std, ea_mean + ea_std,
        alpha=0.2, color="tab:blue",
    )
    ax.plot(sa_mean, color="tab:orange", label="SA (mean)")
    ax.fill_between(
        range(len(sa_mean)), sa_mean - sa_std, sa_mean + sa_std,
        alpha=0.2, color="tab:orange",
    )
    ax.set_title("EA vs. SA Convergence (Fidelity)")
    ax.set_xlabel("Step (EA: generation, SA: iteration)")
    ax.set_ylabel("Best fidelity")
    ax.legend()
    plt.tight_layout()
    overlay_path = path.replace(".png", "_overlay.png")
    plt.savefig(overlay_path, dpi=150)
    plt.close(fig1)
    print(f"Saved overlay convergence plot to {overlay_path}")

    # --- Figure 2: separate subplots (for reference / appendix) ---
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))

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
    plot_convergence(
        ea_histories, sa_histories,
        os.path.join(RUN_DIR, "convergence.png"),
    )

    print(f"\nRun complete. All outputs for this run are in: {RUN_DIR}")
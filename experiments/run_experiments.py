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
import shutil
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
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
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
        ea_fitness_histories, sa_fitness_histories: the same, but
            recording fitness (alpha*fidelity - beta*gate_count) rather
            than pure fidelity. Fitness is what both algorithms actually
            maximise, so the fitness curves show the quantity being
            optimised, while the fidelity curves show one of its two
            components.
    """
    rows = []
    ea_histories = []
    sa_histories = []
    ea_fitness_histories = []
    sa_fitness_histories = []

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
        ea_fitness_histories.append(ea_result["fitness_history"])
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
        sa_fitness_histories.append(sa_result["fitness_history"])
        print(f"  SA -> fidelity={sa_result['best_fidelity']:.4f}, "
              f"gates={sa_result['best_gate_count']}")

    return (rows, ea_histories, sa_histories,
            ea_fitness_histories, sa_fitness_histories)


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


def plot_convergence(ea_histories, sa_histories, path, pop_size=None):
    """
    Saves three convergence plots: a shared-axis EA-vs-SA overlay by raw
    step (used for the thesis figure, see thesis Section 5.2), the same
    overlay but normalised by number of fitness evaluations, and a
    side-by-side pair.

    Note: EA's x-axis is "generation" (1 generation = pop_size fitness
    evaluations, i.e. 67 for the tuned hyperparameters) while SA's is
    "iteration" (1 iteration = 1 evaluation) -- these are not
    computationally equivalent. The raw-step overlay's x-axis label
    states this explicitly rather than leaving it as an implicit
    assumption. The evaluation-normalised overlay divides this out by
    scaling EA's x-axis by pop_size, giving both curves a shared "number
    of fitness evaluations" axis, which is the fair basis to use for any
    claim about relative convergence *speed* (as opposed to final
    fidelity reached, for which the raw-step overlay is fine).

    pop_size must be passed for the normalised plot; if None, the
    normalised plot is skipped (raw overlay and side-by-side are still
    produced).
    """
    ea_arr = np.array(ea_histories)
    sa_arr = np.array(sa_histories)
    ea_mean, ea_std = ea_arr.mean(axis=0), ea_arr.std(axis=0)
    sa_mean, sa_std = sa_arr.mean(axis=0), sa_arr.std(axis=0)

    # --- Overlay plot (raw step count) ---
    fig1, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ea_mean, color="tab:blue", label="EA (mean)")
    ax.fill_between(range(len(ea_mean)), ea_mean - ea_std, ea_mean + ea_std,
                     alpha=0.2, color="tab:blue")
    ax.plot(sa_mean, color="tab:orange", label="SA (mean)")
    ax.fill_between(range(len(sa_mean)), sa_mean - sa_std, sa_mean + sa_std,
                     alpha=0.2, color="tab:orange")
    ax.set_title(f"EA vs. SA convergence, fidelity ({N_QUBITS}-qubit circuits)")
    ax.set_xlabel("Step (EA: generation, SA: iteration -- NOT computationally\n"
                  "equivalent: 1 EA generation costs pop_size fitness "
                  "evaluations, 1 SA iteration costs 1)")
    ax.set_ylabel("Best fidelity")
    ax.legend()
    plt.tight_layout()
    overlay_path = path.replace(".png", "_overlay.png")
    plt.savefig(overlay_path, dpi=150)
    plt.close(fig1)
    print(f"Saved overlay convergence plot to {overlay_path}")

    # --- Overlay normalised by number of fitness evaluations ---
    if pop_size is not None:
        fig1b, ax = plt.subplots(figsize=(7, 5))
        ea_x = np.arange(len(ea_mean)) * pop_size  # 1 generation = pop_size evals
        sa_x = np.arange(len(sa_mean))              # 1 iteration = 1 eval
        ax.plot(ea_x, ea_mean, color="tab:blue", label="EA (mean)")
        ax.fill_between(ea_x, ea_mean - ea_std, ea_mean + ea_std,
                         alpha=0.2, color="tab:blue")
        ax.plot(sa_x, sa_mean, color="tab:orange", label="SA (mean)")
        ax.fill_between(sa_x, sa_mean - sa_std, sa_mean + sa_std,
                         alpha=0.2, color="tab:orange")
        ax.set_title(f"EA vs. SA convergence, fidelity, normalised by fitness\n"
                     f"evaluations ({N_QUBITS}-qubit circuits)")
        ax.set_xlabel(f"Number of fitness evaluations "
                       f"(EA: generation $\\times$ {pop_size}, SA: iteration)")
        ax.set_ylabel("Best fidelity")
        ax.legend()
        plt.tight_layout()
        overlay_evals_path = path.replace(".png", "_overlay_by_evaluations.png")
        plt.savefig(overlay_evals_path, dpi=150)
        plt.close(fig1b)
        print(f"Saved evaluation-normalised overlay plot to {overlay_evals_path}")

    # --- Side-by-side plot ---
    # sharey=True (rather than leaving each panel auto-scaled) so that a
    # reader comparing the two panels visually sees the true difference
    # in final fidelity between EA and SA, not an artefact of each panel
    # independently filling its own y-range -- side-by-side panels invite
    # visual comparison regardless of what the caption says, so the axes
    # should not silently understate a real difference in the data.
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    axes[0].plot(ea_mean, color="tab:blue", label="EA (mean)")
    axes[0].fill_between(range(len(ea_mean)), ea_mean - ea_std, ea_mean + ea_std,
                          alpha=0.2, color="tab:blue")
    axes[0].set_title(f"EA convergence, fidelity ({N_QUBITS}-qubit circuits)")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Best fidelity")
    axes[0].legend()

    axes[1].plot(sa_mean, color="tab:orange", label="SA (mean)")
    axes[1].fill_between(range(len(sa_mean)), sa_mean - sa_std, sa_mean + sa_std,
                          alpha=0.2, color="tab:orange")
    axes[1].set_title(f"SA convergence, fidelity ({N_QUBITS}-qubit circuits)")
    axes[1].set_xlabel("Iteration")
    axes[1].tick_params(labelleft=True)  # sharey hides these by default

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig2)
    print(f"Saved side-by-side convergence plot to {path}")


def plot_fitness_convergence(ea_fitness_histories, sa_fitness_histories, path,
                              pop_size=None):
    """
    Saves convergence plots for fitness rather than fidelity.

    plot_convergence() above tracks fidelity, which is only one of the two
    components of the objective the algorithms actually maximise
    (alpha*fidelity - beta*gate_count). These plots show that objective
    itself, so the curves correspond directly to what the search is
    optimising.

    Produces a separate panel per algorithm (EA and SA are plotted apart,
    since a generation and an iteration are not the same unit of work),
    plus, if pop_size is given, a shared-axis version normalised by number
    of fitness evaluations, which is the only fair basis for comparing the
    two algorithms' progress against each other.
    """
    ea_arr = np.array(ea_fitness_histories)
    sa_arr = np.array(sa_fitness_histories)
    ea_mean, ea_std = ea_arr.mean(axis=0), ea_arr.std(axis=0)
    sa_mean, sa_std = sa_arr.mean(axis=0), sa_arr.std(axis=0)

    # --- Separate panels, one per algorithm ---
    # sharey=True for the same reason as plot_convergence()'s side-by-side
    # panels: side-by-side placement invites visual comparison regardless
    # of caption text, so the axes should show the true difference in
    # fitness reached rather than each panel independently filling its
    # own auto-scaled range.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    axes[0].plot(ea_mean, color="tab:blue", label="EA (mean)")
    axes[0].fill_between(range(len(ea_mean)), ea_mean - ea_std, ea_mean + ea_std,
                          alpha=0.2, color="tab:blue")
    axes[0].set_title(f"EA convergence, fitness ({N_QUBITS}-qubit circuits)")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel(r"Best fitness ($\alpha\cdot$fidelity $-\ \beta\cdot$gate count)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(sa_mean, color="tab:orange", label="SA (mean)")
    axes[1].fill_between(range(len(sa_mean)), sa_mean - sa_std, sa_mean + sa_std,
                          alpha=0.2, color="tab:orange")
    axes[1].set_title(f"SA convergence, fitness ({N_QUBITS}-qubit circuits)")
    axes[1].set_xlabel("Iteration")
    axes[1].tick_params(labelleft=True)  # sharey hides these by default
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved side-by-side fitness convergence plot to {path}")

    # --- Shared axis, normalised by fitness evaluations ---
    if pop_size is not None:
        fig2, ax = plt.subplots(figsize=(7, 5))
        ea_x = np.arange(len(ea_mean)) * pop_size
        sa_x = np.arange(len(sa_mean))
        ax.plot(ea_x, ea_mean, color="tab:blue", label="EA (mean)")
        ax.fill_between(ea_x, ea_mean - ea_std, ea_mean + ea_std,
                         alpha=0.2, color="tab:blue")
        ax.plot(sa_x, sa_mean, color="tab:orange", label="SA (mean)")
        ax.fill_between(sa_x, sa_mean - sa_std, sa_mean + sa_std,
                         alpha=0.2, color="tab:orange")
        ax.set_title(f"EA vs. SA convergence, fitness, normalised by fitness\n"
                     f"evaluations ({N_QUBITS}-qubit circuits)")
        ax.set_xlabel(f"Number of fitness evaluations "
                       f"(EA: generation $\\times$ {pop_size}, SA: iteration)")
        ax.set_ylabel(r"Best fitness ($\alpha\cdot$fidelity $-\ \beta\cdot$gate count)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p2 = path.replace(".png", "_by_evaluations.png")
        plt.savefig(p2, dpi=150)
        plt.close(fig2)
        print(f"Saved evaluation-normalised fitness convergence plot to {p2}")


def plot_final_comparison_bars(rows, path):
    """
    Bar chart comparing EA and SA on the two reported metrics for the
    final circuits: mean fidelity and mean fitness. Error bars are the
    standard deviation across targets.

    Fitness is recomputed here from each final circuit's reported fidelity
    and gate count (fitness = alpha*fidelity - beta*gate_count), using the
    same alpha/beta the run was configured with, so the two bars refer to
    exactly the same circuits.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, (metric, label) in zip(axes, (("fidelity", "Mean fidelity"),
                                           ("fitness", "Mean fitness"))):
        means, stds = [], []
        for algo in ("EA", "SA"):
            vals = []
            for r in rows:
                if r["algorithm"] != algo:
                    continue
                if metric == "fidelity":
                    vals.append(r["fidelity"])
                else:
                    vals.append(ALPHA * r["fidelity"] - BETA * r["gate_count"])
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        ax.bar(["EA", "SA"], means, yerr=stds, capsize=6,
               color=["tab:blue", "tab:orange"])
        ax.set_ylabel(label)
        ax.set_title(f"{label} of final circuits ({N_QUBITS} qubits)")
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved final-comparison bar chart to {path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    (rows, ea_histories, sa_histories,
     ea_fitness_histories, sa_fitness_histories) = run_all()

    save_csv(rows, os.path.join(RUN_DIR, "results.csv"))
    save_config(os.path.join(RUN_DIR, "config.json"))
    append_summary(rows, SUMMARY_PATH)
    plot_convergence(ea_histories, sa_histories,
                      os.path.join(RUN_DIR, "convergence.png"),
                      pop_size=EA_PARAMS["pop_size"])
    plot_fitness_convergence(ea_fitness_histories, sa_fitness_histories,
                              os.path.join(RUN_DIR, "convergence_fitness.png"),
                              pop_size=EA_PARAMS["pop_size"])
    plot_final_comparison_bars(rows,
                                os.path.join(RUN_DIR, "final_comparison_bars_single_run.png"))

    # Also copy every plot to results/figures/ (unlike results/runs/, not
    # gitignored) so this run's figures are actually committed to the
    # repository -- see budget_matched_comparison.py for the same fix and
    # rationale. plot_convergence()/plot_fitness_convergence() each
    # produce more than one file (the base name plus _overlay and
    # _overlay_by_evaluations variants), so all matching files in RUN_DIR
    # are copied rather than listing every variant by hand.
    import glob
    for pattern in ("convergence*.png", "final_comparison_bars_single_run.png"):
        for src in glob.glob(os.path.join(RUN_DIR, pattern)):
            shutil.copy(src, os.path.join(FIGURES_DIR, os.path.basename(src)))
    print(f"Also copied convergence and comparison plots to {FIGURES_DIR}")

    print(f"\nRun complete. All outputs for this run are in: {RUN_DIR}")
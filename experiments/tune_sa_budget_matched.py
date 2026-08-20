"""
tune_sa_budget_matched.py

Re-tunes simulated annealing's initial_temp specifically for the
budget-matched evaluation count (6,700 evaluations, matching the
evolutionary algorithm's budget) used in
experiments/budget_matched_comparison.py, rather than reusing the
initial_temp value tuned for SA's much shorter default budget (~330
evaluations).

Motivation: tune_hyperparams.py's original Optuna search tuned
initial_temp and cooling_rate together for SA's default configuration,
where the cooling schedule reaches min_temp after only ~330 iterations.
The budget-matched control experiment
(experiments/budget_matched_comparison.py) instead fixes cooling_rate
analytically so the schedule reaches min_temp after ~6,700 iterations
(matching the EA's budget), but leaves initial_temp at the value tuned
for the *original*, much shorter schedule. A longer search can
plausibly tolerate a higher initial_temp (more early-stage exploration
before cooling forces exploitation), so the original initial_temp may
not be optimal for the longer, matched schedule -- this was flagged as
an untested limitation (see research_log.md and discussion.tex,
"Budget-Matched SA Was Not Retuned").

This script isolates exactly one variable: only initial_temp is
searched. cooling_rate is held fixed at the same analytically-derived
value used in budget_matched_comparison.py (0.99883, chosen so the
schedule reaches min_temp after exactly 6,700 iterations). Searching
cooling_rate as well would no longer guarantee a budget-matched
comparison, since Optuna could then select a cooling_rate that reaches
min_temp at a different iteration count than EA's 6,700-evaluation
budget -- this script deliberately keeps that one variable fixed so
"budget-matched" continues to mean exactly what it means in
budget_matched_comparison.py.

Same protocol as tune_hyperparams.py: 30 trials, tuning target states
(seeds 100-104, disjoint from the reporting seeds 42-61), objective is
mean fitness across the 5 tuning targets. Search range for initial_temp
matches the original SA tuning search space (0.1-5.0, log scale) for
consistency.

Output:
    - results/runs/<run_id>/sa_budget_matched_tuning.json : best
      initial_temp found, plus the full trial history
    - results/figures/sa_budget_matched_history.png : trial values and
      best-so-far curve, in the same style as sa_history.png

Usage:
    cd thesis-code
    python experiments/tune_sa_budget_matched.py
"""

import sys
import os
import json
import shutil
from datetime import datetime
import numpy as np
import optuna
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "..", "src")
sys.path.insert(0, SRC_DIR)

from circuit_utils import generate_target_state
from sa import simulated_annealing
from fitness import fitness as fitness_fn

# ---------------------------------------------------------------------------
N_QUBITS = 4
TUNING_SEEDS = [100, 101, 102, 103, 104]  # disjoint from reporting seeds 42-61
BASE_SEED = 100

ALPHA = 1.0
BETA = 0.01
N_TRIALS = 30

# The analytically-derived cooling rate from budget_matched_comparison.py,
# solved so the schedule reaches min_temp after exactly 6,700 iterations
# given initial_temp = 0.256 (the value tuned for the ORIGINAL, ~330
# evaluation budget). Kept fixed here deliberately -- see module
# docstring. Note this value was derived assuming initial_temp=0.256;
# since initial_temp is now being retuned, the true "reaches min_temp at
# 6,700" property will shift slightly depending on the new initial_temp
# found, but the schedule will still terminate close to that budget for
# any initial_temp in the searched range, since cooling_rate close to 1
# dominates the iteration count far more than initial_temp does.
COOLING_RATE_MATCHED = 0.99883

SA_FIXED = dict(
    max_gates=15,
    cooling_rate=COOLING_RATE_MATCHED,
    min_temp=1e-4,
    max_iterations=6700,
    alpha=ALPHA,
    beta=BETA,
    verbose=False,
)

TARGETS = [generate_target_state(N_QUBITS, seed=s) for s in TUNING_SEEDS]

RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + "_sa_budget_matched_tuning"
RUN_DIR = os.path.join(RUNS_DIR, RUN_ID)
os.makedirs(RUN_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
def sa_objective(trial: optuna.Trial) -> float:
    """
    One Optuna trial: samples initial_temp only (cooling_rate fixed),
    runs SA with the 6,700-evaluation budget-matched schedule on all
    tuning targets, and returns mean fitness as the objective to
    maximise.
    """
    initial_temp = trial.suggest_float("initial_temp", 0.1, 5.0, log=True)

    scores = []
    for i, target in enumerate(TARGETS):
        np.random.seed(BASE_SEED + i)
        result = simulated_annealing(
            target, n_qubits=N_QUBITS,
            initial_temp=initial_temp,
            **SA_FIXED,
        )
        score = fitness_fn(result["best_circuit"], target, N_QUBITS, ALPHA, BETA)
        scores.append(score)

    return float(np.mean(scores))


def save_results(study: optuna.Study, path: str):
    output = {
        "run_id": RUN_ID,
        "cooling_rate_fixed": COOLING_RATE_MATCHED,
        "search_space": {"initial_temp": [0.1, 5.0, "log"]},
        "n_trials": N_TRIALS,
        "tuning_seeds": TUNING_SEEDS,
        "best_initial_temp": study.best_params["initial_temp"],
        "best_mean_fitness": study.best_value,
        "original_initial_temp": 0.256,
        "trials": [
            {"number": t.number, "initial_temp": t.params.get("initial_temp"),
             "value": t.value}
            for t in study.trials
        ],
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved results to {path}")


def plot_history(study: optuna.Study, path: str):
    values = [t.value for t in study.trials]
    best_so_far = np.maximum.accumulate(values)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(range(len(values)), values, color="grey", alpha=0.6, label="Trial value")
    ax.plot(best_so_far, color="tab:blue", label="Best so far")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Mean fitness")
    ax.set_title("SA initial_temp Re-Tuning (budget-matched, cooling_rate fixed)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved history plot to {path}")


if __name__ == "__main__":
    print(f"Re-tuning SA's initial_temp for the budget-matched (6,700-evaluation) "
          f"schedule, cooling_rate fixed at {COOLING_RATE_MATCHED}.")
    print(f"Original (short-budget-tuned) initial_temp was 0.256, for comparison.\n")

    study = optuna.create_study(direction="maximize")
    study.optimize(sa_objective, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\nBest initial_temp found: {study.best_params['initial_temp']:.4f}")
    print(f"Best mean fitness: {study.best_value:.4f}")
    print(f"(Original initial_temp 0.256, for comparison against tuning-seed "
          f"performance at this budget)")

    results_path = os.path.join(RUN_DIR, "sa_budget_matched_tuning.json")
    save_results(study, results_path)

    plot_path = os.path.join(RUN_DIR, "sa_budget_matched_history.png")
    plot_history(study, plot_path)

    # Copy the figure to results/figures/ (not gitignored, unlike
    # results/runs/) so it is committed to the repository -- see
    # budget_matched_comparison.py for the same fix and rationale.
    shutil.copy(plot_path, os.path.join(FIGURES_DIR, "sa_budget_matched_history.png"))
    print(f"Also copied history plot to {FIGURES_DIR}")

    print(f"\nDone. All outputs in: {RUN_DIR}")
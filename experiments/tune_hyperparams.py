"""
tune_hyperparams.py

Uses Optuna to search for good EA and SA hyperparameters, replacing manual
guess-and-check tuning (per Leo's feedback: "Optuna benutzen um bessere
Ergebnisse zu bekommen, nicht mehr manuell anpassen").

What's tuned vs. fixed:
  Tunable (genuine search-strategy hyperparameters):
    EA: pop_size, mutation_rate
    SA: initial_temp, cooling_rate
  Fixed (these are experiment variables studied elsewhere in the thesis,
  not implementation details to auto-tune away):
    n_qubits, beta, alpha, max_gates/n_gates, generation/iteration budgets

Objective: mean `fitness` (alpha*fidelity - beta*gate_count) across
N_TUNING_TARGETS target states -- the same quantity the algorithms
themselves search for, so "better hyperparameters" means "better at the
actual thing being optimized," not a different metric.

Output:
  - results/optuna_studies/ea_study.db / sa_study.db : persistent Optuna
    storage (SQLite) -- studies can be resumed/inspected later, e.g. with
    `optuna-dashboard sqlite:///results/optuna_studies/ea_study.db`
  - results/optuna_studies/ea_best_params.json / sa_best_params.json
  - results/optuna_studies/ea_history.png / sa_history.png : trial value
    over trial number, to see whether the search actually converged

Usage:
    cd thesis-code
    pip install optuna --break-system-packages   # one-time, see requirements.txt
    python experiments/tune_hyperparams.py
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import optuna

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "..", "src")
sys.path.insert(0, SRC_DIR)

from circuit_utils import generate_target_state
from ea import evolutionary_algorithm
from sa import simulated_annealing
from fitness import fitness as fitness_fn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_QUBITS = 4
N_TUNING_TARGETS = 5     # fewer than the main N_TARGETS=10, since each
                          # Optuna trial runs a *full* EA/SA search
BASE_SEED = 100           # deliberately disjoint from the reporting seeds
                          # (42-51, used in run_experiments.py/sweep_beta.py),
                          # so hyperparameters aren't selected on the same
                          # targets they're later evaluated/reported on

ALPHA = 1.0
BETA = 0.01                # fixed at the value used for reported results

N_TRIALS_EA = 30
N_TRIALS_SA = 30

# Fixed budget: same as the main experiments, so tuning finds hyperparameters
# that work well within the SAME compute budget already used for reporting,
# rather than just favoring "search longer."
EA_FIXED = dict(max_gates=15, n_generations=100, alpha=ALPHA, beta=BETA, verbose=False)
SA_FIXED = dict(n_gates=20, min_temp=1e-4, max_iterations=2000, alpha=ALPHA, beta=BETA, verbose=False)

RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
STUDIES_DIR = os.path.join(RESULTS_DIR, "optuna_studies")
os.makedirs(STUDIES_DIR, exist_ok=True)

TARGETS = [generate_target_state(N_QUBITS, seed=BASE_SEED + i)
           for i in range(N_TUNING_TARGETS)]


# ---------------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------------
def ea_objective(trial: optuna.Trial) -> float:
    """
    One Optuna trial: sample pop_size/mutation_rate, run EA on all tuning
    targets with those hyperparameters, return mean fitness.
    """
    pop_size = trial.suggest_int("pop_size", 10, 100)
    mutation_rate = trial.suggest_float("mutation_rate", 0.01, 0.5)

    scores = []
    for i, target in enumerate(TARGETS):
        np.random.seed(BASE_SEED + i)
        result = evolutionary_algorithm(
            target, n_qubits=N_QUBITS,
            pop_size=pop_size, mutation_rate=mutation_rate,
            **EA_FIXED,
        )
        score = fitness_fn(result["best_circuit"], target, N_QUBITS, ALPHA, BETA)
        scores.append(score)

    return float(np.mean(scores))


def sa_objective(trial: optuna.Trial) -> float:
    """
    One Optuna trial: sample initial_temp/cooling_rate, run SA on all tuning
    targets with those hyperparameters, return mean fitness.
    """
    initial_temp = trial.suggest_float("initial_temp", 0.1, 5.0, log=True)
    cooling_rate = trial.suggest_float("cooling_rate", 0.9, 0.999)

    scores = []
    for i, target in enumerate(TARGETS):
        np.random.seed(BASE_SEED + i)
        result = simulated_annealing(
            target, n_qubits=N_QUBITS,
            initial_temp=initial_temp, cooling_rate=cooling_rate,
            **SA_FIXED,
        )
        score = fitness_fn(result["best_circuit"], target, N_QUBITS, ALPHA, BETA)
        scores.append(score)

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def save_best_params(study: optuna.Study, path: str):
    output = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved best params to {path}")
    print(f"  best_value (mean fitness) = {study.best_value:.4f}")
    print(f"  best_params = {study.best_params}")


def plot_history(study: optuna.Study, title: str, path: str):
    """Plots objective value per trial, to check whether the search converged."""
    values = [t.value for t in study.trials if t.value is not None]
    best_so_far = np.maximum.accumulate(values)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(range(len(values)), values, alpha=0.4, color="tab:gray", label="Trial value")
    ax.plot(range(len(values)), best_so_far, color="tab:blue", label="Best so far")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Mean fitness")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved optimization history plot to {path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Tuning on {N_TUNING_TARGETS} target states (seeds {BASE_SEED}..{BASE_SEED + N_TUNING_TARGETS - 1})\n")

    # --- EA study ---
    print("=" * 60)
    print(f"Tuning EA ({N_TRIALS_EA} trials)")
    print("=" * 60)
    ea_storage = f"sqlite:///{os.path.join(STUDIES_DIR, 'ea_study.db')}"
    ea_study = optuna.create_study(
        study_name="ea_tuning", storage=ea_storage,
        direction="maximize", load_if_exists=True,
    )
    ea_study.optimize(ea_objective, n_trials=N_TRIALS_EA)
    save_best_params(ea_study, os.path.join(STUDIES_DIR, "ea_best_params.json"))
    plot_history(ea_study, "EA Hyperparameter Tuning",
                 os.path.join(STUDIES_DIR, "ea_history.png"))

    # --- SA study ---
    print("\n" + "=" * 60)
    print(f"Tuning SA ({N_TRIALS_SA} trials)")
    print("=" * 60)
    sa_storage = f"sqlite:///{os.path.join(STUDIES_DIR, 'sa_study.db')}"
    sa_study = optuna.create_study(
        study_name="sa_tuning", storage=sa_storage,
        direction="maximize", load_if_exists=True,
    )
    sa_study.optimize(sa_objective, n_trials=N_TRIALS_SA)
    save_best_params(sa_study, os.path.join(STUDIES_DIR, "sa_best_params.json"))
    plot_history(sa_study, "SA Hyperparameter Tuning",
                 os.path.join(STUDIES_DIR, "sa_history.png"))

    print("\nDone. Use the best_params JSON files to update EA_PARAMS/SA_PARAMS")
    print("in run_experiments.py and sweep_beta.py for the final reported results.")
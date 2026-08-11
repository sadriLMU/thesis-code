"""
tune_hyperparams_manual.py

Manual (grid-search) hyperparameter exploration for the EA, complementing
the Optuna-based search in tune_hyperparams.py. Motivated by supervisor
feedback suggesting that literature-typical settings -- in particular a
larger population (200) and a range of mutation rates -- be checked
directly, rather than relying solely on the automated search.

Unlike the Optuna study, this script sweeps a small, explicitly listed
grid so that every tested configuration is visible and reproducible from
the source alone, which makes it straightforward to state in the thesis
exactly which settings were compared.

Evaluation protocol (matching tune_hyperparams.py, deliberately):
  - Tuning target states use seeds 100-104, DISJOINT from the reporting
    seeds 42-61 used for all results in Chapter 5. This preserves the
    tuning/reporting separation described in the Reproducibility Notes;
    running this grid on the reporting seeds would invalidate that
    separation and bias the reported comparison.
  - Each configuration is scored by mean fitness (alpha*fidelity -
    beta*gate_count) across the tuning targets, the same objective the
    Optuna study maximised.

Note on comparing configurations fairly: one EA generation costs
pop_size fitness evaluations, so raising pop_size while holding
n_generations fixed also raises the total search budget (e.g. pop_size
200 x 100 generations = 20,000 evaluations versus the tuned
configuration's 67 x 100 = 6,700). A configuration scoring higher may
therefore simply have been given more compute rather than being
intrinsically better. This script records total_evaluations for every
configuration so the two effects can be told apart, and additionally
reports fitness-per-1000-evaluations as a budget-normalised view.

Output:
  - results/runs/<run_id>/manual_grid.csv : one row per
    (config, target)
  - results/runs/<run_id>/manual_grid_summary.csv : mean/std per config,
    sorted best-first, including total_evaluations
  - results/runs/<run_id>/config.json

Usage:
    cd thesis-code
    python experiments/tune_hyperparams_manual.py
"""

import sys
import os
import csv
import json
import itertools
from datetime import datetime
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "..", "src")
sys.path.insert(0, SRC_DIR)

from circuit_utils import generate_target_state
from ea import evolutionary_algorithm
from fitness import fitness as fitness_fn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_QUBITS = 4

# Tuning seeds -- deliberately disjoint from the reporting seeds (42-61).
TUNING_SEEDS = [100, 101, 102, 103, 104]

ALPHA = 1.0
BETA = 0.01
MAX_GATES = 15
N_GENERATIONS = 100

# The grid. pop_size 67 and mutation_rate 0.0779 are the Optuna-tuned
# values currently used for all reported results; they are included here
# so the grid contains the incumbent as a reference point rather than
# only alternatives.
POP_SIZES = [50, 67, 100, 200]
MUTATION_RATES = [0.02, 0.05, 0.0779, 0.15, 0.30]

RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + "_manual_grid"
RUN_DIR = os.path.join(RUNS_DIR, RUN_ID)
os.makedirs(RUN_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
def run_grid():
    """
    Runs the EA once per (configuration, tuning target).

    Returns:
        A flat list of row-dicts, one per (pop_size, mutation_rate, seed).
    """
    rows = []
    configs = list(itertools.product(POP_SIZES, MUTATION_RATES))
    print(f"Grid: {len(POP_SIZES)} pop sizes x {len(MUTATION_RATES)} mutation "
          f"rates x {len(TUNING_SEEDS)} targets = "
          f"{len(configs) * len(TUNING_SEEDS)} EA runs\n")

    for cfg_i, (pop_size, mutation_rate) in enumerate(configs, start=1):
        total_evaluations = pop_size * N_GENERATIONS
        print(f"[{cfg_i}/{len(configs)}] pop_size={pop_size}, "
              f"mutation_rate={mutation_rate} "
              f"({total_evaluations} fitness evaluations per run)")

        for seed in TUNING_SEEDS:
            target = generate_target_state(N_QUBITS, seed=seed)
            np.random.seed(seed)
            result = evolutionary_algorithm(
                target, n_qubits=N_QUBITS, max_gates=MAX_GATES,
                pop_size=pop_size, n_generations=N_GENERATIONS,
                mutation_rate=mutation_rate, alpha=ALPHA, beta=BETA,
                verbose=False,
            )
            fit = fitness_fn(result["best_circuit"], target, N_QUBITS,
                             alpha=ALPHA, beta=BETA)
            rows.append({
                "pop_size": pop_size,
                "mutation_rate": mutation_rate,
                "total_evaluations": total_evaluations,
                "seed": seed,
                "fidelity": result["best_fidelity"],
                "gate_count": result["best_gate_count"],
                "fitness": fit,
            })

    return rows


def save_raw_csv(rows, path):
    fieldnames = ["pop_size", "mutation_rate", "total_evaluations", "seed",
                  "fidelity", "gate_count", "fitness"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved raw results to {path}")


def save_summary_csv(rows, path):
    """
    Aggregates per configuration and sorts best-first by mean fitness.

    Also reports fitness_per_1k_evals (mean fitness divided by total
    evaluations, scaled to 1000) so configurations can be compared on
    equal compute as well as on raw score -- see the note in the module
    docstring on why raw score alone favours larger populations.
    """
    summary = []
    configs = sorted(set((r["pop_size"], r["mutation_rate"]) for r in rows))
    for pop_size, mutation_rate in configs:
        vals = [r for r in rows
                if r["pop_size"] == pop_size and r["mutation_rate"] == mutation_rate]
        fits = [r["fitness"] for r in vals]
        fids = [r["fidelity"] for r in vals]
        gates = [r["gate_count"] for r in vals]
        total_evaluations = vals[0]["total_evaluations"]
        summary.append({
            "pop_size": pop_size,
            "mutation_rate": mutation_rate,
            "total_evaluations": total_evaluations,
            "n_targets": len(fits),
            "mean_fitness": np.mean(fits),
            "std_fitness": np.std(fits),
            "mean_fidelity": np.mean(fids),
            "mean_gate_count": np.mean(gates),
            "fitness_per_1k_evals": np.mean(fits) / total_evaluations * 1000,
        })

    summary.sort(key=lambda r: r["mean_fitness"], reverse=True)

    fieldnames = ["pop_size", "mutation_rate", "total_evaluations", "n_targets",
                  "mean_fitness", "std_fitness", "mean_fidelity",
                  "mean_gate_count", "fitness_per_1k_evals"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    print(f"Saved summary to {path}")

    print("\nTop configurations by mean fitness:")
    print(f"{'pop':>5} {'mut':>7} {'evals':>7} {'fitness':>9} {'fidelity':>9} "
          f"{'gates':>6} {'fit/1k':>8}")
    for r in summary[:10]:
        print(f"{r['pop_size']:>5} {r['mutation_rate']:>7} "
              f"{r['total_evaluations']:>7} {r['mean_fitness']:>9.4f} "
              f"{r['mean_fidelity']:>9.4f} {r['mean_gate_count']:>6.2f} "
              f"{r['fitness_per_1k_evals']:>8.4f}")

    return summary


def save_config(path):
    config = {
        "run_id": RUN_ID, "n_qubits": N_QUBITS,
        "tuning_seeds": TUNING_SEEDS, "alpha": ALPHA, "beta": BETA,
        "max_gates": MAX_GATES, "n_generations": N_GENERATIONS,
        "pop_sizes": POP_SIZES, "mutation_rates": MUTATION_RATES,
    }
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {path}")


if __name__ == "__main__":
    rows = run_grid()
    save_raw_csv(rows, os.path.join(RUN_DIR, "manual_grid.csv"))
    save_summary_csv(rows, os.path.join(RUN_DIR, "manual_grid_summary.csv"))
    save_config(os.path.join(RUN_DIR, "config.json"))
    print(f"\nDone. All outputs in: {RUN_DIR}")
"""
budget_matched_comparison.py

Controlled follow-up to the main comparison (run_experiments_repeated.py):
tests whether EA's reported fidelity advantage over SA holds when both
algorithms are given the SAME total number of fitness evaluations, rather
than each running to its own natural termination.

Motivation: EA_PARAMS (pop_size=67, n_generations=100) uses 67*100 = 6,700
fitness evaluations per run. SA_PARAMS (cooling_rate=0.9769) reaches
min_temp and terminates after only ~330-336 iterations -- roughly 20x
fewer evaluations. The main comparison does not control for this; this
script does, as a validity check on that result (not a replacement for
it -- see research_log.md and the Limitations discussion this motivates).

How SA's budget is matched: cooling_rate is set analytically (not tuned)
so that, given the already-tuned initial_temp=0.256 and min_temp=1e-4,
temperature decays to min_temp after exactly ~6,700 iterations:

    cooling_rate = (min_temp / initial_temp) ** (1 / target_evaluations)

initial_temp is deliberately left at its Optuna-tuned value -- this
experiment asks "what if SA searched for as long as EA does," not "what
is the best SA configuration for a 6,700-evaluation budget" (a related
but different question; re-tuning for the larger budget is a natural
further follow-up if this comparison suggests it would be worthwhile).

Runs on the REPORTING seeds (42-61), unlike the hyperparameter ablations
in tune_hyperparams_manual.py and crossover_rate_ablation.py, which
deliberately used the disjoint tuning seeds (100-104). The distinction:
those scripts searched over/selected among candidate hyperparameter
values (where using the reporting seeds would risk overfitting the
choice to the reported data), whereas the SA cooling_rate here is
computed analytically from a budget-matching constraint, not selected by
comparing multiple candidates against these seeds. Using the reporting
seeds instead makes this run's numbers directly comparable to the
existing headline result (run_experiments_repeated.py, research_log.md
Entry 9/12/13), which is the point of the comparison.

Also records wall-clock time per run (time.time(), same machine, same
process, sequential execution -- not a rigorously isolated benchmark, but
enough to see the order of magnitude difference between the two search
styles: EA evaluates its whole population every generation, which numpy
does not parallelise here, so wall-clock time and evaluation count should
scale together; SA's per-evaluation cost should be similar to EA's, so
wall-clock time is expected to roughly track n_evaluations for all three
conditions).

Three conditions are run so the effect of budget-matching is directly
visible against a reproducibility check on the existing result:
  - EA            : standard EA_PARAMS (pop_size=67, n_generations=100),
                    unchanged from every other reported experiment
  - SA (standard) : standard SA_PARAMS (cooling_rate=0.9769, ~330
                    evaluations) -- included as a reproducibility check;
                    should match the existing headline SA numbers
  - SA (matched)  : same SA_PARAMS except cooling_rate replaced so total
                    evaluations match EA's 6,700

Output:
  - results/runs/<run_id>/budget_matched_comparison.csv : one row per
    (condition, target, repeat)
  - results/runs/<run_id>/budget_matched_comparison_summary.csv : mean/std
    per condition, with within-/across-target decomposition
  - results/runs/<run_id>/budget_matched_comparison.png : bar chart,
    all three conditions

Usage:
    cd thesis-code
    python experiments/budget_matched_comparison.py
"""

import sys
import os
import csv
import json
import time
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
from fitness import fitness as fitness_fn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_QUBITS = 4
N_TARGETS = 20
BASE_SEED = 42            # reporting seeds -- see module docstring for why
N_REPEATS = 5

ALPHA = 1.0
BETA = 0.01

EA_PARAMS = dict(
    max_gates=15,
    pop_size=67,
    n_generations=100,
    mutation_rate=0.0779,
    alpha=ALPHA,
    beta=BETA,
    verbose=False,
)
EA_TOTAL_EVALUATIONS = EA_PARAMS["pop_size"] * EA_PARAMS["n_generations"]  # 6,700

SA_PARAMS_STANDARD = dict(
    max_gates=15,
    initial_temp=0.256,
    cooling_rate=0.9769,
    min_temp=1e-4,
    max_iterations=2000,
    alpha=ALPHA,
    beta=BETA,
    verbose=False,
)

# cooling_rate solved analytically so temperature reaches min_temp after
# EA_TOTAL_EVALUATIONS iterations; see module docstring.
_MATCHED_COOLING_RATE = (
    (SA_PARAMS_STANDARD["min_temp"] / SA_PARAMS_STANDARD["initial_temp"])
    ** (1.0 / EA_TOTAL_EVALUATIONS)
)
SA_PARAMS_MATCHED = dict(SA_PARAMS_STANDARD)
SA_PARAMS_MATCHED["cooling_rate"] = _MATCHED_COOLING_RATE
SA_PARAMS_MATCHED["max_iterations"] = EA_TOTAL_EVALUATIONS

RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + "_budget_matched"
RUN_DIR = os.path.join(RUNS_DIR, RUN_ID)
os.makedirs(RUN_DIR, exist_ok=True)

print(f"EA total evaluations per run: {EA_TOTAL_EVALUATIONS}")
print(f"SA (standard) cooling_rate: {SA_PARAMS_STANDARD['cooling_rate']} "
      f"(~330-336 evaluations per run, unequal to EA)")
print(f"SA (matched) cooling_rate: {_MATCHED_COOLING_RATE:.6f} "
      f"(~{EA_TOTAL_EVALUATIONS} evaluations per run, matching EA)\n")


# ---------------------------------------------------------------------------
def run_comparison():
    rows = []
    for i in range(N_TARGETS):
        target_seed = BASE_SEED + i
        target = generate_target_state(N_QUBITS, seed=target_seed)
        print(f"=== Target {i + 1}/{N_TARGETS} (seed={target_seed}) ===")

        for rep in range(N_REPEATS):
            repeat_seed = target_seed * 1000 + rep

            # --- EA ---
            np.random.seed(repeat_seed)
            t0 = time.time()
            ea_result = evolutionary_algorithm(target, n_qubits=N_QUBITS, **EA_PARAMS)
            ea_wall_clock = time.time() - t0
            rows.append({
                "condition": "EA", "target_idx": i, "target_seed": target_seed,
                "repeat": rep,
                "fidelity": ea_result["best_fidelity"],
                "gate_count": ea_result["best_gate_count"],
                "fitness": fitness_fn(ea_result["best_circuit"], target, N_QUBITS, ALPHA, BETA),
                "n_evaluations": len(ea_result["fitness_history"]) * EA_PARAMS["pop_size"],
                "wall_clock_seconds": ea_wall_clock,
            })

            # --- SA (standard) ---
            np.random.seed(repeat_seed)
            t0 = time.time()
            sa_std_result = simulated_annealing(target, n_qubits=N_QUBITS, **SA_PARAMS_STANDARD)
            sa_std_wall_clock = time.time() - t0
            rows.append({
                "condition": "SA_standard", "target_idx": i, "target_seed": target_seed,
                "repeat": rep,
                "fidelity": sa_std_result["best_fidelity"],
                "gate_count": sa_std_result["best_gate_count"],
                "fitness": fitness_fn(sa_std_result["best_circuit"], target, N_QUBITS, ALPHA, BETA),
                "n_evaluations": len(sa_std_result["fitness_history"]),
                "wall_clock_seconds": sa_std_wall_clock,
            })

            # --- SA (budget-matched) ---
            np.random.seed(repeat_seed)
            t0 = time.time()
            sa_matched_result = simulated_annealing(target, n_qubits=N_QUBITS, **SA_PARAMS_MATCHED)
            sa_matched_wall_clock = time.time() - t0
            rows.append({
                "condition": "SA_matched", "target_idx": i, "target_seed": target_seed,
                "repeat": rep,
                "fidelity": sa_matched_result["best_fidelity"],
                "gate_count": sa_matched_result["best_gate_count"],
                "fitness": fitness_fn(sa_matched_result["best_circuit"], target, N_QUBITS, ALPHA, BETA),
                "n_evaluations": len(sa_matched_result["fitness_history"]),
                "wall_clock_seconds": sa_matched_wall_clock,
            })

        print(f"  {N_REPEATS} repeats x 3 conditions done")

    return rows


def save_raw_csv(rows, path):
    fieldnames = ["condition", "target_idx", "target_seed", "repeat",
                  "fidelity", "gate_count", "fitness", "n_evaluations",
                  "wall_clock_seconds"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved raw results to {path}")


def save_summary_csv(rows, path):
    """
    Aggregates per condition, with the same within-/across-target std
    decomposition used elsewhere in this project (see
    sweep_beta_repeated.py), so this result is read on the same terms as
    the headline comparison it is checking.
    """
    conditions = ["EA", "SA_standard", "SA_matched"]
    summary = []
    for cond in conditions:
        vals = [r for r in rows if r["condition"] == cond]
        fids = [r["fidelity"] for r in vals]
        fits = [r["fitness"] for r in vals]
        gates = [r["gate_count"] for r in vals]
        n_evals = vals[0]["n_evaluations"]
        wall_clocks = [r["wall_clock_seconds"] for r in vals]

        target_idxs = sorted(set(r["target_idx"] for r in vals))
        per_target = {t: [r["fidelity"] for r in vals if r["target_idx"] == t]
                      for t in target_idxs}
        within_stds = [np.std(v) for v in per_target.values() if len(v) > 1]
        within_target_std = float(np.mean(within_stds)) if within_stds else float("nan")
        target_means = [np.mean(v) for v in per_target.values()]
        across_target_std = float(np.std(target_means))

        summary.append({
            "condition": cond, "n_evaluations_per_run": n_evals,
            "n_samples": len(fids),
            "mean_fidelity": np.mean(fids), "std_fidelity": np.std(fids),
            "within_target_std": within_target_std,
            "across_target_std": across_target_std,
            "mean_fitness": np.mean(fits),
            "mean_gate_count": np.mean(gates),
            "mean_wall_clock_seconds": np.mean(wall_clocks),
            "std_wall_clock_seconds": np.std(wall_clocks),
        })

    fieldnames = ["condition", "n_evaluations_per_run", "n_samples",
                  "mean_fidelity", "std_fidelity", "within_target_std",
                  "across_target_std", "mean_fitness", "mean_gate_count",
                  "mean_wall_clock_seconds", "std_wall_clock_seconds"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    print(f"Saved summary to {path}\n")

    print("=== Summary ===")
    for r in summary:
        print(f"{r['condition']:12s} (n_evals={r['n_evaluations_per_run']:>5.0f}): "
              f"fidelity = {r['mean_fidelity']:.4f} +/- {r['within_target_std']:.4f} "
              f"(within-target), gates = {r['mean_gate_count']:.2f}, "
              f"wall-clock = {r['mean_wall_clock_seconds']:.2f}s "
              f"+/- {r['std_wall_clock_seconds']:.2f}s per run")

    ea = next(r for r in summary if r["condition"] == "EA")
    sa_std = next(r for r in summary if r["condition"] == "SA_standard")
    sa_matched = next(r for r in summary if r["condition"] == "SA_matched")

    gap_standard = ea["mean_fidelity"] - sa_std["mean_fidelity"]
    gap_matched = ea["mean_fidelity"] - sa_matched["mean_fidelity"]
    print(f"\nEA vs SA (standard, unequal budget): gap = {gap_standard:.4f}")
    print(f"EA vs SA (budget-matched):            gap = {gap_matched:.4f}")

    pooled_se = np.sqrt(ea["within_target_std"]**2 / ea["n_samples"]
                         + sa_matched["within_target_std"]**2 / sa_matched["n_samples"])
    ratio = abs(gap_matched) / pooled_se if pooled_se > 0 else float("inf")
    if gap_matched < 0:
        print(f">>> SA outperforms EA once budget is equalised "
              f"(gap/SE = {ratio:.2f}, {'likely real' if ratio > 2 else 'within noise'}).")
    else:
        print(f">>> EA still ahead under equal budget "
              f"(gap/SE = {ratio:.2f}, {'likely real' if ratio > 2 else 'within noise'}).")

    return summary


def save_config(path):
    config = {
        "run_id": RUN_ID, "n_qubits": N_QUBITS, "n_targets": N_TARGETS,
        "base_seed": BASE_SEED, "n_repeats": N_REPEATS, "alpha": ALPHA, "beta": BETA,
        "ea_params": EA_PARAMS, "ea_total_evaluations": EA_TOTAL_EVALUATIONS,
        "sa_params_standard": SA_PARAMS_STANDARD,
        "sa_params_matched": SA_PARAMS_MATCHED,
    }
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {path}")


def plot_comparison(summary, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    labels = ["EA\n(6,700 evals)", "SA standard\n(~330 evals)", "SA matched\n(6,700 evals)"]
    colors = ["tab:blue", "tab:orange", "tab:red"]

    panels = (("mean_fidelity", "Mean fidelity", "within_target_std"),
              ("mean_fitness", "Mean fitness", None),
              ("mean_wall_clock_seconds", "Mean wall-clock time (s)", "std_wall_clock_seconds"))

    for ax, (metric, title, err_key) in zip(axes, panels):
        means = [r[metric] for r in summary]
        errs = [r[err_key] for r in summary] if err_key else [0, 0, 0]
        ax.bar(labels, means, yerr=errs, capsize=6, color=colors)
        ax.set_ylabel(title)
        ax.set_title(f"{title} ({N_QUBITS} qubits)")
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot to {path}")


if __name__ == "__main__":
    rows = run_comparison()
    save_raw_csv(rows, os.path.join(RUN_DIR, "budget_matched_comparison.csv"))
    summary_path = os.path.join(RUN_DIR, "budget_matched_comparison_summary.csv")
    summary = save_summary_csv(rows, summary_path)
    save_config(os.path.join(RUN_DIR, "config.json"))
    plot_path = os.path.join(RUN_DIR, "budget_matched_comparison.png")
    plot_comparison(summary, plot_path)

    # Also copy summary + plot to results/figures/ (unlike results/runs/,
    # not gitignored) so this experiment's results are actually committed
    # to the repository, not just reproducible locally -- see README.md's
    # reproducibility principle. The large raw per-run CSV
    # (budget_matched_comparison.csv) is purely regenerable, so it is left
    # in results/runs/ only.
    shutil.copy(summary_path, os.path.join(FIGURES_DIR, "budget_matched_comparison_summary.csv"))
    shutil.copy(plot_path, os.path.join(FIGURES_DIR, "budget_matched_comparison.png"))
    print(f"Also copied summary and plot to {FIGURES_DIR}")

    print(f"\nDone. All outputs in: {RUN_DIR}")
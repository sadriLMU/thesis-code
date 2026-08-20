"""
gate_budget_ablation.py

Follow-up to the main comparison and Section 5.1 (RQ1) discussion, which
attributes EA/SA's fidelity ceiling (well below 1.0) to the circuit-length
budget used throughout this thesis (max_gates=15). This script tests that
claim directly with a 2x2 factorial design on EA, isolating the effect of
max_gates from the effect of the gate-count penalty beta, since both could
plausibly limit achievable fidelity and the main comparison never varies
either one independently of the other.

Conditions (max_gates, beta):
  - (15, 0.01)   : baseline, identical to the main comparison
                    (run_experiments_repeated.py)
  - (40, 0.01)   : isolates the effect of raising max_gates alone
  - (15, 0.001)  : isolates the effect of lowering beta alone
  - (40, 0.001)  : both loosened together

All other EA hyperparameters (pop_size, n_generations, mutation_rate) are
held at their Optuna-tuned values (Section 4.3), unchanged from the main
comparison, so the total evaluation budget (6,700) stays constant across
all four conditions -- only the circuit-length ceiling and the gate-count
penalty vary. Runs on the REPORTING seeds (42-61), for direct
comparability to the main comparison's reported baseline fidelity.

Output:
  - results/runs/<run_id>/gate_budget_ablation.csv : one row per
    (condition, target, repeat)
  - results/runs/<run_id>/gate_budget_ablation_summary.csv : mean/std
    per condition, with within-target decomposition
  - results/figures/gate_budget_ablation_summary.csv : copy, for citing
    in the thesis

Usage:
    cd thesis-code
    python experiments/gate_budget_ablation.py
"""

import sys, os, csv, json, time, shutil
from datetime import datetime
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "..", "src")
sys.path.insert(0, SRC_DIR)

from circuit_utils import generate_target_state
from ea import evolutionary_algorithm

# ---------------------------------------------------------------------------
N_QUBITS = 4
N_TARGETS = 20
BASE_SEED = 42          # reporting seeds -- for direct comparability to
                         # the main comparison's baseline fidelity
N_REPEATS = 5

ALPHA = 1.0
EA_BASE = dict(pop_size=67, n_generations=100, mutation_rate=0.0779,
               alpha=ALPHA, verbose=False)

# (condition name, max_gates, beta)
CONDITIONS = [
    ("baseline_15_001", 15, 0.01),    # identical to main comparison
    ("maxgates40_001",  40, 0.01),    # isolates max_gates
    ("beta0001_15",     15, 0.001),   # isolates beta
    ("both_loosened",   40, 0.001),   # both loosened together
]

RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + "_gate_budget_ablation"
RUN_DIR = os.path.join(RUNS_DIR, RUN_ID)
os.makedirs(RUN_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
def run_ablation():
    rows = []
    for i in range(N_TARGETS):
        target_seed = BASE_SEED + i
        target = generate_target_state(N_QUBITS, seed=target_seed)
        print(f"=== Target {i + 1}/{N_TARGETS} (seed={target_seed}) ===")

        for rep in range(N_REPEATS):
            repeat_seed = target_seed * 1000 + rep

            for cond_name, max_gates, beta in CONDITIONS:
                np.random.seed(repeat_seed)
                t0 = time.time()
                result = evolutionary_algorithm(
                    target, n_qubits=N_QUBITS, max_gates=max_gates,
                    beta=beta, **EA_BASE)
                wall_clock = time.time() - t0
                rows.append({
                    "condition": cond_name, "max_gates": max_gates, "beta": beta,
                    "target_idx": i, "target_seed": target_seed, "repeat": rep,
                    "fidelity": result["best_fidelity"],
                    "gate_count": result["best_gate_count"],
                    "wall_clock_seconds": wall_clock,
                })

        print(f"  {N_REPEATS} repeats x {len(CONDITIONS)} conditions done")

    return rows


def save_raw_csv(rows, path):
    fieldnames = ["condition", "max_gates", "beta", "target_idx", "target_seed",
                  "repeat", "fidelity", "gate_count", "wall_clock_seconds"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved raw results to {path}")


def save_summary_csv(rows, path):
    summary = []
    for cond_name, max_gates, beta in CONDITIONS:
        vals = [r for r in rows if r["condition"] == cond_name]
        fids = [r["fidelity"] for r in vals]
        gates = [r["gate_count"] for r in vals]

        target_idxs = sorted(set(r["target_idx"] for r in vals))
        per_target = {t: [r["fidelity"] for r in vals if r["target_idx"] == t]
                      for t in target_idxs}
        within_stds = [np.std(v) for v in per_target.values() if len(v) > 1]
        within_target_std = float(np.mean(within_stds)) if within_stds else float("nan")

        summary.append({
            "condition": cond_name, "max_gates": max_gates, "beta": beta,
            "n_samples": len(fids), "mean_fidelity": np.mean(fids),
            "within_target_std": within_target_std,
            "mean_gate_count": np.mean(gates), "max_gate_count_used": max(gates),
        })

    fieldnames = ["condition", "max_gates", "beta", "n_samples", "mean_fidelity",
                  "within_target_std", "mean_gate_count", "max_gate_count_used"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    print(f"Saved summary to {path}\n")

    print("=== Summary ===")
    for r in summary:
        print(f"{r['condition']:20s} (max_gates={r['max_gates']:3d}, beta={r['beta']}): "
              f"fidelity={r['mean_fidelity']:.4f} +/- {r['within_target_std']:.4f}, "
              f"gates={r['mean_gate_count']:.2f} (max used: {r['max_gate_count_used']})")

    base = next(r for r in summary if r["condition"] == "baseline_15_001")
    print()
    for r in summary:
        if r["condition"] == "baseline_15_001":
            continue
        gap = r["mean_fidelity"] - base["mean_fidelity"]
        se = np.sqrt(base["within_target_std"]**2 / base["n_samples"]
                     + r["within_target_std"]**2 / r["n_samples"])
        ratio = abs(gap) / se if se > 0 else float("inf")
        print(f"{r['condition']} vs baseline: gap={gap:.4f}, gap/SE={ratio:.2f} "
              f"({'likely real' if ratio > 2 else 'within noise'})")

    return summary


def save_config(path):
    config = {
        "run_id": RUN_ID, "n_qubits": N_QUBITS, "n_targets": N_TARGETS,
        "base_seed": BASE_SEED, "n_repeats": N_REPEATS, "alpha": ALPHA,
        "ea_base_params": EA_BASE, "conditions": CONDITIONS,
    }
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {path}")


if __name__ == "__main__":
    rows = run_ablation()
    raw_path = os.path.join(RUN_DIR, "gate_budget_ablation.csv")
    save_raw_csv(rows, raw_path)
    summary_path = os.path.join(RUN_DIR, "gate_budget_ablation_summary.csv")
    save_summary_csv(rows, summary_path)
    save_config(os.path.join(RUN_DIR, "config.json"))

    # Copy summary to results/figures/ (not gitignored, unlike results/runs/)
    # so this experiment's results are committed, same convention as
    # budget_matched_comparison.py.
    shutil.copy(summary_path, os.path.join(FIGURES_DIR, "gate_budget_ablation_summary.csv"))
    print(f"Also copied summary to {FIGURES_DIR}")

    print(f"\nDone. All outputs in: {RUN_DIR}")
# Investigation of Metaheuristic Algorithms for Circuit Synthesis in Quantum Computing

Bachelor's thesis code repository. Compares an Evolutionary Algorithm (EA)
and Simulated Annealing (SA) for synthesizing quantum circuits that prepare
Haar-random target states on 4 qubits.

**Author:** Sadri Oueslati
**Supervisor:** Prof. Dr. Claudia Linnhoff-Popien
**Advisor:** Leo Sünkel
**Institution:** LMU Munich

## Overview

Both algorithms search for a short quantum circuit whose output state
approximates a given Haar-random target state, starting from a fully
unstructured random circuit, under a shared fitness function:

```
fitness = alpha * fidelity - beta * gate_count
```

See the thesis PDF for full background, methodology, and results.

## Project Structure

```
thesis-code/
├── src/                        Core library code
│   ├── circuit_utils.py        Haar-random state generation, gate
│   │                           representation, circuit construction
│   ├── fitness.py              Fitness functions (standard + fidelity-floor
│   │                           variant)
│   ├── ea.py                   Evolutionary Algorithm implementation
│   └── sa.py                   Simulated Annealing implementation
├── experiments/                Scripts that run experiments using src/
│   ├── run_experiments.py      Main EA vs. SA comparison (single run/target)
│   ├── run_experiments_repeated.py
│   │                           Same, with multiple repeats per target for
│   │                           statistical robustness
│   ├── sweep_beta.py           Beta (gate-count penalty) sensitivity sweep
│   ├── sweep_beta_repeated.py  Same, with multiple repeats per (beta, target)
│   ├── sweep_beta_floor.py     Compares standard vs. fidelity-floor fitness
│   │                           across beta values (single run)
│   ├── sweep_beta_floor_repeated.py
│   │                           Same, with repeats
│   ├── tune_hyperparams.py     Optuna-based automatic hyperparameter tuning
│   ├── plot_circuits.py        Exports circuit diagrams. Best-circuit
│   │                           examples are picked by scanning all 20
│   │                           target seeds and choosing the one closest
│   │                           to the mean gate count, rather than a
│   │                           fixed seed. Also exports a crossover-trace
│   │                           diagram (ea_crossover_trace.png) showing
│   │                           the gate-list split point directly, since
│   │                           Qiskit's per-qubit circuit diagrams alone
│   │                           cannot show this. Writes suggested LaTeX
│   │                           captions to results/figures/CAPTIONS.md.
│   ├── verify_crossover.py     Prints the exact gate-list origin for a
│   │                           crossover example, to verify correctness
│   │                           independently of the circuit diagram
│   ├── plot_results_evolution.py
│   │                           Supporting chart showing how results changed
│   │                           across development stages. Writes to
│   │                           results/figures_debug/, not results/figures/,
│   │                           since it is explicitly not part of the
│   │                           thesis body (internal reference only).
│   ├── sweep_beta_extended_quick.py / sweep_beta_extended_full.py
│   │                           Extends the beta sweep beyond the original
│   │                           0.01-0.07 range up to 0.30, checking whether
│   │                           the EA/SA gap ever reverses at higher beta
│   │                           (it does not; both algorithms converge
│   │                           toward a near-trivial ~1-gate solution
│   │                           instead). _quick uses fewer repeats for a
│   │                           fast first look; _full matches the main
│   │                           sweep's statistical rigor.
│   ├── tune_hyperparams_manual.py
│   │                           Manual grid search over pop_size and
│   │                           mutation_rate (literature-typical values,
│   │                           e.g. population 200), complementing the
│   │                           Optuna search. Reports fitness-per-1000-
│   │                           evaluations alongside raw fitness, since
│   │                           larger populations get proportionally more
│   │                           compute and would otherwise look better
│   │                           purely from that.
│   ├── crossover_rate_ablation.py
│   │                           Checks EA's crossover_rate=1.0 (current
│   │                           default) against 0.8 (literature-typical,
│   │                           cf. Sünkel et al. 2025's 0.85) on the
│   │                           tuning seeds. No significant difference
│   │                           found; current default retained.
│   └── budget_matched_comparison.py
│                               Controlled follow-up to the main comparison:
│                               EA and SA are given the SAME total number of
│                               fitness evaluations (SA's cooling_rate is
│                               solved analytically so it no longer
│                               terminates early), rather than each running
│                               to its own natural termination. Also
│                               records wall-clock time. Central finding:
│                               SA significantly outperforms EA once
│                               evaluation budget is equalised -- see
│                               research_log.md Entry 17.
├── results/
│   ├── runs/                   Output of each experiment run (gitignored --
│   │                           regenerate by re-running the scripts)
│   ├── archive/                Pre-bugfix experiment outputs, kept for
│   │                           reference (see research_log.md)
│   ├── figures/                Exported plots/diagrams used in the thesis
│   ├── figures_debug/          Supporting/debug charts, not used in the
│   │                           thesis (gitignored, regenerate as needed)
│   └── optuna_studies/         Persistent Optuna tuning databases
├── test_ea.py, test_sa.py      Quick manual sanity-check scripts
├── research_log.md             Dated log of every experiment: what was run,
│                               why, results, and interpretation. Documents
│                               the full development history including bugs
│                               found and fixed.
└── requirements.txt            Exact pinned dependency versions
```

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running Experiments

All scripts are run from the project root:

```powershell
# Main EA vs. SA comparison (fast, single run per target)
python experiments/run_experiments.py

# Statistically robust version (multiple repeats per target -- used for
# the thesis's reported main comparison)
python experiments/run_experiments_repeated.py

# Beta sensitivity sweep
python experiments/sweep_beta.py
python experiments/sweep_beta_repeated.py   # repeated version, used for reported results

# Hyperparameter tuning (takes considerable time -- 30 trials per algorithm)
python experiments/tune_hyperparams.py

# Fidelity-floor fitness variant comparison
python experiments/sweep_beta_floor.py
python experiments/sweep_beta_floor_repeated.py

# Circuit diagrams for the thesis (scans all 20 target seeds to pick a
# representative example; takes ~2 minutes)
python experiments/plot_circuits.py
python experiments/verify_crossover.py

# Extended beta sweep (checks for an EA/SA crossover beyond beta=0.07)
python experiments/sweep_beta_extended_quick.py   # fast, fewer repeats
python experiments/sweep_beta_extended_full.py    # full statistical rigor

# Manual hyperparameter ablations, complementing tune_hyperparams.py
python experiments/tune_hyperparams_manual.py     # pop_size, mutation_rate grid
python experiments/crossover_rate_ablation.py     # crossover_rate 1.0 vs 0.8

# Budget-matched EA/SA comparison (controls for fitness-evaluation count
# and records wall-clock time -- see research_log.md Entry 17)
python experiments/budget_matched_comparison.py
```

Each experiment script creates its own timestamped output folder under
`results/runs/`, containing raw CSV data, a JSON config snapshot of every
parameter used, and generated plots -- so every reported number is
reproducible and traceable back to its exact configuration.

## Key Results Summary

(See the thesis PDF, Chapter 5, for full results and Chapter 6 for
discussion.)

- The evolutionary algorithm achieves consistently higher fidelity and
  fidelity-per-gate than simulated annealing, confirmed via repeated runs
  (5-8 repeats per target across 20 target states, 100-160 runs per
  algorithm), validated both before and after a precision fix to EA's
  mutation length-check (see `research_log.md` Entry 12).
- This advantage holds across the entire tested range of the gate-count
  penalty weight (beta), also confirmed via repeated runs
  (`sweep_beta_repeated.py`). Decomposing run-to-run noise from
  target-to-target variation shows the EA/SA gap exceeds run-to-run noise
  at every tested beta except the highest, where the margin narrows
  alongside the mean gap itself (see `research_log.md` Entry 13).
- An additional fitness-function variant with a hard minimum-fidelity
  constraint (following Sünkel et al.'s QCO fitness design) reliably
  prevents EA's fidelity collapse at high beta, at the cost of roughly
  3-4x more gates. The same constraint makes SA's outcomes highly
  unpredictable (bimodal) rather than moderately worse, plausibly due to
  SA's single-trajectory search lacking the redundancy of EA's
  population-based search. This finding is also confirmed via repeated
  runs, both before and after the EA precision fix; a more precise
  within-target analysis puts SA's floor-induced instability at roughly
  2-3x its standard-fitness run-to-run variance across most of the beta
  range tested, rising sharply at the highest beta value (see
  `research_log.md` Entry 13).

## Reproducibility Notes

- All target states are generated with fixed random seeds (see each
  script's configuration section), so every reported result can be
  reproduced exactly.
- Hyperparameters were tuned via Optuna on target states (seeds 100-104)
  disjoint from those used in the reported comparisons (seeds 42-61), to
  avoid tuning/reporting bias.
- `research_log.md` documents three implementation issues found and fixed
  during development: two bugs in `sa.py` (Entries 1-3) and one precision
  issue in `ea.py`'s mutation length-check (Entry 12), each with a
  before/after comparison. The `ea.py` fix was confirmed result-neutral;
  all affected experiments were nonetheless fully re-run and re-validated
  with the corrected code (see the "UPDATE" sections in Entries 7-11).
  Pre-fix run data is preserved in `results/archive/` for reference, not
  used as reported results.
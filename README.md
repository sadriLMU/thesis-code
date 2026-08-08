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
│   ├── sweep_beta_floor.py     Compares standard vs. fidelity-floor fitness
│   │                           across beta values (single run)
│   ├── sweep_beta_floor_repeated.py
│   │                           Same, with repeats
│   ├── tune_hyperparams.py     Optuna-based automatic hyperparameter tuning
│   ├── plot_circuits.py        Exports circuit diagrams (best circuits,
│   │                           SA neighbor example, EA crossover example)
│   ├── verify_crossover.py     Prints the exact gate-list origin for a
│   │                           crossover example, to verify correctness
│   │                           independently of the circuit diagram
│   └── plot_results_evolution.py
│                               Supporting chart showing how results changed
│                               across development stages (not part of the
│                               thesis body; for internal reference)
├── results/
│   ├── runs/                   Output of each experiment run (gitignored --
│   │                           regenerate by re-running the scripts)
│   ├── archive/                Pre-bugfix experiment outputs, kept for
│   │                           reference (see research_log.md)
│   ├── figures/                Exported plots/diagrams used in the thesis
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

# Hyperparameter tuning (takes considerable time -- 30 trials per algorithm)
python experiments/tune_hyperparams.py

# Fidelity-floor fitness variant comparison
python experiments/sweep_beta_floor.py
python experiments/sweep_beta_floor_repeated.py

# Circuit diagrams for the thesis
python experiments/plot_circuits.py
python experiments/verify_crossover.py
```

Each experiment script creates its own timestamped output folder under
`results/runs/`, containing raw CSV data, a JSON config snapshot of every
parameter used, and generated plots -- so every reported number is
reproducible and traceable back to its exact configuration.

## Key Results Summary

(See the thesis PDF, Chapter 5, for full results and Chapter 6 for
discussion.)

- The evolutionary algorithm achieves consistently higher fidelity and
  fidelity-per-gate than simulated annealing, confirmed via 5 repeated runs
  per target across 20 target states (100 runs per algorithm).
- This advantage holds across the tested range of the gate-count penalty
  weight (beta).
- An additional fitness-function variant with a hard minimum-fidelity
  constraint (following Sünkel et al.'s QCO fitness design) reliably
  prevents EA's fidelity collapse at high beta, at the cost of roughly
  2-3x more gates. The same constraint makes SA's outcomes highly
  unpredictable rather than moderately worse, plausibly due to SA's
  single-trajectory search lacking the redundancy of EA's population-based
  search.

## Reproducibility Notes

- All target states are generated with fixed random seeds (see each
  script's configuration section), so every reported result can be
  reproduced exactly.
- Hyperparameters were tuned via Optuna on target states (seeds 100-104)
  disjoint from those used in the reported comparisons (seeds 42-61), to
  avoid tuning/reporting bias.
- `research_log.md` documents two implementation bugs found and fixed
  during development (see Entries 1-3), along with a before/after
  comparison. All results reported in the thesis reflect the corrected
  implementation.
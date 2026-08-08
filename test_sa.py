"""
test_sa.py

Quick manual sanity check: runs SA once with default-ish parameters and
prints the result. Not a formal test suite (no assertions, no pytest) --
just a fast way to confirm the SA pipeline runs end-to-end without errors
after making code changes. For the actual reported results, see
experiments/run_experiments*.py instead.

Note: uses default initial_temp/cooling_rate (not the Optuna-tuned values
used in the reported experiments) -- this script only checks that the
code runs, not that it reproduces any specific reported number.

Usage:
    cd thesis-code
    python test_sa.py
"""

import sys
sys.path.insert(0, 'src')

from circuit_utils import generate_target_state
from sa import simulated_annealing

target = generate_target_state(4, seed=42)
result = simulated_annealing(target, n_qubits=4, max_gates=15, max_iterations=2000, verbose=True)

print('Final Fidelity:', round(result['best_fidelity'], 4))
print('Gate Count:', result['best_gate_count'])
"""
test_ea.py

Quick manual sanity check: runs EA once with default-ish parameters and
prints the result. Not a formal test suite (no assertions, no pytest) --
just a fast way to confirm the EA pipeline runs end-to-end without errors
after making code changes. For the actual reported results, see
experiments/run_experiments*.py instead.

Usage:
    cd thesis-code
    python test_ea.py
"""

import sys
sys.path.insert(0, 'src')

from circuit_utils import generate_target_state
from ea import evolutionary_algorithm

target = generate_target_state(4, seed=42)
result = evolutionary_algorithm(target, n_qubits=4, max_gates=15, pop_size=30, n_generations=100, verbose=True)

print('Final Fidelity:', round(result['best_fidelity'], 4))
print('Gate Count:', result['best_gate_count'])
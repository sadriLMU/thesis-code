import sys
sys.path.insert(0, 'src')

from circuit_utils import generate_target_state
from ea import evolutionary_algorithm

target = generate_target_state(4, seed=42)
result = evolutionary_algorithm(target, n_qubits=4, n_gates=15, pop_size=30, n_generations=100, verbose=True)

print('Final Fidelity:', round(result['best_fidelity'], 4))
print('Gate Count:', result['best_gate_count'])
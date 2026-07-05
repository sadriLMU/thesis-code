import sys
sys.path.insert(0, 'src')

from circuit_utils import generate_target_state
from sa import simulated_annealing

target = generate_target_state(4, seed=42)
result = simulated_annealing(target, n_qubits=4, n_gates=20, max_iterations=2000, verbose=True)

print('Final Fidelity:', round(result['best_fidelity'], 4))
print('Gate Count:', result['best_gate_count'])
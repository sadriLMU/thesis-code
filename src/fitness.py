import numpy as np
from qiskit.quantum_info import Statevector, state_fidelity
from circuit_utils import build_circuit


def compute_fidelity(gates: list, target_state: Statevector, n_qubits: int) -> float:
    """
    Compute the fidelity between the circuit's output state and the target state.
    F = |<psi_target|psi_circuit>|^2
    Returns a value between 0 (worst) and 1 (perfect).
    """
    qc = build_circuit(n_qubits, gates)
    circuit_state = Statevector.from_instruction(qc)
    return state_fidelity(circuit_state, target_state)


def compute_gate_count(gates: list) -> int:
    """Return the number of gates in the circuit."""
    return len(gates)


def fitness(gates: list, target_state: Statevector, n_qubits: int,
            alpha: float = 1.0, beta: float = 0.01) -> float:
    """
    Combined fitness function.
    
    fitness = alpha * fidelity - beta * gate_count
    
    alpha: weight for fidelity (primary metric)
    beta:  penalty for gate count (secondary metric)
    
    Higher is better.
    """
    f = compute_fidelity(gates, target_state, n_qubits)
    g = compute_gate_count(gates)
    return alpha * f - beta * g
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

    Note: this is a smooth trade-off with no lower bound on fidelity --
    at high beta, the algorithm can sacrifice fidelity arbitrarily far to
    save gates. See fitness_with_floor() for a variant that prevents this.
    """
    f = compute_fidelity(gates, target_state, n_qubits)
    g = compute_gate_count(gates)
    return alpha * f - beta * g


def fitness_with_floor(gates: list, target_state: Statevector, n_qubits: int,
                        alpha: float = 1.0, beta: float = 0.01,
                        min_fidelity: float = 0.3,
                        floor_penalty: float = -100.0) -> float:
    """
    Fitness function with a hard minimum-fidelity constraint, following the
    same pattern used in Suenkel et al.'s QCO fitness (see related_work.tex):
    circuits below a minimum fidelity are penalized heavily and uniformly,
    regardless of gate count, so the gate-count penalty (beta) can never
    incentivize sacrificing fidelity below an acceptable floor.

    fitness = floor_penalty                          if fidelity < min_fidelity
            = alpha * fidelity - beta * gate_count    otherwise

    min_fidelity:  the minimum acceptable fidelity; circuits below this are
                   treated as unacceptable regardless of length.
    floor_penalty: the fixed score assigned to any circuit that violates the
                   floor. Should be low enough that no valid circuit (even a
                   very long, very low-scoring one) could ever score lower,
                   so the search never prefers an invalid circuit over a
                   valid one.

    Higher is better.
    """
    f = compute_fidelity(gates, target_state, n_qubits)
    if f < min_fidelity:
        return floor_penalty
    g = compute_gate_count(gates)
    return alpha * f - beta * g
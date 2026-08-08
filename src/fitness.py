"""
fitness.py

Defines the fitness function(s) that guide both EA and SA's search
(see ea.py, sa.py). Both algorithms are given the identical fitness
function by default; the only difference between them is *how* they
search the space of possible circuits to maximise it (see thesis
Section 4.2, "Fitness Function").

Two variants are provided:
  fitness()            -- the primary fitness function used throughout
                           this thesis (Chapters 4-5).
  fitness_with_floor()  -- an additional variant with a hard minimum-
                           fidelity constraint, following Suenkel et al.'s
                           QCO fitness design (see related_work.tex).
                           Used only in sweep_beta_floor(_repeated).py;
                           does not affect any other reported result.

Used by:
  - ea.py, sa.py     : via the optional fitness_fn parameter, or the
                       default fitness() if none is given
  - experiments/*.py : all experiment scripts import fitness and/or
                       fitness_with_floor directly
"""

from qiskit.quantum_info import Statevector, state_fidelity
from circuit_utils import build_circuit


def compute_fidelity(gates: list, target_state: Statevector, n_qubits: int) -> float:
    """
    Compute the fidelity between a circuit's output state and the target.

    F = |<psi_target|psi_circuit>|^2  (thesis Eq. in Section 2.2.3)

    Args:
        gates: Ordered list of gate dictionaries (see circuit_utils.random_gate).
        target_state: The target Statevector to compare against.
        n_qubits: Number of qubits in the circuit.

    Returns:
        Fidelity in [0, 1], where 1 means the states are identical and 0
        means they are orthogonal.
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
    Primary fitness function used throughout this thesis.

    fitness = alpha * fidelity - beta * gate_count

    Higher is better. alpha weights fidelity (primary objective); beta
    penalises gate count (secondary objective, see thesis Section 5.3
    for the beta sensitivity analysis).

    Note: this is a smooth trade-off with no lower bound on fidelity --
    at high beta, the search can sacrifice fidelity arbitrarily far to
    save gates (see thesis Section 5.4 / research_log.md Entry 10-11 for
    the empirical consequence of this, and fitness_with_floor() for a
    variant that prevents it).

    Args:
        gates: Ordered list of gate dictionaries.
        target_state: The target Statevector.
        n_qubits: Number of qubits in the circuit.
        alpha: Weight for fidelity.
        beta: Weight for the gate-count penalty.

    Returns:
        The fitness score (unbounded above and below).
    """
    f = compute_fidelity(gates, target_state, n_qubits)
    g = compute_gate_count(gates)
    return alpha * f - beta * g


def fitness_with_floor(gates: list, target_state: Statevector, n_qubits: int,
                        alpha: float = 1.0, beta: float = 0.01,
                        min_fidelity: float = 0.3,
                        floor_penalty: float = -100.0) -> float:
    """
    Fitness function with a hard minimum-fidelity constraint.

    Follows the same pattern as Suenkel et al.'s QCO fitness (see
    related_work.tex): circuits below a minimum fidelity are penalised
    heavily and uniformly, regardless of gate count, so beta can never
    incentivise sacrificing fidelity below an acceptable floor.

        fitness = floor_penalty                        if fidelity < min_fidelity
                = alpha * fidelity - beta * gate_count  otherwise

    Higher is better. Not used in the main reported comparison (see
    fitness()); used only for the ablation study in
    sweep_beta_floor(_repeated).py (research_log.md Entries 10-11).

    Args:
        gates: Ordered list of gate dictionaries.
        target_state: The target Statevector.
        n_qubits: Number of qubits in the circuit.
        alpha: Weight for fidelity (only applies above the floor).
        beta: Weight for the gate-count penalty (only applies above the
            floor).
        min_fidelity: Minimum acceptable fidelity; circuits below this
            are treated as unacceptable regardless of length.
        floor_penalty: Fixed score assigned to any circuit violating the
            floor. Must be low enough that no valid (above-floor) circuit
            could ever score lower, so the search never prefers an
            invalid circuit over a valid one.

    Returns:
        The fitness score: floor_penalty if below the floor, otherwise
        the same value as fitness() would return.
    """
    f = compute_fidelity(gates, target_state, n_qubits)
    if f < min_fidelity:
        return floor_penalty
    g = compute_gate_count(gates)
    return alpha * f - beta * g
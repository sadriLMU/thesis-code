import numpy as np
from qiskit.quantum_info import Statevector
from circuit_utils import random_circuit, random_gate
from fitness import fitness, compute_fidelity, compute_gate_count


def neighbor(circuit: list, n_qubits: int) -> list:
    """
    Generate a neighboring circuit by applying one of three operations:
    - Replace a random gate
    - Insert a random gate at a random position
    - Delete a random gate
    """
    new_circuit = circuit[:]
    if len(new_circuit) == 0:
        new_circuit.append(random_gate(n_qubits))
        return new_circuit

    operation = np.random.choice(['replace', 'insert', 'delete'],
                                  p=[0.5, 0.25, 0.25])

    if operation == 'replace':
        idx = np.random.randint(len(new_circuit))
        new_circuit[idx] = random_gate(n_qubits)

    elif operation == 'insert':
        idx = np.random.randint(len(new_circuit) + 1)
        new_circuit.insert(idx, random_gate(n_qubits))

    elif operation == 'delete' and len(new_circuit) > 1:
        idx = np.random.randint(len(new_circuit))
        new_circuit.pop(idx)

    return new_circuit


def acceptance_probability(current_score: float, new_score: float,
                            temperature: float) -> float:
    """
    Metropolis acceptance criterion.
    - Always accept better solutions
    - Accept worse solutions with probability exp((new - current) / T)
    """
    if new_score > current_score:
        return 1.0
    return np.exp((new_score - current_score) / temperature)


def simulated_annealing(
    target_state: Statevector,
    n_qubits: int,
    n_gates: int = 20,
    initial_temp: float = 1.0,
    cooling_rate: float = 0.995,
    min_temp: float = 1e-4,
    max_iterations: int = 5000,
    alpha: float = 1.0,
    beta: float = 0.01,
    verbose: bool = True
) -> dict:
    """
    Run Simulated Annealing for quantum circuit synthesis.

    Returns a dict with:
        - best_circuit: list of gates
        - best_fidelity: float
        - best_gate_count: int
        - history: list of best fidelity per iteration
    """
    # Initialize with a random circuit
    current_circuit = random_circuit(n_qubits, n_gates)
    current_score = fitness(current_circuit, target_state, n_qubits, alpha, beta)

    best_circuit = current_circuit[:]
    best_score = current_score

    temperature = initial_temp
    history = []
    iteration = 0

    while temperature > min_temp and iteration < max_iterations:
        # Generate neighbor
        new_circuit = neighbor(current_circuit, n_qubits)
        new_score = fitness(new_circuit, target_state, n_qubits, alpha, beta)

        # Acceptance
        ap = acceptance_probability(current_score, new_score, temperature)
        if np.random.random() < ap:
            current_circuit = new_circuit
            current_score = new_score

        # Update best
        if current_score > best_score:
            best_circuit = current_circuit[:]
            best_score = current_score

        # Track best fidelity
        history.append(best_score)

        # Cool down
        temperature *= cooling_rate
        iteration += 1

        if verbose and iteration % 500 == 0:
            f = compute_fidelity(best_circuit, target_state, n_qubits)
            print(f"Iter {iteration:5d} | Temp: {temperature:.5f} | "
                  f"Best fitness: {best_score:.4f} | Fidelity: {f:.4f}")

    return {
        'best_circuit': best_circuit,
        'best_fidelity': compute_fidelity(best_circuit, target_state, n_qubits),
        'best_gate_count': compute_gate_count(best_circuit),
        'history': history
    }
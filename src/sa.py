import numpy as np
from qiskit.quantum_info import Statevector
from circuit_utils import random_circuit, random_gate
from fitness import fitness as default_fitness, compute_fidelity, compute_gate_count


def neighbor(circuit: list, n_qubits: int, max_gates: int) -> list:
    """
    Generate a neighboring circuit by applying one of three operations:
    - Replace a random gate
    - Insert a random gate at a random position (capped at max_gates,
      matching EA's mutate() constraint -- without this cap, SA's circuit
      length can grow unboundedly over the course of the search, since
      nothing here previously stopped repeated accepted 'insert' moves)
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

    elif operation == 'insert' and len(new_circuit) < max_gates:
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
    max_gates: int = 15,
    initial_temp: float = 1.0,
    cooling_rate: float = 0.995,
    min_temp: float = 1e-4,
    max_iterations: int = 2000,
    alpha: float = 1.0,
    beta: float = 0.01,
    verbose: bool = True,
    fitness_fn=None,
) -> dict:
    """
    Run Simulated Annealing for quantum circuit synthesis.

    max_gates serves two roles, mirroring EA's max_gates parameter:
    (1) the upper bound for the randomly chosen starting circuit length,
    (2) the hard cap that neighbor() enforces during the search, so the
    circuit can never grow past max_gates via repeated 'insert' moves.

    fitness_fn: optional custom fitness function with signature
                fitness_fn(gates, target_state, n_qubits, alpha, beta).
                Defaults to the standard fitness() from fitness.py if not
                given, e.g. to allow experimenting with fitness_with_floor()
                without changing this module's default behavior.

    Returns a dict with:
        - best_circuit: list of gates
        - best_fidelity: float
        - best_gate_count: int
        - history: list of best fidelity per iteration (pure fidelity,
                   matching ea.py's history -- not the penalized fitness score)
    """
    fn = fitness_fn if fitness_fn is not None else default_fitness

    start_len = np.random.randint(1, max_gates + 1)
    current_circuit = random_circuit(n_qubits, start_len)
    current_score = fn(current_circuit, target_state, n_qubits, alpha, beta)

    best_circuit = current_circuit[:]
    best_score = current_score
    best_fidelity = compute_fidelity(best_circuit, target_state, n_qubits)

    temperature = initial_temp
    history = []
    iteration = 0

    while temperature > min_temp and iteration < max_iterations:
        new_circuit = neighbor(current_circuit, n_qubits, max_gates=max_gates)
        new_score = fn(new_circuit, target_state, n_qubits, alpha, beta)

        ap = acceptance_probability(current_score, new_score, temperature)
        if np.random.random() < ap:
            current_circuit = new_circuit
            current_score = new_score

        if current_score > best_score:
            best_circuit = current_circuit[:]
            best_score = current_score
            best_fidelity = compute_fidelity(best_circuit, target_state, n_qubits)

        history.append(best_fidelity)

        temperature *= cooling_rate
        iteration += 1

        if verbose and iteration % 500 == 0:
            print(f"Iter {iteration:5d} | Temp: {temperature:.5f} | "
                  f"Best fitness: {best_score:.4f} | Fidelity: {best_fidelity:.4f}")

    return {
        'best_circuit': best_circuit,
        'best_fidelity': compute_fidelity(best_circuit, target_state, n_qubits),
        'best_gate_count': compute_gate_count(best_circuit),
        'history': history
    }
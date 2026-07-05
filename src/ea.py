import numpy as np
from qiskit.quantum_info import Statevector
from circuit_utils import random_circuit, random_gate
from fitness import fitness


def initialize_population(pop_size: int, n_qubits: int, n_gates: int) -> list:
    """Create an initial population of random circuits."""
    return [random_circuit(n_qubits, n_gates) for _ in range(pop_size)]


def evaluate_population(population: list, target_state: Statevector,
                         n_qubits: int, alpha: float, beta: float) -> list:
    """Evaluate fitness for each individual in the population."""
    return [fitness(ind, target_state, n_qubits, alpha, beta) for ind in population]


def selection(population: list, scores: list, n_select: int) -> list:
    """Tournament selection — pick the best n_select individuals."""
    paired = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
    return [ind for _, ind in paired[:n_select]]


def crossover(parent1: list, parent2: list) -> tuple:
    """Single-point crossover between two parent circuits."""
    if len(parent1) < 2 or len(parent2) < 2:
        return parent1[:], parent2[:]
    point = np.random.randint(1, min(len(parent1), len(parent2)))
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def mutate(circuit: list, n_qubits: int, mutation_rate: float = 0.1) -> list:
    """
    Mutate a circuit by randomly replacing gates.
    Each gate is replaced with probability mutation_rate.
    """
    mutated = []
    for gate in circuit:
        if np.random.random() < mutation_rate:
            mutated.append(random_gate(n_qubits))
        else:
            mutated.append(gate)
    return mutated


def evolutionary_algorithm(
    target_state: Statevector,
    n_qubits: int,
    n_gates: int = 20,
    pop_size: int = 50,
    n_generations: int = 200,
    mutation_rate: float = 0.1,
    alpha: float = 1.0,
    beta: float = 0.01,
    verbose: bool = True
) -> dict:
    """
    Run the Evolutionary Algorithm for quantum circuit synthesis.

    Returns a dict with:
        - best_circuit: list of gates
        - best_fidelity: float
        - best_gate_count: int
        - history: list of best fitness per generation
    """
    # Initialize
    population = initialize_population(pop_size, n_qubits, n_gates)
    history = []

    for gen in range(n_generations):
        # Evaluate
        scores = evaluate_population(population, target_state, n_qubits, alpha, beta)
        best_idx = int(np.argmax(scores))
        best_score = scores[best_idx]
        history.append(best_score)

        if verbose and gen % 20 == 0:
            from fitness import compute_fidelity
            f = compute_fidelity(population[best_idx], target_state, n_qubits)
            print(f"Gen {gen:4d} | Best fitness: {best_score:.4f} | Fidelity: {f:.4f}")

        # Selection — keep top 50%
        n_select = pop_size // 2
        selected = selection(population, scores, n_select)

        # Crossover — fill population back up
        new_population = selected[:]
        while len(new_population) < pop_size:
            p1, p2 = selected[np.random.randint(len(selected))], \
                     selected[np.random.randint(len(selected))]
            c1, c2 = crossover(p1, p2)
            new_population.extend([c1, c2])
        population = new_population[:pop_size]

        # Mutation
        population = [mutate(ind, n_qubits, mutation_rate) for ind in population]

        # Elitism — keep best individual unchanged
        population[0] = selected[0]

    # Final evaluation
    scores = evaluate_population(population, target_state, n_qubits, alpha, beta)
    best_idx = int(np.argmax(scores))
    best_circuit = population[best_idx]

    from fitness import compute_fidelity, compute_gate_count
    return {
        'best_circuit': best_circuit,
        'best_fidelity': compute_fidelity(best_circuit, target_state, n_qubits),
        'best_gate_count': compute_gate_count(best_circuit),
        'history': history
    }
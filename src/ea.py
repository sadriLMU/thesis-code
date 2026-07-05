import numpy as np
from qiskit.quantum_info import Statevector
from circuit_utils import random_circuit, random_gate
from fitness import fitness, compute_fidelity, compute_gate_count


def initialize_population(pop_size: int, n_qubits: int, max_gates: int) -> list:
    """
    Create an initial population of random circuits.
    Each circuit has a random length between 1 and max_gates.
    """
    population = []
    for _ in range(pop_size):
        # Random length between 1 and max_gates
        n_gates = np.random.randint(1, max_gates + 1)
        population.append(random_circuit(n_qubits, n_gates))
    return population


def evaluate_population(population: list, target_state: Statevector,
                         n_qubits: int, alpha: float, beta: float) -> list:
    """Evaluate fitness for each individual in the population."""
    return [fitness(ind, target_state, n_qubits, alpha, beta) for ind in population]


def selection(population: list, scores: list, n_select: int) -> list:
    """Select the best n_select individuals by fitness score."""
    paired = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
    return [ind for _, ind in paired[:n_select]]


def crossover(parent1: list, parent2: list) -> tuple:
    """
    Single-point crossover between two parent circuits.
    Works with variable lengths — the crossover point is chosen
    within the shorter parent to avoid index errors.
    """
    if len(parent1) < 2 or len(parent2) < 2:
        return parent1[:], parent2[:]
    point = np.random.randint(1, min(len(parent1), len(parent2)))
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def mutate(circuit: list, n_qubits: int, max_gates: int,
           mutation_rate: float = 0.1) -> list:
    """
    Mutate a circuit using three operations:
    - Replace a gate (most common)
    - Insert a new random gate
    - Delete a gate (only if circuit has more than 1 gate)

    Each gate is affected with probability mutation_rate.
    Circuit length stays between 1 and max_gates.
    """
    mutated = []
    for gate in circuit:
        r = np.random.random()
        if r < mutation_rate:
            operation = np.random.choice(['replace', 'insert', 'delete'],
                                          p=[0.5, 0.25, 0.25])
            if operation == 'replace':
                # Replace this gate with a new random one
                mutated.append(random_gate(n_qubits))
            elif operation == 'insert' and len(mutated) + len(circuit) < max_gates:
                # Insert a new gate before this one, then keep this one too
                mutated.append(random_gate(n_qubits))
                mutated.append(gate)
            elif operation == 'delete' and len(circuit) > 1:
                # Skip this gate (delete it)
                pass
            else:
                mutated.append(gate)
        else:
            mutated.append(gate)

    # Safety: ensure circuit is never empty
    if len(mutated) == 0:
        mutated.append(random_gate(n_qubits))

    return mutated


def evolutionary_algorithm(
    target_state: Statevector,
    n_qubits: int,
    max_gates: int = 20,
    pop_size: int = 50,
    n_generations: int = 200,
    mutation_rate: float = 0.1,
    alpha: float = 1.0,
    beta: float = 0.01,
    verbose: bool = True
) -> dict:
    """
    Run the Evolutionary Algorithm for quantum circuit synthesis.

    Key idea:
    - Start with a population of random circuits (variable length)
    - Each generation: evaluate, select best, crossover, mutate
    - Elitism: always keep the best individual unchanged

    Returns:
        best_circuit:   list of gates
        best_fidelity:  float (0 to 1)
        best_gate_count: int
        history:        list of best fidelity per generation
    """
    # Step 1: Initialize population with variable circuit lengths
    population = initialize_population(pop_size, n_qubits, max_gates)
    history = []

    for gen in range(n_generations):
        # Step 2: Evaluate fitness of every circuit
        scores = evaluate_population(population, target_state, n_qubits, alpha, beta)
        best_idx = int(np.argmax(scores))
        best_individual = population[best_idx][:]

        # Track best fidelity for convergence plot
        best_fidelity = compute_fidelity(best_individual, target_state, n_qubits)
        history.append(best_fidelity)

        if verbose and gen % 20 == 0:
            print(f"Gen {gen:4d} | Fidelity: {best_fidelity:.4f} | "
                  f"Gates: {compute_gate_count(best_individual)}")

        # Step 3: Selection — keep top 50%
        n_select = pop_size // 2
        selected = selection(population, scores, n_select)

        # Step 4: Crossover — fill population back to pop_size
        new_population = selected[:]
        while len(new_population) < pop_size:
            p1 = selected[np.random.randint(len(selected))]
            p2 = selected[np.random.randint(len(selected))]
            c1, c2 = crossover(p1, p2)
            new_population.extend([c1, c2])
        population = new_population[:pop_size]

        # Step 5: Mutation
        population = [mutate(ind, n_qubits, max_gates, mutation_rate)
                      for ind in population]

        # Step 6: Elitism — best individual always survives unchanged
        population[0] = best_individual

    # Final evaluation
    scores = evaluate_population(population, target_state, n_qubits, alpha, beta)
    best_idx = int(np.argmax(scores))
    best_circuit = population[best_idx]

    return {
        'best_circuit':    best_circuit,
        'best_fidelity':   compute_fidelity(best_circuit, target_state, n_qubits),
        'best_gate_count': compute_gate_count(best_circuit),
        'history':         history
    }
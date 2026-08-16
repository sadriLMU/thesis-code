import numpy as np
from qiskit.quantum_info import Statevector
from circuit_utils import random_circuit, random_gate
from fitness import fitness as default_fitness, compute_fidelity, compute_gate_count


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
                         n_qubits: int, alpha: float, beta: float,
                         fitness_fn=None) -> list:
    """
    Evaluate fitness for each individual in the population.

    fitness_fn: optional custom fitness function with signature
                fitness_fn(gates, target_state, n_qubits, alpha, beta).
                Defaults to the standard fitness() from fitness.py if not
                given, e.g. to allow experimenting with fitness_with_floor()
                without changing this module's default behavior.
    """
    fn = fitness_fn if fitness_fn is not None else default_fitness
    return [fn(ind, target_state, n_qubits, alpha, beta) for ind in population]


def selection(population: list, scores: list, n_select: int) -> list:
    """Select the best n_select individuals by fitness score."""
    paired = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
    return [ind for _, ind in paired[:n_select]]


def crossover(parent1: list, parent2: list, return_point: bool = False):
    """
    Single-point crossover between two parent circuits.
    Works with variable lengths -- the crossover point is chosen
    within the shorter parent to avoid index errors.

    Args:
        return_point: If True, also returns the chosen split index (or
            None for the degenerate case where either parent has fewer
            than 2 gates and both are copied unchanged). Used by
            experiments/plot_circuits.py to draw the crossover-trace
            figure, where the split point needs to be known explicitly
            rather than reconstructed after the fact.
    """
    if len(parent1) < 2 or len(parent2) < 2:
        if return_point:
            return parent1[:], parent2[:], None
        return parent1[:], parent2[:]
    point = np.random.randint(1, min(len(parent1), len(parent2)))
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    if return_point:
        return child1, child2, point
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

    The insert check (len(mutated) < max_gates - 1) tests the actual
    accumulated length directly, ensuring the two gates about to be added
    (new gate + the current one) fit within max_gates. This matches SA's
    equivalent check in neighbor() (len(new_circuit) < max_gates), so
    both algorithms enforce the same length cap with the same precision --
    a previous version of this check compared against the original
    circuit's full length instead of the true accumulated length, which
    could block valid inserts prematurely (see research_log.md for
    details and the before/after validation of this fix).
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
            elif operation == 'insert' and len(mutated) < max_gates - 1:
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
    crossover_rate: float = 1.0,
    alpha: float = 1.0,
    beta: float = 0.01,
    verbose: bool = True,
    fitness_fn=None,
) -> dict:
    """
    Run the Evolutionary Algorithm for quantum circuit synthesis.

    Key idea:
    - Start with a population of random circuits (variable length)
    - Each generation: evaluate, select best, crossover, mutate
    - Elitism: always keep the best individual unchanged

    crossover_rate: probability, per pair of selected parents, that
                    crossover() is applied at all; with probability
                    (1 - crossover_rate) the pair is instead cloned
                    unchanged into the next generation (mutation still
                    applies afterwards either way, see Step 5). Defaults
                    to 1.0 (crossover always applied), matching every
                    result reported in this thesis up to and including
                    the main comparison (research_log.md Entry 9/12/13)
                    -- passing any other value is an explicit ablation,
                    not a change to those results.

    fitness_fn: optional custom fitness function (see evaluate_population).
                Defaults to the standard alpha*fidelity - beta*gate_count
                fitness from fitness.py.

    Returns:
        best_circuit:   list of gates
        best_fidelity:  float (0 to 1)
        best_gate_count: int
        history:        list of best fidelity per generation
        fitness_history: list of best fitness per generation -- the
                        penalised score alpha*fidelity - beta*gate_count
                        that the search actually maximises, as opposed to
                        `history`, which records pure fidelity. Both refer
                        to the same individual (the generation's best by
                        fitness), so they can be plotted against each
                        other directly.
    """
    # Step 1: Initialize population with variable circuit lengths
    population = initialize_population(pop_size, n_qubits, max_gates)
    history = []
    fitness_history = []

    for gen in range(n_generations):
        # Step 2: Evaluate fitness of every circuit
        scores = evaluate_population(population, target_state, n_qubits,
                                      alpha, beta, fitness_fn=fitness_fn)
        best_idx = int(np.argmax(scores))
        best_individual = population[best_idx][:]

        # Track best fidelity and best fitness for convergence plots.
        # scores[best_idx] is already the fitness of best_individual --
        # it is what selection below acts on -- so recording it costs
        # nothing extra and does not affect the search in any way.
        best_fidelity = compute_fidelity(best_individual, target_state, n_qubits)
        history.append(best_fidelity)
        fitness_history.append(float(scores[best_idx]))

        if verbose and gen % 20 == 0:
            print(f"Gen {gen:4d} | Fidelity: {best_fidelity:.4f} | "
                  f"Gates: {compute_gate_count(best_individual)}")

        # Step 3: Selection -- keep top 50%
        n_select = pop_size // 2
        selected = selection(population, scores, n_select)

        # Step 4: Crossover -- fill population back to pop_size.
        # With probability crossover_rate, recombine the pair; otherwise
        # clone both parents unchanged (still subject to mutation below).
        # At the default crossover_rate=1.0, np.random.random() is never
        # called here, so this is byte-for-byte the pre-existing behaviour.
        new_population = selected[:]
        while len(new_population) < pop_size:
            p1 = selected[np.random.randint(len(selected))]
            p2 = selected[np.random.randint(len(selected))]
            if crossover_rate >= 1.0 or np.random.random() < crossover_rate:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]
            new_population.extend([c1, c2])
        population = new_population[:pop_size]

        # Step 5: Mutation
        population = [mutate(ind, n_qubits, max_gates, mutation_rate)
                      for ind in population]

        # Step 6: Elitism -- best individual always survives unchanged
        population[0] = best_individual

    # Final evaluation
    scores = evaluate_population(population, target_state, n_qubits,
                                  alpha, beta, fitness_fn=fitness_fn)
    best_idx = int(np.argmax(scores))
    best_circuit = population[best_idx]

    return {
        'best_circuit':    best_circuit,
        'best_fidelity':   compute_fidelity(best_circuit, target_state, n_qubits),
        'best_gate_count': compute_gate_count(best_circuit),
        'history':         history,
        'fitness_history': fitness_history
    }
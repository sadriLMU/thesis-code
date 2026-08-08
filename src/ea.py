"""
ea.py

Evolutionary Algorithm (EA) for quantum circuit synthesis. Searches the
space of circuits (represented as gate-dict lists, see circuit_utils.py)
for one that maximises a given fitness function (see fitness.py), using
a population-based search with selection, crossover, mutation, and
elitism (see thesis Section 4.3, "Evolutionary Algorithm Design").

Compare to sa.py, which solves the identical problem via a single-
trajectory search (Simulated Annealing) instead of a population.

Used by:
  - experiments/run_experiments.py, run_experiments_repeated.py,
    sweep_beta.py, sweep_beta_floor(_repeated).py, tune_hyperparams.py
"""

import numpy as np
from qiskit.quantum_info import Statevector
from circuit_utils import random_circuit, random_gate
from fitness import fitness as default_fitness, compute_fidelity, compute_gate_count


def initialize_population(pop_size: int, n_qubits: int, max_gates: int) -> list:
    """
    Create an initial population of random circuits.

    Each individual's length is drawn independently and uniformly from
    {1, ..., max_gates} -- matching SA's initialization convention (see
    sa.py) so neither algorithm starts with a systematic length advantage.

    Args:
        pop_size: Number of individuals in the population.
        n_qubits: Number of qubits available for gate placement.
        max_gates: Maximum circuit length; each individual's length is
            drawn from {1, ..., max_gates}.

    Returns:
        A list of pop_size circuits (each a list of gate dictionaries).
    """
    population = []
    for _ in range(pop_size):
        n_gates = np.random.randint(1, max_gates + 1)
        population.append(random_circuit(n_qubits, n_gates))
    return population


def evaluate_population(population: list, target_state: Statevector,
                         n_qubits: int, alpha: float, beta: float,
                         fitness_fn=None) -> list:
    """
    Evaluate fitness for every individual in the population.

    Args:
        population: List of circuits to evaluate.
        target_state: The target Statevector.
        n_qubits: Number of qubits in each circuit.
        alpha: Fidelity weight, passed through to the fitness function.
        beta: Gate-count penalty weight, passed through to the fitness
            function.
        fitness_fn: Optional custom fitness function with signature
            fitness_fn(gates, target_state, n_qubits, alpha, beta).
            Defaults to the standard fitness() from fitness.py. Used to
            swap in fitness_with_floor() for the ablation study in
            sweep_beta_floor(_repeated).py without changing this
            module's default behaviour elsewhere.

    Returns:
        A list of fitness scores, one per individual, in the same order
        as population.
    """
    fn = fitness_fn if fitness_fn is not None else default_fitness
    return [fn(ind, target_state, n_qubits, alpha, beta) for ind in population]


def selection(population: list, scores: list, n_select: int) -> list:
    """
    Select the n_select individuals with the highest fitness score.

    Args:
        population: List of circuits.
        scores: Fitness scores, same order/length as population.
        n_select: Number of top individuals to keep.

    Returns:
        The n_select best individuals, sorted best-first.
    """
    paired = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
    return [ind for _, ind in paired[:n_select]]


def crossover(parent1: list, parent2: list) -> tuple:
    """
    Single-point crossover between two parent circuits.

    The crossover point is chosen within the length of the shorter
    parent, so this works correctly for parents of different lengths
    (circuits are variable-length; see thesis Section 2.3.3).

    Args:
        parent1: First parent circuit.
        parent2: Second parent circuit.

    Returns:
        A (child1, child2) tuple: child1 = parent1[:point] + parent2[point:],
        child2 is the complementary split. If either parent has fewer than
        2 genes, returns unmodified copies of both parents instead (a
        crossover point cannot be meaningfully chosen).
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
    Mutate a circuit gate-by-gate, matching thesis Section 4.3.4.

    Each gate is independently affected with probability mutation_rate.
    When triggered, one of three operations applies: replace (p=0.5),
    insert a new gate before this one (p=0.25), or delete this gate
    (p=0.25, only if the circuit has more than one gate).

    Note: the insert-operation's length check
    (len(mutated) + len(circuit) < max_gates) uses the growing partial
    result plus the *original* circuit length, not a direct check against
    the final length. This is more conservative than strictly necessary
    (it can block some inserts that would still respect max_gates), but
    it guarantees the output never exceeds max_gates. Left as originally
    implemented and tested (see research_log.md) rather than altered.

    Args:
        circuit: The circuit to mutate (not modified in place).
        n_qubits: Number of qubits available for new gates.
        max_gates: Maximum circuit length (see note above).
        mutation_rate: Per-gate probability of mutation.

    Returns:
        A new, mutated circuit. Never empty (a random gate is inserted
        as a fallback if mutation would otherwise remove all gates).
    """
    mutated = []
    for gate in circuit:
        r = np.random.random()
        if r < mutation_rate:
            operation = np.random.choice(['replace', 'insert', 'delete'],
                                          p=[0.5, 0.25, 0.25])
            if operation == 'replace':
                mutated.append(random_gate(n_qubits))
            elif operation == 'insert' and len(mutated) + len(circuit) < max_gates:
                mutated.append(random_gate(n_qubits))
                mutated.append(gate)
            elif operation == 'delete' and len(circuit) > 1:
                pass
            else:
                mutated.append(gate)
        else:
            mutated.append(gate)

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
    verbose: bool = True,
    fitness_fn=None,
) -> dict:
    """
    Run the Evolutionary Algorithm for quantum circuit synthesis.

    Each generation: evaluate the population, select the top half,
    refill via crossover, mutate, then reinsert the previous generation's
    best individual unchanged (elitism) -- guaranteeing fitness never
    decreases across generations (thesis Section 4.3.5).

    Args:
        target_state: The Haar-random target Statevector to approximate.
        n_qubits: Number of qubits.
        max_gates: Maximum circuit length (used for both initialization
            and the mutation length cap).
        pop_size: Number of individuals per generation.
        n_generations: Number of generations to run.
        mutation_rate: Per-gate mutation probability (see mutate()).
        alpha: Fidelity weight in the fitness function.
        beta: Gate-count penalty weight in the fitness function.
        verbose: If True, print progress every 20 generations.
        fitness_fn: Optional custom fitness function (see
            evaluate_population()). Defaults to fitness() from fitness.py.

    Returns:
        A dict with:
            best_circuit: The best circuit found (list of gate dicts).
            best_fidelity: Its fidelity (float, 0 to 1).
            best_gate_count: Its gate count (int).
            history: List of best fidelity per generation, for
                convergence plots (see experiments/*.py).
    """
    population = initialize_population(pop_size, n_qubits, max_gates)
    history = []

    for gen in range(n_generations):
        scores = evaluate_population(population, target_state, n_qubits,
                                      alpha, beta, fitness_fn=fitness_fn)
        best_idx = int(np.argmax(scores))
        best_individual = population[best_idx][:]

        best_fidelity = compute_fidelity(best_individual, target_state, n_qubits)
        history.append(best_fidelity)

        if verbose and gen % 20 == 0:
            print(f"Gen {gen:4d} | Fidelity: {best_fidelity:.4f} | "
                  f"Gates: {compute_gate_count(best_individual)}")

        n_select = pop_size // 2
        selected = selection(population, scores, n_select)

        new_population = selected[:]
        while len(new_population) < pop_size:
            p1 = selected[np.random.randint(len(selected))]
            p2 = selected[np.random.randint(len(selected))]
            c1, c2 = crossover(p1, p2)
            new_population.extend([c1, c2])
        population = new_population[:pop_size]

        population = [mutate(ind, n_qubits, max_gates, mutation_rate)
                      for ind in population]

        population[0] = best_individual  # elitism

    scores = evaluate_population(population, target_state, n_qubits,
                                  alpha, beta, fitness_fn=fitness_fn)
    best_idx = int(np.argmax(scores))
    best_circuit = population[best_idx]

    return {
        'best_circuit':    best_circuit,
        'best_fidelity':   compute_fidelity(best_circuit, target_state, n_qubits),
        'best_gate_count': compute_gate_count(best_circuit),
        'history':         history
    }
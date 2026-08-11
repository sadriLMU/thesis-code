"""
sa.py

Simulated Annealing (SA) for quantum circuit synthesis. Solves the
identical problem as ea.py -- searching for a circuit maximising a given
fitness function (see fitness.py) -- but via a single-trajectory search
with a Metropolis acceptance criterion and a cooling schedule, instead of
a population (see thesis Section 4.4, "Simulated Annealing Design").

Used by:
  - experiments/run_experiments.py, run_experiments_repeated.py,
    sweep_beta.py, sweep_beta_floor(_repeated).py, tune_hyperparams.py
"""

import numpy as np
from qiskit.quantum_info import Statevector
from circuit_utils import random_circuit, random_gate
from fitness import fitness as default_fitness, compute_fidelity, compute_gate_count


def neighbor(circuit: list, n_qubits: int, max_gates: int) -> list:
    """
    Generate a neighboring circuit via one of three operations.

    Exactly one operation is applied per call: replace a random gate
    (p=0.5), insert a random gate at a random position (p=0.25, capped at
    max_gates -- see note below), or delete a random gate (p=0.25, only
    if more than one gate remains).

    Note: the insert cap (len(new_circuit) < max_gates) was added to fix
    a bug where SA's circuit length could grow unboundedly over the
    course of a search -- see research_log.md Entry 3 for the empirical
    impact of this fix (gate-count std dropped ~3.6x).

    Args:
        circuit: The current circuit (not modified in place).
        n_qubits: Number of qubits available for new/replacement gates.
        max_gates: Maximum circuit length; insert is skipped if it would
            exceed this.

    Returns:
        A new circuit with exactly one operation applied (or unchanged,
        if the sampled operation's precondition wasn't met). If the input
        circuit is empty, a single random gate is inserted unconditionally.
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
    Metropolis acceptance criterion, adapted for fitness maximisation
    (thesis Section 4.4.3 -- note this is the maximisation form, not the
    classical cost-minimisation form the criterion is usually written in).

    Always accepts an improving or equal move; accepts a worse move with
    probability exp((new_score - current_score) / temperature), so
    acceptance of worse moves becomes less likely as temperature cools.

    Args:
        current_score: Fitness of the current solution.
        new_score: Fitness of the candidate neighbor.
        temperature: Current annealing temperature (> 0).

    Returns:
        Acceptance probability in (0, 1].
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

    Starts from a single random circuit (length drawn uniformly from
    {1, ..., max_gates}, matching EA's initialization convention -- see
    ea.py's initialize_population() -- so neither algorithm starts with a
    systematic advantage). Each iteration proposes a neighbor(), accepts
    or rejects it via the Metropolis criterion, and cools the temperature
    geometrically (temperature *= cooling_rate) until either min_temp or
    max_iterations is reached.

    max_gates serves two roles, mirroring EA's max_gates: (1) the upper
    bound for the randomly chosen starting length, (2) the hard cap
    neighbor() enforces during the search.

    Args:
        target_state: The Haar-random target Statevector to approximate.
        n_qubits: Number of qubits.
        max_gates: Maximum circuit length (see note above).
        initial_temp: Starting temperature.
        cooling_rate: Geometric cooling factor per iteration, in (0, 1).
        min_temp: Temperature at which the search terminates.
        max_iterations: Maximum number of iterations (secondary
            termination condition).
        alpha: Fidelity weight in the fitness function.
        beta: Gate-count penalty weight in the fitness function.
        verbose: If True, print progress every 500 iterations.
        fitness_fn: Optional custom fitness function (see ea.py's
            evaluate_population() for the equivalent EA-side parameter).
            Defaults to fitness() from fitness.py.

    Returns:
        A dict with:
            best_circuit: The best circuit found (list of gate dicts).
            best_fidelity: Its fidelity (float, 0 to 1).
            best_gate_count: Its gate count (int).
            history: List of best fidelity per iteration -- pure
                fidelity, matching ea.py's history convention, not the
                penalized fitness score (see research_log.md Entry 3).
            fitness_history: List of best fitness per iteration -- the
                penalised score alpha*fidelity - beta*gate_count that the
                search actually maximises, as opposed to `history`, which
                records pure fidelity. Both refer to the same circuit
                (the incumbent best), so they can be plotted against each
                other directly. Mirrors ea.py's fitness_history.
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
    fitness_history = []
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

        # best_score is already the fitness of best_circuit; recording
        # it costs nothing extra and does not affect the search.
        history.append(best_fidelity)
        fitness_history.append(float(best_score))

        temperature *= cooling_rate
        iteration += 1

        if verbose and iteration % 500 == 0:
            print(f"Iter {iteration:5d} | Temp: {temperature:.5f} | "
                  f"Best fitness: {best_score:.4f} | Fidelity: {best_fidelity:.4f}")

    return {
        'best_circuit': best_circuit,
        'best_fidelity': compute_fidelity(best_circuit, target_state, n_qubits),
        'best_gate_count': compute_gate_count(best_circuit),
        'history': history,
        'fitness_history': fitness_history
    }
"""
plot_circuits.py

Generates visual circuit diagrams for the thesis, addressing Leo's feedback
that results should be shown concretely (actual circuits), not just numbers:
  - The best circuit found by EA and by SA for a chosen target state.
  - An example of SA's neighbor() operation: a circuit before and after one
    neighbor step, so the reader can see concretely what "replace/insert/
    delete a gate" looks like.
  - An example of EA's crossover: two parent circuits and the resulting
    child, so the reader can see concretely how single-point crossover
    combines two circuits.

Uses Qiskit's built-in circuit drawer (matplotlib backend) to render each
circuit as an image.

Output (all in results/figures/):
  - best_circuit_ea.png
  - best_circuit_sa.png
  - sa_neighbor_before.png / sa_neighbor_after.png
  - ea_crossover_parent1.png / ea_crossover_parent2.png / ea_crossover_child.png

Usage:
    cd thesis-code
    python experiments/plot_circuits.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "..", "src")
sys.path.insert(0, SRC_DIR)

from circuit_utils import generate_target_state, build_circuit, random_circuit
from ea import evolutionary_algorithm, crossover
from sa import simulated_annealing, neighbor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_QUBITS = 4
TARGET_SEED = 42  # same seed used as "Target 1" throughout the thesis,
                   # so the reader can cross-reference this circuit with the
                   # numbers already reported in Chapter 5

ALPHA = 1.0
BETA = 0.01

EA_PARAMS = dict(
    max_gates=15,
    pop_size=67,
    n_generations=100,
    mutation_rate=0.0779,
    alpha=ALPHA,
    beta=BETA,
    verbose=False,
)

SA_PARAMS = dict(
    max_gates=15,
    initial_temp=0.256,
    cooling_rate=0.9769,
    min_temp=1e-4,
    max_iterations=2000,
    alpha=ALPHA,
    beta=BETA,
    verbose=False,
)

FIGURES_DIR = os.path.join(SCRIPT_DIR, "..", "results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: draw and save a circuit
# ---------------------------------------------------------------------------
def save_circuit_diagram(gates, title, filename):
    """
    Builds a Qiskit circuit from a gate-dict list and saves a diagram.
    """
    qc = build_circuit(N_QUBITS, gates)
    qc.name = title
    fig = qc.draw(output="mpl")
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {title} ({len(gates)} gates) to {path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    target = generate_target_state(N_QUBITS, seed=TARGET_SEED)

    # --- Best EA circuit ---
    np.random.seed(TARGET_SEED)
    ea_result = evolutionary_algorithm(target, n_qubits=N_QUBITS, **EA_PARAMS)
    save_circuit_diagram(
        ea_result["best_circuit"],
        f"Best EA circuit (fidelity={ea_result['best_fidelity']:.3f})",
        "best_circuit_ea.png",
    )

    # --- Best SA circuit ---
    np.random.seed(TARGET_SEED)
    sa_result = simulated_annealing(target, n_qubits=N_QUBITS, **SA_PARAMS)
    save_circuit_diagram(
        sa_result["best_circuit"],
        f"Best SA circuit (fidelity={sa_result['best_fidelity']:.3f})",
        "best_circuit_sa.png",
    )

    # --- SA neighbor() example: before and after one step ---
    np.random.seed(123)  # separate seed, just for a clean illustrative example
    example_circuit = random_circuit(N_QUBITS, 5)
    neighbor_circuit = neighbor(example_circuit, N_QUBITS, max_gates=15)
    save_circuit_diagram(example_circuit, "Before neighbor() step",
                          "sa_neighbor_before.png")
    save_circuit_diagram(neighbor_circuit, "After neighbor() step",
                          "sa_neighbor_after.png")

    # --- EA crossover() example: two parents and the child ---
    np.random.seed(456)
    parent1 = random_circuit(N_QUBITS, 6)
    parent2 = random_circuit(N_QUBITS, 6)
    child1, _child2 = crossover(parent1, parent2)
    save_circuit_diagram(parent1, "Crossover: Parent 1", "ea_crossover_parent1.png")
    save_circuit_diagram(parent2, "Crossover: Parent 2", "ea_crossover_parent2.png")
    save_circuit_diagram(child1, "Crossover: Child", "ea_crossover_child.png")

    print(f"\nAll circuit diagrams saved to {FIGURES_DIR}")
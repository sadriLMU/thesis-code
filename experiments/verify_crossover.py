"""
verify_crossover.py

The circuit diagram from plot_circuits.py cannot unambiguously show where
crossover() split the two parents, because Qiskit's drawer groups gates by
qubit wire, not by their true position in the underlying gate list -- two
different splice points can look identical in the diagram. This script
resolves that by printing the actual gate list, gate by gate, labelled with
which parent it came from and its index -- a direct, unambiguous check of
what crossover() actually did, independent of any drawing/rendering step.

Usage:
    cd thesis-code
    python experiments/verify_crossover.py
"""

import sys
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "..", "src")
sys.path.insert(0, SRC_DIR)

from circuit_utils import random_circuit
from ea import crossover


def gate_str(gate: dict) -> str:
    """Short human-readable description of one gate dict."""
    if gate["gate"] in ("rx", "ry", "rz"):
        return f"{gate['gate'].upper()}(q{gate['qubit']}, angle={gate['angle']:.2f})"
    elif gate["gate"] == "cx":
        return f"CX(control=q{gate['control']}, target=q{gate['target']})"
    else:
        return f"{gate['gate'].upper()}(q{gate['qubit']})"


def print_labelled(circuit: list, label: str):
    print(f"\n{label} ({len(circuit)} gates):")
    for i, gate in enumerate(circuit):
        print(f"  [{i}] {gate_str(gate)}")


if __name__ == "__main__":
    np.random.seed(456)  # same seed as plot_circuits.py's crossover example,
                          # so this verifies exactly the circuits shown there

    parent1 = random_circuit(4, 6)
    parent2 = random_circuit(4, 6)
    child1, child2 = crossover(parent1, parent2)

    print_labelled(parent1, "PARENT 1")
    print_labelled(parent2, "PARENT 2")

    # Directly compare child1's gates against both parents, index by index,
    # to find exactly where the splice happened.
    print(f"\nCHILD 1 ({len(child1)} gates) -- origin of each gate:")
    for i, gate in enumerate(child1):
        from_p1 = i < len(parent1) and gate == parent1[i]
        from_p2 = i < len(parent2) and gate == parent2[i]
        if from_p1 and not from_p2:
            origin = "parent1"
        elif from_p2 and not from_p1:
            origin = "parent2"
        elif from_p1 and from_p2:
            origin = "parent1==parent2 (ambiguous, gates happened to match)"
        else:
            origin = "MISMATCH -- does not match either parent at this index!"
        print(f"  [{i}] {gate_str(gate)}  <-- {origin}")

    print("\nCHILD 2 (for reference):")
    for i, gate in enumerate(child2):
        print(f"  [{i}] {gate_str(gate)}")

    print("\nConclusion: child1 should match parent1 up to some crossover")
    print("point, then match parent2 from that point onward. Check the")
    print("'origin' column above -- it should switch from parent1 to parent2")
    print("exactly once, confirming single-point crossover behaves correctly.")
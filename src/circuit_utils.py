"""
circuit_utils.py

Foundational module for the circuit representation used throughout this
project: quantum circuits are represented as ordered lists of gate
dictionaries (see random_gate() for the exact schema), not as Qiskit
QuantumCircuit objects directly. This keeps circuits easy to manipulate
(slice, mutate, recombine) for EA and SA, independent of Qiskit's API.

Used by:
  - fitness.py      : build_circuit() to construct a circuit for fidelity
                       evaluation
  - ea.py, sa.py     : random_gate(), random_circuit() for initialization
                       and mutation/neighbor operations
  - experiments/*.py : generate_target_state() for reproducible Haar-random
                       targets
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_statevector, Statevector

# The 8-gate universal basis used for circuit synthesis throughout this
# project (see thesis Section 2.1, "Quantum Gates and Circuits").
GATE_SET = ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz', 'cx']


def generate_target_state(n_qubits: int, seed: int = None) -> Statevector:
    """
    Generate a Haar-random target state for the given number of qubits.

    Args:
        n_qubits: Number of qubits; the resulting state has 2**n_qubits
            complex amplitudes.
        seed: Random seed for reproducibility. Two calls with the same
            seed and n_qubits always return the identical state.

    Returns:
        A Haar-random Statevector, drawn uniformly from the full state
        space (see thesis Section 2.2.2).
    """
    return random_statevector(2**n_qubits, seed=seed)


def random_gate(n_qubits: int) -> dict:
    """
    Generate a single random gate, uniformly sampled from GATE_SET.

    Gate dictionary schema (all gates include the key 'gate'):
        Single-qubit, non-parameterised (h, x, y, z):
            {'gate': str, 'qubit': int}
        Rotation gates (rx, ry, rz):
            {'gate': str, 'qubit': int, 'angle': float}  # angle in [0, 2*pi)
        Two-qubit gate (cx):
            {'gate': 'cx', 'control': int, 'target': int}  # control != target

    Args:
        n_qubits: Number of qubits available; qubit indices are sampled
            from range(n_qubits).

    Returns:
        A single gate dictionary matching the schema above.
    """
    gate = np.random.choice(GATE_SET)
    qubit = np.random.randint(0, n_qubits)

    if gate in ['rx', 'ry', 'rz']:
        angle = np.random.uniform(0, 2 * np.pi)
        return {'gate': gate, 'qubit': qubit, 'angle': angle}
    elif gate == 'cx':
        target = np.random.randint(0, n_qubits)
        while target == qubit:
            target = np.random.randint(0, n_qubits)
        return {'gate': gate, 'control': qubit, 'target': target}
    else:
        return {'gate': gate, 'qubit': qubit}


def build_circuit(n_qubits: int, gates: list) -> QuantumCircuit:
    """
    Convert a list of gate dictionaries into an executable Qiskit circuit.

    This is the single place where the gate-dict representation (used by
    EA/SA for search) is translated into Qiskit's native representation
    (used for simulation/fidelity evaluation). See fitness.py.

    Args:
        n_qubits: Number of qubits in the circuit.
        gates: Ordered list of gate dictionaries, each matching the schema
            described in random_gate().

    Returns:
        A Qiskit QuantumCircuit with the given gates applied in order,
        starting from the all-zero state.
    """
    qc = QuantumCircuit(n_qubits)
    for g in gates:
        if g['gate'] == 'h':
            qc.h(g['qubit'])
        elif g['gate'] == 'x':
            qc.x(g['qubit'])
        elif g['gate'] == 'y':
            qc.y(g['qubit'])
        elif g['gate'] == 'z':
            qc.z(g['qubit'])
        elif g['gate'] == 'rx':
            qc.rx(g['angle'], g['qubit'])
        elif g['gate'] == 'ry':
            qc.ry(g['angle'], g['qubit'])
        elif g['gate'] == 'rz':
            qc.rz(g['angle'], g['qubit'])
        elif g['gate'] == 'cx':
            qc.cx(g['control'], g['target'])
    return qc


def random_circuit(n_qubits: int, n_gates: int) -> list:
    """
    Generate a random circuit of a given fixed length.

    Note: the *length* passed here is fixed by the caller -- EA and SA
    each independently randomize the starting length (see
    initialize_population() in ea.py and the start_len logic in
    simulated_annealing() in sa.py) before calling this function.

    Args:
        n_qubits: Number of qubits available for gate placement.
        n_gates: Exact number of gates to generate.

    Returns:
        A list of n_gates gate dictionaries (see random_gate()).
    """
    return [random_gate(n_qubits) for _ in range(n_gates)]
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_statevector, Statevector

# Gate set used for circuit synthesis
GATE_SET = ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz', 'cx']

def generate_target_state(n_qubits: int, seed: int = None) -> Statevector:
    """Generate a Haar-random target state."""
    return random_statevector(2**n_qubits, seed=seed)

def random_gate(n_qubits: int) -> dict:
    """Generate a random gate from the gate set."""
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
    """Build a Qiskit QuantumCircuit from a list of gate dicts."""
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
    """Generate a random circuit as a list of gate dicts."""
    return [random_gate(n_qubits) for _ in range(n_gates)]
"""
plot_circuits.py

Generates visual circuit diagrams for the thesis, addressing Leo's
feedback that results should be shown concretely (actual circuits), not
just numbers. Produces the figures used in thesis Section 5.2 and
Chapter 4 (illustrating EA's crossover and SA's neighbor() operator).

Best-circuit example target: rather than always using Target 1 (seed
42), this script scans all 20 main-comparison seeds (42-61), runs EA and
SA once on each, and picks the seed whose EA and SA gate counts are
jointly closest to the reported means (research_log.md Entry 9), so the
featured example is representative of a typical run rather than
whichever single-run outlier happens to fall on seed 42. See
select_representative_target().

Crossover illustration: a Qiskit circuit diagram groups gates by qubit
wire, not by their position in the underlying flat gate list that
crossover() actually operates on (see verify_crossover.py, which checks
this directly), so two different split points can render identically.
The three per-qubit diagrams (parent1/parent2/child) are kept as a
"what does an actual circuit look like" illustration, but
plot_crossover_trace() additionally draws the flat gate list directly,
colouring each gate by which parent it came from and marking the split
point -- this is the figure that actually explains the crossover
mechanism.

Generates (all in results/figures/):
  - best_circuit_ea.png / best_circuit_sa.png
  - sa_neighbor_before.png / sa_neighbor_after.png
  - ea_crossover_parent1.png / ea_crossover_parent2.png /
    ea_crossover_child.png            (per-qubit Qiskit diagrams, reference only)
  - ea_crossover_trace.png            (explains the crossover split itself)
  - CAPTIONS.md                       (suggested LaTeX caption text for
    every figure above, filled in with the actual numbers from this
    run -- Qiskit's drawer does not render circuit.name as visible text
    in the saved PNG, so fidelity/gate-count values need to come from
    the caption)

Usage:
    cd thesis-code
    python experiments/plot_circuits.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
BASE_SEED = 42
N_TARGETS = 20  # scan seeds 42-61, matching the main comparison (Entry 9)

# Reference means/stds from the main comparison (research_log.md Entry 9,
# 8-repeat, post-Entry-12-fix numbers) -- used only to pick a representative
# example target below, not used anywhere in the actual fidelity/fitness
# computation.
EA_MEAN_GATES, EA_STD_GATES = 4.96, 1.66
SA_MEAN_GATES, SA_STD_GATES = 6.26, 3.19

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

CAPTION_LINES = []  # collects suggested LaTeX captions as we go


def log_caption(figure_name: str, text: str):
    CAPTION_LINES.append(f"### `{figure_name}`\n\n{text}\n")


# ---------------------------------------------------------------------------
# Step 1: pick a representative target instead of always using seed 42
# ---------------------------------------------------------------------------
def select_representative_target():
    """
    Runs EA and SA once (single run, matching the original script's
    convention) for each of the 20 main-comparison seeds, and picks the
    seed whose (EA gates, SA gates) are jointly closest to the reported
    means, normalised by each algorithm's own std so neither metric
    dominates the score.

    Returns:
        (best_seed, results_dict) where results_dict maps seed ->
        (ea_result, sa_result, target).
    """
    print("Scanning seeds 42-61 for a representative example target "
          "(this takes a couple of minutes)...")
    candidates = []
    cache = {}

    for i in range(N_TARGETS):
        seed = BASE_SEED + i
        target = generate_target_state(N_QUBITS, seed=seed)

        np.random.seed(seed)
        ea_result = evolutionary_algorithm(target, n_qubits=N_QUBITS, **EA_PARAMS)

        np.random.seed(seed)
        sa_result = simulated_annealing(target, n_qubits=N_QUBITS, **SA_PARAMS)

        ea_gates = ea_result["best_gate_count"]
        sa_gates = sa_result["best_gate_count"]
        score = (abs(ea_gates - EA_MEAN_GATES) / EA_STD_GATES
                 + abs(sa_gates - SA_MEAN_GATES) / SA_STD_GATES)

        cache[seed] = (ea_result, sa_result, target)
        candidates.append((score, seed, ea_gates, sa_gates))
        print(f"  seed {seed}: EA {ea_gates} gates, SA {sa_gates} gates, "
              f"representativeness score {score:.3f} (lower is better)")

    candidates.sort(key=lambda c: c[0])
    best_score, best_seed, best_ea_gates, best_sa_gates = candidates[0]
    print(f"\nChosen target: seed {best_seed} "
          f"(EA {best_ea_gates} gates vs. mean {EA_MEAN_GATES}, "
          f"SA {best_sa_gates} gates vs. mean {SA_MEAN_GATES})\n")
    return best_seed, cache


# ---------------------------------------------------------------------------
# Helper: draw and save a circuit (Qiskit per-qubit diagram)
# ---------------------------------------------------------------------------
def save_circuit_diagram(gates, title, filename):
    """
    Builds a Qiskit circuit from a gate-dict list and saves a diagram.

    Note: `title` is descriptive metadata used in the console log and in
    CAPTIONS.md; Qiskit's mpl drawer does not render circuit.name as
    visible text in the output image itself, so the actual numbers
    (fidelity, gate count) must come from the caption in the LaTeX
    source, not the image -- see CAPTIONS.md for ready-to-use text.
    """
    qc = build_circuit(N_QUBITS, gates)
    qc.name = title
    fig = qc.draw(output="mpl")
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {title} ({len(gates)} gates) to {path}")


# ---------------------------------------------------------------------------
# Helper: gate -> short label, reused for the trace figure
# ---------------------------------------------------------------------------
def gate_label(gate: dict) -> str:
    if gate["gate"] in ("rx", "ry", "rz"):
        return f"{gate['gate'].upper()}\nq{gate['qubit']}\n{gate['angle']:.2f}"
    elif gate["gate"] == "cx":
        return f"CX\nq{gate['control']}$\\to$q{gate['target']}"
    else:
        return f"{gate['gate'].upper()}\nq{gate['qubit']}"


# ---------------------------------------------------------------------------
# Crossover trace figure: shows the split explicitly (list order, not
# grouped by qubit), see module docstring above.
# ---------------------------------------------------------------------------
def plot_crossover_trace(parent1, parent2, child, point, path):
    """
    Draws the flat gate list of parent1, parent2, and the resulting
    child as three aligned rows of labelled boxes (list order, *not*
    grouped by qubit), colouring each box by which parent it came from
    and marking the split point with a vertical dashed line. This is the
    figure that should be used to explain the crossover mechanism --
    unlike a Qiskit circuit diagram, the split point and gate origin are
    directly readable here.

    If point is None (degenerate case, one parent had <2 gates), the
    figure instead states that no split occurred and both parents were
    copied unchanged.
    """
    color_p1 = "#3b6fb5"   # blue -- from parent 1
    color_p2 = "#e08a2b"   # orange -- from parent 2
    color_neutral = "#c9c9c9"

    rows = [("Parent 1", parent1), ("Parent 2", parent2), ("Child", child)]
    max_len = max(len(g) for _, g in rows)

    fig, axes = plt.subplots(3, 1, figsize=(max(6, max_len * 1.1), 6.5),
                              sharex=True)

    for ax, (label, gates) in zip(axes, rows):
        for idx, gate in enumerate(gates):
            if label == "Parent 1":
                color = color_p1 if (point is not None and idx < point) else color_neutral
            elif label == "Parent 2":
                color = color_p2 if (point is not None and idx >= point) else color_neutral
            else:  # Child
                color = color_p1 if (point is not None and idx < point) else color_p2

            rect = mpatches.FancyBboxPatch(
                (idx, 0), 0.85, 0.85, boxstyle="round,pad=0.02",
                linewidth=1.0, edgecolor="black", facecolor=color)
            ax.add_patch(rect)
            ax.text(idx + 0.425, 0.425, gate_label(gate), ha="center",
                     va="center", fontsize=7.5, color="white",
                     fontweight="bold")

        ax.set_xlim(-0.2, max_len + 0.2)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=11)
        ax.set_xticks(np.arange(max_len) + 0.425)
        ax.set_xticklabels([str(i) for i in range(max_len)], fontsize=8)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)

        if point is not None:
            ax.axvline(point, color="black", linestyle="--", linewidth=1.2)

    axes[-1].set_xlabel("Index in gate list")
    legend_handles = [
        mpatches.Patch(color=color_p1, label="from Parent 1"),
        mpatches.Patch(color=color_p2, label="from Parent 2"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.02), frameon=False)

    title = (f"Single-point crossover, split at index {point}"
             if point is not None else
             "Crossover: one parent had fewer than 2 gates -- "
             "both copied unchanged")
    fig.suptitle(title, y=1.08, fontsize=12)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved crossover trace figure to {path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    best_seed, cache = select_representative_target()
    ea_result, sa_result, target = cache[best_seed]

    # --- Best EA / SA circuits, for the representative target ---
    save_circuit_diagram(
        ea_result["best_circuit"],
        f"Best EA circuit (target seed={best_seed}, "
        f"fidelity={ea_result['best_fidelity']:.3f})",
        "best_circuit_ea.png",
    )
    log_caption(
        "best_circuit_ea.png",
        f"Best circuit found by EA for target seed {best_seed} "
        f"(fidelity = {ea_result['best_fidelity']:.3f}, "
        f"{ea_result['best_gate_count']} gates -- close to the main "
        f"comparison's mean of {EA_MEAN_GATES} gates, Table "
        f"\\ref{{tab:main-comparison}}, chosen as a representative "
        f"rather than a best- or worst-case example)."
    )

    save_circuit_diagram(
        sa_result["best_circuit"],
        f"Best SA circuit (target seed={best_seed}, "
        f"fidelity={sa_result['best_fidelity']:.3f})",
        "best_circuit_sa.png",
    )
    log_caption(
        "best_circuit_sa.png",
        f"Best circuit found by SA for the same target seed {best_seed} "
        f"(fidelity = {sa_result['best_fidelity']:.3f}, "
        f"{sa_result['best_gate_count']} gates -- close to the main "
        f"comparison's mean of {SA_MEAN_GATES} gates). Shown for the "
        f"same target as Figure~\\ref{{fig:best-circuit-ea}} to allow a "
        f"direct visual comparison of circuit length and gate "
        f"composition between the two algorithms."
    )

    # --- SA neighbor() example: before and after one step ---
    np.random.seed(123)  # separate seed, just for a clean illustrative example
    example_circuit = random_circuit(N_QUBITS, 5)
    neighbor_circuit = neighbor(example_circuit, N_QUBITS, max_gates=15)
    save_circuit_diagram(example_circuit, "Before neighbor() step",
                          "sa_neighbor_before.png")
    save_circuit_diagram(neighbor_circuit, "After neighbor() step",
                          "sa_neighbor_after.png")
    log_caption(
        "sa_neighbor_before.png / sa_neighbor_after.png",
        "One neighbor() step (Section~\\ref{sec:sa-design}), shown before "
        "and after. Compare the two diagrams gate-by-gate to see which "
        "single operation (replace/insert/delete) was applied."
    )

    # --- EA crossover() example: two parents, child, and the trace figure ---
    np.random.seed(456)
    parent1 = random_circuit(N_QUBITS, 6)
    parent2 = random_circuit(N_QUBITS, 6)
    child1, _child2, point = crossover(parent1, parent2, return_point=True)
    save_circuit_diagram(parent1, "Crossover: Parent 1", "ea_crossover_parent1.png")
    save_circuit_diagram(parent2, "Crossover: Parent 2", "ea_crossover_parent2.png")
    save_circuit_diagram(child1, "Crossover: Child", "ea_crossover_child.png")
    log_caption(
        "ea_crossover_parent1.png / ea_crossover_parent2.png / "
        "ea_crossover_child.png",
        "Two parent circuits and the resulting child, drawn per-qubit "
        "for reference. \\textbf{Caution:} Qiskit's circuit drawer "
        "groups gates by qubit wire, not by their position in the flat "
        "gate list crossover() actually operates on, so the split point "
        "is not visible in these three diagrams alone -- use "
        "Figure~\\ref{fig:crossover-trace} "
        "(\\texttt{ea\\_crossover\\_trace.png}) to explain the "
        "mechanism, and cite these three only as \"what a resulting "
        "circuit looks like.\""
    )

    plot_crossover_trace(parent1, parent2, child1, point,
                          os.path.join(FIGURES_DIR, "ea_crossover_trace.png"))
    log_caption(
        "ea_crossover_trace.png",
        f"Single-point crossover (Section~\\ref{{sec:ea-design}}), shown "
        f"as the flat gate list rather than per-qubit, so the split "
        f"point (index {point}) and each gate's parental origin are "
        f"directly visible. This is the recommended figure for "
        f"explaining the crossover mechanism; "
        f"Figure~\\ref{{fig:crossover-parents}} shows the same example "
        f"as an actual per-qubit circuit for reference."
    )

    # --- Write CAPTIONS.md ---
    captions_path = os.path.join(FIGURES_DIR, "CAPTIONS.md")
    with open(captions_path, "w") as f:
        f.write("# Suggested captions for results/figures/*.png\n\n")
        f.write("Auto-generated by plot_circuits.py from the actual "
                "numbers of this run. Qiskit's circuit drawer does not "
                "render fidelity/gate-count text into the saved PNGs, "
                "so these numbers must come from the LaTeX caption, not "
                "the image. Copy/adapt into the relevant \\caption{} "
                "calls; \\ref{} targets assume label names matching the "
                "filenames (e.g. fig:best-circuit-ea).\n\n")
        f.write("\n".join(CAPTION_LINES))
    print(f"\nWrote suggested captions to {captions_path}")

    print(f"\nAll circuit diagrams saved to {FIGURES_DIR}")
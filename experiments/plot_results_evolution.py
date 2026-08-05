"""
plot_results_evolution.py

Visualizes how the reported EA/SA numbers changed across each corrected
stage of the project (research_log.md Entries 1, 4, 6, 7, 9), as line
charts -- one chart per metric (fidelity, gate count), each showing EA's
and SA's trend as a separate curve.

This is NOT meant for the thesis body itself (Leo's guidance was to keep
the debugging process out of the main narrative) -- this is a supporting
set of charts for your own understanding and for meeting/presentation use,
e.g. an appendix figure or backup slide if asked "how did the numbers
change over the course of the project."

All values below are taken directly from research_log.md; if you add more
corrected entries later, update the STAGES list accordingly.

Output (in results/figures/):
  - fidelity_evolution.png : EA vs. SA mean fidelity across stages
  - gate_count_evolution.png : EA vs. SA mean gate count across stages

Usage:
    cd thesis-code
    python experiments/plot_results_evolution.py
"""

import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "..", "results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# (label, EA fidelity, SA fidelity, EA gate count, SA gate count, note)
# Sourced directly from research_log.md entries.
STAGES = [
    ("Entry 1\n(buggy SA)", 0.406, 0.541, 4.7, 14.8,
     "Initial run, SA bugs still present"),
    ("Entry 4\n(SA bugs fixed)", 0.406, 0.475, 4.7, 7.8,
     "start-length + growth-cap fixes applied"),
    ("Entry 6\n(overlapping-seed\ntuning)", 0.499, 0.514, 5.8, 8.0,
     "Optuna tuning, but tuning/reporting seeds overlapped"),
    ("Entry 7\n(disjoint-seed\ntuning, 20 targets)", 0.462, 0.384, 4.85, 6.05,
     "Overlap fixed, sample size doubled -- EA now ahead"),
    ("Entry 9\n(5 repeats/target)", 0.4728, 0.3790, 5.29, 6.36,
     "Statistically confirmed via repeated runs"),
]


def plot_metric(values_ea, values_sa, ylabel, title, filename, fmt="{:.3f}"):
    labels = [s[0] for s in STAGES]
    x = range(len(labels))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x, values_ea, marker="o", color="tab:blue", linewidth=2, label="EA")
    ax.plot(x, values_sa, marker="o", color="tab:orange", linewidth=2, label="SA")

    for i, (v_ea, v_sa) in enumerate(zip(values_ea, values_sa)):
        ax.annotate(fmt.format(v_ea), (i, v_ea), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="tab:blue")
        ax.annotate(fmt.format(v_sa), (i, v_sa), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color="tab:orange")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title + "\n(supporting chart -- not for thesis body per supervisor guidance)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {title} to {path}")


if __name__ == "__main__":
    ea_fidelity = [s[1] for s in STAGES]
    sa_fidelity = [s[2] for s in STAGES]
    ea_gates = [s[3] for s in STAGES]
    sa_gates = [s[4] for s in STAGES]

    plot_metric(ea_fidelity, sa_fidelity, "Mean fidelity",
                "Fidelity trend across corrected stages",
                "fidelity_evolution.png", fmt="{:.3f}")

    plot_metric(ea_gates, sa_gates, "Mean gate count",
                "Gate count trend across corrected stages",
                "gate_count_evolution.png", fmt="{:.2f}")

    print("\nStage notes:")
    for label, ea_f, sa_f, ea_g, sa_g, note in STAGES:
        print(f"  {label.splitlines()[0]}: {note}")
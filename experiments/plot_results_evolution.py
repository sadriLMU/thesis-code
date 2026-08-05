"""
plot_results_evolution.py

Visualizes how the reported EA/SA fidelity numbers changed across each
corrected stage of the project (research_log.md Entries 1, 4, 6, 7, 9).
This is NOT meant for the thesis body itself (Leo's guidance was to keep
the debugging process out of the main narrative) -- this is a supporting
chart for your own understanding and for meeting/presentation use, e.g. an
appendix figure or backup slide if asked "how did the numbers change over
the course of the project."

The values below are taken directly from research_log.md; if you add more
corrected entries later, update the STAGES list accordingly.

Output:
  - results/figures/results_evolution.png

Usage:
    cd thesis-code
    python experiments/plot_results_evolution.py
"""

import os
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "..", "results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# (label, EA mean fidelity, SA mean fidelity, note)
# Sourced directly from research_log.md entries.
STAGES = [
    ("Entry 1\n(initial, buggy SA,\n10 targets)", 0.406, 0.541,
     "SA bugs: fixed start length + no growth cap"),
    ("Entry 4\n(SA bugs fixed,\n10 targets)", 0.406, 0.475,
     "Same EA seeds/params as Entry 1"),
    ("Entry 6\n(Optuna tuning,\noverlapping seeds,\n10 targets)", 0.499, 0.514,
     "Tuning/reporting seed overlap (train/test leakage)"),
    ("Entry 7\n(disjoint-seed tuning,\n20 targets)", 0.462, 0.384,
     "Reversal: EA now ahead of SA"),
    ("Entry 9\n(5 repeats/target,\n20 targets)", 0.4728, 0.3790,
     "Statistically confirmed: EA advantage exceeds run-to-run noise"),
]

if __name__ == "__main__":
    labels = [s[0] for s in STAGES]
    ea_values = [s[1] for s in STAGES]
    sa_values = [s[2] for s in STAGES]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    bars_ea = ax.bar(x - width / 2, ea_values, width, label="EA", color="tab:blue")
    bars_sa = ax.bar(x + width / 2, sa_values, width, label="SA", color="tab:orange")

    for bars in (bars_ea, bars_sa):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Mean fidelity")
    ax.set_title("Reported EA/SA fidelity across each corrected stage of the project\n"
                  "(supporting chart -- not for thesis body per supervisor guidance)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "results_evolution.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved results-evolution chart to {path}")

    print("\nStage notes:")
    for label, ea, sa, note in STAGES:
        print(f"  {label.splitlines()[0]}: {note}")
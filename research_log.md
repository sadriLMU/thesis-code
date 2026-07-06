# Research Log

Purpose: this file preserves the *interpretation* of each experiment, not just the
raw numbers (those already live in `results/runs/<run_id>/` automatically).
Fill in the "Your interpretation" sections in your own words before moving on —
this is the material you'll draw on when writing the Discussion chapter, so it
needs to sound like your own reasoning, not a paraphrase of what Claude said.

Each entry: what was run, why, the key numbers, and open questions.

---

## Entry 1 — 2026-07-06 — Initial EA vs. SA comparison

**Run ID:** `20260706_143856`
**Script:** `run_experiments.py`
**Config:** N_QUBITS=4, N_TARGETS=10, alpha=1.0, beta=0.01,
EA: max_gates=15, pop_size=30, n_generations=100
SA: n_gates=20, initial_temp=1.0, cooling_rate=0.995, max_iterations=2000

**Why this run:** first sanity check that the EA/SA pipeline works end-to-end
and produces a CSV + convergence plots.

**Raw results (fidelity / gate_count per target, beta=0.01):**

| target | seed | EA fidelity | EA gates | SA fidelity | SA gates |
|---|---|---|---|---|---|
| 0 | 42 | 0.3794 | 5 | 0.5215 | 39 |
| 1 | 43 | 0.3359 | 4 | 0.5173 | 16 |
| 2 | 44 | 0.3483 | 2 | 0.6252 | 33 |
| 3 | 45 | 0.3173 | 4 | 0.3762 | 8 |
| 4 | 46 | 0.3876 | 4 | 0.5475 | 8 |
| 5 | 47 | 0.4215 | 5 | 0.2928 | 4 |
| 6 | 48 | 0.4335 | 8 | 0.6961 | 10 |
| 7 | 49 | 0.2998 | 4 | 0.5808 | 5 |
| 8 | 50 | 0.6006 | 6 | 0.6580 | 17 |
| 9 | 51 | 0.5330 | 5 | 0.5936 | 8 |

**Mean fidelity:** EA ≈ 0.406, SA ≈ 0.541
**Gate count range:** EA 2–8 (tight), SA 4–39 (wide)

**Your interpretation (fill in):**
- Why do you think SA reaches higher raw fidelity here? _______________
- Is "higher fidelity" alone a fair comparison given SA's much larger gate counts? _______________
- What does the tight EA range vs. wide SA range suggest about each algorithm's search behavior? _______________

**Open question carried to next entry:** does penalizing gate count more heavily
(higher beta) shrink SA's gate count without EA changing much? → tested in Entry 2.

---

## Entry 2 — 2026-07-06 — Beta sweep

**Run ID:** `20260706_164922_beta_sweep`
**Script:** `sweep_beta.py`
**Config:** same as Entry 1, beta ∈ {0.01, 0.05, 0.1, 0.2, 0.4}, same 10 targets/seeds

**Why this run:** test the hypothesis from Entry 1 — does increasing beta shrink
SA's gate count while EA stays roughly flat?

**Aggregated results (mean across 10 targets):**

| beta | EA mean gates | SA mean gates | EA mean fidelity | SA mean fidelity |
|---|---|---|---|---|
| 0.01 | 4.7 | 14.8 | 0.406 | 0.541 |
| 0.05 | 2.7 | 3.1 | 0.341 | 0.372 |
| 0.1 | 1.6 | 1.7 | 0.249 | 0.273 |
| 0.2 | 1.0 | 1.1 | 0.152 | 0.180 |
| 0.4 | 1.0 | 1.0 | 0.152 | 0.155 |

**Gate-count std at beta=0.01:** EA ≈ 1.5, SA ≈ 11.4 (SA ~7.6x more variable)

**Fidelity-per-gate (fidelity / mean gate count):**

| beta | EA | SA |
|---|---|---|
| 0.01 | 0.086 | 0.037 |
| 0.05 | 0.126 | 0.120 |
| ≥0.1 | ~equal | ~equal |

**Key finding — hypothesis was only half-right:** EA's gate count does NOT stay
flat; it collapses in lockstep with SA's as beta increases. Both converge to
1-gate circuits by beta≈0.2. The real difference at low beta is **variance**
(SA far less predictable target-to-target) and **efficiency** (EA gets ~2x
the fidelity per gate at beta=0.01), not "SA ignores the penalty."

**Saturation observed:** beta=0.2 and beta=0.4 give nearly identical results
per-target → no new information gained above beta≈0.2. A finer sweep between
0.01–0.1 would resolve the actual transition point better than testing 0.4 again.

**Your interpretation (fill in):**
- Why would SA's single-trajectory random walk produce much higher run-to-run
  variance in circuit length than EA's population-based search? _______________
- For this thesis, is "fidelity per gate" or "raw fidelity" the more meaningful
  metric? Why? _______________
- What does the beta≥0.2 saturation tell you about the fitness landscape —
  is there a point where no achievable fidelity gain can outweigh the gate
  penalty, regardless of search algorithm? _______________

**Next step:** finer sweep, beta ∈ {0.01, 0.02, 0.03, 0.05, 0.07}, to pin down
the transition point precisely.

---

## Entry 3 — [date] — [experiment name]

**Run ID:**
**Script:**
**Config:**

**Why this run:**

**Raw results:**

**Your interpretation:**

**Open question / next step:**

---
# Research Log

Purpose: this file preserves the *interpretation* of each experiment, not just the
raw numbers (those already live in `results/runs/<run_id>/` automatically).
Fill in the "Your interpretation" sections in your own words before moving on -
this is the material you'll draw on when writing the Discussion chapter, so it
needs to sound like your own reasoning, not a paraphrase of what Claude said.

Each entry: what was run, why, the key numbers, and open questions.

---

## SUPERSESSION NOTICE - read before using Entries 1-2

Entries 1 and 2 below were generated with a **buggy `sa.py`** and should not be
cited as final results. Two bugs were found after these runs (see Entry 3):

1. SA always started from a fixed 20-gate circuit, while EA started from a
   random length (1 to max_gates) - a biased comparison.
2. SA's `neighbor()` had no upper bound on circuit length, so accepted
   `insert` moves could make the circuit grow indefinitely over the run -
   this is the main reason SA's gate counts in Entry 1 reached up to 39.

The raw data and plots from these runs are preserved, unmodified, at
`results/archive/20260706_143856_PRE-FIX-buggy-sa-init/` and
`results/archive/20260706_164922_beta_sweep_PRE-FIX/` for reference (e.g. if
you want to show a before/after comparison in the thesis to demonstrate the
debugging process). **Do not use their numbers as reported results.**
Corrected re-runs are in Entries 4-5.

---

## Entry 1 - 2026-07-06 - Initial EA vs. SA comparison [SUPERSEDED, see above]

**Run ID:** `20260706_143856` (archived)
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

**Mean fidelity:** EA = 0.406, SA = 0.541
**Gate count range:** EA 2-8 (tight), SA 4-39 (wide) - **later found to be
inflated by the two SA bugs above.**

**Your interpretation (fill in, noting these numbers are superseded):**
- _______________

**Open question carried forward:** does penalizing gate count more heavily
(higher beta) shrink SA's gate count without EA changing much? -> tested in
Entry 2, then re-tested correctly in Entry 5.

---

## Entry 2 - 2026-07-06 - Beta sweep [SUPERSEDED, see above]

**Run ID:** `20260706_164922_beta_sweep` (archived)
**Script:** `sweep_beta.py`
**Config:** same as Entry 1, beta in {0.01, 0.05, 0.1, 0.2, 0.4}, same 10 targets/seeds

**Aggregated results (mean across 10 targets):**

| beta | EA mean gates | SA mean gates | EA mean fidelity | SA mean fidelity |
|---|---|---|---|---|
| 0.01 | 4.7 | 14.8 | 0.406 | 0.541 |
| 0.05 | 2.7 | 3.1 | 0.341 | 0.372 |
| 0.1 | 1.6 | 1.7 | 0.249 | 0.273 |
| 0.2 | 1.0 | 1.1 | 0.152 | 0.180 |
| 0.4 | 1.0 | 1.0 | 0.152 | 0.155 |

**Gate-count std at beta=0.01:** EA = 1.5, SA = 11.4 (SA ~7.6x more variable)
- **this variance figure was inflated by the growth-cap bug; see Entry 5 for
the corrected figure.**

**Saturation observed:** beta=0.2 and beta=0.4 give nearly identical results
per-target -> motivated the finer sweep in Entry 5.

**Your interpretation (fill in, noting these numbers are superseded):**
- _______________

---

## Entry 3 - 2026-07-09 - SA implementation bugs found and fixed

**Files changed:** `src/sa.py`
**Commits:** fix(sa): random start length + growth cap in neighbor(); finer
beta sweep values (and earlier fix(sa): track pure fidelity in history
instead of penalized fitness score)

**Why this matters:** Entries 1-2 showed SA reaching higher raw fidelity than
EA, but with much larger and far more variable gate counts (up to 39 gates,
std = 11.4 vs. EA's std = 1.5). Investigating why SA's gate counts were so
erratic surfaced two real implementation bugs, not just parameter differences:

**Bug 1 - fixed starting length.** `simulated_annealing()` always built its
initial circuit via `random_circuit(n_qubits, n_gates)` with `n_gates=20`
fixed, while `initialize_population()` in `ea.py` draws a random starting
length via `np.random.randint(1, max_gates + 1)` for every individual. This
gave SA a systematically longer, fixed starting point than EA.

Fix: `start_len = np.random.randint(1, n_gates + 1)` before generating the
initial circuit, matching EA's convention.

**Bug 2 - unbounded circuit growth.** `neighbor()`'s insert operation had no
upper bound at all. Compare to `ea.py`'s `mutate()`, which caps insertion at
max_gates. With no equivalent cap, SA's circuit could grow arbitrarily large
over the course of up to max_iterations steps, any time an insert move was
accepted by the Metropolis criterion - regardless of where the search started.
This, not the starting length, was likely the dominant cause of the extreme
39-gate outlier in Entry 1.

Fix: added a max_gates parameter to neighbor(), reusing n_gates as the cap.

**A debugging note worth keeping for the Methodology chapter:** actually
verifying these fixes took much longer than writing them, because of an
unrelated infrastructure issue - the project existed in two separate folders
on disk (C:\Users\sadri\thesis-code, connected to git, and
C:\Users\sadri\PycharmProjects\thesis-code, where PyCharm was actually
saving edits), which silently diverged. Several "confirmed fixed" code
reviews and re-runs kept reproducing bug-era numbers because they were
running against the untouched copy. This was only conclusively diagnosed by
checking os.path.abspath(sa.__file__) at runtime rather than trusting file
contents shown in an editor or pasted into chat.

**Your interpretation (fill in):**
- Why might "insert always succeeds, no cap" be an easy category of bug to
  miss when writing a neighbor-generating function? _______________
- What does the two-folder incident suggest about how you should verify
  results going forward? _______________

---

## Entry 4 - 2026-07-09 - Corrected EA vs. SA comparison

**Run ID:** `20260709_005722`
**Script:** `run_experiments.py`
**Config:** identical to Entry 1 - only the sa.py bugs from Entry 3 differ.

**Raw results:**

| target | seed | EA fidelity | EA gates | SA fidelity | SA gates |
|---|---|---|---|---|---|
| 0 | 42 | 0.3794 | 5 | 0.2990 | 3 |
| 1 | 43 | 0.3359 | 4 | 0.4619 | 6 |
| 2 | 44 | 0.3483 | 2 | 0.4233 | 5 |
| 3 | 45 | 0.3173 | 4 | 0.3762 | 8 |
| 4 | 46 | 0.3876 | 4 | 0.5616 | 9 |
| 5 | 47 | 0.4215 | 5 | 0.3436 | 6 |
| 6 | 48 | 0.4335 | 8 | 0.6914 | 11 |
| 7 | 49 | 0.2998 | 4 | 0.3975 | 7 |
| 8 | 50 | 0.6006 | 6 | 0.6081 | 15 |
| 9 | 51 | 0.5330 | 5 | 0.5861 | 8 |

**Comparison to Entry 1 (buggy):**

| | Pre-fix SA | Fixed SA |
|---|---|---|
| Mean gate count | 14.8 | 7.8 |
| Gate count std | ~11.4 | ~3.2 |
| Mean fidelity | 0.541 | 0.475 |
| Fidelity-per-gate | 0.037 | 0.061 |

**Your interpretation (fill in):**
- Which matters more for your thesis: the fidelity drop or the gate-count/variance improvement? _______________
- Why would SA still show more variance than EA even with both bugs fixed? _______________

---

## Entry 5 - 2026-07-09 - Corrected, finer beta sweep

**Run ID:** `20260709_005834_beta_sweep`
**Script:** `sweep_beta.py`
**Config:** beta in {0.01, 0.02, 0.03, 0.05, 0.07}

**Aggregated results (mean across 10 targets):**

| beta | EA gates | SA gates | EA fidelity | SA fidelity | EA fid/gate | SA fid/gate |
|---|---|---|---|---|---|---|
| 0.01 | 4.7 | 7.8 | 0.406 | 0.475 | 0.086 | 0.061 |
| 0.02 | 4.1 | 5.6 | 0.430 | 0.492 | 0.105 | 0.088 |
| 0.03 | 3.4 | 3.6 | 0.363 | 0.415 | 0.107 | 0.115 |
| 0.05 | 2.7 | 2.7 | 0.341 | 0.369 | 0.126 | 0.137 |
| 0.07 | 1.9 | 2.3 | 0.274 | 0.330 | 0.144 | 0.143 |

**Possible efficiency crossover:** SA's fidelity-per-gate overtakes EA's
around beta=0.02-0.03, hidden previously by the growth-cap bug.

**Caveat:** single run per (target, beta) - crossover is a signal, not yet confirmed.

**Your interpretation (fill in):**
- Does the crossover make sense given how the cap changes SA's exploration? _______________
- Repeated runs next, or move to writing? _______________

---

## Entry 6 - 2026-07-18 - Optuna hyperparameter tuning (initial run)

**Run ID:** N/A (Optuna studies, not a single run) - `results/optuna_studies/`
**Script:** `tune_hyperparams.py`
**Config:** N_TUNING_TARGETS=5, seeds 42-46, N_TRIALS_EA=30, N_TRIALS_SA=30,
beta=0.01 fixed, EA budget: max_gates=15/n_generations=100 fixed, SA budget:
n_gates=20/max_iterations=2000 fixed. Objective: mean fitness
(alpha*fidelity - beta*gate_count) across the 5 tuning targets.

**Why this run:** per Leo's feedback (week 2 meeting), replace manual
hyperparameter guessing (mutation_rate=0.1, cooling_rate=0.995, etc.) with
a systematic search using Optuna.

**Optuna results:**

| | EA | SA |
|---|---|---|
| Tuned params | pop_size=57, mutation_rate=0.127 | initial_temp=0.173, cooling_rate=0.992 |
| Previous (manual) params | pop_size=30, mutation_rate=0.1 | initial_temp=1.0, cooling_rate=0.995 |
| Best mean fitness (tuning set) | 0.4227 | 0.4042 |

**Applied the tuned params and re-ran the main 10-target comparison
(run_id `20260718_162342`), compared to Entry 4 (manual params):**

| | EA (manual) | EA (tuned) | SA (manual) | SA (tuned) |
|---|---|---|---|---|
| Mean fidelity | 0.406 | 0.499 | 0.475 | 0.514 |
| Mean gate count | 4.7 | 5.8 | 7.8 | 8.0 |
| Gate count std | 1.49 | ~1.75 | 3.19 | ~2.79 |

Both algorithms improved in fidelity; EA improved more (+23%) than SA (+8%).
Plausible explanation: EA's population size nearly doubled (30->57), a bigger
structural change to the search than SA's tuning (lower starting temperature,
same search mechanics).

**Important methodological issue found (to be fixed in Entry 7):** the 5
tuning target seeds (42-46) overlap with 5 of the 10 seeds (42-51) used for
the reported comparison. This means hyperparameters were partly selected on
the same targets they are later evaluated on - similar to a train/test
leakage problem. Fix in progress: re-tune on a disjoint seed range (100+).

**Your interpretation (fill in):**
- Does the size of EA's improvement vs. SA's make sense given what actually
  changed in each algorithm's tuned parameters? _______________
- How big a deal is the tuning/reporting seed overlap for the validity of
  these numbers - cosmetic concern or a real thing to fix before reporting? _______________

---

## Entry 7 - 2026-07-18 - Re-tuned on disjoint seeds, expanded to 20 targets

**Commits:**
- `fix(sa): restore missing return statement (file was truncated, causing NoneType error)` (0187f30)
- `fix: rename n_gates to max_gates in SA, remove debug print` (a69b418)
- `feat: add Optuna hyperparameter tuning for EA and SA` (30e299e, re-tuned on seeds 100-104)
- `feat: update final EA/SA params from Optuna, N_TARGETS=20` (3888b95)
- `chore: add Optuna study databases (disjoint seed tuning run)` (0ee44e3)

**Run ID:** `20260718_175554`
**Script:** `run_experiments.py`
**Config:** N_QUBITS=4, N_TARGETS=20 (up from 10), alpha=1.0, beta=0.01,
EA: max_gates=15, pop_size=67, mutation_rate=0.0779 (Optuna-tuned, seeds 100-104),
SA: max_gates=15, initial_temp=0.256, cooling_rate=0.9769 (Optuna-tuned, seeds 100-104)

**Why this run:** Entry 6's tuning had a train/test overlap (tuning seeds 42-46
were a subset of the 10 reporting seeds 42-51). Re-tuned on a disjoint seed
range (100-104) to remove this leakage, and expanded reporting from 10 to 20
targets for a more robust sample. Also fixed an unrelated bug found in this
process: `sa.py` had been accidentally truncated (missing its `return`
statement entirely), causing a `NoneType` crash - unrelated to the tuning
work, just discovered while editing the file for the `n_gates`->`max_gates`
rename.

**Aggregated results (20 targets):**

| | EA | SA |
|---|---|---|
| Mean fidelity | 0.462 | 0.384 |
| Mean gate count | 4.85 | 6.05 |
| Gate count std | ~1.24 | ~3.19 |
| Fidelity-per-gate | 0.095 | 0.064 |

**Comparison to Entry 6 (overlapping-seed tuning, 10 targets):**

| | EA (Entry 6) | EA (this entry) | SA (Entry 6) | SA (this entry) |
|---|---|---|---|---|
| Mean fidelity | 0.499 | 0.462 | 0.514 | 0.384 |

**Important reversal:** in every prior entry (1, 4, 6), SA had higher raw
fidelity than EA. Here - with the tuning/reporting overlap removed and a
larger 20-target sample - **EA now clearly outperforms SA** on both fidelity
(+20% relative) and fidelity-per-gate (+48% relative). Gate-count variance
is also still much higher for SA (std ~3.19 vs EA's ~1.24), consistent with
every earlier entry.

Three possible explanations, not yet distinguished from each other:
1. Larger sample (20 vs 10 targets) reduced noise that previously happened
   to favor SA.
2. Entry 6's SA tuning may have partly overfit to the overlapping seeds,
   inflating SA's apparent performance in that specific comparison.
3. EA's hyperparameter search may have simply had more genuine room to
   improve than SA's - EA's best tuning-set fitness (0.386) was notably
   higher than SA's (0.314) even during tuning itself, before this
   full run happened.

**Your interpretation (fill in):**
- Which of the three explanations above (or combination) do you find most
  plausible, and why? _______________
- Does this change how you'd frame the EA vs. SA comparison in your
  Discussion chapter - e.g., from "SA wins on fidelity but wastefully" to
  something else? _______________
- Is 20 targets/1 run each still enough, or does this reversal make the
  case for repeated runs per target stronger than it was before? _______________

**Open question / next step:** decide whether to also re-run the finer beta
sweep (sweep_beta.py) with these newly re-tuned hyperparameters, for
consistency with this entry, before moving fully into writing the
Experiments chapter.

---

## Entry 8 - [date] - [experiment name]

**Run ID:**
**Script:**
**Config:**

**Why this run:**

**Raw results:**

**Your interpretation:**

**Open question / next step:**

---
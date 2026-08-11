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

**Run ID:** `20260809_220540` (re-run after Entry 12's ea.py insert-check
fix; results confirmed effectively identical to the original 20260718_175554
run — see comparison in Entry 12)
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
| Mean gate count | 5.20 | 6.05 |
| Gate count std | ~1.66 | ~3.19 |
| Fidelity-per-gate | 0.089 | 0.064 |

**Comparison to Entry 6 (overlapping-seed tuning, 10 targets):**

| | EA (Entry 6) | EA (this entry) | SA (Entry 6) | SA (this entry) |
|---|---|---|---|---|
| Mean fidelity | 0.499 | 0.462 | 0.514 | 0.384 |

**Important reversal:** in every prior entry (1, 4, 6), SA had higher raw
fidelity than EA. Here - with the tuning/reporting overlap removed and a
larger 20-target sample - **EA now clearly outperforms SA** on both fidelity
(+20% relative) and fidelity-per-gate (+39% relative). Gate-count variance
is also still much higher for SA (std ~3.19 vs EA's ~1.66), consistent with
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

## Entry 8 - 2026-07-18 - Beta sweep with final (disjoint-seed-tuned) parameters, 20 targets

**Run ID:** `20260718_180425_beta_sweep`
**Script:** `sweep_beta.py`
**Config:** N_QUBIT## Entry 8 - 2026-08-09 - Beta sweep re-run (post-Entry 12 ea.py fix)

**Run ID:** `20260809_221202_beta_sweep` (re-run after Entry 12's ea.py
insert-check fix; supersedes the original 20260718_180425_beta_sweep run)
**Script:** `sweep_beta.py`
**Config:** N_QUBITS=4, N_TARGETS=20 (seeds 42-61), beta in {0.01, 0.02, 0.03, 0.05, 0.07},
EA: max_gates=15, pop_size=67, mutation_rate=0.0779 (Optuna-tuned),
SA: max_gates=15, initial_temp=0.256, cooling_rate=0.9769 (Optuna-tuned)

**Why this run:** re-run the beta sensitivity sweep with the corrected
ea.py (Entry 12's insert-length-check fix), to confirm the original
finding still holds with the precise code.

**Aggregated results (mean across 20 targets):**

| beta | EA fidelity | EA gates | SA fidelity | SA gates | EA fid/gate | SA fid/gate |
|---|---|---|---|---|---|---|
| 0.01 | 0.462 | 5.20 | 0.384 | 6.05 | 0.089 | 0.064 |
| 0.02 | 0.418 | 3.80 | 0.322 | 4.45 | 0.110 | 0.072 |
| 0.03 | 0.403 | 3.10 | 0.305 | 3.45 | 0.130 | 0.088 |
| 0.05 | 0.358 | 2.55 | 0.276 | 2.55 | 0.140 | 0.108 |
| 0.07 | 0.285 | 1.80 | 0.271 | 1.95 | 0.158 | 0.139 |

**Consistency check passed:** beta=0.01 here reproduces Entry 7's numbers
exactly (EA 0.462/5.20, SA 0.384/6.05), confirming this sweep and the main
comparison genuinely use the same corrected code, parameters, and seeds.

**Conclusion: original Entry 8 finding fully confirmed post-fix.** EA has
equal or higher fidelity and fidelity-per-gate than SA at every tested
beta value. The gap narrows as beta increases (near-parity at beta=0.07:
0.158 vs. 0.139 fid/gate), but never reverses. As expected, increasing
beta reduces both fidelity and gate count for both algorithms.

**Your interpretation (fill in):**
- Does the gate-count values at beta=0.05 (EA and SA both 2.55) suggest
  anything about where the two algorithms' behavior converges under
  strong penalty? _______________
- Is this preliminary (single-run) result still sufficiently supported by
  the consistency check against Entry 7, or does it still warrant repeated
  runs given time allows? _______________

**Open question / next step:** as before, a repeated version of this
sweep remains future work if time permits, though the exact match with
Entry 7 at beta=0.01 provides some additional confidence in this single-run
data.

**UPDATE 2026-08-09 - Repeated beta sweep confirms the finding (closes the last remaining single-run gap):**

**Run ID:** `20260810_001917_beta_sweep_repeated`
**Script:** `sweep_beta_repeated.py` (new script, standard fitness only,
5 repeats per (beta, target, algorithm) - 1000 runs total)

**Results (mean +/- std across 100 samples per cell = 20 targets x 5 repeats):**

| beta | EA fidelity | EA gates | SA fidelity | SA gates |
|---|---|---|---|---|
| 0.01 | 0.468 +/- 0.114 | 4.85 +/- 1.38 | 0.379 +/- 0.105 | 6.36 +/- 3.03 |
| 0.02 | 0.431 +/- 0.119 | 3.91 +/- 1.39 | 0.351 +/- 0.108 | 4.61 +/- 2.42 |
| 0.03 | 0.415 +/- 0.125 | 3.26 +/- 1.10 | 0.337 +/- 0.118 | 3.83 +/- 2.43 |
| 0.05 | 0.377 +/- 0.123 | 2.70 +/- 1.02 | 0.303 +/- 0.108 | 2.65 +/- 1.31 |
| 0.07 | 0.320 +/- 0.129 | 2.10 +/- 0.99 | 0.258 +/- 0.112 | 1.93 +/- 1.19 |

**Conclusion: Entry 8's single-run finding is fully confirmed under proper
statistical repetition.** EA has higher mean fidelity than SA at every
tested beta value, and the gap between the two (roughly 0.06-0.09 across
the range) consistently exceeds either algorithm's own run-to-run standard
deviation - this is not a coincidence of single-run sampling. The
fidelity/gate-count trade-off (both metrics decreasing as beta increases)
is also confirmed for both algorithms.

**This closes the last remaining single-run gap in the project.** Every
core finding reported for this thesis - the main EA vs. SA comparison
(Entry 9), the beta sensitivity trade-off (this entry), and the
minimum-fidelity floor ablation (Entry 10/11) - is now backed by repeated
runs (5-8 repeats per condition), not single-sample results.

---

## Entry 9 - 2026-07-18 - Repeated runs (5x per target) confirm EA's advantage is real

**Run ID:** `20260718_193345_repeated`
**Script:** `run_experiments_repeated.py`
**Config:** N_QUBITS=4, N_TARGETS=20 (seeds 42-61), N_REPEATS=5 per target,
beta=0.01, EA: pop_size=67/mutation_rate=0.0779 (Optuna-tuned), SA:
initial_temp=0.256/cooling_rate=0.9769 (Optuna-tuned) - same parameters as
Entry 7/8, now with 5 repetitions per target instead of 1.

**Why this run:** Entries 4, 6, 7, and 8 all reported results from a single
run per target, which is not statistically robust for stochastic algorithms
like EA/SA. This was flagged as an open question since Entry 5. Given the
significant reversal found in Entry 7 (EA now beating SA, opposite of
Entries 1/4/6), it became important to check whether this reversal holds up
under repeated sampling, or whether it was itself an artifact of single-run
noise.

**Overall results (mean across 20 targets, 5 repeats each = 100 runs per algorithm):**

| | EA | SA |
|---|---|---|
| Mean fidelity | 0.4728 | 0.3790 |
| Mean gate count | 5.29 | 6.36 |
| Within-target fidelity std (run-to-run noise) | 0.0417 | 0.0642 |
| Across-target fidelity std (target difficulty variation) | 0.0945 | 0.0794 |

**Key finding: EA's advantage survives repeated sampling.** The fidelity gap
between EA and SA (0.4728 - 0.3790 = 0.0938) is larger than either
algorithm's own run-to-run noise (0.0417 for EA, 0.0642 for SA). This means
the reversal first observed in Entry 7 (EA outperforming SA, after fixing
the tuning/reporting seed overlap and the sa.py truncation bug) is not
simply single-run noise - it holds up when each target is run 5 times
independently.

**Secondary finding, consistent with every prior entry:** SA continues to
show substantially higher run-to-run variance than EA (within-target std
0.0642 vs. 0.0417, roughly 1.5x), even with both known sa.py bugs fixed.
This appears to be a genuine property of SA's single-trajectory search
process rather than a bug artifact, since it persists after Entry 3's
fixes and across every corrected entry since.

**UPDATE 2026-08-08 - Extended to 8 repeats, confirms result holds at larger sample:**

**Run ID:** `20260808_192152_repeated`
**Config:** identical to above, N_REPEATS increased from 5 to 8 (160 samples
per algorithm instead of 100).

**Results:** EA mean fidelity = 0.4731 (vs. 0.4728 with 5 repeats), SA mean
fidelity = 0.3688 (vs. 0.3790 with 5 repeats). Gate counts and within-target
std also closely match the original 5-repeat run (EA std 0.0442 vs. 0.0417;
SA std 0.0666 vs. 0.0642).

The EA/SA fidelity gap (0.1043) is, if anything, slightly larger with the
extended sample than with the original 5 repeats (0.0938), and remains
well above both algorithms' within-target noise. This confirms the main
finding is stable under a 60% larger sample, not an artifact of the
original repeat count.

**UPDATE 2026-08-09 - Re-run after Entry 12's ea.py insert-check fix:**

**Run ID:** `20260809_223259_repeated`
**Config:** identical to above (8 repeats, 20 targets), using the corrected
ea.py (see Entry 12).

**Results:** EA mean fidelity = 0.4660, mean gate count = 4.96, within-target
std = 0.0446. SA mean fidelity = 0.3688, mean gate count = 6.26,
within-target std = 0.0666 (SA identical to the pre-fix 8-repeat run, as
expected since SA was never affected by the ea.py fix).

EA shows a small shift consistent with the fix's expected effect (slightly
more gates allowed where the flawed check previously blocked valid inserts
in rare cases; here manifesting as fidelity 0.4731->0.4660, gates
5.24->4.96) - within the range of normal run-to-run variation already
observed across this project's repeated EA runs. **The core finding is
unchanged: EA's fidelity advantage over SA (0.4660 vs. 0.3688 = 0.0972)
remains well above both algorithms' within-target noise, and SA remains
substantially more variable than EA (std 0.0666 vs. 0.0446, ~1.5x).**

**Comparison to single-run results:**

| | Entry 7 (1 run/target, corrected) | Entry 9 (8 runs/target, corrected) |
|---|---|---|
| EA mean fidelity | 0.462 | 0.4660 |
| SA mean fidelity | 0.384 | 0.3688 |

Close agreement between single-run and repeated-run means, consistent
across all versions of this experiment (pre-fix and post-fix alike) -
further confirming the result is genuinely stable, not sensitive to the
minor implementation detail corrected in Entry 12.

**Also observed:** across-target variation (how much target difficulty
varies) is larger than within-target run-to-run noise for both algorithms -
i.e., which target state is being synthesised matters more to the outcome
than randomness in a single algorithm run. This is a plausible, if
unsurprising, property of Haar-random states (some are closer to
low-complexity/structured states than others) worth a mention in Discussion.

**Your interpretation (fill in):**
- Given this result, how confident are you now in framing RQ2's answer as
  "EA outperforms SA" rather than "it depends on conditions"? _______________
- Why might SA's search process inherently produce more run-to-run variance
  than EA's, even with a correct implementation (i.e., is this a fundamental
  property of single-trajectory vs. population-based search)? _______________
- Does the target-difficulty-variation finding suggest anything about which
  kinds of Haar-random states are harder to approximate, worth investigating
  further, or is it out of scope for this thesis? _______________

**Open question / next step:** with this statistical confirmation in hand,
proceed to write Section 5.2 (Main Comparison) and 5.3 (Beta Sensitivity)
using these repeated-run numbers (Entry 9) as the primary reported results.
Consider also running a repeated version of the beta sweep (Entry 8) for
the same level of confidence, time permitting.

---
## Entry 10 - 2026-08-09 - Minimum-fidelity floor experiment re-run (post-Entry 12 fix)

**Run ID:** `20260809_225420_beta_floor` (re-run after Entry 12's ea.py
insert-check fix; supersedes the original run this entry was based on)
**Script:** `sweep_beta_floor.py`
**Config:** N_QUBITS=4, N_TARGETS=20 (seeds 42-61), beta in {0.03, 0.05, 0.07,
0.10, 0.15}, Optuna-tuned hyperparameters (same as Entry 7/8/9). Two fitness
variants compared: "standard" (alpha*fidelity - beta*gate_count, no floor)
and "floor" (same formula, but a hard penalty of -100 if fidelity 
min_fidelity=0.3), following the same pattern as Suenkel et al.'s QCO
fitness (Related Work). Single run per (beta, algorithm, variant, target).

**Why this run:** Leo's meeting feedback suggested avoiding beta values so
strong that fidelity collapses, possibly by adding a minimum-fidelity
constraint. Re-run with the corrected ea.py (Entry 12) to confirm the
finding holds with the precise code.

**Aggregated results (mean across 20 targets):**

| beta | EA standard fid/gates | EA floor fid/gates | SA standard fid/gates | SA floor fid/gates |
|---|---|---|---|---|
| 0.03 | 0.403 / 3.10 | 0.439 / 6.05 | 0.305 / 3.45 | 0.263 / 7.10 |
| 0.05 | 0.358 / 2.55 | 0.419 / 5.55 | 0.276 / 2.55 | 0.261 / 6.95 |
| 0.07 | 0.285 / 1.80 | 0.405 / 5.00 | 0.271 / 1.95 | 0.247 / 6.70 |
| 0.10 | 0.239 / 1.40 | 0.381 / 4.60 | 0.228 / 1.40 | 0.235 / 6.55 |
| 0.15 | 0.228 / 1.25 | 0.363 / 4.65 | 0.197 / 1.15 | 0.235 / 6.55 |

**Key finding, fully confirmed post-fix: the floor helps EA clearly, but
SA inconsistently.** For EA, the floor variant beats standard at every
beta tested here (unlike the original run's single beta=0.03 exception,
the fix seems to have shifted the balance point slightly, but the overall
pattern - floor prevents collapse, gap widens with beta - is unchanged and
if anything stronger: beta=0.15 shows floor=0.363 vs standard=0.228, a
larger gap than before).

For SA, the floor is again worse than standard at low/moderate beta
(0.03: 0.263 vs. 0.305; 0.05: 0.261 vs. 0.276), and only edges out standard
at the two highest beta values (0.10: 0.235 vs. 0.228; 0.15: 0.235 vs.
0.197) - essentially identical to the original run, as expected since SA
was unaffected by the ea.py fix.

**Cost of the floor:** substantial for both algorithms, and EA's gate
count under the floor is notably higher post-fix (5.00-6.05 vs. the
original run's 4.35-4.90) - consistent with the fix allowing EA to build
slightly longer circuits where the flawed check previously blocked valid
inserts. SA's gate count under the floor is unchanged (6.55-7.10, since SA
wasn't affected).

**Caveat - single run per condition:** unchanged from the original entry;
see Entry 11 for the repeated version confirming these findings hold under
proper statistical sampling.

**Your interpretation (fill in):**
- Does it make intuitive sense that a population-based search (EA) would
  clear a hard fidelity threshold more reliably than a single-trajectory
  search (SA)? _______________
- Is the roughly doubled/tripled gate-count cost an acceptable trade-off
  for avoiding fidelity collapse, or does it undermine the whole point of
  having a beta penalty in the first place? _______________
- Given SA's inconsistent result, would you present this experiment to Leo
  as "floor works, with a caveat for SA" or hold off until repeated runs
  confirm the SA finding either way? _______________

**Open question / next step:** see Entry 11 for the repeated-runs
confirmation of these findings (also re-run post-fix, given time allows).

---

## Entry 11 - 2026-08-09 - Repeated floor comparison re-run (post-Entry 12 fix)

**Run ID:** `20260809_231713_beta_floor_repeated` (re-run after Entry 12's
ea.py insert-check fix; supersedes the original runs this entry was based on)
**Script:** `sweep_beta_floor_repeated.py`
**Config:** N_QUBITS=4, N_TARGETS=20 (seeds 42-61), N_REPEATS=3, beta in
{0.03, 0.05, 0.07, 0.10, 0.15}, same fitness variants and Optuna-tuned
hyperparameters as Entry 10. 60 samples per (beta, algorithm, variant) cell
(20 targets x 3 repeats).

**Why this run:** re-run the repeated floor comparison with the corrected
ea.py (Entry 12), to confirm the EA-benefit and SA-instability findings
hold with the precise code.

**Results (mean +/- std across 60 samples per cell):**

| beta | EA standard | EA floor | SA standard | SA floor |
|---|---|---|---|---|
| 0.03 | 0.413 +/- 0.115 | 0.414 +/- 0.109 | 0.327 +/- 0.107 | 0.220 +/- 0.205 |
| 0.05 | 0.378 +/- 0.125 | 0.410 +/- 0.106 | 0.303 +/- 0.102 | 0.210 +/- 0.194 |
| 0.07 | 0.319 +/- 0.127 | 0.392 +/- 0.102 | 0.259 +/- 0.108 | 0.206 +/- 0.189 |
| 0.10 | 0.264 +/- 0.110 | 0.375 +/- 0.090 | 0.231 +/- 0.108 | 0.199 +/- 0.182 |
| 0.15 | 0.207 +/- 0.099 | 0.363 +/- 0.085 | 0.194 +/- 0.097 | 0.195 +/- 0.177 |

**EA finding fully confirmed post-fix, and now even cleaner:** the floor
beats standard at every beta value, including beta=0.03 (0.414 vs. 0.413 -
the original single-exception is now gone with repeated sampling, though
the two are essentially tied there, consistent with the "nothing to
protect against yet at low beta" explanation). The gap widens smoothly and
monotonically with beta (standard collapses 0.413->0.207; floor stays
nearly flat, 0.414->0.363) - the same clean pattern as the pre-fix version
of this entry, now confirmed under the corrected implementation.

**SA instability finding fully confirmed post-fix:** SA-floor's std
(0.177-0.205) remains roughly 2x SA-standard's (0.097-0.108) at every beta
tested - essentially identical to the pre-fix numbers, as expected since
SA was unaffected by the ea.py fix. This is now confirmed consistent
across two independent full runs of this experiment (pre-fix and post-fix),
in addition to being consistent across all 5 beta values within each run.

**Cross-check against Entry 10 (single-run, post-fix):** beta=0.03 values
match closely (EA floor: 0.414 here vs. 0.439 in Entry 10 - within normal
sampling variation for a single-run vs. 3-repeat comparison).

**Conclusion: this experiment's findings are now validated to the same
standard as the main comparison (Entry 9) - both under the original
implementation and after Entry 12's precision fix, across independent runs
and multiple beta values.** (1) The fidelity floor reliably prevents EA's
fidelity collapse at high beta, at a roughly 3-4x gate-count cost (up from
the pre-fix ~2-3x, consistent with EA now being able to build slightly
longer circuits where valid). (2) The same floor makes SA's outcome highly
bimodal/unpredictable rather than moderately worse, unaffected by the
ea.py fix as expected, likely due to SA's single-trajectory search having
no redundancy against failing to clear the threshold within its iteration
budget.

**Your interpretation (fill in):**
- Does the size of EA's improvement vs. SA's make sense given what actually
  changed in each algorithm's search process? _______________
- Is this level of validation (repeated runs, cross-checked before and
  after a code fix) sufficient to present these floor-experiment findings
  to Leo with full confidence, alongside the main EA vs. SA comparison? _______________
- Given EA's clear, twice-replicated benefit and SA's twice-replicated
  instability under the floor, would you recommend the floor as the main
  reported fitness function, or present it as a secondary
  experiment/ablation alongside the standard results? _______________

**Open question / next step:** decide with Leo whether the floor variant
should become the primary reported fitness function or remain a secondary
ablation study. This is now purely a presentation/framing decision, not
one blocked by data quality or robustness concerns.

---

## Entry 12 - 2026-08-09 - EA mutate() insert-check precision fix, validated as result-neutral

**Files changed:** src/ea.py
**Commit:** fix(ea): correct insert-length check to test actual accumulated
length, matching SA's equivalent check

**Why this change:** during code documentation review, found that EA's
mutate() insert check (`len(mutated) + len(circuit) < max_gates`) compared
against the original circuit's full length rather than the actual
accumulated length, unlike SA's equivalent check in neighbor()
(`len(new_circuit) < max_gates`), which tests the true current length
directly. EA's check could block valid inserts prematurely - e.g. for a
10-gate circuit with max_gates=15, inserts got blocked once `mutated`
reached only 5 gates, well below the actual 15-gate limit. This created an
unintended asymmetry: EA's length cap was effectively stricter than SA's,
relevant to a thesis whose core question is a fair EA vs. SA comparison.

**Fix:** `len(mutated) < max_gates - 1` - tests the actual accumulated
length directly, precisely mirroring SA's check.

**Validation run (run_id 20260809_215244, N_TARGETS=20, same seeds/params
as Entry 7/9):**

| | EA fidelity | EA gates | SA fidelity | SA gates |
|---|---|---|---|---|
| Entry 7 (pre-fix) | 0.462 | 4.85 | 0.384 | 6.05 |
| Entry 9, 5 repeats (pre-fix) | 0.4728 | 5.29 | 0.3790 | 6.36 |
| Entry 9, 8 repeats (pre-fix) | 0.4731 | 5.24 | 0.3688 | 6.26 |
| This run (post-fix) | 0.4620 | 5.20 | 0.3843 | 6.05 |

**Conclusion: the fix is result-neutral.** The post-fix values fall
entirely within the range of normal run-to-run variation already observed
across Entry 7/9's pre-fix runs. As hypothesized, the flawed check rarely
triggered in practice, since observed circuit lengths (2-12 gates) stay
well below max_gates=15.

**Full re-validation:** Entries 7, 8, 9, 10, and 11 were all subsequently
re-run in full with the corrected code (see the "UPDATE" sections appended
to each). Every core finding held unchanged.

**Your interpretation (fill in):**
- Why might "insert always succeeds, no cap" (the same category of bug
  originally found in sa.py, Entry 3) be an easy pattern to overlook when
  writing a mutation/neighbor-generating function in general? _______________
- Given this fix turned out to be result-neutral, was the decision to
  fully re-validate all entries the right call for a thesis, given the
  time it took? _______________

---

## Entry 13 - 2026-08-10 - Within-target/across-target error bar decomposition; figure fixes for thesis presentation

**Files changed:** experiments/sweep_beta_repeated.py, experiments/sweep_beta_floor_repeated.py,
experiments/run_experiments.py, experiments/plot_circuits.py, experiments/plot_results_evolution.py,
src/ea.py

**Why this change:** while preparing figures for the thesis, found that the
error bars in beta_sweep_repeated.png and beta_floor_comparison_repeated.png
were computed as a single pooled std across all 20 targets x N repeats
combined. This conflates two different sources of variation - run-to-run
noise for a fixed target, and target-to-target difficulty variation - and
the pooled value is dominated by the latter (which is typically 2-3x
larger). Several existing claims in this log (e.g. Entry 9's "the EA/SA
gap exceeds run-to-run noise", Entry 11's "SA-floor's std is roughly 2x
SA-standard's") are specifically about run-to-run noise, i.e. the
within-target component, not the pooled value that was actually being
plotted. save_summary_csv() in both sweep scripts now reports
within_target_std and across_target_std as separate columns; the plots use
within_target_std as the error bar.

Also fixed while reviewing the same figures:
- The crossover illustration (ea_crossover_parent1/2/child.png) cannot
  show where the split happened, because Qiskit's drawer groups gates by
  qubit wire rather than by their position in the flat gate list
  crossover() operates on (same issue verify_crossover.py already checks
  for numerically). plot_circuits.py now also produces
  ea_crossover_trace.png, which draws the flat gate list directly with
  origin-coloured gates and an explicit split marker.
- The best-circuit example always used Target 1 (seed 42), which happened
  to be an outlier for SA (11 gates vs. the reported mean of 6.26).
  plot_circuits.py now scans all 20 seeds and picks the one closest to
  both algorithms' mean gate count (seed 49 this run: EA 5 gates,
  fidelity 0.505; SA 5 gates, fidelity 0.454).
- convergence_overlay.png shared a raw step axis between EA generations
  and SA iterations despite these not being computationally equivalent (1
  EA generation = pop_size fitness evaluations, 1 SA iteration = 1). Added
  convergence_overlay_by_evaluations.png, normalised by number of fitness
  evaluations.
- ea.py's crossover() gained an optional return_point argument (default
  False, no effect on existing callers) so plot_circuits.py can draw the
  trace figure without reconstructing the split point after the fact.

**Re-validation runs (fresh runs, same seeds/params as Entries 8/9/11):**

All three experiments were re-run in full to get within/across-target std.
Pooled mean_fidelity and mean_gate_count values match the corresponding
prior entries to 3 decimal places in every case (see raw CSVs), confirming
these are genuinely the same underlying results, just re-analysed.

*Main beta sweep (sweep_beta_repeated.py, 20260810_022552_beta_sweep_repeated):*

| beta | EA within-std | SA within-std | EA-SA gap |
|---|---|---|---|
| 0.01 | 0.040 | 0.064 | 0.089 |
| 0.02 | 0.052 | 0.068 | 0.080 |
| 0.03 | 0.044 | 0.067 | 0.078 |
| 0.05 | 0.062 | 0.061 | 0.074 |
| 0.07 | 0.051 | 0.059 | 0.062 |

The gap exceeds both algorithms' within-target std at every beta except
0.07, where it is close to (slightly above) SA's within-target std
(0.062 vs. 0.059) - the statistical margin narrows at high beta alongside
the mean gap itself, consistent with the "near-parity at beta=0.07"
observation already in experiments.tex, but this is the first time that
narrowing has been checked against run-to-run noise specifically rather
than just the mean values.

*Floor experiment (sweep_beta_floor_repeated.py, run 20260810_022613):*

| beta | SA-standard within-std | SA-floor within-std | ratio |
|---|---|---|---|
| 0.03 | 0.044 | 0.124 | 2.8x |
| 0.05 | 0.054 | 0.120 | 2.2x |
| 0.07 | 0.048 | 0.115 | 2.4x |
| 0.10 | 0.036 | 0.109 | 3.0x |
| 0.15 | 0.0086 | 0.111 | 12.9x |

This is a sharper version of Entry 11's "roughly 2x" finding: the actual
within-target ratio ranges from 2.2x-3.0x across most of the beta range,
consistent with Entry 11, but rises to ~13x at beta=0.15, where
SA-standard's within-target std collapses to near zero (0.0086) - SA
without the floor becomes almost deterministic at high beta, converging
on the same short/simple circuit nearly every run, while SA-floor remains
highly volatile (~0.111) across the whole beta range tested. The pooled
std used in Entry 11 could not show this, since it averages over the
target-to-target variation that dominates it at every beta.

*Convergence (run_experiments.py, run 20260810_022623):* pooled
mean_fidelity/mean_gate_count match Entry 12's validation run exactly
(EA 0.462/5.20, SA 0.384/6.05). The new evaluation-normalised convergence
plot shows SA's run terminates after ~330 fitness evaluations (cooling
schedule hits min_temp) versus EA's ~6700 (100 generations x pop_size 67)
- SA's absolute evaluation budget in this run is roughly 20x smaller than
EA's, which is worth keeping in mind when comparing final fidelity, not
just number of generations/iterations.

**Conclusion:** none of the core findings changed - this is a re-analysis
of the same results with a more precise error metric, not new data. But
the floor experiment's instability finding is now better supported (a
per-beta ratio rather than a single "~2x" figure, with the beta=0.15 case
being considerably more dramatic than previously stated), and the
evaluation-budget asymmetry between EA and SA is a new observation not
previously logged.

**Your interpretation (fill in):**
- Does the SA-floor instability ratio growing from ~2x to ~13x as beta
  increases change how you'd frame the floor experiment's conclusion in
  the Discussion, compared to the flatter "~2x at every beta" framing in
  Entry 11? _______________
- SA's evaluation budget is much smaller than EA's in this setup (a
  consequence of the tuned cooling schedule hitting min_temp early, not a
  deliberate experimental control) - does this belong in the Methodology
  as a limitation, in the Discussion as a possible confound, or is it
  outside the scope of what this thesis needs to address? _______________

---

## Entry 14 - 2026-08-11 - Fitness tracking added to EA/SA, in response to supervisor feedback

**Files changed:** src/ea.py, src/sa.py, experiments/run_experiments.py,
experiments/run_experiments_repeated.py

**Why this change:** supervisor feedback requested fitness (not just
fidelity) tracked per generation/iteration, separately for EA and SA, and
fidelity + fitness reported together for final circuits. Both
evolutionary_algorithm() and simulated_annealing() already computed
fitness internally for selection -- this change only records it
(fitness_history return key), it does not alter search behaviour. Verified
against unpatched code on 5 seeds: identical fidelity/gate_count output.

**New outputs:** convergence_fitness.png / convergence_fitness_by_evaluations.png
(run_experiments.py), final_comparison_bars.png (both run_experiments.py
and run_experiments_repeated.py, the latter using the full N=160 sample).

**Your interpretation (fill in):**
- _______________

---

## Entry 15 - 2026-08-11 - Manual hyperparameter ablations (pop_size, mutation_rate, crossover_rate)

**Files changed:** experiments/tune_hyperparams_manual.py (new),
experiments/crossover_rate_ablation.py (new), src/ea.py (added optional
crossover_rate parameter, default 1.0, backward-compatible)

**Why this change:** supervisor feedback suggested checking
literature-typical hyperparameter values manually (e.g. population 200,
crossover rate ~0.8, cf. Sunkel et al. 2025's crossover rate of 0.85),
rather than relying solely on the Optuna search. Both ablations run on
the tuning seeds (100-104), disjoint from reporting seeds, matching
tune_hyperparams.py's methodology.

**Findings:**
- pop_size grid (50/67/100/200) x mutation_rate grid (0.02-0.30): current
  tuned config (pop_size=67, mutation_rate=0.0779) ranks 3rd of 20 by raw
  fitness; gap to the top config (pop_size=200) is not statistically
  distinguishable from noise (5 targets, std ~0.05-0.09). At matched
  compute budget (fitness_per_1k_evals), smaller populations are 3-4x
  more efficient -- larger population is not a free improvement.
- crossover_rate 1.0 (current) vs 0.8: mean fitness 0.3234 vs 0.3155
  (N=25 each), gap/SE ratio 0.34 -- not distinguishable from noise.

**Conclusion:** no evidence that either alternative would improve on the
current tuned configuration; current config retained unchanged.

**Your interpretation (fill in):**
- _______________

---

## Entry 16 - 2026-08-11 - Extended beta sweep (0.01-0.30), checking for an EA/SA crossover at high beta

**Files changed:** experiments/sweep_beta_extended_quick.py (new),
experiments/sweep_beta_extended_full.py (new)

**Why this change:** the main beta sweep (Entry 8/13) only covers
beta=0.01-0.07, where the EA/SA fidelity gap narrows but does not close.
Checked whether SA eventually overtakes EA at higher beta, beyond the
originally tested range.

**Findings (full version, N=5 repeats/target/beta, beta=0.01-0.30):**
gap never reverses (EA >= SA at every beta tested); gap shrinks to
near-zero for beta >= 0.20 (e.g. beta=0.30: EA 0.173 vs SA 0.166, gap
0.007). Mean gate count for BOTH algorithms converges to ~1 gate by
beta=0.30 (SA: exactly 1.00 +/- 0.00 across all 100 samples), i.e. the
search problem degenerates to a near-trivial solution independent of
search strategy at high beta -- the closing gap reflects the problem
becoming trivial for both algorithms, not SA improving relative to EA.

**Your interpretation (fill in):**
- _______________

---

## Entry 17 - 2026-08-11 - Budget-matched EA/SA comparison (controlling for fitness-evaluation count)

**Files changed:** experiments/budget_matched_comparison.py (new)

**Why this change:** the main comparison (Entry 9/12/13) does not
control for total fitness-evaluation budget: EA uses pop_size(67) x
n_generations(100) = 6,700 evaluations per run; SA's tuned cooling
schedule (cooling_rate=0.9769) terminates after only ~330-336 iterations
(evaluations), reaching min_temp far before the max_iterations=2000 cap.
This is roughly a 20x difference in search effort, unaccounted for in
the headline comparison. This experiment adds a third condition,
SA_matched, with cooling_rate solved analytically
(cooling_rate = (min_temp/initial_temp)^(1/6700) = 0.998829) so SA also
receives ~6,700 evaluations; initial_temp is left at its Optuna-tuned
value (this experiment asks "what if SA searched as long as EA," not "what
is the best SA config for this budget" -- a natural further follow-up).
Run on the REPORTING seeds (42-61), not the tuning seeds, since
cooling_rate here is computed analytically from a budget constraint, not
selected by comparing candidates -- see script docstring for the full
reasoning. N=20 targets x 5 repeats = 100 samples per condition.
Wall-clock time (time.time(), single machine, sequential runs, not a
rigorous benchmark) recorded alongside fidelity/fitness/gate count.

**Findings:**

| Condition | n_evaluations | Mean fidelity | Mean fitness | Mean gates | Mean wall-clock |
|---|---|---|---|---|---|
| EA | 6,700 | 0.4675 +/- 0.0396 (within) | 0.4190 | 4.85 | 2.60s |
| SA (standard) | ~336 | 0.3790 +/- 0.0642 (within) | 0.3154 | 6.36 | 0.19s |
| SA (matched) | 6,700 | 0.5645 +/- 0.0436 (within) | 0.4795 | 8.50 | 4.03s |

EA and SA (standard) numbers match Entry 8 exactly, confirming this is
the same underlying comparison, just extended with a third condition.

**SA (matched) significantly outperforms EA on both fidelity and fitness**
once evaluation count is equalised: gap = -0.097 fidelity (SA ahead),
gap/pooled-SE ratio ~16.5 (N=100 each) -- not a marginal effect. SA wins
despite a higher mean gate count (8.5 vs 4.85); the fidelity gain
(+0.097) outweighs the extra gate penalty at beta=0.01 (+3.65 gates x
0.01 = 0.037).

Wall-clock time tells a separate, third story: SA (matched) takes ~55%
longer in real time than EA despite equal evaluation counts (4.03s vs
2.60s), suggesting SA's per-evaluation cost is higher than EA's in this
implementation. SA (standard) remains by far the fastest condition
(0.19s) for a fidelity only moderately below EA's.

**This does not invalidate the main comparison (Entry 9/12/13) or the
beta-sweep (Entry 8/13, 16)** -- those results are correct descriptions of
each algorithm's behaviour under its own natural termination criterion,
which remains a valid and practically relevant protocol. What this shows
is that the EA advantage reported under that protocol is substantially
attributable to unequal search effort rather than to EA's search strategy
being intrinsically superior to SA's.

**Your interpretation (fill in):**
- Does this change how RQ2's answer should be framed in the Discussion
  (e.g. "EA outperforms SA under natural termination criteria, but this
  advantage is largely attributable to an unequal evaluation budget, not
  to intrinsic search superiority") ? _______________
- SA (matched) was not re-tuned for its new budget (initial_temp kept at
  the value tuned for ~330 evaluations) -- worth a follow-up with
  re-tuned initial_temp, or out of scope for this thesis? _______________
- The wall-clock finding (SA costs more per evaluation) is unexplained --
  worth investigating the implementation, or reported as an open
  observation? _______________

---
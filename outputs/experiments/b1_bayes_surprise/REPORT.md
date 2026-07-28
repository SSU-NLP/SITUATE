# B1-BayesSurprise (Kumar et al. 2023) — reimplementation, validation, deployable comparison

**Date:** 2026-07-15 · **Scope:** EACL Phase 0 · dev docs (Pieman/Tunnel/Monkey, all dev per D-M5)
**Status:** B1 reimplemented, **fidelity-validated at word resolution**, integrated into deployable pipeline.

## 1. Setup

- **Goal.** Add B1-BayesSurprise (Kumar et al. 2023, *Cognitive Science*) as a baseline — the strongest
  scalar-surprise method and the direct test bed for **H1 (scalar surprise vs multi-axis reset)**. Code
  unpublished → reimplemented from the paper's exact equations. (`kenmcgarry/BayesSurprise` is unrelated; unused.)
- **Method (faithful to paper §2.5.1, verified verbatim).**
  - Model: **GPT-2-XL (1.5B)**, HuggingFace, moving 1024-token context window (Kumar §2.4). fp16==fp32 (checked).
  - Bayesian surprise `S_i = KL(P_i ‖ P_{i-1})` between consecutive next-word predictive distributions
    (full ~50k vocab), taken at each **word endpoint** (last BPE token). Surprisal `= -log p(actual next word)`.
  - Transient normalization (Reynolds 2007): `mu_t = mu_{t-1}+(S_t-mu_{t-1})*drift`, `Transient=S_t/mu_{t-1}`,
    `drift=0.10` (paper's best), `mu_1 =` global mean.
- **Data.** Michelmann sentence packs (`{doc}_michelmann_master.jsonl`) for Pieman/Tunnel/Monkey; gold =
  author-filtered 18/18/30 boundaries (reset_matrix `boundary`). Word-resolution fidelity uses
  `data/eacl_phase0/pieman/pieman_rep_results.csv` (24 human boundary words + `agreement`).
- **Scripts.** `scripts/eacl_phase0/b1_bayes_surprise.py` (signal + `--dump-words`, `--fp32`),
  `b1_kumar_regression.py` (trailing-window L1 regression, dev-fit), `b1_deployable.py` (deployable Score/AUC).
- **Metric.** Main = official DTS Score (`hi_ontop.dts_scoring`), deployable (μ+cσ for continuous, count≥t* for M3);
  AUC supplement. All knobs frozen on dev=Pieman.

## 2. Results

### 2a. Fidelity check — word resolution vs 24 human boundary words (Pieman) — **PASS**

| signal | word-res AUC (best trailing-win) |
|---|---|
| **Bayesian surprise (raw)** | **0.705** |
| Bayesian surprise (transient) | 0.682 |
| surprisal | 0.52 (≈chance) |
| surprisal (transient) | 0.49 |

→ Reproduces Kumar's signature: **boundaries track transient Bayesian surprise, NOT surprisal.** Implementation validated.
michelmann-18 boundaries mapped to words also predicted (AUC 0.662) and coincide with the 24 CSV boundaries
(median 1-word offset) — same boundaries, different processing.

### 2b. Deployable per-transition comparison (dev; B1 readout = `newsent_max` transient)

frozen on Pieman: M3-S5 t\*=3 · M1-Cos c=1.0 · B1 c=0.75

| doc | M3-S5 Score | M1-Cos Score | B1 Score | AUC S5 | AUC M1 | AUC B1 |
|---|---|---|---|---|---|---|
| pieman (dev) | **0.634** | 0.585 | 0.374 | **0.757** | 0.706 | 0.533 |
| tunnel | **0.383** | 0.208 | 0.200 | **0.672** | 0.553 | 0.563 |
| monkey | **0.359** | 0.307 | 0.277 | **0.597** | 0.525 | 0.542 |

## 3. Interpretation

- **B1 is validated but weak at sentence-level episode segmentation.** Word-resolution AUC ~0.70 (strong,
  reproduces Kumar); per-transition deployable AUC ~0.53–0.56 (≈chance). The gap is **not a bug** — it is a
  real property: the only difference between the two evaluations is the negative set. At word resolution the
  negatives are mostly mid-sentence words (easy to beat); at transition resolution the negatives are
  **non-boundary sentence junctions, which are homogeneous in Bayesian surprise**, so scalar surprise cannot
  tell which sentence junction is an episode boundary. New-event onsets ("And so…", "Now…") are even
  *predictable* (low surprise) → onset-max readout is anti-correlated (AUC 0.35).
- **Kumar's trailing-window L1 regression** (faithfully reproduced) overfits Pieman (in-sample AUC 0.75) but
  transfers below chance (tunnel 0.40, monkey 0.44) → no generalizable sentence-level boundary signal. Not used
  as the deployable readout; `newsent_max` (best/robust simple readout on dev) used instead.
- **Supports H1.** The strongest scalar-surprise method (B1) is near-chance for sentence-level episode
  segmentation, while multi-axis situation-model reset (M3-S5) reaches AUC 0.60–0.76 and wins Score on all 3
  dev docs. Scalar surprise ≠ multi-axis coordinated reset.

## 4. Judgment

- **M3-S5 > M1-CosDrift > B1** on all 3 dev docs (Score and AUC). B1 added and ready for Tier-2 (frozen knobs
  transfer unchanged).
- B1 implementation is **defensible against "you crippled the baseline"**: it reproduces Kumar's published
  word-resolution result; its per-transition weakness is an honest property of scalar surprise, reported as such.

## 5. Limitations / open

- **All 3 docs are dev** (D-M5, selection-on-test). These are **exploratory**; the defensible held-out B1 number
  comes from **Tier-2** (`tier2-heldout-spec.md`) with the frozen config, once built.
- B1 deployable readout (`newsent_max` transient) chosen on Pieman from a small principled set (onset/junction/
  sentence family); documented before Tier-2. drift=0.10 frozen; 0.03/0.05 available as robustness.
- Michelmann text may differ slightly from Kumar's exact transcript (disfluencies); did not affect the fidelity result.
- Kumar's dense per-word button-press DV was unavailable (CSV gives 24 boundary words + agreement only); fidelity
  used the binary word-resolution boundaries, which sufficed to reproduce the Bayesian-vs-surprisal dissociation.

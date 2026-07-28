# M5-FactoredSlot — Phase 2 (slot-alignment de-risk) REPORT

**Date:** 2026-07-15 · **Verdict:** typed-slot factoring VIABLE (factoring gain robust >0); strong axes
align; weak axes (place/time/goal) bottlenecked by silver-label quality, not architecture.

## Setup
- Question (plan Phase 2): does typed-slot + anchor loss align K=5 slots to EISM axes on FROZEN mpnet
  embeddings, WITHOUT dynamics/switching? Cheapest make-or-break before the heavy model.
- Data: 10 narrative-fiction docs (pieman/tunnel/monkey michelmann + 7 Tier2), 3100 sentences / 3090
  transitions. Anchor labels = M3(S5) per-transition RESET (0/1 ×5) + per-sentence enum states.
  `build_slot_dataset.py`. Reset rates: protagonist 0.5–0.7, causality 0.3–0.8, goal 0.1–0.4,
  **place 0.02–0.18, time 0.02–0.07 (rare → class imbalance).**
- Model `src/hi_ontop/factored_slot.py`; train `train_factored_slot.py`; probe `probe_slot_alignment.py`;
  seed sweep `slot_seed_sweep.py`. Anchor loss = codex 2026-07-15 (per-axis RESET BCE primary + aux
  state CE + variance-floor collapse; head locality; D3→D1).

## Result 1 — v1 (shared input) COLLAPSED
Naive EntNet keyed slots (all slots see same full z_t, keys differ): own-axis AUPRC decent, but
cross-axis probe showed **every slot decodes every axis identically** → slots = redundant copies,
**M5 ≈ M4** (0/5 separation). Diagnosis (codex): head locality stops a head peeking at other slots but
does NOT stop a slot from encoding everything; all slots see full high-capacity input.

## Result 2 — D1 fix (per-axis extractor routing) SEPARATES
Fix (codex): per-axis extractor `E_k: 768→32→GELU→Dropout(0.4)→8D` routed to slot k ONLY (separate the
INFORMATION path, not just parameters); no shared proj/residual. Cross-axis probe with **label-correlation
ceiling** `B(k→j)=AUPRC(reset_k→reset_j)` removed (EISM axes are correlated, so off-diagonal ≠ 0 even
when separated). Metric = excess leakage `slot_k→j − B(k→j)`.

| | M5 (D1) excess off-axis leakage | M4 (mono) excess | factoring gain (M4−M5) |
|---|---|---|---|
| single run | +0.072 | +0.114 | +0.042 |

## Result 3 — seed stability (A) + weak-axis variant (B)
**A (seed stability):** factoring gain base +0.033/+0.063; weak variant **+0.044 ± 0.005 (min +0.038)
→ ROBUST >0.** The D1 factoring benefit is NOT noise — M5 separates slots measurably better than
monolithic across seeds/variants.

**B (weak-axis tuning: bottleneck 16 + stronger pos_weight):** own-axis AUPRC (M5, held-out val):
| axis | own-axis (weak) | own-axis (base) | base rate |
|---|---|---|---|
| protagonist | 0.70 | 0.70 | 0.54 |
| causality | 0.62 | 0.57 | 0.42 |
| goal | 0.32 | 0.30 | 0.22 |
| place | 0.34 | 0.35 | 0.09 |
| time | 0.13 | 0.12 | 0.03 |
Tuning lifts causality/goal slightly; **place/time stay weak** → the bottleneck is the rare/noisy silver
labels (place/time were weak throughout the project), NOT the architecture.

## Verdict
- ✅ **Typed-slot factored approach is viable:** after the D1 fix, M5 slots separate (factoring gain
  robustly ~+0.04 > 0) and strong axes align (protagonist 0.70, causality 0.62 ≫ base). The de-risk's
  core question — "can typed slots align to EISM axes" — is YES for the well-labeled axes.
- ⚠️ **Weak axes (place/time/goal) are label-quality-limited**, not architecture-limited. Per plan
  Phase 2 ladder: scale anchor labels (Phase 1 large corpus → more rare-axis positives) / improve M3
  extraction, OR descope v1 to the strong axes (protagonist/causality [+time-marker]).
- **Scope honesty (codex):** D1 injects the EISM ontology into the observation model (typed sufficient
  statistics) — it validates the typed-slot MACHINE, not axis *discovery*. Discovery (D2/D3 with
  competitive binding on token/span input) is a separate later phase. Anchor loss ∉ SEM (sCRP/Bayes/
  local-MAP untouched) → no conflict yet; decision-log entry warranted.

## Limitations / next
- Frozen encoder, anchor-only (no dynamics). Small silver data (~3000 transitions, 10 docs), 2-doc val.
- Next (plan): (a) scale weak-axis anchors (Phase 1 corpus, budget); (b) Phase 3 dynamics (per-slot
  predictive + sticky-CRP) on the D1 typed slots; (c) multi-seed (5+) full confirmation.

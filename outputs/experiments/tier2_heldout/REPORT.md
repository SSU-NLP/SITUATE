# Tier 2 held-out — preliminary REPORT (paused at 7/12 for M5 pivot)

**Date:** 2026-07-15 · **Status:** gold 12/12 done · S5(M3) 8/12 done · eval on **7 complete docs (preliminary)**.
Paused to move to M5; remaining 4-5 docs' S5 need API budget. Canonical decisions: ledger D-M9.

## Setup
- **12 public-domain short stories**, 4 authors (O.Henry/Maupassant/Chekhov/Crane ×3 each), 94–442 sentences
  (dev range). Poe gothic 3편 replaced (gpt-4.1 content-filter blocked verbatim copy) — D-M9 amendment 2.
- **Gold** = GPT-4.1 × B2 prompt × 20 reps temp 0 → B3 top-n consensus (D-M9). Cross-family (gold=OpenAI,
  M3=claude-fable-5). `build_tier2_stories.py`/`build_tier2_pack.py`/`build_tier2_gold.py`.
- **Frozen-on-dev knobs (seal):** M3-S5 count≥t\*=3 · M1-Cos μ+cσ c=1.0 · M2-LAM c=0.0 · B1 newsent_max c=0.75.
- Main metric = official DTS Score (deployable, self-determined count). Eval: `tier2_eval.py`.
- ⚠️ **All Phase-0 config is provisional/exploratory** (final method = M5; t\*=3 is a known-provisional
  operating point that M5's sticky-CRP κ replaces). These are **fresh-data exploratory** numbers, not final.

## Results — deployable Score, held-out (n=7 complete docs)
| method | mean Score | vs M3 (doc-level bootstrap 95% CI) |
|---|---|---|
| **M3-S5** | **0.344** | — |
| M1-CosDrift | 0.223 | M3−M1 **+0.121 [+0.063, +0.172]** → Go |
| M2-MonoLAM | 0.135 | M3−M2 **+0.209 [+0.179, +0.242]** → Go |
| B1-BayesSurprise | 0.200 | M3−B1 **+0.144 [+0.098, +0.200]** → Go |

**M3 beats M1, M2, B1 with 95% CI excluding 0 (3/3 Go)** — project's first statistically defensible held-out
signal (all fresh-data). M3 wins 6/7 docs. H1 (multi-axis > scalar Bayesian surprise) supported on held-out.

### Signal-quality companion (defuses "Score too low")
Low absolute Score = honest deployable + silver-gold + exact-F1 (all methods low). AUC shows signal ranks well:
AUC M3 0.717 / M1 0.680 / B1 0.558 / M2 0.470. Low deployable = operating-point cost (which M5 κ fixes).
Nuance: matched@gold M3≈M1 (0.34 vs 0.33) → M3's win over M1 is partly operating-point; M3 > B1/M2 on ALL
metrics. Headline comparison = **M3 vs B1 (EISM vs EST)**, clean on Score+AUC.

## B1 model robustness (GPT-2-XL → modern open LM) — "not a crippled baseline"
B1's Bayesian surprise needs full next-word distribution (open logits) → closed APIs (GPT-5, proxy Qwen3-235B:
logprobs=None) CANNOT run B1. Tested open LMs on Pieman dev (deployable Score / word-res AUC):
| B1 LM | Score | AUC |
|---|---|---|
| GPT-2-XL (1.5B, Kumar orig) | 0.374 | 0.705 |
| Qwen2.5-7B | 0.419 | 0.709 |
| Qwen2.5-14B | 0.309 | 0.750 |
→ Scaling the LM does NOT lift B1 to M3's level (all ≪ M3 0.634 on Pieman); scalar-surprise ceiling is the
**method**, not model size. `b1_bayes_surprise.py --model ...`.

## B2/B3 topline vs LLM-free (dev human gold, non-circular; B3 reps=10, exploratory)
| doc | B3 (GPT-4.1 direct, topline) | M3-S5 (LLM-free) | M1 | B1 |
|---|---|---|---|---|
| pieman | 0.434 | **0.634** | 0.585 | 0.374 |
| tunnel | **0.509** | 0.383 | 0.208 | 0.200 |
| monkey | **0.478** | 0.359 | 0.307 | 0.277 |
| **mean** | **0.474** | **0.459** | 0.367 | 0.284 |
→ **RQ2:** LLM-free M3 (0.459) ≈ GPT-4.1-direct offline topline (0.474); **M3 > topline on Pieman**. M3 tuned
on dev (esp. Pieman home-advantage) — exploratory. `dev_b2b3_topline.py`.

## Limitations / open
- **Preliminary (7/12 docs).** Final 12-doc number pending S5 on 4-5 remaining (budget). the_wreck gold is a
  density outlier (60/249, kept per D-M9 no-adjustment rule).
- Silver gold (GPT-4.1); LLM shared-bias mitigated by cross-family, not eliminated. B2/B3 barred as Tier2
  baselines (circular). Tier3 (Korean human) = final defense.
- Config provisional; definitive held-out = M5-era on Tier2/Tier3.

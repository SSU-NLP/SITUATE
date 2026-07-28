#!/usr/bin/env python3
"""Propagate measured temp-0 extraction jitter onto gold-scored dev-3 → honest ΔScore noise σ.

The self-agreement floor (noise_floor_run2run.py) OVERSTATES ΔScore noise because it uses one
run as its own target. Gate C's gold is EXTERNAL (independent of both runs), so the decision-
relevant question is: how much does Score(run, human_gold) move across re-extractions?

We answer it with budget 0:
  1. Measure per-axis reset FLIP rate from the 3 famous literary docs (run1 vs run2, shared trans).
  2. On each dev-3 doc (real human gold + real frozen axis columns), Monte-Carlo perturb each axis
     column by its flip rate, recompute count>=t*=3, score vs that doc's gold → Score distribution.
  3. σ = SD of Score under re-extraction. Two independent runs differ by ~√2·σ; report the
     empirical 95% spread of (Score_a − Score_b). Compare to 관문① decision deltas.

Model = independent per-axis Bernoulli flips at the famous-measured rate (first-order; real jitter
may be correlated/asymmetric — famous run2 fired fewer boundaries, see limitation).
"""
from __future__ import annotations
import json, sys, random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402

FROZEN_AXES = ["protagonist_present", "space", "time", "causality", "intentionality"]
T_STAR = 3
N_REP = 3000
SEED = 20260723
RUN1 = REPO / "outputs/eacl_phase0/repcorpus_s5_eismv2_run1/reset_matrix.jsonl"
RUN2 = REPO / "outputs/eacl_phase0/repcorpus_s5_eismv2_run2/reset_matrix.jsonl"
FAMOUS = ["fam_doyle_speckled", "fam_gilman_wallpaper", "fam_london_fire"]
DEV3 = {"pieman": "pieman_michelmann_s5_eismv2", "tunnel": "tunnel_michelmann_s5_eismv2",
        "monkey": "monkey_michelmann_s5_eismv2"}


def load_by_doc(p: Path):
    docs = {}
    for l in p.read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            docs.setdefault(r["doc_id"], {})[int(r["transition"])] = r
    return docs


def end_turn_labels(n_sent, bt):
    y = [0] * n_sent
    for i in bt:
        y[i - 1] = 1
    return y


def measure_flip_rates():
    d1, d2 = load_by_doc(RUN1), load_by_doc(RUN2)
    rates = {}
    for a in FROZEN_AXES:
        dis = tot = 0
        for doc in FAMOUS:
            st = sorted(set(d1[doc]) & set(d2[doc]))
            for i in st:
                tot += 1
                dis += int(d1[doc][i][a]) != int(d2[doc][i][a])
        rates[a] = dis / tot
    return rates


def doc_score(rows_sorted, gold_labels, n_sent):
    count = [sum(int(r[a]) for a in FROZEN_AXES) for r in rows_sorted]
    bt = [i + 1 for i, c in enumerate(count) if c >= T_STAR]
    return score_dialogues([gold_labels], [end_turn_labels(n_sent, bt)])["score"]


def main():
    rng = random.Random(SEED)
    flip = measure_flip_rates()
    print("=" * 86)
    print("HONEST ΔScore NOISE via jitter propagation onto gold-scored dev-3")
    print("=" * 86)
    print("per-axis FLIP rate measured on 3 famous literary docs (run1 vs run2):")
    for a in FROZEN_AXES:
        print(f"  {a:22s} flip={flip[a]:.3f}  (agree {1-flip[a]:.3f})")

    print(f"\nMonte-Carlo N={N_REP}, seed={SEED}, t*={T_STAR}\n")
    print(f"{'doc':8s} {'gold':>4} {'baseScore':>9} {'MC_mean':>8} {'MC_sd(σ)':>9} "
          f"{'2.5%':>6} {'97.5%':>6} {'ΔScore95(√2σ pair)':>18}")
    all_sd = []
    for name, sub in DEV3.items():
        rows = [json.loads(l) for l in (REPO / "outputs/eacl_phase0" / sub / "reset_matrix.jsonl")
                .read_text().splitlines() if l.strip()]
        rows.sort(key=lambda r: int(r["transition"]))
        n_sent = len(rows) + 1
        gold = end_turn_labels(n_sent, [int(r["transition"]) for r in rows if int(r["boundary"])])
        n_gold = sum(int(r["boundary"]) for r in rows)
        base = doc_score(rows, gold, n_sent)

        scores = []
        for _ in range(N_REP):
            pert = []
            for r in rows:
                pr = dict(r)
                for a in FROZEN_AXES:
                    if rng.random() < flip[a]:
                        pr[a] = 1 - int(r[a])
                pert.append(pr)
            scores.append(doc_score(pert, gold, n_sent))
        scores.sort()
        mean = sum(scores) / len(scores)
        sd = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
        lo, hi = scores[int(0.025 * N_REP)], scores[int(0.975 * N_REP)]
        # empirical ΔScore between two independent runs: draw random pairs
        diffs = sorted(abs(scores[rng.randrange(N_REP)] - scores[rng.randrange(N_REP)])
                       for _ in range(N_REP))
        d95 = diffs[int(0.95 * N_REP)]
        all_sd.append(sd)
        print(f"{name:8s} {n_gold:>4} {base:>9.3f} {mean:>8.3f} {sd:>9.3f} "
              f"{lo:>6.3f} {hi:>6.3f} {d95:>18.3f}")

    sigma = sum(all_sd) / len(all_sd)
    pair_sd = (2 ** 0.5) * sigma
    print("-" * 86)
    print(f"mean single-run σ = {sigma:.3f}   |   two-run ΔScore SD ≈ √2σ = {pair_sd:.3f}   "
          f"|   ~2σ_pair ≈ {2*pair_sd:.3f}")
    print("\n" + "=" * 86)
    print(f"VERDICT — 관문① decision deltas vs run-to-run ΔScore noise (√2σ={pair_sd:.3f}):")
    print("=" * 86)
    checks = [
        ("Gate C pieman  deepseek vs fable5", 0.041, "declared MISS (≥−0.03 bar)"),
        ("Gate C monkey", 0.072, "declared MISS"),
        ("Gate C tunnel", 0.078, "declared PASS"),
        ("protagonist_present vs perspective (dev-3 mean)", 0.070, "adopted D-B10"),
        ("causality variant band", 0.006, "called ±noise §5"),
    ]
    for nm, mag, tag in checks:
        z = mag / pair_sd
        rel = ("WITHIN noise" if z < 1 else "~1–2σ (ambiguous)" if z < 2 else "ABOVE noise (real)")
        print(f"  |Δ|={mag:.3f}  = {z:.1f}·(√2σ)  ->  {rel:20s} [{nm}; {tag}]")
    print("\nLimitation: independent-Bernoulli flip model; real jitter is partly correlated and")
    print("asymmetric (famous run2 fired fewer boundaries) — true σ may be modestly larger.")


if __name__ == "__main__":
    main()

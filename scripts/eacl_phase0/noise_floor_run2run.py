#!/usr/bin/env python3
"""Run-to-run Score noise floor from two identical-config temp-0 extractions (D-B11 run1/run2).

Purpose (예산 0): quantify the extractor's OWN reproducibility ceiling so we can tell whether
관문① decision deltas are signal or jitter:
  - Gate C  Δ = −0.041 (deepseek 0.593 vs fable5 0.634 on pieman)
  - protagonist_present swap  = +0.070 (present 0.447 vs perspective 0.377, dev-3 mean)
  - causality variants        = ±0.006 band (0.447 / 0.440 / 0.438 / 0.433 / inceptive分해)

Method: for each doc present in BOTH runs, derive frozen-config boundaries (count>=t*=3) from
run1 and from run2, then score one against the other with the OFFICIAL scorer (same units as
eval_seg_score / Gate C). S_self < 1 => extraction jitter; the (1 - S_self) gap is the noise
floor. A ΔScore smaller than this gap is indistinguishable from re-running the SAME config.

No gold needed (repcorpus is unlabeled by design) — self-agreement is gold-free.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402

FROZEN_AXES = ["protagonist_present", "space", "time", "causality", "intentionality"]  # D-B10
T_STAR = 3
RUN1 = REPO / "outputs/eacl_phase0/repcorpus_s5_eismv2_run1/reset_matrix.jsonl"
RUN2 = REPO / "outputs/eacl_phase0/repcorpus_s5_eismv2_run2/reset_matrix.jsonl"
FAMOUS = {"fam_doyle_speckled", "fam_gilman_wallpaper", "fam_london_fire"}


def load(p: Path) -> dict[str, dict[int, dict]]:
    docs: dict[str, dict[int, dict]] = {}
    for l in p.read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        docs.setdefault(r["doc_id"], {})[int(r["transition"])] = r
    return docs


def end_turn_labels(n_sent: int, bt: list[int]) -> list[int]:
    y = [0] * n_sent
    for i in bt:
        y[i - 1] = 1
    return y


def boundaries(rows: dict[int, dict], shared_t: list[int]) -> list[int]:
    return [i for i in shared_t if sum(int(rows[i][a]) for a in FROZEN_AXES) >= T_STAR]


def main() -> None:
    d1, d2 = load(RUN1), load(RUN2)
    shared_docs = sorted(set(d1) & set(d2))
    print("=" * 88)
    print("RUN-TO-RUN Score NOISE FLOOR  (same frozen config p0s5-eism-v2, temp0, t*=3)")
    print("  S_self = official Score of run2 boundaries vs run1 boundaries (and reverse, mean)")
    print("  noise floor = 1 - S_self  (a ΔScore below this is re-extraction jitter)")
    print("=" * 88)
    print(f"{'doc':24s} {'nTrans':>6} {'bnd1':>5} {'bnd2':>5} {'S(2|1)':>7} {'S(1|2)':>7} "
          f"{'S_self':>7} {'1-S_self':>9}")

    per_doc = []
    golds_fam, preds_fam = [], []   # pooled-as-Gate-C (per-dialogue aggregation) for famous
    for doc in shared_docs:
        shared_t = sorted(set(d1[doc]) & set(d2[doc]))
        n_sent = len(shared_t) + 1
        b1 = boundaries(d1[doc], shared_t)
        b2 = boundaries(d2[doc], shared_t)
        y1 = end_turn_labels(n_sent, b1)
        y2 = end_turn_labels(n_sent, b2)
        s21 = score_dialogues([y1], [y2])["score"]   # run1 as ref, run2 as hyp
        s12 = score_dialogues([y2], [y1])["score"]   # symmetric
        s_self = (s21 + s12) / 2
        per_doc.append((doc, s_self))
        grp = "FAM" if doc in FAMOUS else "obs"
        print(f"{doc:24s} {len(shared_t):>6} {len(b1):>5} {len(b2):>5} {s21:>7.3f} {s12:>7.3f} "
              f"{s_self:>7.3f} {1 - s_self:>9.3f}   {grp}")
        if doc in FAMOUS:
            golds_fam.append(y1); preds_fam.append(y2)

    print("-" * 88)
    # pooled famous (per-dialogue aggregation = exactly how Gate C 3편평균 is computed)
    fam_pool = score_dialogues(golds_fam, preds_fam)["score"]
    fam_mean = sum(s for d, s in per_doc if d in FAMOUS) / len(FAMOUS)
    print(f"FAMOUS (3 literary docs, ~ dev-3 register):")
    print(f"  per-doc mean S_self = {fam_mean:.3f}   =>  noise floor 1-S_self = {1 - fam_mean:.3f}")
    print(f"  pooled (per-dialogue agg) S_self = {fam_pool:.3f}  =>  noise floor = {1 - fam_pool:.3f}")
    obs = [s for d, s in per_doc if d not in FAMOUS]
    if obs:
        print(f"obscure (obs_28922): S_self = {obs[0]:.3f}  =>  noise floor = {1 - obs[0]:.3f}")

    floor = 1 - fam_mean
    print("\n" + "=" * 88)
    print(f"VERDICT — comparing 관문① decision deltas to the famous noise floor ({floor:.3f}):")
    print("=" * 88)
    checks = [
        ("Gate C  deepseek vs fable5 (pieman)", 0.041, "miss"),
        ("Gate C  monkey", 0.072, "miss"),
        ("Gate C  tunnel", 0.078, "pass"),
        ("protagonist_present vs perspective (dev-3 mean)", 0.070, "adopt"),
        ("causality variant band (frozen/OR/marker/flip)", 0.006, "±noise"),
        ("causality remove-axis (0.447->0.448)", 0.001, "±noise"),
    ]
    for name, mag, tag in checks:
        rel = "WITHIN noise" if mag < floor else ("~AT noise" if mag < 1.5 * floor else "ABOVE noise")
        print(f"  |Δ|={mag:.3f}  vs floor {floor:.3f}  ->  {rel:14s}  [{name}; {tag}]")
    print("\nNote: floor is measured on 3 literary docs (register-matched to Gate C dev-3),")
    print("frozen config, temp0. It is the SAME-config re-extraction jitter — a hard lower")
    print("bound on what any single-run ΔScore comparison can resolve.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase-0 MAIN metric: the project's official DTS segmentation Score.

Score = 0.5*F1 + 0.25*(1-Pk) + 0.25*(1-WD)  (SuperDialseg total_score, HANDOFF_04).
Computed via the single official scorer `hi_ontop.dts_scoring.score_dialogues` so EACL
Phase-0 is scored identically to every other benchmark in the repo (per-dialogue aggregation,
nltk pk/windowdiff, F1 binary, end-turn convention, last turn forced 0).

Coordinated count uses the frozen config: protagonist = protagonist_present (D-B10, 2026-07-22;
supersedes D-B8 perspective-only — present 0.447 > perspective 0.377 on dev-3, 3/3 improve).
AUC kept as a threshold-free supplement.

Threshold-freeze protocol (D-B6 seal): t* = argmax Score(count>=t) on the DEV doc (Pieman);
frozen and applied UNCHANGED to sealed test docs. --freeze-threshold N applies a pre-frozen t.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402

FROZEN_AXES = ["protagonist_present", "space", "time", "causality", "intentionality"]  # D-B10


def read(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def auc(y, s):
    pos = [a for yy, a in zip(y, s) if yy]; neg = [a for yy, a in zip(y, s) if not yy]
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def end_turn_labels(n_sent: int, boundary_transitions: list[int]) -> list[int]:
    """per-sentence 0/1, end-turn convention: sentence i-1 is segment-final iff transition i is a boundary."""
    y = [0] * n_sent
    for i in boundary_transitions:
        y[i - 1] = 1
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reset_matrix", type=Path)
    ap.add_argument("--freeze-threshold", type=int, default=None)
    ap.add_argument("--gold-master", type=Path, default=None,
                    help="override boundary labels from this master (resets/idm are gold-independent)")
    args = ap.parse_args()
    M = read(args.reset_matrix)
    M.sort(key=lambda r: int(r["transition"]))
    if args.gold_master:  # swap in a different gold's boundary column (gold B = raw+5s)
        gb = {int(r["transition"]): int(r["boundary"]) for r in read(args.gold_master)}
        for r in M:
            r["boundary"] = gb.get(int(r["transition"]), 0)
    n_sent = len(M) + 1
    ref_bt = [int(r["transition"]) for r in M if int(r["boundary"])]
    count = {int(r["transition"]): sum(int(r[a]) for a in FROZEN_AXES) for r in M}
    idm = {int(r["transition"]): float(r["idm_score"]) for r in M}
    gold = end_turn_labels(n_sent, ref_bt)

    def score_at_count(t):
        bt = [i for i in count if count[i] >= t]
        return score_dialogues([gold], [end_turn_labels(n_sent, bt)])

    # freeze threshold by argmax Score on this (dev) doc, unless supplied
    if args.freeze_threshold is not None:
        tstar = args.freeze_threshold
    else:
        tstar = max(range(1, 6), key=lambda t: score_at_count(t)["score"])

    print(f"doc={M[0]['doc_id']}  n_sentences={n_sent}  n_boundaries={len(ref_bt)}")
    print("\n=== count threshold sweep (MAIN metric = Score) ===")
    print(f"{'t':>2} {'nPred':>5} {'Score':>7} {'Pk':>6} {'WD':>6} {'F1':>6}")
    for t in range(1, 6):
        bt = [i for i in count if count[i] >= t]
        if not bt:
            continue
        r = score_dialogues([gold], [end_turn_labels(n_sent, bt)])
        mark = "  <- t*" if t == tstar else ""
        print(f"{t:>2} {len(bt):>5} {r['score']:7.3f} {r['pk']:6.3f} {r['wd']:6.3f} {r['f1']:6.3f}{mark}")

    ours = score_at_count(tstar)
    # IDM: argmax Score over its thresholds
    idm_bt_by_t = {}
    for thr in sorted(set(idm.values())):
        bt = [i for i in idm if idm[i] >= thr]
        if bt:
            idm_bt_by_t[thr] = bt
    idm_best = max(idm_bt_by_t, key=lambda thr: score_dialogues([gold], [end_turn_labels(n_sent, idm_bt_by_t[thr])])["score"])
    idmr = score_dialogues([gold], [end_turn_labels(n_sent, idm_bt_by_t[idm_best])])

    print("\n=== MAIN METRIC — official Score = 0.5*F1 + 0.25*(1-Pk) + 0.25*(1-WD) ===")
    print(f"{'method':16s} {'Score':>7} | {'Pk↓':>6} {'WD↓':>6} {'F1↑':>6} {'nPred':>5}")
    print(f"{'ours (5-axis)':16s} {ours['score']:7.3f} | {ours['pk']:6.3f} {ours['wd']:6.3f} "
          f"{ours['f1']:6.3f} {sum(count[i]>=tstar for i in count):>5}   (t*={tstar})")
    print(f"{'IDM':16s} {idmr['score']:7.3f} | {idmr['pk']:6.3f} {idmr['wd']:6.3f} "
          f"{idmr['f1']:6.3f} {len(idm_bt_by_t[idm_best]):>5}   (thr={idm_best:.3f})")
    print(f"\nScore gain (ours - IDM) = {ours['score'] - idmr['score']:+.3f}")
    ry = [int(r['boundary']) for r in M]
    print(f"supplementary AUC: ours {auc(ry, [count[int(r['transition'])] for r in M]):.3f}  "
          f"IDM {auc(ry, [idm[int(r['transition'])] for r in M]):.3f}")

    # matched prediction-count: both predict the SAME number of boundaries, isolating
    # signal quality (which boundaries) from count effects (Pk/WD reward fewer predictions).
    def idm_topN(N):
        order = sorted(idm, key=lambda i: idm[i], reverse=True)[:N]
        return end_turn_labels(n_sent, order)
    tx = [int(r["transition"]) for r in M]
    def count_topN(N):  # our count, ties broken by transition order (deterministic, gold-free)
        order = sorted(tx, key=lambda i: (count[i], -i), reverse=True)[:N]
        return end_turn_labels(n_sent, order)
    print("\n=== matched prediction-count (both predict top-N; isolates signal quality) ===")
    print(f"{'N':>4} {'ours_Score':>10} {'IDM_Score':>10} | {'ours F1/Pk/WD':>18} {'IDM F1/Pk/WD':>18}")
    Ns = sorted({sum(c >= t for c in count.values()) for t in range(1, 6)} | {len(ref_bt)})
    for N in Ns:
        if N == 0:
            continue
        o = score_dialogues([gold], [count_topN(N)]); d = score_dialogues([gold], [idm_topN(N)])
        tag = " (=gold)" if N == len(ref_bt) else ""
        print(f"{N:>4} {o['score']:>10.3f} {d['score']:>10.3f} | "
              f"{o['f1']:.3f}/{o['pk']:.3f}/{o['wd']:.3f}  {d['f1']:.3f}/{d['pk']:.3f}/{d['wd']:.3f}{tag}")


if __name__ == "__main__":
    main()

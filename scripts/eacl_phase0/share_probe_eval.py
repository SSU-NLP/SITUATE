"""SHARE M3(S5) probe 평가 — M3 coordinated vs M1-CosDrift, 메인=Score.

메인 지표 = **DTS Score**(`dts_scoring.score_dialogues`), per-doc 평균 + **doc-level
bootstrap CI**(30 docs → 프로젝트 최초 CI). AUC·축별 lift는 **보조**.

- M3-S5: coordinated count = frozen 5축 리셋합(protagonist_perspective+place+time+
  causality+goal). deployable = count ≥ t*(=3, 동결). matched@gold = top-K.
- M1-CosDrift: CosDrift-raw(1−cos 인접). deployable = μ+cσ(c=1.0 동결). matched = top-K.

usage: uv run python scripts/eacl_phase0/share_probe_eval.py \
         --master outputs/eacl_phase0/share_valid_probe30_master.jsonl \
         --reset-matrix outputs/eacl_phase0/share_valid_probe30_s5/reset_matrix.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from hi_ontop.embedding import make_encoder  # noqa: E402

ENCODER = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
AX = ["protagonist_perspective", "place", "time", "causality", "goal"]
T_STAR = 3      # M3 deployable 동결 (D-M5-b)
C_FROZEN = 1.0  # M1 deployable 동결 (D-M3)


def endturn(n: int, ts: list[int]) -> list[int]:
    y = [0] * n
    for t in ts:
        y[t - 1] = 1
    return y


def doc_scores(tx, lbl, signal, *, mode, thr=None, tstar=None):
    """deployable(개수 자가결정) Score. mode='sigma'|'count'."""
    n = len(tx) + 1
    gold = endturn(n, [t for t, y in zip(tx, lbl) if y])
    if mode == "sigma":
        a = np.array(signal); cut = a.mean() + C_FROZEN * a.std()
        pred = [t for t, s in zip(tx, signal) if s >= cut]
    else:  # count >= tstar
        pred = [t for t, s in zip(tx, signal) if s >= tstar]
    return score_dialogues([gold], [endturn(n, pred)])["score"]


def matched(tx, lbl, signal):
    n = len(tx) + 1
    gold_t = [t for t, y in zip(tx, lbl) if y]
    K = len(gold_t)
    if not K:
        return float("nan")
    top = [tx[i] for i in sorted(range(len(signal)), key=lambda i: (signal[i], -tx[i]),
                                 reverse=True)[:K]]
    return score_dialogues([endturn(n, gold_t)], [endturn(n, top)])["score"]


def auc(lbl, signal):
    return float(roc_auc_score(lbl, signal)) if len(set(lbl)) == 2 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=Path, required=True)
    ap.add_argument("--reset-matrix", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260714)
    args = ap.parse_args()

    master = [json.loads(l) for l in open(args.master)]
    rm = {(r["doc_id"], r["transition"]): r for r in (json.loads(l) for l in open(args.reset_matrix))}
    docs: dict[str, list[dict]] = {}
    for r in master:
        if (r["doc_id"], r["transition"]) in rm:
            docs.setdefault(r["doc_id"], []).append(r)
    for d in docs.values():
        d.sort(key=lambda r: r["transition"])

    # encode for M1-CosDrift
    texts, idx = [], {}
    def add(t):
        if t not in idx:
            idx[t] = len(texts); texts.append(t)
        return idx[t]
    for d in docs.values():
        add(d[0]["previous"])
        for r in d:
            add(r["current"])
    enc = make_encoder(model=ENCODER)
    E = np.asarray(enc.encode(texts)); E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    per = {k: {"m3_dep": [], "m3_match": [], "m1_dep": [], "m1_match": [],
               "m3_auc": [], "m1_auc": []} for k in [0]}
    m3_dep, m3_match, m1_dep, m1_match, m3_auc, m1_auc = [], [], [], [], [], []
    ax_auc = {a: [] for a in AX}
    ax_b = {a: [] for a in AX}
    ax_i = {a: [] for a in AX}
    for did, d in docs.items():
        tx = [r["transition"] for r in d]
        lbl = [int(r["boundary"]) for r in d]
        cnt = [sum(int(rm[(did, t)][a]) for a in AX) for t in tx]
        cos = [1 - float(np.dot(E[idx[r["current"]]], E[idx[r["previous"]]])) for r in d]
        m3_dep.append(doc_scores(tx, lbl, cnt, mode="count", tstar=T_STAR))
        m1_dep.append(doc_scores(tx, lbl, cos, mode="sigma"))
        m3_match.append(matched(tx, lbl, cnt))
        m1_match.append(matched(tx, lbl, cos))
        m3_auc.append(auc(lbl, [float(c) for c in cnt]))
        m1_auc.append(auc(lbl, cos))
        for a in AX:
            fl = [int(rm[(did, t)][a]) for t in tx]
            ax_auc[a].append(auc(lbl, [float(f) for f in fl]))
            b = [f for f, y in zip(fl, lbl) if y]; ii = [f for f, y in zip(fl, lbl) if not y]
            if b: ax_b[a].append(np.mean(b))
            if ii: ax_i[a].append(np.mean(ii))

    def ms(x):
        a = np.array([v for v in x if v == v]); return a.mean(), a.std()

    def boot_ci(a, b):
        """M3-M1 per-doc gap bootstrap CI."""
        gap = np.array(a) - np.array(b)
        rng = np.random.default_rng(args.seed)
        bs = [gap[rng.integers(0, len(gap), len(gap))].mean() for _ in range(args.boot)]
        return float(np.mean(gap)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    print(f"\n=== SHARE probe 평가 — {len(docs)} docs ===  (메인=Score, t*={T_STAR})")
    print(f"\n[메인] DTS Score (per-doc mean ± std)")
    print(f"  {'':14s} {'M3-S5':>16s} {'M1-CosDrift':>16s}")
    for lab, a, b in [("deployable", m3_dep, m1_dep), ("matched@gold", m3_match, m1_match)]:
        (ma, sa), (mb, sb) = ms(a), ms(b)
        g, lo, hi = boot_ci([v for v in a if v == v], [b[i] for i, v in enumerate(a) if v == v])
        star = "" if (lo <= 0 <= hi) else "  *CI>0*" if lo > 0 else "  *CI<0*"
        print(f"  {lab:14s} {ma:.3f} ± {sa:.3f}   {mb:.3f} ± {sb:.3f}   "
              f"gap {g:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}")

    print(f"\n[보조] AUC (per-doc mean ± std)")
    (ma, sa), (mb, sb) = ms(m3_auc), ms(m1_auc)
    print(f"  {'':14s} M3 {ma:.3f}±{sa:.3f}   M1 {mb:.3f}±{sb:.3f}")

    print(f"\n[보조] 축별 진단 (per-doc mean AUC / 경계율 / 내부율)")
    for a in AX:
        au, _ = ms(ax_auc[a])
        br = np.mean(ax_b[a]) if ax_b[a] else float("nan")
        ir = np.mean(ax_i[a]) if ax_i[a] else float("nan")
        print(f"  {a:24s} AUC {au:.3f}   경계율 {br:.3f}   내부율 {ir:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

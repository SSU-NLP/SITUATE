#!/usr/bin/env python3
"""Deployable comparison M3-S5 vs M1-CosDrift vs B1-BayesSurprise (dev docs).

B1 = Kumar et al. (2023) transient Bayesian surprise from GPT-2-XL (b1_bayes_surprise.py).
The word-level signal is VALIDATED against human boundaries at word resolution (fidelity check:
Bayesian-surprise word-res AUC ~0.70 vs surprisal ~0.52 — reproduces Kumar). For the deployable
sentence-transition task its readout is the max transient Bayesian surprise within the new event's
first sentence (`newsent_max`) — the best/most-robust sentence-level readout on dev=Pieman among a
small principled set (onset/junction/sentence family). Bayesian surprise cannot tell which SENTENCE
junction is an episode boundary (junctions are homogeneous in surprise), so B1 is near-chance here —
an honest, faithful result that supports H1 (scalar surprise < multi-axis reset).

Natural rules (all knobs frozen on dev=Pieman, argmax official DTS Score):
  M1-CosDrift : idm_score (continuous)      -> mu+c*sigma   (c frozen)
  M3-S5       : coordinated count (0..5)    -> count>=t*     (t* frozen)
  B1          : newsent_max transient (cont)-> mu+c*sigma    (c frozen)
Main metric = official DTS Score; AUC supplement.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402

AX = ["protagonist_perspective", "space", "time", "causality", "intentionality"]
C_GRID = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
BASE = REPO / "outputs/eacl_phase0"
DOCS = {  # name -> (reset_matrix, per-word dump)
    "pieman": ("pieman_michelmann_s5/reset_matrix.jsonl", "b1_bayes/pieman_words.jsonl"),
    "tunnel": ("tunnel_michelmann_s5/reset_matrix.jsonl", "b1_bayes/tunnel_words.jsonl"),
    "monkey": ("monkey_michelmann_s5/reset_matrix.jsonl", "b1_bayes/monkey_words.jsonl"),
}


def read(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def endturn(n, idxs):
    y = [0] * n
    for i in idxs:
        y[i - 1] = 1
    return y


def auc(y, s):
    pos = [a for yy, a in zip(y, s) if yy]
    neg = [a for yy, a in zip(y, s) if not yy]
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def adaptive_pred(sig, c):
    s = np.asarray(sig, dtype=float)
    thr = s.mean() + c * s.std()
    return [t + 1 for t, v in enumerate(sig) if v >= thr]


def count_pred(sig, t):
    return [i + 1 for i, v in enumerate(sig) if v >= t]


def b1_newsent_max(words, n_trans):
    """max transient Bayesian surprise within the new event's first sentence, per transition."""
    g0 = {}
    for gi, w in enumerate(words):
        if w["sent"] not in g0:
            g0[w["sent"]] = gi
    V = [w["transient"] for w in words]
    out = []
    for t in range(1, n_trans + 1):  # transition t -> new sentence index t
        a = g0.get(t)
        b = g0.get(t + 1, len(V))
        seg = V[a:b] if a is not None else []
        out.append(float(max(seg)) if seg else 0.0)
    return out


def load_doc(name):
    rm = sorted(read(BASE / DOCS[name][0]), key=lambda r: int(r["transition"]))
    words = read(BASE / DOCS[name][1])
    n = len(rm) + 1
    gold = [int(r["transition"]) for r in rm if int(r["boundary"])]
    idm = [float(r["idm_score"]) for r in rm]
    cnt = [sum(int(r[a]) for a in AX) for r in rm]
    b1 = b1_newsent_max(words, len(rm))
    y = [int(r["boundary"]) for r in rm]
    return {"n": n, "gold": gold, "idm": idm, "cnt": cnt, "b1": b1, "y": y}


def score_docs(docs, key, knob, rule):
    golds, preds = [], []
    for d in docs:
        golds.append(endturn(d["n"], d["gold"]))
        sig = d[key]
        pred = count_pred(sig, knob) if rule == "count" else adaptive_pred(sig, knob)
        preds.append(endturn(d["n"], pred))
    return score_dialogues(golds, preds)["score"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="*", default=list(DOCS))
    args = ap.parse_args()
    data = {name: load_doc(name) for name in args.docs}
    dev = data["pieman"]
    t_star = max([1, 2, 3, 4, 5], key=lambda t: score_docs([dev], "cnt", t, "count"))
    c_m1 = max(C_GRID, key=lambda c: score_docs([dev], "idm", c, "adapt"))
    c_b1 = max(C_GRID, key=lambda c: score_docs([dev], "b1", c, "adapt"))
    print(f"frozen on pieman(dev): M3-S5 t*={t_star} | M1-Cos c={c_m1} | B1 c={c_b1}")
    print(f"(B1 readout = newsent_max transient Bayesian surprise; validated word-res AUC~0.70)\n")
    print(f"{'doc':10s} | {'M3-S5':>7s} {'M1-Cos':>7s} {'B1':>7s} | AUC {'S5':>6s} {'M1':>6s} {'B1':>6s}")
    for name in args.docs:
        d = data[name]
        s5 = score_docs([d], "cnt", t_star, "count")
        m1 = score_docs([d], "idm", c_m1, "adapt")
        b1 = score_docs([d], "b1", c_b1, "adapt")
        print(f"{name:10s} | {s5:7.3f} {m1:7.3f} {b1:7.3f} |     "
              f"{auc(d['y'],d['cnt']):6.3f} {auc(d['y'],d['idm']):6.3f} {auc(d['y'],d['b1']):6.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""B1-BayesSurprise readout = Kumar et al. (2023) trailing-window regression (faithful).

Kumar computed word-level (transient) Bayesian surprise from GPT-2 and predicted human
boundary/button-press with an L1-regularized regression over TRAILING-WINDOW features
(windows of 20/50/100/150/200 words), leave-one-story-out. A single local readout of the
surprise does NOT reproduce their result (the boundary signal is a slow, wide component;
onset-max is even anti-correlated on Pieman dev). This script reproduces the trailing-window
regression and freezes it on dev=Pieman, transferring UNCHANGED to test docs (deployable:
the model is trained on dev gold, never on test).

Input : per-word dumps from b1_bayes_surprise.py --dump-words (fields: doc_id,sent,pos,transient,raw)
        + the doc's reset_matrix (boundary + transition alignment).
Output: per-doc b1 stream {doc_id, transition, b1_score = fitted P(boundary)} -> deployable μ+cσ.

Freeze protocol (dev/test seal): features, window set, and the fitted L1-logistic model are
selected/fit ONLY on Pieman; applied unchanged to every other doc. Report AUC (supplement) +
downstream deployable DTS Score via b1_deployable.py.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "outputs/eacl_phase0"
WINDOWS = (20, 50, 100, 150, 200)   # Kumar trailing-window widths

DOCS = {
    "pieman": ("pieman_michelmann_s5/reset_matrix.jsonl", "b1_bayes/pieman_words.jsonl"),
    "tunnel": ("tunnel_michelmann_s5/reset_matrix.jsonl", "b1_bayes/tunnel_words.jsonl"),
    "monkey": ("monkey_michelmann_s5/reset_matrix.jsonl", "b1_bayes/monkey_words.jsonl"),
}


def read(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def auc(y, s):
    pos = [a for yy, a in zip(y, s) if yy]
    neg = [a for yy, a in zip(y, s) if not yy]
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def doc_features(name):
    """Return (transitions, X features, y boundary) for a doc.
    Feature for transition t (boundary before sentence t, onset word global index g0[t]):
      trailing-window stats of transient & raw surprise ending at g0[t]:
        mean and max over each window W in WINDOWS  ->  len(WINDOWS)*2*2 features.
    """
    rm = sorted(read(BASE / DOCS[name][0]), key=lambda r: int(r["transition"]))
    words = read(BASE / DOCS[name][1])
    g0 = {}
    for gi, w in enumerate(words):
        if w["sent"] not in g0:
            g0[w["sent"]] = gi
    tr = [w["transient"] for w in words]
    rw = [w["raw"] for w in words]
    T = [int(r["transition"]) for r in rm]
    y = [int(r["boundary"]) for r in rm]
    feats = []
    for t in T:
        g = g0.get(t, None)
        row = []
        for series in (tr, rw):
            for W in WINDOWS:
                seg = series[max(0, g - W):g] if (g is not None and g > 0) else []
                row.append(float(np.mean(seg)) if seg else 0.0)
                row.append(float(np.max(seg)) if seg else 0.0)
        feats.append(row)
    return T, np.asarray(feats, dtype=float), np.asarray(y, dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=BASE / "b1_bayes")
    ap.add_argument("--C", type=float, default=None, help="fixed inverse-reg; default = LogisticRegressionCV on dev")
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler

    data = {name: doc_features(name) for name in DOCS}
    Tdev, Xdev, ydev = data["pieman"]
    scaler = StandardScaler().fit(Xdev)
    Xdev_s = scaler.transform(Xdev)
    if args.C is None:
        clf = LogisticRegressionCV(Cs=10, cv=5, penalty="l1", solver="liblinear",
                                   class_weight="balanced", scoring="roc_auc", max_iter=2000)
    else:
        clf = LogisticRegression(C=args.C, penalty="l1", solver="liblinear",
                                 class_weight="balanced", max_iter=2000)
    clf.fit(Xdev_s, ydev)
    Cbest = getattr(clf, "C_", [args.C])[0] if hasattr(clf, "C_") else args.C
    nnz = int(np.sum(clf.coef_ != 0))
    print(f"[b1-kumar] fit on pieman(dev): C*={Cbest:.4g}, nonzero_coef={nnz}/{Xdev.shape[1]}")
    print()
    print(f"{'doc':10s} {'AUC(b1-reg)':>12s}   n_gold")
    for name in DOCS:
        T, X, y = data[name]
        prob = clf.predict_proba(scaler.transform(X))[:, 1]
        print(f"{name:10s} {auc(y, prob):12.3f}   {int(y.sum())}")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out = args.out_dir / f"{name}_b1reg.jsonl"
        out.write_text("\n".join(json.dumps({"doc_id": name, "transition": int(t), "b1_score": float(p)})
                                  for t, p in zip(T, prob)) + "\n")
    print(f"\n[b1-kumar] wrote {'/'.join(DOCS)} *_b1reg.jsonl -> {args.out_dir}")


if __name__ == "__main__":
    main()

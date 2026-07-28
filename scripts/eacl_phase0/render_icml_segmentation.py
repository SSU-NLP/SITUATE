#!/usr/bin/env python3
"""Qualitative side-by-side segmentation of the ICML Korean diary by M1 / M3 / B1.

No gold (inspection only). Each method self-determines boundaries (deployable), knobs frozen on
dev=Pieman (domain transfer to Korean diary — qualitative only):
  M1-CosDrift : idm_score (already in annotations) -> mu+c*sigma, c=1.0
  M3-S5       : coordinated 5-axis reset count      -> count >= t*=3   (built from icml_diary_s5 states)
  B1          : newsent_max transient Bayesian surprise (polyglot-ko-1.3b) -> mu+c*sigma, c=0.75

Prints every sentence with a marker per method that placed a boundary BEFORE it, so boundaries can
be eyeballed. B1 uses a KOREAN autoregressive LM (polyglot-ko), the Korean analog of Kumar's GPT-2-XL.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from llm_s5_states import (carry_forward, reset_place, reset_time, reset_causality,  # noqa: E402
                           reset_perspective, reset_protagonist_charset, AXES)

BASE = REPO / "outputs/eacl_phase0"
ANN = BASE / "icml_phase0_annotations.jsonl"
S5_DIR = BASE / "icml_diary_s5/icml_diary_2026"
B1_WORDS = BASE / "b1_bayes/icml_words_ko.jsonl"
AX = ["protagonist_perspective", "place", "time", "causality", "goal"]


def read(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def adaptive(sig, c):
    s = np.asarray(sig, dtype=float)
    thr = s.mean() + c * s.std()
    return set(t + 1 for t, v in enumerate(sig) if v >= thr)  # transition indices (1-based)


def count_pred(sig, t):
    return set(i + 1 for i, v in enumerate(sig) if v >= t)


def m3_counts(rows):
    """Build coordinated 5-axis reset count per transition from cached S5 states."""
    st = {}
    for ax in AXES:
        p = S5_DIR / f"{ax}_states.jsonl"
        st[ax] = {x["item_id"]: x for x in read(p)}
    st["place"] = carry_forward(st["place"], "location", "unspecified")
    st["goal"] = carry_forward(st["goal"], "active_goal", "none")
    n = len(rows) + 1
    counts = []
    for i in range(1, n):
        pp = reset_perspective(st["protagonist"][i - 1], st["protagonist"][i])
        row = {
            "protagonist_perspective": pp,
            "place": reset_place(st["place"][i - 1], st["place"][i]),
            "time": reset_time(st["time"][i - 1], st["time"][i]),
            "causality": reset_causality(st["causality"][i - 1], st["causality"][i]),
            "goal": reset_goal_safe(st["goal"][i - 1], st["goal"][i]),
        }
        counts.append(sum(int(row[a]) for a in AX))
    return counts


def reset_goal_safe(a, b):
    from llm_s5_states import reset_goal
    return reset_goal(a, b)


def b1_newsent_max(words, n_trans):
    g0 = {}
    for gi, w in enumerate(words):
        if w["sent"] not in g0:
            g0[w["sent"]] = gi
    V = [w["transient"] for w in words]
    out = []
    for t in range(1, n_trans + 1):
        a = g0.get(t)
        b = g0.get(t + 1, len(V))
        seg = V[a:b] if a is not None else []
        out.append(float(max(seg)) if seg else 0.0)
    return out


def main():
    rows = sorted(read(ANN), key=lambda r: int(r["transition"]))
    sents = [rows[0]["previous"], *[r["current"] for r in rows]]
    n_trans = len(rows)
    idm = [float(r["idm_score"]) for r in rows]
    m1 = adaptive(idm, 1.0)
    b1 = adaptive(b1_newsent_max(read(B1_WORDS), n_trans), 0.75)
    m3 = count_pred(m3_counts(rows), 3) if (S5_DIR / "goal_states.jsonl").exists() else set()
    print(f"ICML diary: {len(sents)} sentences, {n_trans} transitions. Boundaries self-determined (deployable).")
    print(f"M1={len(m1)}  M3={len(m3)}  B1={len(b1)} boundaries.  markers before each sentence:\n")
    for si, s in enumerate(sents):
        marks = []
        if si in m1: marks.append("M1")
        if si in m3: marks.append("M3")
        if si in b1: marks.append("B1")
        if marks:
            print(f"   {'─' * 30} [{'|'.join(marks)}]")
        print(f"{si:3d}  {s}")


if __name__ == "__main__":
    main()

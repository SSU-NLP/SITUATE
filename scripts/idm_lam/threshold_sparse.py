"""Sparse-robust no-parameter count calibration on the ms_mean signal.
Candidate peaks = local maxima. Then decide how many to keep WITHOUT a tuned parameter:
  otsu_all : Otsu over all transitions + NMS (current capstone; over-fires on sparse AMI)
  otsu_pk  : keep local-max peaks whose value > Otsu(of the PEAK VALUES only)
  knee_pk  : keep peaks above the largest-gap (knee) in sorted peak values
Peaks-only calibration adapts to sparsity (few prominent peaks -> few boundaries).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from dts_common import embed_split, ENCODER_NAME  # noqa: E402
from gate2_heldout import load_ami  # noqa: E402
from adaptive_robust import otsu, _nms_from_thresh  # noqa: E402
from capstone_segment import ms_mean  # noqa: E402
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from hi_ontop.ami_scoring import score_meetings  # noqa: E402

DEV = "cuda"


def local_maxima(s):
    pk = []
    for t in range(len(s)):
        left = s[t] >= s[t - 1] if t > 0 else True
        right = s[t] >= s[t + 1] if t + 1 < len(s) else True
        if left and right:
            pk.append(t)
    return pk


def otsu_all(s):
    return _nms_from_thresh(s, np.full(len(s), otsu(s)))


def otsu_pk(s):
    pk = local_maxima(s)
    if len(pk) < 3:
        return set(pk)
    vals = np.array([s[t] for t in pk]); th = otsu(vals)
    return {t for t in pk if s[t] > th}


def knee_pk(s):
    pk = local_maxima(s)
    if len(pk) < 3:
        return set(pk)
    vals = np.sort(np.array([s[t] for t in pk]))[::-1]     # descending
    gaps = vals[:-1] - vals[1:]
    # search knee in the upper region [1 .. len/2] to avoid trivial top-1 cut
    hi = max(2, len(gaps) // 2)
    cut = 1 + int(np.argmax(gaps[1:hi])) if hi > 1 else 1
    thr = vals[cut]
    return {t for t in pk if s[t] > thr}


RULES = {"otsu_all": otsu_all, "otsu_pk": otsu_pk, "knee_pk": knee_pk}


def main():
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(ENCODER_NAME, device="cuda")
    for name, kind in [("tiage_test", None), ("superseg_test", None),
                       ("dialseg711_test", None), ("ami_eval", "ami")]:
        data = load_ami("eval", enc) if kind == "ami" else embed_split(name)
        golds = []; preds = {m: [] for m in RULES}; npred = {m: 0 for m in RULES}; ng = 0
        for d in data:
            n = len(d["turns"]); emb = d["emb"]; s = ms_mean(emb)
            if kind == "ami":
                bt = list(d["bnd_top"]); bt[-1] = 0; golds.append(bt); ng += sum(bt)
            else:
                g = [1 if (i + 1) in d["boundaries"] else 0 for i in range(n)]; g[-1] = 0
                golds.append(g); ng += sum(g)
            for m in RULES:
                sel = RULES[m](s)
                if kind == "ami":
                    p = [0] * n
                    for t in sel:
                        if t + 1 < n:
                            p[t + 1] = 1
                    p[-1] = 0
                else:
                    p = [1 if t in sel else 0 for t in range(n)]; p[-1] = 0
                preds[m].append(p); npred[m] += sum(p)
        scorer = score_meetings if kind == "ami" else score_dialogues
        print(f"\n=== {name} ({'NOISY' if kind=='ami' else 'clean'}) === gold={ng}")
        print(f"{'rule':10s}{'F1':>8s}{'Score':>8s}{'n_pred':>8s}")
        for m in RULES:
            r = scorer(golds, preds[m])
            print(f"{m:10s}{r['f1']:>8.3f}{r['score']:>8.3f}{npred[m]:>8d}")


if __name__ == "__main__":
    main()

"""No-parameter adaptive threshold robust to SPARSE boundaries (fix AMI over-segmentation).

Compares boundary DECISION rules (all parameter-free, no per-dataset tuning) on cosine & FDM_best:
  mu1sig     : mu + 1*sigma over full past (current; over-fires on sparse-boundary streams).
  MAD+NMS    : median + 1.4826*MAD (robust ~1 sigma) + non-maximum suppression (one peak / excursion).
  otsu+NMS   : per-dialogue Otsu threshold (parameter-free bimodal split) + NMS.
NMS collapses each above-threshold run to its single local max => self-limits the boundary count.
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
from lam import LAM  # noqa: E402
from gate2_heldout import load_ami  # noqa: E402
from sticky_sem import error_matrix  # noqa: E402
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from hi_ontop.ami_scoring import score_meetings  # noqa: E402

OUT = HERE.parents[1] / "outputs/experiments/2026-07-09_idm_lam_gonogo/results"
DEV = "cuda"
WARMUP = 5


def _nms_from_thresh(s, thresh):
    """thresh: per-index threshold array. Keep local-max of each contiguous above-thresh run."""
    above = s > thresh
    bounds = set()
    t = 0
    n = len(s)
    while t < n:
        if above[t]:
            j = t
            while j + 1 < n and above[j + 1]:
                j += 1
            seg = s[t:j + 1]
            bounds.add(t + int(np.argmax(seg)))   # single peak per excursion
            t = j + 1
        else:
            t += 1
    return {b for b in bounds if b >= WARMUP}


def rule_mu1sig(s):
    th = np.full(len(s), np.inf)
    for t in range(WARMUP, len(s)):
        past = s[:t]; th[t] = past.mean() + past.std()
    return {t for t in range(WARMUP, len(s)) if s[t] > th[t]}


def rule_mad_nms(s):
    th = np.full(len(s), np.inf)
    for t in range(WARMUP, len(s)):
        past = s[:t]; med = np.median(past); mad = np.median(np.abs(past - med))
        th[t] = med + 1.4826 * mad
    return _nms_from_thresh(s, th)


def otsu(x):
    """Parameter-free bimodal threshold (maximize between-class variance)."""
    lo, hi = x.min(), x.max()
    if hi <= lo:
        return hi
    bins = np.linspace(lo, hi, 64)
    best_t, best_v = bins[0], -1
    for t in bins:
        a, b = x[x <= t], x[x > t]
        if len(a) == 0 or len(b) == 0:
            continue
        v = len(a) * len(b) * (a.mean() - b.mean()) ** 2
        if v > best_v:
            best_v, best_t = v, t
    return best_t


def rule_otsu_nms(s):
    th = np.full(len(s), otsu(s))
    return _nms_from_thresh(s, th)


RULES = {"mu1sig": rule_mu1sig, "MAD+NMS": rule_mad_nms, "otsu+NMS": rule_otsu_nms}


def main():
    ck = torch.load(OUT / "lam_corpus_f1.0.pt", weights_only=False)
    a = ck["args"]
    model = LAM(code_dim=a["code_dim"], codebook_size=a["codebook_size"],
               commitment_weight=a["commitment"]).to(DEV)
    model.load_state_dict(ck["state"]); model.eval()
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(ENCODER_NAME, device="cuda")

    for name, kind in [("tiage_test", None), ("superseg_test", None),
                       ("dialseg711_test", None), ("ami_eval", "ami")]:
        data = load_ami("eval", enc) if kind == "ami" else embed_split(name)
        golds = []
        combos = [(sig, rule) for sig in ["cosine", "FDM_best"] for rule in RULES]
        preds = {c: [] for c in combos}; npred = {c: 0 for c in combos}
        ng = 0
        for d in data:
            n = len(d["turns"]); emb = d["emb"]
            E, delta = error_matrix(model, emb)
            sigs = {"cosine": delta, "FDM_best": E.min(1)}
            if kind == "ami":
                bt = list(d["bnd_top"]); bt[-1] = 0; golds.append(bt); ng += sum(bt)
            else:
                g = [1 if (i + 1) in d["boundaries"] else 0 for i in range(n)]; g[-1] = 0
                golds.append(g); ng += sum(g)
            for (sig, rule) in combos:
                sel = RULES[rule](sigs[sig])
                if kind == "ami":
                    p = [0] * n
                    for t in sel:
                        if t + 1 < n:
                            p[t + 1] = 1
                    p[-1] = 0
                else:
                    p = [1 if t in sel else 0 for t in range(n)]; p[-1] = 0
                preds[(sig, rule)].append(p); npred[(sig, rule)] += sum(p)
        scorer = score_meetings if kind == "ami" else score_dialogues
        print(f"\n=== {name} ===  (gold boundaries={ng}, parameter-free, no tuning)")
        print(f"{'signal':10s}{'rule':10s}{'F1':>7s}{'Pk':>7s}{'WD':>7s}{'Score':>7s}{'n_pred':>8s}")
        for c in combos:
            r = scorer(golds, preds[c])
            print(f"{c[0]:10s}{c[1]:10s}{r['f1']:>7.3f}{r['pk']:>7.3f}{r['wd']:>7.3f}{r['score']:>7.3f}{npred[c]:>8d}")


if __name__ == "__main__":
    main()

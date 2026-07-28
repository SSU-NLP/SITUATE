"""Online (causal) count calibration for IDM signals, and re-eval on 4 held-out sets.
online_otsu_pk: confirm a local peak with 1-turn delay (no future beyond that), then boundary iff
  peak value > Otsu(PAST peak values seen so far) — self-calibrates count causally.
Compare vs online mu+1sigma, offline otsu_pk (global, reference), density-matched (ceiling).
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
from adaptive_robust import otsu  # noqa: E402
from threshold_sparse import otsu_pk as otsu_pk_offline  # noqa: E402
from online_survey import mu1sig  # noqa: E402
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from hi_ontop.ami_scoring import score_meetings  # noqa: E402

OUT = HERE.parents[1] / "outputs/experiments/2026-07-09_idm_lam_gonogo/results"
DEV = "cuda"
WARMUP = 5


def online_otsu_pk(s, min_peaks=5):
    """Causal: peak at p confirmed at p+1; boundary iff s[p] > Otsu(past peaks). Look-ahead <=1 turn."""
    bounds = set(); past = []
    n = len(s)
    for t in range(2, n):
        if s[t - 1] >= s[t] and s[t - 1] > s[t - 2]:      # t-1 is a local max (known at t)
            p = t - 1; v = s[p]
            if p < WARMUP:
                past.append(v); continue
            if len(past) >= min_peaks:
                thr = otsu(np.asarray(past))
            elif past:
                med = np.median(past); mad = np.median(np.abs(np.asarray(past) - med))
                thr = med + 1.4826 * mad
            else:
                thr = -np.inf
            if v > thr:
                bounds.add(p)
            past.append(v)
    return bounds


def topn(s, N):
    return set(np.argsort(-s)[:N].tolist()) if N > 0 else set()


def main():
    ck = torch.load(OUT / "lam_corpus_f1.0.pt", weights_only=False)
    a = ck["args"]
    model = LAM(code_dim=a["code_dim"], codebook_size=a["codebook_size"],
               commitment_weight=a["commitment"]).to(DEV)
    model.load_state_dict(ck["state"]); model.eval()
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(ENCODER_NAME, device="cuda")

    RULES = {
        "on:mu1sig": ("online", mu1sig),
        "on:otsu_pk": ("online", online_otsu_pk),
        "off:otsu_pk": ("offline", otsu_pk_offline),
        "oracle(dens)": ("dens", None),
    }
    print(f"{'set':16s}{'signal':10s}{'rule':14s}{'F1':>7s}{'Score':>7s}{'npred':>7s}{'gold':>6s}")
    for name, kind in [("tiage_test", None), ("superseg_test", None),
                       ("dialseg711_test", None), ("ami_eval", "ami")]:
        data = load_ami("eval", enc) if kind == "ami" else embed_split(name)
        cache = []
        for d in data:
            E, delta = error_matrix(model, d["emb"])
            cache.append((d, {"cosine": delta, "FDM_best": E.min(1)}))
        for sig_name in ["FDM_best", "cosine"]:
            for rname, (kind2, rule) in RULES.items():
                golds, preds = [], []; npred = 0; ng = 0
                for d, ss in cache:
                    n = len(d["turns"]); s = ss[sig_name]
                    gold = {t for t in range(n - 1) if (t + 1) in d["boundaries"]}
                    if kind2 == "dens":
                        sel = topn(s, len(gold))
                    else:
                        sel = rule(s)
                    if kind == "ami":
                        bt = list(d["bnd_top"]); bt[-1] = 0; golds.append(bt); ng += sum(bt)
                        p = [0] * n
                        for t in sel:
                            if t + 1 < n:
                                p[t + 1] = 1
                        p[-1] = 0
                    else:
                        g = [1 if (i + 1) in d["boundaries"] else 0 for i in range(n)]; g[-1] = 0
                        golds.append(g); ng += sum(g)
                        p = [1 if t in sel else 0 for t in range(n)]; p[-1] = 0
                    preds.append(p); npred += sum(p)
                scorer = score_meetings if kind == "ami" else score_dialogues
                r = scorer(golds, preds)
                print(f"{name:16s}{sig_name:10s}{rname:14s}{r['f1']:>7.3f}{r['score']:>7.3f}"
                      f"{npred:>7d}{ng:>6d}", flush=True)


if __name__ == "__main__":
    main()

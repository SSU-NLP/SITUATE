"""DEPLOYABLE, PARAMETER-FREE evaluation: online adaptive threshold mu + c*sigma with c=1.0
FIXED (calibration-free, project convention), full-past causal stats, warmup=5. NO density-matching
(the method decides its OWN boundary count), NO per-dataset tuning. Same for all held-out sets.

Compares the boundary SIGNAL only:
  cosine  = 1 - cos(z_t, z_{t+1})            [= project delta_eff core / method A]
  FDM_best= min_k ||FDM(z_t,c_k) - z_{t+1}||^2  [SEM-style learned-dynamics surprise]
Scale-invariant threshold (normalizes by sigma) => no per-signal parameter either.
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
C = 1.0        # FIXED, calibration-free
WARMUP = 5


def adaptive_bounds(s):
    """Online mu+C*sigma over full past (causal). Return list of transition indices flagged."""
    flagged = []
    for t in range(len(s)):
        if t < WARMUP:
            continue
        past = s[:t]
        if s[t] > past.mean() + C * past.std():
            flagged.append(t)
    return set(flagged)


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
        preds = {"cosine": [], "FDM_best": []}
        ng, npred = 0, {"cosine": 0, "FDM_best": 0}
        for d in data:
            n = len(d["turns"]); emb = d["emb"]
            E, delta = error_matrix(model, emb)
            sigs = {"cosine": delta, "FDM_best": E.min(1)}
            if kind == "ami":
                bt = list(d["bnd_top"]); bt[-1] = 0; golds.append(bt); ng += sum(bt)
            else:
                g = [1 if (i + 1) in d["boundaries"] else 0 for i in range(n)]; g[-1] = 0
                golds.append(g); ng += sum(g)
            for m, s in sigs.items():
                sel = adaptive_bounds(s)
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
        print(f"\n=== {name} ===  (c={C} fixed, no density-match, no tuning; gold boundaries={ng})")
        print(f"{'signal':10s}{'F1':>8s}{'Pk':>8s}{'WD':>8s}{'Score':>8s}{'n_pred':>8s}")
        for m in ["cosine", "FDM_best"]:
            r = scorer(golds, preds[m])
            print(f"{m:10s}{r['f1']:>8.3f}{r['pk']:>8.3f}{r['wd']:>8.3f}{r['score']:>8.3f}{npred[m]:>8d}")


if __name__ == "__main__":
    main()

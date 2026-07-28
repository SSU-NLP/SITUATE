"""Test the noise hypothesis: does WINDOW-AGGREGATION rescue embedding signals on noisy AMI?
block-cosine(w) = 1 - cos( mean(z[t-w+1..t]), mean(z[t+1..t+w]) )  -- aggregates turn-level churn.
Density-matched, ±2 tolerant (AMI) / DTS official. Compares raw vs block(w) vs C_lex.
"""
from __future__ import annotations

import re
from collections import Counter
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
WORD = re.compile(r"[a-zA-Z']+")


def block_cos(emb, w):
    n = len(emb); out = np.zeros(n - 1)
    for t in range(n - 1):
        L = emb[max(0, t - w + 1):t + 1].mean(0)
        R = emb[t + 1:min(n, t + 1 + w)].mean(0)
        L = L / (np.linalg.norm(L) + 1e-9); R = R / (np.linalg.norm(R) + 1e-9)
        out[t] = 1 - float(L @ R)
    return out


def lex(turns, w=2):
    toks = [Counter(x.lower() for x in WORD.findall(s)) for s in turns]
    n = len(turns); out = []
    for t in range(n - 1):
        L = Counter(); R = Counter()
        for i in range(max(0, t - w + 1), t + 1):
            L.update(toks[i])
        for i in range(t + 1, min(n, t + 1 + w)):
            R.update(toks[i])
        ks = set(L) | set(R)
        if not ks:
            out.append(0.0); continue
        a = np.array([L[k] for k in ks], float); b = np.array([R[k] for k in ks], float)
        out.append(1 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
    return np.array(out)


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

    METH = ["cos_raw", "cos_w2", "cos_w3", "cos_w5", "FDM_best", "C_lex", "random"]
    for name, kind in [("ami_eval", "ami"), ("tiage_test", None)]:
        data = load_ami("eval", enc) if kind == "ami" else embed_split(name)
        golds = []; preds = {m: [] for m in METH}; rng = np.random.default_rng(0)
        for d in data:
            n = len(d["turns"]); emb = d["emb"]
            gold = {t for t in range(n - 1) if (t + 1) in d["boundaries"]}; N = len(gold)
            E, delta = error_matrix(model, emb)
            sig = {"cos_raw": delta, "cos_w2": block_cos(emb, 2), "cos_w3": block_cos(emb, 3),
                   "cos_w5": block_cos(emb, 5), "FDM_best": E.min(1), "C_lex": lex(d["turns"])}
            sel = {m: topn(sig[m], N) for m in sig}
            sel["random"] = set(rng.permutation(n - 1)[:N].tolist())
            if kind == "ami":
                bt = list(d["bnd_top"]); bt[-1] = 0; golds.append(bt)
            else:
                g = [1 if (i + 1) in d["boundaries"] else 0 for i in range(n)]; g[-1] = 0; golds.append(g)
            for m in METH:
                if kind == "ami":
                    p = [0] * n
                    for t in sel[m]:
                        if t + 1 < n:
                            p[t + 1] = 1
                    p[-1] = 0
                else:
                    p = [1 if t in sel[m] else 0 for t in range(n)]; p[-1] = 0
                preds[m].append(p)
        scorer = score_meetings if kind == "ami" else score_dialogues
        print(f"\n=== {name} === (density-matched, {'±2 tol' if kind=='ami' else 'DTS'})")
        print(f"{'method':10s}{'F1':>8s}{'Score':>8s}")
        for m in METH:
            r = scorer(golds, preds[m])
            print(f"{m:10s}{r['f1']:>8.3f}{r['score']:>8.3f}")


if __name__ == "__main__":
    main()

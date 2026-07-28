"""Performance push #2: apply the project's proven trick -- LOCAL ADAPTIVE normalization
(trailing-window z-score, ~ delta_eff's mu+c*sigma) -- to the signals before density-matched
top-N. Tests raw vs adaptive-W for cosine-delta, FDM_best, and fusion(0.5). Goal: crack the
human-TIAGE ceiling.
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


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def adaptive(s, W):
    """Local z-score: (s[t] - trailing mean) / trailing std over window W (causal)."""
    if W is None:
        return s
    out = np.zeros_like(s)
    for t in range(len(s)):
        lo = max(0, t - W)
        win = s[lo:t] if t > lo else s[:1]
        m, sd = win.mean(), win.std()
        out[t] = (s[t] - m) / (sd + 1e-9)
    return out


def topn(sig, N):
    return set(np.argsort(-sig)[:N].tolist()) if N > 0 else set()


def build_signals(model, emb):
    E, delta = error_matrix(model, emb)
    fdm_best = E.min(1)
    fusion = 0.5 * zscore(delta) + 0.5 * zscore(fdm_best)
    return {"cosine": delta, "FDM_best": fdm_best, "fusion0.5": fusion}


def eval_set(model, name, kind, enc, configs):
    data = load_ami("eval", enc) if kind == "ami" else embed_split(name)
    golds = []
    preds = {c: [] for c in configs}
    f1 = {c: [] for c in configs}
    for d in data:
        n = len(d["turns"]); emb = d["emb"]
        gold = {t for t in range(n - 1) if (t + 1) in d["boundaries"]}
        sigs = build_signals(model, emb)
        if kind == "ami":
            bt = list(d["bnd_top"]); bt[-1] = 0; N = sum(bt); golds.append(bt)
        else:
            N = len(gold); g = [1 if (i + 1) in d["boundaries"] else 0 for i in range(n)]; g[-1] = 0
            golds.append(g)
        for cname, (sig, W) in configs.items():
            sel = topn(adaptive(sigs[sig], W), N)
            if kind == "ami":
                p = [0] * n
                for t in sel:
                    if t + 1 < n:
                        p[t + 1] = 1
                p[-1] = 0
            else:
                p = [1 if t in sel else 0 for t in range(n)]; p[-1] = 0
            preds[cname].append(p)
            f1[cname].append(len(sel & gold) / len(gold) if gold else 0.0)
    scorer = score_meetings if kind == "ami" else score_dialogues
    return {c: scorer(golds, preds[c]) for c in configs}


def main():
    ck = torch.load(OUT / "lam_corpus_f1.0.pt", weights_only=False)
    a = ck["args"]
    model = LAM(code_dim=a["code_dim"], codebook_size=a["codebook_size"],
               commitment_weight=a["commitment"]).to(DEV)
    model.load_state_dict(ck["state"]); model.eval()
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(ENCODER_NAME, device="cuda")

    configs = {}
    for sig in ["cosine", "FDM_best", "fusion0.5"]:
        for W in [None, 10, 20, 40]:
            configs[f"{sig}/{'raw' if W is None else f'W{W}'}"] = (sig, W)

    for name, kind in [("tiage_test", None), ("ami_eval", "ami"),
                       ("superseg_test", None), ("dialseg711_test", None)]:
        out = eval_set(model, name, kind, enc, configs)
        print(f"\n=== {name} ===")
        print(f"{'config':20s}{'F1':>8s}{'Score':>8s}")
        best = max(out, key=lambda c: out[c]["score"])
        for c in configs:
            star = "  <== best" if c == best else ""
            print(f"{c:20s}{out[c]['f1']:>8.3f}{out[c]['score']:>8.3f}{star}")


if __name__ == "__main__":
    main()

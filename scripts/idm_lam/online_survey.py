"""ONLINE-only survey of our methods on the 4 held-out sets, deployable (own count), same scorers.
Signals (causal): cosine=1-cos(z_t,z_{t+1}); FDM_best=min_k||FDM(z_t,c_k)-z_{t+1}|| (lam_corpus_f1.0).
Online thresholds (no lookahead, no oracle count):
  mu1sig  : mu + 1*sigma over full past (causal).
  robpeak : median + 1.4826*MAD over trailing W past + rising (causal robust peak).
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


def mu1sig(s):
    out = []
    for t in range(len(s)):
        if t < WARMUP:
            continue
        past = s[:t]
        if s[t] > past.mean() + past.std():
            out.append(t)
    return set(out)


def robpeak(s, W=20):
    out = []
    for t in range(len(s)):
        if t < WARMUP:
            continue
        past = s[max(0, t - W):t]
        med = np.median(past); mad = np.median(np.abs(past - med))
        thr = med + 1.4826 * mad
        if s[t] > thr and s[t] >= s[t - 1]:
            out.append(t)
    return set(out)


RULES = {"mu1sig": mu1sig, "robpeak": robpeak}


def main():
    ck = torch.load(OUT / "lam_corpus_f1.0.pt", weights_only=False)
    a = ck["args"]
    model = LAM(code_dim=a["code_dim"], codebook_size=a["codebook_size"],
               commitment_weight=a["commitment"]).to(DEV)
    model.load_state_dict(ck["state"]); model.eval()
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(ENCODER_NAME, device="cuda")

    def sigs(emb):
        E, delta = error_matrix(model, emb)
        return {"cosine": delta, "FDM_best": E.min(1)}

    print(f"{'set':16s}{'signal':10s}{'rule':9s}{'F1':>7s}{'Pk':>7s}{'WD':>7s}{'Score':>7s}{'npred':>7s}{'gold':>6s}")
    for name, kind in [("tiage_test", None), ("superseg_test", None),
                       ("dialseg711_test", None), ("ami_eval", "ami")]:
        data = load_ami("eval", enc) if kind == "ami" else embed_split(name)
        # cache signals per dialogue
        cache = [(d, sigs(d["emb"])) for d in data]
        for sig_name in ["cosine", "FDM_best"]:
            for rule_name, rule in RULES.items():
                golds, preds = [], []; npred = 0; ng = 0
                for d, ss in cache:
                    n = len(d["turns"]); sel = rule(ss[sig_name])
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
                print(f"{name:16s}{sig_name:10s}{rule_name:9s}{r['f1']:>7.3f}{r['pk']:>7.3f}"
                      f"{r['wd']:>7.3f}{r['score']:>7.3f}{npred:>7d}{ng:>6d}", flush=True)


if __name__ == "__main__":
    main()

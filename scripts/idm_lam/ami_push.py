"""How high can AMI go? Multimodal fusion (lexical + embedding-multiscale + learned-dynamics),
density-matched ±2 tolerant (oracle-count ceiling). If ceiling << 0.5, need LLM-level signal.
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
from dts_common import ENCODER_NAME  # noqa: E402
from lam import LAM  # noqa: E402
from gate2_heldout import load_ami  # noqa: E402
from capstone_segment import ms_mean  # noqa: E402
from sticky_sem import error_matrix  # noqa: E402
from hi_ontop.ami_scoring import score_meetings  # noqa: E402

OUT = HERE.parents[1] / "outputs/experiments/2026-07-09_idm_lam_gonogo/results"
DEV = "cuda"
WORD = re.compile(r"\w+", re.UNICODE)


def z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def lex_ms(turns, wins=(2, 3, 5)):
    """Multi-scale lexical incohesion, z-averaged (mirror of ms_mean but lexical)."""
    toks = [Counter(w.lower() for w in WORD.findall(s)) for s in turns]
    n = len(turns)
    zs = []
    for w in wins:
        out = np.zeros(n - 1)
        for t in range(n - 1):
            L = Counter(); R = Counter()
            for i in range(max(0, t - w + 1), t + 1):
                L.update(toks[i])
            for i in range(t + 1, min(n, t + 1 + w)):
                R.update(toks[i])
            ks = set(L) | set(R)
            if ks:
                a = np.array([L[k] for k in ks], float); b = np.array([R[k] for k in ks], float)
                out[t] = 1 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        zs.append(z(out))
    return np.stack(zs, 0).mean(0)


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
    data = load_ami("eval", enc)

    METH = ["emb(ms_mean)", "lex(ms)", "fdm_best",
            "emb+lex", "emb+lex+fdm", "ORACLE", "random"]
    golds = []; preds = {m: [] for m in METH}; rng = np.random.default_rng(0)
    for d in data:
        n = len(d["turns"]); emb = d["emb"]
        gold_tr = {t for t in range(n - 1) if (t + 1) in d["boundaries"]}; N = len(gold_tr)
        E, _ = error_matrix(model, emb)
        e = z(ms_mean(emb)); lx = z(lex_ms(d["turns"])); fb = z(E.min(1))
        sig = {"emb(ms_mean)": e, "lex(ms)": lx, "fdm_best": fb,
               "emb+lex": e + lx, "emb+lex+fdm": e + lx + fb}
        sel = {m: topn(sig[m], N) for m in sig}
        sel["ORACLE"] = set(gold_tr)
        sel["random"] = set(rng.permutation(n - 1)[:N].tolist())
        bt = list(d["bnd_top"]); bt[-1] = 0; golds.append(bt)
        for m in METH:
            p = [0] * n
            for t in sel[m]:
                if t + 1 < n:
                    p[t + 1] = 1
            p[-1] = 0; preds[m].append(p)
    print("AMI eval, density-matched (oracle count), ±2 tolerant. Ceiling check.")
    print(f"{'signal':16s}{'F1':>8s}{'Score':>8s}")
    for m in METH:
        r = score_meetings(golds, preds[m])
        print(f"{m:16s}{r['f1']:>8.3f}{r['score']:>8.3f}")


if __name__ == "__main__":
    main()

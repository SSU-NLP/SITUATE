"""LLM-FREE push for AMI: add MEETING STRUCTURE signals (timing gaps, speaker novelty) to the
embedding signal. Meeting topic shifts often coincide with pauses / a new speaker taking the floor.
Density-matched ±2 tolerant (ceiling check). All parameter-light, no LLM.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from dts_common import ENCODER_NAME  # noqa: E402
from capstone_segment import ms_mean  # noqa: E402
from hi_ontop.ami_scoring import score_meetings  # noqa: E402

ROOT = HERE.parents[1]
DEV = "cuda"


def z(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-9)


def load_raw_eval(enc):
    sp = json.loads((ROOT / "data/ami/split.json").read_text())["eval"]
    out = []
    for mid in sp:
        d = json.loads((ROOT / f"data/ami/topic/{mid}.json").read_text())
        turns = [" ".join(t["text"].split()) for t in d["turns"]]
        starts = [float(t["start"]) for t in d["turns"]]
        spk = [t["speaker"] for t in d["turns"]]
        emb = enc.encode(turns, batch_size=256, normalize_embeddings=True,
                         convert_to_numpy=True).astype(np.float32)
        bnd = {i for i, b in enumerate(d["bnd_top"]) if b == 1}
        out.append({"turns": turns, "emb": emb, "starts": starts, "spk": spk,
                    "boundaries": bnd, "bnd_top": list(d["bnd_top"])})
    return out


def struct_signals(d):
    n = len(d["turns"]); starts = d["starts"]; spk = d["spk"]
    gap = np.array([starts[t + 1] - starts[t] for t in range(n - 1)])   # inter-turn interval
    W = 5
    spk_new = np.zeros(n - 1)
    for t in range(n - 1):
        recent = set(spk[max(0, t + 1 - W):t + 1])
        spk_new[t] = 1.0 if spk[t + 1] not in recent else 0.0
    return {"gap": gap, "spk_new": spk_new}


def topn(s, N):
    return set(np.argsort(-s)[:N].tolist()) if N > 0 else set()


def main():
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(ENCODER_NAME, device="cuda")
    data = load_raw_eval(enc)
    METH = ["emb(ms_mean)", "gap", "spk_new", "emb+gap", "emb+spk", "emb+gap+spk", "ORACLE", "random"]
    golds = []; preds = {m: [] for m in METH}; rng = np.random.default_rng(0)
    for d in data:
        n = len(d["turns"])
        e = z(ms_mean(d["emb"])); st = struct_signals(d)
        g_, sp_ = z(st["gap"]), z(st["spk_new"])
        gold = {t for t in range(n - 1) if (t + 1) in d["boundaries"]}; N = len(gold)
        sig = {"emb(ms_mean)": e, "gap": g_, "spk_new": sp_,
               "emb+gap": e + g_, "emb+spk": e + sp_, "emb+gap+spk": e + g_ + sp_}
        sel = {m: topn(sig[m], N) for m in sig}
        sel["ORACLE"] = set(gold); sel["random"] = set(rng.permutation(n - 1)[:N].tolist())
        bt = list(d["bnd_top"]); bt[-1] = 0; golds.append(bt)
        for m in METH:
            p = [0] * n
            for t in sel[m]:
                if t + 1 < n:
                    p[t + 1] = 1
            p[-1] = 0; preds[m].append(p)
    print("AMI eval, density-matched, ±2 tolerant. LLM-free structure signals.")
    print(f"{'signal':14s}{'F1':>8s}{'Score':>8s}")
    for m in METH:
        r = score_meetings(golds, preds[m])
        print(f"{m:14s}{r['f1']:>8.3f}{r['score']:>8.3f}")


if __name__ == "__main__":
    main()

"""Is the AMI 'IDM win' independent of gap(timing), or a subset of it?
Train noise β=10, get IDM boundary signal A=1-cos(ā_t,ā_{t-1}) on AMI eval, plus gap signal.
Density-matched top-N each. Partition GOLD boundaries into {both / only-IDM / only-gap / neither}
(±2 tolerant). If only-IDM > 0 substantially -> IDM catches what gap misses (independent).
Also: fraction of IDM predictions that coincide (±2) with a gap prediction (IDM ⊂ gap?).
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
from train_corpus import load_corpus  # noqa: E402
from confirm_noise import train_seed  # noqa: E402

ROOT = HERE.parents[1]
DEV = "cuda"
TOL = 2


def load_ami_raw(enc):
    sp = json.loads((ROOT / "data/ami/split.json").read_text())["eval"]
    out = []
    for mid in sp:
        d = json.loads((ROOT / f"data/ami/topic/{mid}.json").read_text())
        turns = [" ".join(t["text"].split()) for t in d["turns"]]
        starts = [float(t["start"]) for t in d["turns"]]
        emb = enc.encode(turns, batch_size=256, normalize_embeddings=True,
                         convert_to_numpy=True).astype(np.float32)
        out.append({"emb": emb, "starts": starts, "bnd_top": list(d["bnd_top"]), "n": len(turns)})
    return out


def topn_turns(sig, N):
    """sig over transitions t=0..n-2; boundary turn = t+1. Return set of predicted turn idx."""
    if N <= 0:
        return set()
    return {int(t) + 1 for t in np.argsort(-sig)[:N]}


def hit(pred_set, g):
    return any(abs(p - g) <= TOL for p in pred_set)


def main():
    emb, pl = load_corpus()
    m, recon = train_seed(emb, pl, 0.3, 8, 10.0, 64, 0)
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(ENCODER_NAME, device="cuda")
    data = load_ami_raw(enc)

    both = only_idm = only_gap = neither = 0
    tot_gold = 0
    idm_pred_total = 0
    idm_pred_near_gap = 0
    idm_correct = gap_correct = 0
    for d in data:
        n = d["n"]; emb_d = d["emb"]; starts = d["starts"]
        bt = list(d["bnd_top"]); bt[-1] = 0
        gold = [i for i in range(n) if bt[i] == 1]
        N = len(gold); tot_gold += N
        if N == 0:
            continue
        r = m.repr_seq(emb_d, DEV)
        sigA = np.array([1 - float(r[t] @ r[t - 1] / (np.linalg.norm(r[t]) * np.linalg.norm(r[t - 1]) + 1e-9))
                         if t >= 1 else 0.0 for t in range(len(r))])
        gap = np.array([starts[t + 1] - starts[t] for t in range(n - 1)])
        P_idm = topn_turns(sigA, N)
        P_gap = topn_turns(gap, N)
        for g in gold:
            hi, hg = hit(P_idm, g), hit(P_gap, g)
            if hi and hg:
                both += 1
            elif hi:
                only_idm += 1
            elif hg:
                only_gap += 1
            else:
                neither += 1
        idm_correct += sum(hit(P_idm, g) for g in gold)
        gap_correct += sum(hit(P_gap, g) for g in gold)
        # predicted-set overlap: IDM preds near a gap pred
        for p in P_idm:
            idm_pred_total += 1
            if any(abs(p - q) <= TOL for q in P_gap):
                idm_pred_near_gap += 1

    print(f"=== AMI eval: IDM(noise β=10, frac0.3) vs gap(timing), density-matched, ±{TOL} ===")
    print(f"gold boundaries total: {tot_gold}")
    print(f"IDM recall(맞힌 gold): {idm_correct} ({idm_correct/tot_gold:.2%})")
    print(f"gap recall(맞힌 gold): {gap_correct} ({gap_correct/tot_gold:.2%})")
    print(f"\ngold 분할 (±{TOL}):")
    print(f"  둘 다 맞힘        : {both} ({both/tot_gold:.1%})")
    print(f"  IDM만 맞힘(gap놓침): {only_idm} ({only_idm/tot_gold:.1%})   <-- IDM 독립 기여")
    print(f"  gap만 맞힘(IDM놓침): {only_gap} ({only_gap/tot_gold:.1%})")
    print(f"  둘 다 놓침        : {neither} ({neither/tot_gold:.1%})")
    print(f"\nIDM 예측 중 gap 예측과 ±{TOL} 겹침: {idm_pred_near_gap}/{idm_pred_total} "
          f"({idm_pred_near_gap/max(idm_pred_total,1):.1%})  (높을수록 IDM⊂gap)")


if __name__ == "__main__":
    main()

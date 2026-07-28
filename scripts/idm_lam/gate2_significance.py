"""Fast per-dialogue matched-boundary-F1 + Wilcoxon (A_typed vs Bp vs random), all K.

Density-matched => per dialogue pred count == gold count => precision=recall=F1=hits/N.
Position match uses the transition index directly (end-turn convention identical for all).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dts_common import embed_split  # noqa: E402
from lam import LAM  # noqa: E402
from gate2_segmentation import code_seq, calibrate  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "outputs/experiments/2026-07-09_idm_lam_gonogo/results"
DEV = "cuda"


def topn_set(scores, N):
    if N <= 0:
        return set()
    return set(np.argsort(-scores)[:N].tolist())


def dlg_f1(pred_set, gold_set):
    if not gold_set:
        return 1.0 if not pred_set else 0.0
    return len(pred_set & gold_set) / len(gold_set)


def run_tag(tag):
    ck = torch.load(OUT / f"lam_{tag}.pt", weights_only=False)
    a = ck["args"]; K = a["codebook_size"]
    model = LAM(code_dim=a["code_dim"], codebook_size=K, commitment_weight=a["commitment"]).to(DEV)
    model.load_state_dict(ck["state"]); model.eval()
    train = embed_split("superseg_train")
    shiftness, P, uni = calibrate(model, train, DEV, K)

    res = {}
    for ds in ["tiage_test", "dialseg711_test", "superseg_test"]:
        data = embed_split(ds)
        f1 = {"A_typed": [], "A_bigram": [], "Bp": [], "D_random": []}
        for d in data:
            emb = d["emb"]; n = len(d["turns"])
            gold = {t for t in range(n - 1) if (t + 1) in d["boundaries"]}  # transition idx
            N = len(gold)
            c = code_seq(model, emb, DEV)
            delta = 1.0 - (emb[:-1] * emb[1:]).sum(1)
            s_typed = shiftness[c]
            s_big = np.array([-np.log(P[c[t - 1], c[t]]) if t > 0 else -np.log(uni[c[t]])
                              for t in range(len(c))])
            f1["A_typed"].append(dlg_f1(topn_set(s_typed, N), gold))
            f1["A_bigram"].append(dlg_f1(topn_set(s_big, N), gold))
            f1["Bp"].append(dlg_f1(topn_set(delta, N), gold))
            rng = np.random.default_rng(7)
            f1["D_random"].append(dlg_f1(set(rng.permutation(len(delta))[:N].tolist()), gold))
        def w(x, y):
            d = np.array(x) - np.array(y)
            p = 1.0 if np.all(d == 0) else float(wilcoxon(x, y).pvalue)
            return {"dF1": float(d.mean()), "p": p}
        res[ds] = {
            "meanF1": {k: float(np.mean(v)) for k, v in f1.items()},
            "A_typed_vs_Bp": w(f1["A_typed"], f1["Bp"]),
            "A_typed_vs_Drandom": w(f1["A_typed"], f1["D_random"]),
        }
    return res


if __name__ == "__main__":
    allres = {}
    for tag in ["K8", "K16", "K32"]:
        allres[tag] = run_tag(tag)
        print(f"\n############ {tag} ############")
        for ds, r in allres[tag].items():
            mf = r["meanF1"]
            print(f"{ds:16s} A_typed={mf['A_typed']:.3f} A_bigram={mf['A_bigram']:.3f} "
                  f"Bp={mf['Bp']:.3f} rand={mf['D_random']:.3f}")
            print(f"{'':16s}  A-Bp dF1={r['A_typed_vs_Bp']['dF1']:+.3f} p={r['A_typed_vs_Bp']['p']:.1e}"
                  f" | A-rand dF1={r['A_typed_vs_Drandom']['dF1']:+.3f} p={r['A_typed_vs_Drandom']['p']:.1e}")
    (OUT / "gate2_significance.json").write_text(json.dumps(allres, indent=2))
    print("\nsaved -> gate2_significance.json")

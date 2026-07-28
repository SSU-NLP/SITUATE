"""Honest head-to-head: our no-parameter (ms_mean + otsu_pk) vs the project's REAL delta_eff
(HiOnTopV2: ctx_window=2, decay=0.7, blend=0.5, adaptive mu+c*sigma, c=1.0 fixed).
Both deployable (own count), no tuning, same encoder/embeddings, official DTS scorer, held-out.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from dts_common import embed_split  # noqa: E402
from capstone_segment import ms_mean  # noqa: E402
from threshold_sparse import otsu_pk  # noqa: E402
from hi_ontop.hi_ontop_v2 import HiOnTopV2  # noqa: E402
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402


def deltaeff_pred(emb):
    """Real delta_eff (HiOnTopV2 c=1.0). is_b[i]=new seg at turn i -> end-turn[i-1]."""
    seg = HiOnTopV2(dim=emb.shape[1], c=1.0, warmup=5)
    isb = []
    for s in emb:
        _, b = seg.assign(np.asarray(s, dtype=np.float64))
        isb.append(bool(b))
    n = len(emb)
    p = [1 if (t + 1 < n and isb[t + 1]) else 0 for t in range(n)]
    p[-1] = 0
    return p


def ours_pred(emb):
    sel = otsu_pk(ms_mean(emb))
    n = len(emb)
    p = [1 if t in sel else 0 for t in range(n)]
    p[-1] = 0
    return p


def main():
    print("Deployable, no-tuning, held-out. Official DTS scorer (no density-match).")
    print(f"{'set':16s}{'method':20s}{'F1':>7s}{'Pk':>7s}{'WD':>7s}{'Score':>7s}{'n_pred':>8s}")
    for ds in ["tiage_test", "dialseg711_test", "superseg_test"]:
        data = embed_split(ds)
        golds = []
        preds = {"delta_eff(HiOnTopV2)": [], "ours(ms_mean+otsu_pk)": []}
        npred = {k: 0 for k in preds}
        for d in data:
            n = len(d["turns"])
            g = [1 if (i + 1) in d["boundaries"] else 0 for i in range(n)]; g[-1] = 0
            golds.append(g)
            pe = deltaeff_pred(d["emb"]); po = ours_pred(d["emb"])
            preds["delta_eff(HiOnTopV2)"].append(pe); npred["delta_eff(HiOnTopV2)"] += sum(pe)
            preds["ours(ms_mean+otsu_pk)"].append(po); npred["ours(ms_mean+otsu_pk)"] += sum(po)
        for m in preds:
            r = score_dialogues(golds, preds[m])
            print(f"{ds if m==list(preds)[0] else '':16s}{m:20s}{r['f1']:>7.3f}{r['pk']:>7.3f}"
                  f"{r['wd']:>7.3f}{r['score']:>7.3f}{npred[m]:>8d}")


if __name__ == "__main__":
    main()

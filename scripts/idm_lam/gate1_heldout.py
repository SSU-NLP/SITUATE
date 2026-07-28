"""Gate 1 leakage diagnostic on HELD-OUT eval sets, corpus-trained IDM.

Per eval set (tiage_test, superseg_test, dialseg711_test, ami_eval): does the code encode
CONTENT (leakage) or TRANSITION CHARACTER?  Fair control vs surprise delta binned to K.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dts_common import embed_split, ENCODER_NAME  # noqa: E402
from lam import LAM  # noqa: E402
from gate1_diagnostics import mutual_info, ridge_r2, entropy  # noqa: E402
from gate2_segmentation import code_seq  # noqa: E402
from gate2_heldout import load_ami  # noqa: E402

OUT = HERE.parents[1] / "outputs/experiments/2026-07-09_idm_lam_gonogo/results"
DEV = "cuda"


def transitions(data, model):
    Zt, Z1, bl = [], [], []
    idxs = []
    for d in data:
        e = d["emb"]; bnd = d["boundaries"]
        c = code_seq(model, e, DEV)
        for t in range(len(e) - 1):
            Zt.append(e[t]); Z1.append(e[t + 1]); bl.append(1 if (t + 1) in bnd else 0)
            idxs.append(c[t])
    return np.stack(Zt), np.stack(Z1), np.array(bl), np.array(idxs)


def diag(name, data, model, K):
    Zt, Z1, bl, idx = transitions(data, model)
    from sklearn.cluster import KMeans
    ntop = min(30, max(5, len(Z1) // 200))
    topic = KMeans(ntop, n_init=4, random_state=0).fit_predict(Z1)
    delta = 1 - (Zt * Z1).sum(1)
    qs = np.quantile(delta, np.linspace(0, 1, K + 1)[1:-1]); dbin = np.digitize(delta, qs)
    _, nmi_ct = mutual_info(idx, topic)
    _, nmi_cb = mutual_info(idx, bl)
    _, nmi_dt = mutual_info(dbin, topic)
    _, nmi_db = mutual_info(dbin, bl)

    def cond(code):
        v, tot = [], 0
        for c in np.unique(topic):
            m = topic == c
            if m.sum() < 30 or len(np.unique(bl[m])) < 2 or len(np.unique(code[m])) < 2:
                continue
            _, nm = mutual_info(code[m].astype(int), bl[m].astype(int)); v.append(nm * m.sum()); tot += m.sum()
        return float(sum(v) / tot) if tot else 0.0
    oh = np.eye(K)[idx]
    r2c = ridge_r2(oh, Z1); r2ch = ridge_r2(oh, Z1 - Zt)
    counts = np.bincount(idx, minlength=K)
    return {
        "n_transitions": int(len(idx)), "codes_used": int((counts > 0).sum()),
        "perplexity": float(np.exp(entropy(counts))),
        "nmi_code_topic": nmi_ct, "nmi_code_boundary": nmi_cb,
        "nmi_delta_topic": nmi_dt, "nmi_delta_boundary": nmi_db,
        "cond_code_bnd": cond(idx), "cond_delta_bnd": cond(dbin),
        "r2_code_content": r2c, "r2_code_change": r2ch, "base_rate": float(bl.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="corpus_K16")
    args = ap.parse_args()
    ck = torch.load(OUT / f"lam_{args.tag}.pt", weights_only=False)
    a = ck["args"]; K = a["codebook_size"]
    model = LAM(code_dim=a["code_dim"], codebook_size=K, commitment_weight=a["commitment"]).to(DEV)
    model.load_state_dict(ck["state"]); model.eval()

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(ENCODER_NAME, device="cuda")

    sets = {ds: embed_split(ds) for ds in ["tiage_test", "superseg_test", "dialseg711_test"]}
    sets["ami_eval"] = load_ami("eval", enc)
    res = {name: diag(name, data, model, K) for name, data in sets.items()}
    (OUT / f"gate1_heldout_{args.tag}.json").write_text(json.dumps(res, indent=2))
    print(f"=== Gate-1 held-out leakage ({args.tag}, K={K}) ===")
    print(f"{'set':16s}{'codes':>6s}{'ppl':>6s}{'NMI(c;top)':>11s}{'NMI(d;top)':>11s}"
          f"{'cond(c;b)':>10s}{'cond(d;b)':>10s}{'R2c-cont':>9s}")
    for name, r in res.items():
        print(f"{name:16s}{r['codes_used']:>6d}{r['perplexity']:>6.1f}{r['nmi_code_topic']:>11.3f}"
              f"{r['nmi_delta_topic']:>11.3f}{r['cond_code_bnd']:>10.3f}{r['cond_delta_bnd']:>10.3f}"
              f"{r['r2_code_content']:>9.3f}")


if __name__ == "__main__":
    main()

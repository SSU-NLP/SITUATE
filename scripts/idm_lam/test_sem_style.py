"""Test the SEM-style reframe: use LEARNED forward-dynamics prediction error as the
boundary signal (vs cosine-delta surprise), and code-identity change with stickiness.

Signals per transition t (higher = more likely boundary), density-matched, held-out:
  Bp_cos    : 1 - cos(z_t, z_{t+1})                    [existing delta_eff surprise]
  FDM_err   : ||FDM(z_t, c_mean) - z_{t+1}||^2          [learned forward-dynamics surprise, SEM-style]
  FDM_best  : min over codes ||FDM(z_t, c_k) - z_{t+1}||^2  (best any event explains it; low=explained)
              -> boundary signal = FDM_best (high = no event predicts it well)
  code_chg  : 1 if code_t != code_{t-1} else 0, ranked by -log P(c_t|c_{t-1}) among changes
Compares matched-F1 on TIAGE(human)/superseg/ami(spoken) vs Bp_cos.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dts_common import embed_split, ENCODER_NAME  # noqa: E402
from lam import LAM  # noqa: E402
from gate2_segmentation import code_seq  # noqa: E402
from gate2_heldout import load_ami  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

OUT = HERE.parents[1] / "outputs/experiments/2026-07-09_idm_lam_gonogo/results"
DEV = "cuda"


def signals(model, emb):
    z = torch.tensor(emb, device=DEV)
    zt, z1 = z[:-1], z[1:]
    with torch.no_grad():
        _, q, idx, _ = model.encode_transition(zt, z1)
        # learned forward dynamics with mean code (no future info leaked into the code)
        qmean = model.vq._codebook.embed.reshape(-1, model.code_dim).mean(0, keepdim=True).expand(len(zt), -1)
        fdm_mean = model.fdm(torch.cat([zt, qmean], -1))
        err_mean = ((fdm_mean - z1) ** 2).sum(-1).cpu().numpy()
        # best-code explanation: for each transition, min recon over all K codes
        cb = model.vq._codebook.embed.reshape(-1, model.code_dim)  # (K, d)
        K = cb.shape[0]
        errs = []
        for k in range(K):
            ck = cb[k:k+1].expand(len(zt), -1)
            fk = model.fdm(torch.cat([zt, ck], -1))
            errs.append(((fk - z1) ** 2).sum(-1))
        err_best = torch.stack(errs, 0).min(0).values.cpu().numpy()
    delta = 1.0 - (emb[:-1] * emb[1:]).sum(1)
    return {"Bp_cos": delta, "FDM_err": err_mean, "FDM_best": err_best}, idx.cpu().numpy()


def matched_f1(sig, gold, N):
    if N <= 0:
        return 1.0 if len(sig) == 0 else 0.0
    sel = set(np.argsort(-sig)[:N].tolist())
    return len(sel & gold) / len(gold) if gold else 0.0


def run(model, data, is_ami=False):
    res = {k: [] for k in ["Bp_cos", "FDM_err", "FDM_best"]}
    for d in data:
        emb = d["emb"]; n = len(d["turns"])
        gold = {t for t in range(n - 1) if (t + 1) in d["boundaries"]}
        N = len(gold)
        sig, idx = signals(model, emb)
        for k in res:
            res[k].append(matched_f1(sig[k], gold, N))
    return {k: (np.array(v)) for k, v in res.items()}


def main():
    ck = torch.load(OUT / "lam_corpus_f1.0.pt", weights_only=False)
    a = ck["args"]
    model = LAM(code_dim=a["code_dim"], codebook_size=a["codebook_size"],
               commitment_weight=a["commitment"]).to(DEV)
    model.load_state_dict(ck["state"]); model.eval()

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(ENCODER_NAME, device="cuda")

    sets = {ds: embed_split(ds) for ds in ["tiage_test", "superseg_test", "dialseg711_test"]}
    sets["ami_eval"] = load_ami("eval", enc)

    print(f"{'set':16s}{'Bp_cos':>9s}{'FDM_err':>9s}{'FDM_best':>9s}  (matched-F1; vs Bp_cos p)")
    for name, data in sets.items():
        r = run(model, data)
        def p(a, b):
            d = a - b
            return 1.0 if np.all(d == 0) else float(wilcoxon(a, b).pvalue)
        print(f"{name:16s}{r['Bp_cos'].mean():>9.3f}{r['FDM_err'].mean():>9.3f}{r['FDM_best'].mean():>9.3f}"
              f"   FDM_err p={p(r['FDM_err'], r['Bp_cos']):.2e}  FDM_best p={p(r['FDM_best'], r['Bp_cos']):.2e}")


if __name__ == "__main__":
    main()

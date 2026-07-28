"""Full SEM-style sticky inference: K codes = event vocabulary, event k's dynamics = FDM(.,c_k).

Maintain an active event; at each transition compute error e_k = ||FDM(z_t,c_k)-z_{t+1}||^2 under
each event. Stay in the active event (sticky sCRP: staying gets a +lambda prior bonus) unless
another event explains the transition enough better. Boundary = active-event switch. This is the
completion of the "IDM code as SEM event model" idea, WITH stickiness (unlike the per-step FDM_best).

Density-matched: rank transitions by switch-evidence, take top-N. Swept over stickiness lambda.
Compared to Bp_cos (cosine-delta) with official DTS/AMI scorers on held-out sets.
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
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from hi_ontop.ami_scoring import score_meetings  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

OUT = HERE.parents[1] / "outputs/experiments/2026-07-09_idm_lam_gonogo/results"
DEV = "cuda"


def error_matrix(model, emb):
    """Return E (T x K): squared error of each event-code's forward prediction, + cosine delta."""
    z = torch.tensor(emb, device=DEV); zt, z1 = z[:-1], z[1:]
    with torch.no_grad():
        cb = model.vq._codebook.embed.reshape(-1, model.code_dim)  # (K,d)
        cols = []
        for k in range(cb.shape[0]):
            ck = cb[k:k+1].expand(len(zt), -1)
            cols.append(((model.fdm(torch.cat([zt, ck], -1)) - z1) ** 2).sum(-1))
        E = torch.stack(cols, 1).cpu().numpy()  # (T,K)
    delta = 1.0 - (emb[:-1] * emb[1:]).sum(1)
    return E, delta


def sticky_scores(E, lam):
    """Sequential sticky event inference. Return per-transition switch-evidence (higher=boundary)."""
    T, K = E.shape
    scale = E.mean() + 1e-9
    C = E / scale                      # normalized costs
    active = int(np.argmin(C[0]))
    ev = np.zeros(T)
    for t in range(T):
        stay_cost = C[t, active] - lam           # stickiness: staying is cheaper by lam
        others = C[t].copy(); others[active] = np.inf
        k = int(np.argmin(others))
        switch_cost = others[k]
        ev[t] = stay_cost - switch_cost          # >0 => switching favored => boundary evidence
        if ev[t] > 0:
            active = k                            # switch (greedy MAP)
    return ev


def topn(sig, N, rng=None):
    if N <= 0:
        return set()
    if rng is not None:
        return set(rng.permutation(len(sig))[:N].tolist())
    return set(np.argsort(-sig)[:N].tolist())


def eval_set(model, name, kind, enc, lams):
    data = load_ami("eval", enc) if kind == "ami" else embed_split(name)
    methods = ["Bp_cos"] + [f"sticky_l{l}" for l in lams]
    golds, preds = [], {m: [] for m in methods}
    f1 = {m: [] for m in methods}
    for d in data:
        n = len(d["turns"]); emb = d["emb"]
        gold = {t for t in range(n - 1) if (t + 1) in d["boundaries"]}
        E, delta = error_matrix(model, emb)
        if kind == "ami":
            bt = list(d["bnd_top"]); bt[-1] = 0; N = sum(bt); golds.append(bt)
        else:
            N = len(gold); g = [1 if (i + 1) in d["boundaries"] else 0 for i in range(n)]; g[-1] = 0
            golds.append(g)
        sels = {"Bp_cos": topn(delta, N)}
        for l in lams:
            sels[f"sticky_l{l}"] = topn(sticky_scores(E, l), N)
        for m in methods:
            if kind == "ami":
                p = [0] * n
                for t in sels[m]:
                    if t + 1 < n:
                        p[t + 1] = 1
                p[-1] = 0
            else:
                p = [1 if t in sels[m] else 0 for t in range(n)]; p[-1] = 0
            preds[m].append(p)
            f1[m].append(len(sels[m] & gold) / len(gold) if gold else 0.0)
    scorer = score_meetings if kind == "ami" else score_dialogues
    out = {m: scorer(golds, preds[m]) for m in methods}
    base = np.array(f1["Bp_cos"])
    for m in methods:
        arr = np.array(f1[m]); d = arr - base
        out[m]["dF1"] = float(d.mean())
        out[m]["p"] = 1.0 if np.all(d == 0) else float(wilcoxon(arr, base).pvalue)
    return out, methods


def main():
    ck = torch.load(OUT / "lam_corpus_f1.0.pt", weights_only=False)
    a = ck["args"]
    model = LAM(code_dim=a["code_dim"], codebook_size=a["codebook_size"],
               commitment_weight=a["commitment"]).to(DEV)
    model.load_state_dict(ck["state"]); model.eval()
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(ENCODER_NAME, device="cuda")

    lams = [0.0, 0.5, 1.0, 2.0]
    for name, kind in [("tiage_test", None), ("ami_eval", "ami"),
                       ("superseg_test", None), ("dialseg711_test", None)]:
        out, methods = eval_set(model, name, kind, enc, lams)
        print(f"\n=== {name} ===  (Bp_cos = existing cosine-delta)")
        print(f"{'method':12s}{'F1':>8s}{'Score':>8s}   vs Bp_cos(ΔF1,p)")
        for m in methods:
            r = out[m]; tag = "" if m == "Bp_cos" else f"   ΔF1={r['dF1']:+.3f} p={r['p']:.1e}"
            print(f"{m:12s}{r['f1']:>8.3f}{r['score']:>8.3f}{tag}")


if __name__ == "__main__":
    main()

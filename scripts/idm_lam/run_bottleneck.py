"""Re-run the go/no-go with CONTINUOUS bottlenecks (sparse 1st, noise 2nd) vs VQ (control).
Loads corpus once, trains all three, reports Gate-1 (leakage) + Gate-2 (transition-change boundary).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from dts_common import embed_split  # noqa: E402
from train_corpus import load_corpus  # noqa: E402
from lam_bottleneck import LAMBottleneck  # noqa: E402
from gate1_diagnostics import ridge_r2, mutual_info  # noqa: E402
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from hi_ontop.ami_scoring import score_meetings  # noqa: E402
from gate2_heldout import load_ami  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import adjusted_mutual_info_score as AMI  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

OUT = HERE.parents[1] / "outputs/experiments/2026-07-09_idm_lam_gonogo/results"
DEV = "cuda"


def train(bottleneck, emb, pair_left, frac, epochs, lam, beta, latent_dim):
    torch.manual_seed(0); np.random.seed(0)
    rng = np.random.default_rng(0); pl = pair_left.copy(); rng.shuffle(pl)
    pl = pl[: int(len(pl) * frac)]
    nval = min(150_000, len(pl) // 20); val, tr = pl[:nval], pl[nval:]
    embt = torch.from_numpy(emb)
    m = LAMBottleneck(latent_dim=latent_dim, bottleneck=bottleneck, lam=lam, beta=beta).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    B = 8192
    def batch(idx):
        return embt[idx].to(DEV, torch.float32), embt[idx + 1].to(DEV, torch.float32)
    for ep in range(epochs):
        m.train(); perm = np.random.permutation(len(tr))
        for i in range(0, len(tr), B):
            idx = tr[perm[i:i + B]]; zt, z1 = batch(idx)
            out = m(zt, z1); opt.zero_grad(); out["loss"].backward(); opt.step()
    m.eval()
    with torch.no_grad():
        zt, z1 = batch(val); vo = m(zt, z1)
    return m, vo["recon"].item()


def gate1(m, va):
    Zt, Z1, abar, bl = [], [], [], []
    for d in va:
        e, bnd = d["emb"], d["boundaries"]
        r = m.repr_seq(e, DEV)
        for t in range(len(e) - 1):
            Zt.append(e[t]); Z1.append(e[t + 1]); abar.append(r[t])
            bl.append(1 if (t + 1) in bnd else 0)
    Zt = np.stack(Zt); Z1 = np.stack(Z1); A = np.stack(abar)
    r2_content = ridge_r2(A, Z1)                       # leakage: ā -> next content
    r2_change = ridge_r2(A, Z1 - Zt)                   # ā -> transition delta
    topic = KMeans(30, n_init=3, random_state=0).fit_predict(Z1[:20000])
    clab = KMeans(16, n_init=3, random_state=0).fit_predict(A)   # cluster ā into 16
    ami_topic = AMI(clab[:20000], topic)
    # active-dim stats (sparsity)
    frac_active = float((np.abs(A) > 0.01 * np.abs(A).max()).mean())
    return {"r2_content": r2_content, "r2_change": r2_change,
            "ami_cluster_topic": float(ami_topic), "frac_active_dims": frac_active}


def gate2(m, datasets, enc):
    def sigA(r):
        s = np.zeros(len(r))
        for t in range(1, len(r)):
            a, b = r[t], r[t - 1]
            s[t] = 1 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        return s
    res = {}
    for ds, kind in datasets:
        data = load_ami("eval", enc) if kind == "ami" else embed_split(ds)
        golds = []; preds = {"A": [], "Bp": [], "rand": []}
        f1 = {"A": [], "Bp": [], "rand": []}
        rng = np.random.default_rng(0)
        for d in data:
            n = len(d["turns"]); emb = d["emb"]
            gold = {t for t in range(n - 1) if (t + 1) in d["boundaries"]}; N = len(gold)
            sa = sigA(m.repr_seq(emb, DEV)); delta = 1 - (emb[:-1] * emb[1:]).sum(1)
            sel = {"A": set(np.argsort(-sa)[:N]), "Bp": set(np.argsort(-delta)[:N]),
                   "rand": set(rng.permutation(n - 1)[:N])}
            if kind == "ami":
                bt = list(d["bnd_top"]); bt[-1] = 0; golds.append(bt)
            else:
                g = [1 if (i + 1) in d["boundaries"] else 0 for i in range(n)]; g[-1] = 0; golds.append(g)
            for k in preds:
                if kind == "ami":
                    p = [0] * n
                    for t in sel[k]:
                        if t + 1 < n:
                            p[t + 1] = 1
                    p[-1] = 0
                else:
                    p = [1 if t in sel[k] else 0 for t in range(n)]; p[-1] = 0
                preds[k].append(p)
                f1[k].append(len(sel[k] & gold) / N if N else 0.0)
        scorer = score_meetings if kind == "ami" else score_dialogues
        sc = {k: scorer(golds, preds[k]) for k in preds}
        def p(a, b):
            d = np.array(a) - np.array(b)
            return 1.0 if np.all(d == 0) else float(wilcoxon(a, b).pvalue)
        res[ds] = {"A": sc["A"]["score"], "Bp": sc["Bp"]["score"], "rand": sc["rand"]["score"],
                   "A_F1": sc["A"]["f1"], "Bp_F1": sc["Bp"]["f1"],
                   "p_A_Bp": p(f1["A"], f1["Bp"]), "p_A_rand": p(f1["A"], f1["rand"])}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--latent-dim", type=int, default=64)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these bottlenecks")
    args = ap.parse_args()

    emb, pair_left = load_corpus()
    va = embed_split("superseg_validation")
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", device="cuda")
    dsets = [("tiage_test", None), ("superseg_test", None), ("dialseg711_test", None), ("ami_eval", "ami")]

    for bn in (args.only or ["sparse", "noise", "vq"]):
        m, recon = train(bn, emb, pair_left, args.frac, args.epochs, args.lam, args.beta, args.latent_dim)
        g1 = gate1(m, va)
        g2 = gate2(m, dsets, enc)
        print(f"\n######## bottleneck={bn}  (frac={args.frac}, recon={recon:.4f}) ########")
        print(f"  [Gate1] R2(ā→내용)={g1['r2_content']:.3f}  R2(ā→변화)={g1['r2_change']:.3f}  "
              f"AMI(clust;topic)={g1['ami_cluster_topic']:.3f}  active_dims={g1['frac_active_dims']:.2f}")
        for ds in g2:
            r = g2[ds]
            print(f"  [Gate2] {ds:16s} A={r['A']:.3f}(F1 {r['A_F1']:.3f}) Bp={r['Bp']:.3f}(F1 {r['Bp_F1']:.3f}) "
                  f"rand={r['rand']:.3f}  p(A-Bp)={r['p_A_Bp']:.1e} p(A-rand)={r['p_A_rand']:.1e}")


if __name__ == "__main__":
    main()

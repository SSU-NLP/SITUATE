"""Gate 2: does the IDM code-transition boundary BEAT surprise and density-matched random?

All methods are DENSITY-MATCHED: per dialogue each method emits exactly N = (#gold
boundaries) predicted boundaries (top-N by its score). Under equal counts the official
DTS scorer (Pk/WD/F1, per-dialogue avg) measures POSITION accuracy only -> isolates
"did it put boundaries in the right place" from "did it emit the right count".

Methods (score per transition t, between turn t and t+1; boundary => new seg starts at t+1):
  A_typed  : ours, most-favorable. shiftness(c_t) = TRAIN-calibrated boundary-rate of code c_t.
  A_bigram : ours, unsupervised. -log P(c_t | c_{t-1}) from TRAIN code-bigram (type-novelty).
  Bp       : surprise, SAME embedding. delta_t = 1 - cos(z_t, z_{t+1}).   [fair surprise rival]
  C_lex    : lexical cohesion (TextTiling-style block cosine over token TF).
  D_random : density-matched random (avg over seeds).  ** critical control **
  E_unif   : evenly spaced N boundaries.

Verdict needs BOTH: A > Bp (beats surprise) AND A > D (beats density-matched random).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dts_common import embed_split  # noqa: E402
from lam import LAM  # noqa: E402
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/experiments/2026-07-09_idm_lam_gonogo/results"


def code_seq(model, emb, dev):
    """Return code index array c_t for transitions t=0..n-2 of one dialogue."""
    z = torch.tensor(emb, device=dev)
    with torch.no_grad():
        _, _, idx, _ = model.encode_transition(z[:-1], z[1:])
    return idx.cpu().numpy()


def calibrate(model, train, dev, K):
    """TRAIN-set stats: per-code boundary rate (shiftness) + code-bigram transition probs."""
    bnd_hit = np.zeros(K); bnd_tot = np.zeros(K)
    bigram = np.ones((K, K))  # laplace
    unigram = np.ones(K)
    for d in train:
        c = code_seq(model, d["emb"], dev)
        bnd = d["boundaries"]
        for t in range(len(c)):
            unigram[c[t]] += 1
            bnd_tot[c[t]] += 1
            if (t + 1) in bnd:
                bnd_hit[c[t]] += 1
            if t > 0:
                bigram[c[t - 1], c[t]] += 1
    shiftness = bnd_hit / np.maximum(bnd_tot, 1)
    P = bigram / bigram.sum(1, keepdims=True)
    return shiftness, P, unigram / unigram.sum()


_WORD = re.compile(r"\w+")


def lex_scores(turns):
    """TextTiling-style: 1 - cosine of block-TF (w=2 each side) at each transition t."""
    toks = [Counter(_WORD.findall(s.lower())) for s in turns]
    n = len(turns)
    w = 2
    out = []
    for t in range(n - 1):
        left = Counter()
        for i in range(max(0, t - w + 1), t + 1):
            left.update(toks[i])
        right = Counter()
        for i in range(t + 1, min(n, t + 1 + w)):
            right.update(toks[i])
        keys = set(left) | set(right)
        if not keys:
            out.append(0.0); continue
        a = np.array([left[k] for k in keys], float)
        b = np.array([right[k] for k in keys], float)
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        out.append(1 - cos)
    return np.array(out)


def topn_pred(scores, n_turns, N, rng=None):
    """Pick top-N transitions -> end-turn pred (len n_turns, last=0)."""
    pred = [0] * n_turns
    if N <= 0:
        return pred
    order = np.argsort(-scores)
    if rng is not None:  # random: permute
        order = rng.permutation(len(scores))
    for t in order[:N]:
        pred[t] = 1  # transition t => end-turn label at t (new seg starts t+1)
    pred[-1] = 0
    return pred


def gold_endturn(d):
    n = len(d["turns"])
    g = [1 if (i + 1) in d["boundaries"] else 0 for i in range(n)]
    g[-1] = 0
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="K16")
    ap.add_argument("--datasets", nargs="+", default=["tiage_test", "dialseg711_test", "superseg_test"])
    args = ap.parse_args()
    dev = "cuda"

    ck = torch.load(OUT / f"lam_{args.tag}.pt", weights_only=False)
    a = ck["args"]; K = a["codebook_size"]
    model = LAM(code_dim=a["code_dim"], codebook_size=K, commitment_weight=a["commitment"]).to(dev)
    model.load_state_dict(ck["state"]); model.eval()

    train = embed_split("superseg_train")
    shiftness, P, uni = calibrate(model, train, dev, K)

    results = {}
    for ds in args.datasets:
        data = embed_split(ds)
        golds = [gold_endturn(d) for d in data]
        Ns = [sum(g) for g in golds]
        methods = {k: [] for k in ["A_typed", "A_bigram", "Bp", "C_lex", "E_unif"]}
        rand_runs = {s: [] for s in range(20)}
        for d, N in zip(data, Ns):
            n = len(d["turns"]); emb = d["emb"]
            c = code_seq(model, emb, dev)
            z = emb
            delta = 1.0 - (z[:-1] * z[1:]).sum(1)
            s_typed = shiftness[c]
            s_bigram = np.array([-np.log(P[c[t - 1], c[t]]) if t > 0 else -np.log(uni[c[t]])
                                 for t in range(len(c))])
            s_lex = lex_scores(d["turns"])
            methods["A_typed"].append(topn_pred(s_typed, n, N))
            methods["A_bigram"].append(topn_pred(s_bigram, n, N))
            methods["Bp"].append(topn_pred(delta, n, N))
            methods["C_lex"].append(topn_pred(s_lex, n, N))
            # E uniform: evenly spaced N transitions in [0, n-2]
            if N > 0:
                pos = np.linspace(0, n - 2, N + 2)[1:-1].round().astype(int)
                eu = [0] * n
                for t in pos:
                    eu[t] = 1
                eu[-1] = 0
                methods["E_unif"].append(eu)
            else:
                methods["E_unif"].append([0] * n)
            for s in range(20):
                rng = np.random.default_rng(1000 * s + 7)
                rand_runs[s].append(topn_pred(delta, n, N, rng=rng))

        ds_res = {}
        for m, preds in methods.items():
            ds_res[m] = score_dialogues(golds, preds)
        # per-dialogue F1 for paired significance (A_typed vs Bp, vs one random seed)
        from scipy.stats import wilcoxon

        def perdlg_f1(preds):
            return [score_dialogues([g], [p])["f1"] for g, p in zip(golds, preds)]
        f1_typed = perdlg_f1(methods["A_typed"])
        f1_bp = perdlg_f1(methods["Bp"])
        f1_rand = perdlg_f1(rand_runs[0])
        def wtest(x, y):
            d = np.array(x) - np.array(y)
            if np.all(d == 0):
                return {"delta_mean": 0.0, "p": 1.0}
            try:
                p = float(wilcoxon(x, y, zero_method="wilcox").pvalue)
            except Exception:
                p = float("nan")
            return {"delta_mean": float(np.mean(d)), "p": p}
        ds_res["_sig"] = {"A_typed_vs_Bp": wtest(f1_typed, f1_bp),
                          "A_typed_vs_Drandom": wtest(f1_typed, f1_rand)}
        # random: average scorer output over seeds
        rscores = [score_dialogues(golds, rand_runs[s]) for s in range(20)]
        ds_res["D_random"] = {
            k: float(np.mean([r[k] for r in rscores])) for k in ["pk", "wd", "f1", "score"]
        } | {"std_score": float(np.std([r["score"] for r in rscores])), "n": rscores[0]["n"]}

        # per-dialogue F1 paired significance A_typed vs Bp and vs D_random (Wilcoxon)
        results[ds] = ds_res
        print(f"\n===== {ds}  (n={len(data)}, mean gold boundaries/dlg={np.mean(Ns):.1f}) =====")
        print(f"{'method':10s} {'F1':>7s} {'Pk':>7s} {'WD':>7s} {'Score':>7s}")
        for m in ["A_typed", "A_bigram", "Bp", "C_lex", "D_random", "E_unif"]:
            r = ds_res[m]
            print(f"{m:10s} {r['f1']:7.4f} {r['pk']:7.4f} {r['wd']:7.4f} {r['score']:7.4f}")
        sg = ds_res["_sig"]
        print(f"  sig A_typed-Bp: dF1={sg['A_typed_vs_Bp']['delta_mean']:+.4f} p={sg['A_typed_vs_Bp']['p']:.2e}"
              f" | A_typed-Drand: dF1={sg['A_typed_vs_Drandom']['delta_mean']:+.4f} p={sg['A_typed_vs_Drandom']['p']:.2e}")

    (OUT / f"gate2_{args.tag}.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> gate2_{args.tag}.json")


if __name__ == "__main__":
    main()

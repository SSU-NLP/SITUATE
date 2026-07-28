"""Gate 2 on FULLY HELD-OUT eval, corpus-trained IDM (no eval data in IDM training).

Calibration of the boundary DECISION rule (shiftness / code-bigram) uses TRAIN splits only
(superseg_train, tiage_train, AMI-train meetings) -> proper train/test separation; the IDM
representation itself was trained only on the large unlabeled corpus.

Eval (all held out): tiage_test, dialseg711_test, superseg_test  [DTS scorer, end-turn, -1]
                     AMI 35 meetings                              [AMI scorer, start-turn, +2 tol]
Methods (density-matched): A_bigram (unsup), A_typed (calib), Bp=surprise delta, C_lex, D_random, E_unif.
Verdict: A > Bp AND A > D_random (esp. on human TIAGE & noisy AMI).
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
sys.path.insert(0, str(HERE.parents[1] / "src"))
from dts_common import embed_split, load_split, ENCODER_NAME  # noqa: E402
from lam import LAM  # noqa: E402
from gate2_segmentation import code_seq, calibrate, lex_scores  # noqa: E402
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from hi_ontop.ami_scoring import score_meetings  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

ROOT = HERE.parents[1]
OUT = ROOT / "outputs/experiments/2026-07-09_idm_lam_gonogo/results"
DEV = "cuda"


def load_ami(split_key, enc):
    sp = json.loads((ROOT / "data/ami/split.json").read_text())[split_key]
    out = []
    for mid in sp:
        d = json.loads((ROOT / f"data/ami/topic/{mid}.json").read_text())
        turns = [" ".join(t["text"].split()) for t in d["turns"]]
        emb = enc.encode(turns, batch_size=256, normalize_embeddings=True,
                         convert_to_numpy=True).astype(np.float32)
        bnd = {i for i, b in enumerate(d["bnd_top"]) if b == 1}
        out.append({"id": mid, "turns": turns, "emb": emb, "boundaries": bnd,
                    "bnd_top": list(d["bnd_top"])})
    return out


def scores_for(d, model, shiftness, P, uni):
    emb = d["emb"]; c = code_seq(model, emb, DEV)
    delta = 1.0 - (emb[:-1] * emb[1:]).sum(1)
    s_typed = shiftness[c]
    s_big = np.array([-np.log(P[c[t - 1], c[t]]) if t > 0 else -np.log(uni[c[t]])
                      for t in range(len(c))])
    s_lex = lex_scores(d["turns"])
    return {"A_typed": s_typed, "A_bigram": s_big, "Bp": delta, "C_lex": s_lex}


def topn_transitions(scores, N, rng=None):
    if N <= 0:
        return set()
    if rng is not None:
        return set(rng.permutation(len(scores))[:N].tolist())
    return set(np.argsort(-scores)[:N].tolist())


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

    # ---- calibrate decision rule on TRAIN splits only ----
    calib = embed_split("superseg_train") + embed_split("tiage_train") + load_ami("train", enc)
    shiftness, P, uni = calibrate(model, calib, DEV, K)

    METHODS = ["A_typed", "A_bigram", "Bp", "C_lex", "D_random", "E_unif"]
    results = {}

    def eval_dts(ds):
        data = embed_split(ds)
        golds = []
        preds = {m: [] for m in METHODS}
        f1pd = {m: [] for m in ["A_typed", "A_bigram", "Bp", "D_random"]}
        for d in data:
            n = len(d["turns"])
            gold_tr = {t for t in range(n - 1) if (t + 1) in d["boundaries"]}
            N = len(gold_tr)
            g = [1 if (i + 1) in d["boundaries"] else 0 for i in range(n)]; g[-1] = 0
            golds.append(g)
            sc = scores_for(d, model, shiftness, P, uni)
            sel = {}
            for m in ["A_typed", "A_bigram", "Bp", "C_lex"]:
                sel[m] = topn_transitions(sc[m], N)
            sel["D_random"] = topn_transitions(sc["Bp"], N, rng=np.random.default_rng(7))
            # E uniform
            eu = set(np.linspace(0, n - 2, N + 2)[1:-1].round().astype(int).tolist()) if N > 0 else set()
            sel["E_unif"] = eu
            for m in METHODS:
                p = [1 if t in sel[m] else 0 for t in range(n)]; p[-1] = 0
                preds[m].append(p)
            for m in f1pd:
                inter = len(sel[m] & gold_tr)
                f1pd[m].append(inter / N if N else (1.0 if not sel[m] else 0.0))
        r = {m: score_dialogues(golds, preds[m]) for m in METHODS}
        r["_sig"] = sig_block(f1pd)
        return r

    def eval_ami():
        data = load_ami("eval", enc)
        golds = []
        preds = {m: [] for m in METHODS}
        f1pd = {m: [] for m in ["A_typed", "A_bigram", "Bp", "D_random"]}
        for d in data:
            n = len(d["turns"]); bt = list(d["bnd_top"]); bt[-1] = 0
            gold_tr = {t for t in range(n - 1) if (t + 1) in d["boundaries"]}
            N = sum(bt)  # gold boundary count (start-turn)
            golds.append(bt)
            sc = scores_for(d, model, shiftness, P, uni)
            sel = {}
            for m in ["A_typed", "A_bigram", "Bp", "C_lex"]:
                sel[m] = topn_transitions(sc[m], N)
            sel["D_random"] = topn_transitions(sc["Bp"], N, rng=np.random.default_rng(7))
            eu = set(np.linspace(0, n - 2, N + 2)[1:-1].round().astype(int).tolist()) if N > 0 else set()
            sel["E_unif"] = eu
            for m in METHODS:
                # transition t -> new segment starts at turn t+1 (start-turn pred)
                p = [0] * n
                for t in sel[m]:
                    if t + 1 < n:
                        p[t + 1] = 1
                p[-1] = 0
                preds[m].append(p)
            for m in f1pd:
                inter = len(sel[m] & gold_tr)
                f1pd[m].append(inter / len(gold_tr) if gold_tr else (1.0 if not sel[m] else 0.0))
        r = {m: score_meetings(golds, preds[m]) for m in METHODS}
        r["_sig"] = sig_block(f1pd)
        return r

    def sig_block(f1pd):
        def w(x, y):
            d = np.array(x) - np.array(y)
            p = 1.0 if np.all(d == 0) else float(wilcoxon(x, y).pvalue)
            return {"dF1": float(d.mean()), "p": p}
        return {"A_typed_vs_Bp": w(f1pd["A_typed"], f1pd["Bp"]),
                "A_bigram_vs_Bp": w(f1pd["A_bigram"], f1pd["Bp"]),
                "A_typed_vs_Drandom": w(f1pd["A_typed"], f1pd["D_random"])}

    for ds in ["tiage_test", "dialseg711_test", "superseg_test"]:
        results[ds] = eval_dts(ds)
    results["ami_eval"] = eval_ami()

    (OUT / f"gate2_heldout_{args.tag}.json").write_text(json.dumps(results, indent=2))
    for ds, r in results.items():
        print(f"\n===== {ds} =====")
        print(f"{'method':10s}{'F1':>8s}{'Pk':>8s}{'WD':>8s}{'Score':>8s}")
        for m in METHODS:
            print(f"{m:10s}{r[m]['f1']:8.3f}{r[m]['pk']:8.3f}{r[m]['wd']:8.3f}{r[m]['score']:8.3f}")
        s = r["_sig"]
        print(f"  A_typed-Bp dF1={s['A_typed_vs_Bp']['dF1']:+.3f} p={s['A_typed_vs_Bp']['p']:.1e} | "
              f"A_bigram-Bp dF1={s['A_bigram_vs_Bp']['dF1']:+.3f} p={s['A_bigram_vs_Bp']['p']:.1e} | "
              f"A_typed-rand dF1={s['A_typed_vs_Drandom']['dF1']:+.3f} p={s['A_typed_vs_Drandom']['p']:.1e}")
    print(f"\nsaved -> gate2_heldout_{args.tag}.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Tier-2 held-out FROZEN 1-time evaluation: M3-S5 vs M1-CosDrift vs M2-MonoLAM vs B1-BayesSurprise.

The paper's real held-out number. All knobs FROZEN on dev=Pieman (D-M9 seal), applied UNCHANGED:
  M1-CosDrift  : idm_score (raw-cosine mpnet drift)      -> mu+c*sigma, c=1.0
  M2-MonoLAM   : 1-cos(a_t,a_{t-1}) (LAM transition repr)-> mu+c*sigma, c=0.0
  M3-S5        : coordinated 5-axis reset count (0..5)   -> count >= t*=3
  B1-BayesSurp : newsent_max transient Bayesian surprise -> mu+c*sigma, c=0.75
Main metric = official DTS Score (deployable, self-determined counts). Reports per-doc Score +
**doc-level bootstrap 95% CI** of the gain M3-vs-M1 and M3-vs-B1 (Tier 2's reason for existing).
B2/B3 are the gold generators -> NOT baselines here (D-M6-c circularity).

Requires (run before): gold applied to masters (build_tier2_gold --apply), S5 reset_matrix
(tier2_s5/reset_matrix.jsonl), B1 words (tier2/b1/{sid}_words.jsonl).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/idm_lam"))
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from hi_ontop.embedding import make_encoder  # noqa: E402
from lam import LAM  # noqa: E402

AX = ["protagonist_perspective", "place", "time", "causality", "goal"]
BASE = REPO / "outputs/eacl_phase0/tier2"
S5_RM = REPO / "outputs/eacl_phase0/tier2_s5/reset_matrix.jsonl"
# frozen-on-dev knobs (D-M9 seal)
C_M1, C_M2, T_M3, C_B1 = 1.0, 0.0, 3, 0.75
N_BOOT, SEED = 5000, 20260715


def read(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def endturn(n, idxs):
    y = [0] * n
    for i in idxs:
        y[i - 1] = 1
    return y


def adaptive(sig, c):
    s = np.asarray(sig, float)
    thr = s.mean() + c * s.std()
    return [t + 1 for t, v in enumerate(sig) if v >= thr]


def count_pred(sig, t):
    return [i + 1 for i, v in enumerate(sig) if v >= t]


def b1_newsent_max(words, n_trans):
    g0 = {}
    for gi, w in enumerate(words):
        g0.setdefault(w["sent"], gi)
    V = [w["transient"] for w in words]
    out = []
    for t in range(1, n_trans + 1):
        a, b = g0.get(t), g0.get(t + 1, len(V))
        seg = V[a:b] if a is not None else []
        out.append(float(max(seg)) if seg else 0.0)
    return out


def main():
    enc = make_encoder(model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    ck = torch.load(REPO / "outputs/experiments/2026-07-09_idm_lam_gonogo/results/lam_corpus_f1.0.pt",
                    map_location="cpu", weights_only=False)
    a = ck["args"]
    lam = LAM(code_dim=a["code_dim"], codebook_size=a["codebook_size"], commitment_weight=a["commitment"])
    lam.load_state_dict(ck["state"]); lam.eval()

    s5 = {}
    for r in read(S5_RM):
        s5.setdefault(r["doc_id"], {})[int(r["transition"])] = sum(int(r[ax]) for ax in AX)

    manifest = json.loads((REPO / "data/eacl_phase0/tier2/manifest.json").read_text())
    per_doc = {}  # sid -> {method: score}
    for m in manifest:
        sid = m["story_id"]
        if sid not in s5:  # only evaluate docs whose S5 (M3) is complete (in reset_matrix)
            continue
        rows = sorted(read(BASE / f"{sid}_master.jsonl"), key=lambda r: int(r["transition"]))
        n = len(rows) + 1
        gold = endturn(n, [int(r["transition"]) for r in rows if int(r["boundary"])])
        idm = [float(r["idm_score"]) for r in rows]
        sents = [rows[0]["previous"], *[r["current"] for r in rows]]
        Z = np.asarray(enc.encode(sents), float); Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
        with torch.no_grad():
            A = lam.idm(torch.cat([torch.tensor(Z[:-1], dtype=torch.float32),
                                   torch.tensor(Z[1:] - Z[:-1], dtype=torch.float32)], -1)).numpy()
        A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
        m2 = [0.0] + [float(1 - A[k] @ A[k - 1]) for k in range(1, n - 1)]
        cnt = [s5.get(sid, {}).get(int(r["transition"]), 0) for r in rows]
        b1 = b1_newsent_max(read(BASE / f"b1/{sid}_words.jsonl"), len(rows))
        sc = {
            "M3": score_dialogues([gold], [endturn(n, count_pred(cnt, T_M3))])["score"],
            "M1": score_dialogues([gold], [endturn(n, adaptive(idm, C_M1))])["score"],
            "M2": score_dialogues([gold], [endturn(n, adaptive(m2, C_M2))])["score"],
            "B1": score_dialogues([gold], [endturn(n, adaptive(b1, C_B1))])["score"],
        }
        per_doc[sid] = sc
        print(f"{sid:22s} M3={sc['M3']:.3f} M1={sc['M1']:.3f} M2={sc['M2']:.3f} B1={sc['B1']:.3f}")

    sids = list(per_doc)
    def mean(meth, idx): return float(np.mean([per_doc[sids[i]][meth] for i in idx]))
    allidx = list(range(len(sids)))
    print(f"\nMEAN (n={len(sids)} docs): " + " ".join(
        f"{m}={mean(m, allidx):.3f}" for m in ["M3", "M1", "M2", "B1"]))

    rng = np.random.default_rng(SEED)
    for opp in ["M1", "M2", "B1"]:
        gains = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(sids), len(sids))
            gains.append(mean("M3", idx) - mean(opp, idx))
        lo, hi = np.percentile(gains, [2.5, 97.5])
        pt = mean("M3", allidx) - mean(opp, allidx)
        verdict = "Go" if lo > 0 else ("No-Go" if hi < 0 else "gray")
        print(f"  M3 - {opp}: {pt:+.3f}  95%CI [{lo:+.3f}, {hi:+.3f}]  -> {verdict}")


if __name__ == "__main__":
    main()

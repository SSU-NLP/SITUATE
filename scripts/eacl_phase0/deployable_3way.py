#!/usr/bin/env python3
"""Deployable 3-way Phase-0 evaluation: ① delta_eff(IDM) vs ② LAM vs ③ S5 factored-slot.

DEPLOYABLE (CLAUDE.md 필수): 각 방법이 경계 *개수를 스스로 결정*한다. gold 개수 미제공.
공통 규칙 = per-doc 적응임계 boundary(t) ⟺ signal_t ≥ μ_doc + c·σ_doc,  c 는 dev(Pieman)에서
argmax-Score 로 방법별 동결 → 모든 test 에 그대로 적용. (matched@gold/top-N 아님.)

signals per transition t (1..n-1), z = L2-normalized mpnet sentence embeddings:
  ① IDM   = 1 - cos(z_t, z_{t-1})                      # raw-cosine drift (프로젝트 HP 무관)
  ② LAM   = 1 - cos(a_t, a_{t-1}), a_t = LAM.idm(cat(z_{t-1}, z_t - z_{t-1}))   # 전이표현 변화
  ③ S5    = Σ_5axes reset_t (coordinated count, protagonist=perspective)
Score = 공식 DTS score_dialogues (per-doc 평균, 끝-turn).
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

AX = ["protagonist_perspective", "space", "time", "causality", "intentionality"]
C_GRID = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

enc = make_encoder(model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
_ck = torch.load(REPO / "outputs/experiments/2026-07-09_idm_lam_gonogo/results/lam_corpus_f1.0.pt",
                 map_location="cpu", weights_only=False)
_a = _ck["args"]
lam = LAM(code_dim=_a["code_dim"], codebook_size=_a["codebook_size"], commitment_weight=_a["commitment"])
lam.load_state_dict(_ck["state"]); lam.eval()


def read(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def endturn(n, idxs):
    y = [0] * n
    for i in idxs:
        y[i - 1] = 1
    return y


def doc_signals(rows):
    """rows sorted (one doc). Return (n, gold_transitions, {method: per-transition signal list})."""
    rows = sorted(rows, key=lambda r: int(r["transition"]))
    seq = [rows[0]["previous"], *[r["current"] for r in rows]]
    Z = np.asarray(enc.encode(seq), dtype=np.float64)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    n = len(seq)
    idm = [float(1 - Z[t] @ Z[t - 1]) for t in range(1, n)]          # transition t=1..n-1
    with torch.no_grad():
        zt = torch.tensor(Z[:-1], dtype=torch.float32)
        ztp1 = torch.tensor(Z[1:], dtype=torch.float32)
        A = lam.idm(torch.cat([zt, ztp1 - zt], dim=-1)).numpy()       # a per transition (n-1, cd)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    lam_sig = [0.0] + [float(1 - A[k] @ A[k - 1]) for k in range(1, n - 1)]  # transition t: 1-cos(a_t,a_{t-1})
    cnt = [sum(int(r[ax]) for ax in AX) for r in rows]                 # ③ coordinated count
    resets = {ax: [int(r[ax]) for r in rows] for ax in AX}            # per-axis 0/1 (S5w 가중용)
    gold = [int(r["transition"]) for r in rows if int(r["boundary"])]
    return n, gold, {"IDM": idm, "LAM": lam_sig, "S5": [float(c) for c in cnt]}, resets


def adaptive_pred(sig, c):
    """per-doc μ+cσ deployable boundaries (연속 신호 IDM/LAM용) -> transition indices (1-based)."""
    s = np.asarray(sig)
    thr = s.mean() + c * s.std()
    return [t + 1 for t, v in enumerate(sig) if v >= thr]


def count_pred(sig, t):
    """정수 count ≥ t deployable boundaries (S5 정수 신호용) -> transition indices (1-based)."""
    return [i + 1 for i, v in enumerate(sig) if v >= t]


def load_docs(master_path):
    rows = read(master_path)
    by = {}
    for r in rows:
        by.setdefault(r["doc_id"], []).append(r)
    out = []
    for doc, rs in by.items():
        n, gold, sigs = doc_signals(rs)
        if gold:
            out.append((n, gold, sigs))
    return out


def score_method(docs, method, knob):
    """method S5 -> count≥t (knob=정수 t); IDM/LAM -> μ+cσ (knob=c)."""
    golds, preds = [], []
    for n, gold, sigs in docs:
        golds.append(endturn(n, gold))
        pred = count_pred(sigs[method], knob) if method == "S5" else adaptive_pred(sigs[method], knob)
        preds.append(endturn(n, pred))
    return score_dialogues(golds, preds)["score"]


DATASETS = [
    ("pieman", "fiction(dev)", "outputs/eacl_phase0/pieman_michelmann_s5/reset_matrix.jsonl_MASTER"),
    ("tunnel", "fiction", None), ("monkey", "journalism", None),
    ("tiage", "dialogue", None), ("dialseg711", "dialogue", None),
    ("superseg", "dialogue", None), ("ami", "meeting", None),
]
MASTER = {
    "pieman": "outputs/eacl_phase0/pieman_michelmann_s5/reset_matrix.jsonl",
    "tunnel": "outputs/eacl_phase0/tunnel_michelmann_s5/reset_matrix.jsonl",
    "monkey": "outputs/eacl_phase0/monkey_michelmann_s5/reset_matrix.jsonl",
    "tiage": "outputs/eacl_phase0/dts_tiage_s5/reset_matrix.jsonl",
    "dialseg711": "outputs/eacl_phase0/dts_dialseg711_s5/reset_matrix.jsonl",
    "superseg": "outputs/eacl_phase0/dts_superseg_s5/reset_matrix.jsonl",
    "ami": "outputs/eacl_phase0/dts_ami_s5/reset_matrix.jsonl",
}
# reset_matrix has previous/current? No — need master text. Use the s5 reset_matrix's doc but master for text.
MASTER_TXT = {
    "pieman": "outputs/eacl_phase0/pieman_michelmann_master.jsonl",
    "tunnel": "outputs/eacl_phase0/tunnel_michelmann_master.jsonl",
    "monkey": "outputs/eacl_phase0/monkey_michelmann_master.jsonl",
    "tiage": "outputs/eacl_phase0/dts_tiage_master.jsonl",
    "dialseg711": "outputs/eacl_phase0/dts_dialseg711_master.jsonl",
    "superseg": "outputs/eacl_phase0/dts_superseg_master.jsonl",
    "ami": "outputs/eacl_phase0/dts_ami_master.jsonl",
}


def build_docs(name):
    """join master text + reset_matrix resets by (doc_id, transition). S5w(정보량 가중) 주입."""
    rm = {(r["doc_id"], int(r["transition"])): r for r in read(MASTER[name])}
    rows = read(MASTER_TXT[name])
    by = {}
    for r in rows:
        key = (r["doc_id"], int(r["transition"]))
        if key in rm:
            m = rm[key]
            r = {**r, **{ax: m[ax] for ax in AX}}
            by.setdefault(r["doc_id"], []).append(r)
    raw = []
    for doc, rs in by.items():
        n, gold, sigs, resets = doc_signals(rs)
        if gold:
            raw.append((n, gold, sigs, resets))
    # 데이터셋 pooled 축별 리셋 기저율 p_k → 정보량 가중 w_k = -log p_k
    tot = sum(len(r[3][AX[0]]) for r in raw)
    fired = {ax: sum(sum(r[3][ax]) for r in raw) for ax in AX}
    p = {ax: fired[ax] / tot for ax in AX}
    w = {ax: (-np.log(p[ax]) if 0 < p[ax] < 1 else 0.0) for ax in AX}
    docs = []
    for n, gold, sigs, resets in raw:
        T = len(resets[AX[0]])
        sigs["S5w"] = [sum(resets[ax][t] * w[ax] for ax in AX) for t in range(T)]
        docs.append((n, gold, sigs))
    return docs, w, p


def main():
    built = {name: build_docs(name) for name, _, _ in DATASETS}
    all_docs = {k: v[0] for k, v in built.items()}
    # freeze knob per method on dev = pieman (S5: t∈1..5 정수; IDM/LAM/S5w: c∈C_GRID)
    frozen = {}
    for meth in ["IDM", "LAM", "S5", "S5w"]:
        grid = [1, 2, 3, 4, 5] if meth == "S5" else C_GRID
        frozen[meth] = max(grid, key=lambda k: score_method(all_docs["pieman"], meth, k))
    print(f"frozen: S5 t*={frozen['S5']}, S5w c={frozen['S5w']}, IDM c={frozen['IDM']}, LAM c={frozen['LAM']}")
    print(f"pieman 정보량 가중 w_k = -log p_k: " +
          ", ".join(f"{ax.split('_')[0]} {built['pieman'][1][ax]:.2f}(p{built['pieman'][2][ax]:.2f})" for ax in AX))
    print()
    print(f"{'dataset':12s} {'type':12s} {'③S5':>7s} {'③S5w':>7s} {'①IDM':>7s} {'②LAM':>7s}  winner")
    for name, typ, _ in DATASETS:
        docs = all_docs[name]
        r = {m: score_method(docs, m, frozen[m]) for m in ["S5", "S5w", "IDM", "LAM"]}
        win = max(r.items(), key=lambda x: x[1])[0]
        print(f"{name:12s} {typ:12s} {r['S5']:7.3f} {r['S5w']:7.3f} {r['IDM']:7.3f} {r['LAM']:7.3f}  {win}")


if __name__ == "__main__":
    main()

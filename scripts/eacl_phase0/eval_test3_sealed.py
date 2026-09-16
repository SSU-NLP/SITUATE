"""봉인 시험지(listener-test3) 채점 — Score / BOR / Lift.

🔓 2026-08-04 개봉(사용자 지시). 개봉 선언과 caveat 는 `EACL/eval-suite-spec.md` 말미 참조.
작동점은 **앵커에서 동결한 적응임계**(문서 내 상위 rate 분위수)만 쓴다. test3 gold 밀도를
쓰면 oracle 누출이므로 쓰지 않는다 — 따라서 BOR<1(과소분절)이 구조적으로 예상된다.
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
import train_anchor_m4m5 as T  # noqa: E402


def met(ys, preds):
    ng = sum(sum(y[:-1]) for y in ys); ntr = sum(len(y) - 1 for y in ys)
    npd = sum(sum(p[:-1]) for p in preds)
    hit = sum(int(a) * int(b) for y, p in zip(ys, preds) for a, b in zip(y[:-1], p[:-1]))
    dens = ng / max(ntr, 1); prec = hit / npd if npd else float("nan")
    return {"score": float(np.mean([score_dialogues([y], [p])["score"] for y, p in zip(ys, preds)])),
            "bor": (npd / ng) if ng else float("nan"),
            "lift": (prec / dens) if (npd and dens > 0) else float("nan"), "n_pred": int(npd)}


ap = argparse.ArgumentParser()
ap.add_argument("--ckpt-glob", required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--rate", type=float, default=0.0657)
a = ap.parse_args()

t3 = T.Corpus("test3")
rows = []
for path in sorted(glob.glob(a.ckpt_glob)):
    b = torch.load(path, map_location=a.device, weights_only=False)
    ta = argparse.Namespace(**b["args"])
    m = T.FactoredSlot(slot_dim=ta.slot_dim, mono=b["mono"], pos_weight=b["pos_weight"],
                       d1=(ta.input_path == "axis_extractor"),
                       shared_dim=(ta.shared_proj_dim or None),
                       tagger=(None if ta.tagger == "none" else ta.tagger),
                       tagger_axes=ta.tagger_axes, block_readout=ta.block_readout,
                       dynamics=b["dynamics"], pred_weight=ta.pred_weight,
                       pred_space=ta.pred_space, head_feat=ta.head_feat,
                       slot_reset=(None if ta.slot_reset == "none" else ta.slot_reset),
                       slot_reset_dropout=ta.slot_reset_dropout,
                       psi=(None if ta.psi == "none" else ta.psi), psi_beta=ta.psi_beta,
                       psi_dropout=ta.psi_dropout, bottleneck=ta.bottleneck,
                       dropout=ta.dropout).to(a.device)
    m.load_state_dict(b["state"])
    probs = T.axis_probs(m, t3, list(range(t3.n_docs)), a.device, 4, with_anchor=False)
    thr = np.asarray(b["axis_thresholds"]); ys, pri, m3, per_doc = [], [], [], {}
    for i, pr in enumerate(probs):
        g = t3.gold[int(t3.to[i]):int(t3.to[i + 1])].astype(np.int8)
        s = pr.sum(1)
        bp = (s >= np.quantile(s, 1.0 - a.rate)).astype(np.int8)
        bm = ((pr >= thr[None, :]).sum(1) >= T.T_STAR).astype(np.int8)
        ys.append(np.concatenate([g, [0]]).tolist())
        pri.append(np.concatenate([bp, [0]]).tolist())
        m3.append(np.concatenate([bm, [0]]).tolist())
        per_doc[t3.doc_ids[i]] = {"n_gold": int(g.sum()), "n_trans": len(g),
                                  "gold_rate": float(g.mean()),
                                  "primary": met([ys[-1]], [pri[-1]])}
    zero = [[0] * len(y) for y in ys]
    rows.append({"ckpt": Path(path).name, "floor": score_dialogues(ys, zero)["score"],
                 "primary": met(ys, pri), "m3_faithful": met(ys, m3), "per_doc": per_doc})
    print(f"  {Path(path).name}: primary Score {rows[-1]['primary']['score']:.4f}")

a.out.mkdir(parents=True, exist_ok=True)
(a.out / "test3_sealed.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
import statistics as st
def agg(rule, k):
    v = [r[rule][k] for r in rows if r[rule][k] == r[rule][k]]
    return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)
print(f"\n퇴화 하한 {rows[0]['floor']:.4f}")
for rule in ("primary", "m3_faithful"):
    s, sd = agg(rule, "score"); b, _ = agg(rule, "bor"); l, _ = agg(rule, "lift")
    print(f"{rule:14s} Score {s:.4f}±{sd:.4f}  BOR {b:.3f}  Lift {l:.3f}")
print(f"-> {a.out/'test3_sealed.json'}")

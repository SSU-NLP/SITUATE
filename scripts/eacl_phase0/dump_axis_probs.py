#!/usr/bin/env python3
"""M5 의 **축별 전이 확률**을 전이 단위로 디스크에 남긴다.

## 왜 필요한가
appendix 주장 *"다섯 차원 중 어떤 축이 대화에서 정의가 붕괴한다"* 를 확률 분포로 보이려면
축별 확률이 있어야 하는데, **현재 저장소 어디에도 남아 있지 않다.** `axis_probs()` 를 부르는
13개 호출부가 전부 `[n_trans, 5]` 를 즉시 `.sum(1)` 로 붕괴시킨다
(`m4m5_dts_transfer.py:57`, `baseline_compare.py:92-94`). 축별 AUC 스칼라만 남긴 곳이
`2026-07-31_channel_observability` 하나 있는데, 그것으로는 분포도 신뢰구간도 그릴 수 없다.

## 무엇을 하는가
동결 체크포인트(`configs/m5_default.json` → `models/m5_default/M5.pt`)를 그대로 올려
`train_anchor_m4m5.axis_probs()` 를 **재구현 없이 호출**하고, 문서·전이 단위로 덤프한다.
추론에는 앵커가 없다(`with_anchor=False`) — 앵커를 쓰면 누출이다.

    JSONL 한 줄 = {"doc_id", "transition", "p_protagonist", "p_space", "p_time",
                   "p_causality", "p_intentionality", "gold"}

🔴 **5축 전부 유지한다.** 약축(causality)이라도 신호에서 빼지 않는다.
🔴 DTS 는 D-M5 로 scope-out 이다 — 이 산출은 **탐색적 분석 전용**이며 성능 주장에 쓰지 않는다.

    env -C /tmp CUDA_VISIBLE_DEVICES=0 /root/Hi-OnTop/.venv/bin/python \\
        /root/Hi-OnTop/scripts/eacl_phase0/dump_axis_probs.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))

from hi_ontop.factored_slot import AXES, FactoredSlot  # noqa: E402
from train_anchor_m4m5 import Corpus, axis_probs       # noqa: E402

CONFIG = REPO / "configs/m5_default.json"
OUT = REPO / "outputs/experiments/2026-08-04_axis_collapse"

# 서사(학습 도메인) 1 + 대화(전이 도메인) 3. dev3 = 사람 gold 서사, dts_full_* = DTS 공식 test.
DEFAULT_SETS = ("dev3", "test3", "dts_full_tiage", "dts_full_dialseg711", "dts_full_superseg")


def load_model(dev: str):
    """`channel_observability_v2.py:73-82` 과 **같은** 로드 경로 (분기 방지)."""
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    ck = torch.load(REPO / cfg["checkpoint"], map_location=dev, weights_only=False)
    m = FactoredSlot(
        slot_dim=ck["args"]["slot_dim"], mono=ck["mono"], pos_weight=ck["pos_weight"],
        dynamics=ck.get("dynamics"),
        d1=ck["args"].get("input_path", "axis_extractor") == "axis_extractor",
        shared_dim=ck["args"].get("shared_proj_dim") or None,
        pred_space=ck["args"].get("pred_space", "obs")).to(dev)
    m.load_state_dict(ck["state"])
    m.eval()
    return m, cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default=",".join(DEFAULT_SETS))
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, cfg = load_model(dev)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"체크포인트 {cfg['checkpoint']} · device={dev} · 축 {AXES}\n", flush=True)

    summary = {}
    for name in args.sets.split(","):
        c = Corpus(name)
        idx = list(range(c.n_docs))
        probs = axis_probs(m, c, idx, dev, bs=args.bs, with_anchor=False)

        path = OUT / f"axis_probs_{name}.jsonl"
        n_tr = 0
        per_axis_pos, per_axis_neg = {a: [] for a in AXES}, {a: [] for a in AXES}
        with path.open("w") as fp:
            for i, pr in zip(idx, probs):
                g = c.gold[int(c.to[i]):int(c.to[i + 1])].astype(np.int8)
                did = c.doc_ids[i]
                for t in range(len(g)):
                    row = {"doc_id": did, "transition": t, "gold": int(g[t])}
                    for k, a in enumerate(AXES):
                        row[f"p_{a}"] = round(float(pr[t, k]), 6)
                    fp.write(json.dumps(row) + "\n")
                n_tr += len(g)
                for k, a in enumerate(AXES):
                    per_axis_pos[a] += pr[g == 1, k].tolist()
                    per_axis_neg[a] += pr[g == 0, k].tolist()

        # 축별 분리도: gold=1 과 gold=0 의 확률 분포가 얼마나 갈라지는가.
        # AUC 는 순위만 보므로, 붕괴를 보이려면 **평균 차이**도 함께 봐야 한다.
        st = {}
        for a in AXES:
            p, n = np.asarray(per_axis_pos[a]), np.asarray(per_axis_neg[a])
            auc = (float((p[:, None] > n[None, :]).mean()
                         + 0.5 * (p[:, None] == n[None, :]).mean())
                   if len(p) and len(n) else float("nan"))
            st[a] = {"mean_pos": float(p.mean()) if len(p) else None,
                     "mean_neg": float(n.mean()) if len(n) else None,
                     "delta": float(p.mean() - n.mean()) if len(p) and len(n) else None,
                     "auc": auc}
        summary[name] = {"n_docs": c.n_docs, "n_trans": n_tr,
                         "n_gold": int(sum(int(x) for x in
                                           (c.gold[:c.to[-1]] if c.gold is not None else []))),
                         "axes": st, "path": str(path.relative_to(REPO))}
        print(f"{name:22s} 문서 {c.n_docs:>5} · 전이 {n_tr:>7} → {path.name}", flush=True)
        for a in AXES:
            s = st[a]
            print(f"    {a:<16} AUC {s['auc']:.3f} · Δmean {s['delta']:+.4f} "
                  f"(gold1 {s['mean_pos']:.4f} / gold0 {s['mean_neg']:.4f})", flush=True)

    (OUT / "axis_collapse_summary.json").write_text(
        json.dumps({"config": vars(args), "checkpoint": cfg["checkpoint"],
                    "axes": AXES, "sets": summary}, ensure_ascii=False, indent=2))
    print(f"\n→ {OUT/'axis_collapse_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

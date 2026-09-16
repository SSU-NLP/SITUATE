#!/usr/bin/env python3
r"""D-B36 — **구조화 벡터의 블록이 실제로 축별로 분화됐는가**를 잰다.

`probe_slot_alignment_phase1.py` 와 **지표 정의가 동일**하다(excess off-axis leakage: 읽기 단위
k 의 전이 특징으로 다른 축 j 를 얼마나 읽어내는가, 라벨 상관 천장 B 를 차감). 다른 점은 읽기
단위가 슬롯이 아니라 **한 벡터 안의 블록**이라는 것뿐이며, 그 처리는 `FactoredSlot.axis_state`
가 이미 담당한다.

왜 필요한가: dev-3 Score 가 동률이어도 두 해석이 갈린다 —
  ① **구획 압력이 작동은 했는데**(블록이 분화) 분절 성능으로 이어지지 않았다
  ② **구획 자체가 안 생겼다**(블록이 여전히 얽힘) → 단일 재귀로는 분화가 불가능
프로브가 이 둘을 가른다.

🔒 앵커 코퍼스 내부에서만 계산 — dev-3·시험지 미사용(D-B16).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
from db29_final_eval import load  # noqa: E402
from hi_ontop.factored_slot import AXES  # noqa: E402
from probe_slot_alignment_phase1 import _fit_ap, label_corr_ceiling, slot_feats  # noqa: E402
from train_anchor_m4m5 import Corpus  # noqa: E402

OUT = REPO / "outputs/experiments/schema_vector_ablation"
CELLS = [
    ("구조화 벡터 (블록 26×5)", "M4", "_per_slot_slot_sd130_blocks"),
    ("단일 벡터 130 (대조)",     "M4", "_per_slot_slot_sd130"),
    ("M5 슬롯 5개 (참조)",       "M5", "_per_slot_slot"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--probe-train-docs", type=int, default=200)
    ap.add_argument("--probe-val-docs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    c = Corpus("phase1")

    rng = np.random.default_rng(20260728)          # 학습과 동일 분할
    perm = rng.permutation(c.n_docs)
    n_val = max(1, int(c.n_docs * 0.05))
    va_all, tr_all = perm[:n_val].tolist(), perm[n_val:].tolist()
    sub = np.random.default_rng(args.seed)
    tr = sub.choice(tr_all, min(args.probe_train_docs, len(tr_all)), replace=False).tolist()
    va = sub.choice(va_all, min(args.probe_val_docs, len(va_all)), replace=False).tolist()
    print(f"[probe] fit {len(tr)} docs / eval {len(va)} docs", flush=True)

    rows, mats = [], {}
    Ytr = Yva = None
    for label, tag, ns in CELLS:
        m, _ = load(tag, args.seed, dev, ns)
        Xtr, Ytr = slot_feats(m, c, tr, dev, args.batch)
        Xva, Yva = slot_feats(m, c, va, dev, args.batch)
        n_unit = len(Xtr)
        M = np.full((n_unit, 5), np.nan)
        for s in range(n_unit):
            for j in range(5):
                M[s, j] = _fit_ap(Xtr[s], Ytr[:, j], Xva[s], Yva[:, j])
        mats[label] = M
        print(f"[probe] {label} 완료 (읽기 단위 {n_unit}개)", flush=True)

    B = label_corr_ceiling(Ytr, Yva)
    off = ~np.eye(5, dtype=bool)
    L = ["# D-B36 프로브 — 블록이 축별로 분화됐는가 (앵커 내부, seed %d)\n" % args.seed,
         "> 지표 = `probe_slot_alignment_phase1.py` 와 **동일**(excess off-axis leakage,",
         "> 라벨 상관 천장 B 차감). 읽기 단위만 슬롯 → **블록**으로 바뀐다.",
         "> ≈0 = 자기 축에 특화 · 높을수록 얽힘.\n",
         "| 셀 | 읽기 단위 | 평균 EXCESS 누출 | own-axis (대각) |", "|---|---|---|---|"]
    for label in mats:
        M = mats[label]
        if M.shape[0] == 5:
            ex = float(np.nanmean(np.where(off, M - B, np.nan)))
            own = " ".join(f"{AXES[k][:5]} {M[k, k]:.3f}" for k in range(5))
        else:                                    # 단일 잠재 — 한 행이 모든 축을 읽는다
            ex = float(np.nanmean([M[0, j] - np.nanmax([B[k, j] for k in range(5) if k != j])
                                   for j in range(5)]))
            own = " ".join(f"{AXES[j][:5]} {M[0, j]:.3f}" for j in range(5))
        L.append(f"| {label} | {M.shape[0]} | **{ex:+.4f}** | {own} |")
        rows.append((label, ex))

    L += ["\n## 해석 기준 (사전 등록 D-B36)\n",
          "- 구조화 벡터의 누출이 **단일 벡터보다 뚜렷이 낮으면** → 구획 압력이 작동했다",
          "  (그런데 Score 가 동률이면 *축 분리 ≠ 분절 성능* 이 다시 확인된 것).",
          "- **비슷하면** → 단일 재귀 위에서는 구획을 지정해도 분화가 일어나지 않는다.\n",
          "⚠️ 앵커 k=1 · S_self 0.659(상한) · seed 1개 · 탐색적."]
    (OUT / "PROBE.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n-> {OUT / 'PROBE.md'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
r"""D-B35 — 태거가 정의한 축 공간(설계문서 §4.6 **진짜 D1**)을 dev-3 에서 평가한다.

D-B33/34 평가기(`inputpath_dynamics_eval.py`)의 **지표·판정 규칙을 그대로 재사용**하고 셀 목록만
바꾼다(재구현 금지 · 같은 규약이어야 D-B34 표와 나란히 읽을 수 있다).

## 🔒 지표 규약 (D-B33 부칙 1·2 + D-B34 정정 3 승계 · 사후 교체 금지)
- 주지표 = dev-3 Score, 전 셀 동일 규칙 3종(otsu · **online μ+1.0σ** · offline μ+1.0σ).
- 판정 = 규칙 3종 중 ≥2 · seed 3 중 ≥2 · 문서 3편 중 ≥2 방향 일치 (셋 다 만족해야 우세/열세).
- **AUC 가 이번 라운드의 핵심 관전 포인트**다(D-B35 사전 해석 규약): D-B34 에서 전 셀 ±0.007 로
  평평했으므로, 태거가 *순위력*을 더하면 여기서 움직여야 한다.
- gold 밀도 matched 는 ⚠️ oracle 개수 — 성능 주장 금지, 순위 진단 전용.
- t\*=3 · r=0.0642 폐기 상태 유지. J_anchor 는 판정 제외(진단 기록).
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
from db29_final_eval import load  # noqa: E402
from inputpath_dynamics_eval import DEPLOYABLE, DIAG, eval_cell, j_anchor, verdict  # noqa: E402
from train_anchor_m4m5 import Corpus  # noqa: E402

OUT = REPO / "outputs/experiments/tagger_d1_ablation"
SEEDS = (0, 1, 2)

# (표시명, 체크포인트 tag, 네임스페이스)
CELLS = [
    ("학습형 E_k (기준선 M5)", "M5", "_per_slot_slot"),
    ("태거 replace 3축",       "M5", "_per_slot_slot_tagreplace3"),
    ("태거 replace 5축",       "M5", "_per_slot_slot_tagreplace5"),
    ("태거 hybrid 3축",        "M5", "_per_slot_slot_taghybrid3"),
    ("태거 hybrid 5축",        "M5", "_per_slot_slot_taghybrid5"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dev3 = Corpus("dev3")
    if dev3.feat is None:
        raise SystemExit("🔴 dev3_axisfeat.npy 가 없다 — extract_axis_features.py --split dev3")
    OUT.mkdir(parents=True, exist_ok=True)

    raw: dict = {}
    agg: dict = {}
    for label, tag, ns in CELLS:
        got = {}
        for s in seeds:
            try:
                m, _ = load(tag, s, dev, ns)
            except FileNotFoundError:
                print(f"[skip] {label} seed{s} — 체크포인트 없음", flush=True)
                continue
            got[s] = eval_cell(m, dev3, dev)
            got[s]["J_anchor(진단)"] = j_anchor(tag, ns, s)
            print(f"  {label} seed{s} · otsu {got[s]['otsu']['pooled_score']:.4f}"
                  f" · onμ+σ {got[s]['online μ+1.0σ']['pooled_score']:.4f}"
                  f" · AUC {got[s]['auc_pooled']:.4f}", flush=True)
        if not got:
            continue
        raw[label] = got
        docs = list(next(iter(got.values()))["per_doc"])
        agg[label] = {
            "n_seed": len(got),
            "by_rule": {r: [got[s][r]["pooled_score"] for s in got] for r in DEPLOYABLE + DIAG},
            "n_pred": {r: [got[s][r]["n_pred_total"] for s in got] for r in DEPLOYABLE + DIAG},
            "auc": [got[s]["auc_pooled"] for s in got],
            "by_doc": {d: float(np.mean([got[s]["per_doc"][d][r]["score"]
                                         for s in got for r in DEPLOYABLE])) for d in docs},
            "J_anchor(진단)": [got[s]["J_anchor(진단)"] for s in got],
        }

    ref_label = CELLS[0][0]
    L = ["# D-B35 — 태거가 정의한 축 공간 (설계문서 §4.6 진짜 D1) · dev-3\n",
         "> 🔒 지표 = D-B33 부칙 1·2 + D-B34 정정 3 승계. 임계 선택 없음, 전 셀 동일 규칙 3종.",
         "> ⚠️ dev-3 = 3편 연습장(pieman 선택압 9회) — **절대 Score 인용 금지, 셀 간 Δ·방향만**.",
         "> ⚠️ 앵커 k=1 · S_self 0.659(상한). 봉인 시험지 미개봉.\n",
         "## 1. 규칙별 pooled Score (mean ± std over seed)\n",
         "| 셀 | n | " + " | ".join(DEPLOYABLE) + " | **AUC** | *[oracle] gold밀도* |",
         "|---|---|" + "---|" * (len(DEPLOYABLE) + 2)]
    for label in agg:
        a = agg[label]
        cols = [f"{np.mean(a['by_rule'][r]):.4f} ± {np.std(a['by_rule'][r]):.4f}"
                for r in DEPLOYABLE]
        cols += [f"**{np.mean(a['auc']):.4f}**", f"*{np.mean(a['by_rule'][DIAG[0]]):.4f}*"]
        L.append(f"| {label} | {a['n_seed']} | " + " | ".join(cols) + " |")

    L += ["\n## 2. 기준선(학습형 E_k) 대비 Δ + 🔒 사전 판정\n",
          "| 셀 | " + " | ".join(f"Δ {r}" for r in DEPLOYABLE) + " | **Δ AUC** | 판정 |",
          "|---|" + "---|" * (len(DEPLOYABLE) + 2)]
    for label in agg:
        if label == ref_label:
            continue
        a, ref = agg[label], agg[ref_label]
        cols = [f"{np.mean(a['by_rule'][r]) - np.mean(ref['by_rule'][r]):+.4f}" for r in DEPLOYABLE]
        cols.append(f"**{np.mean(a['auc']) - np.mean(ref['auc']):+.4f}**")
        L.append(f"| {label} | " + " | ".join(cols) + f" | {verdict(a, ref)} |")

    L += ["\n## 3. 문서별 (deployable 3종 평균, seed 평균)\n",
          "| 셀 | " + " | ".join(agg[ref_label]["by_doc"]) + " |",
          "|---|" + "---|" * len(agg[ref_label]["by_doc"])]
    for label in agg:
        L.append(f"| {label} | " + " | ".join(
            f"{agg[label]['by_doc'][d]:.4f}" for d in agg[label]["by_doc"]) + " |")

    L += ["\n## 4. 예측 경계 수 (gold 66)\n",
          "| 셀 | " + " | ".join(DEPLOYABLE) + " |", "|---|" + "---|" * len(DEPLOYABLE)]
    for label in agg:
        L.append(f"| {label} | " + " | ".join(
            f"{np.mean(agg[label]['n_pred'][r]):.0f}" for r in DEPLOYABLE) + " |")

    L += ["\n## 5. [진단 기록] J_anchor — 판정에 쓰이지 않음\n", "| 셀 | J_anchor |", "|---|---|"]
    for label in agg:
        v = [x for x in agg[label]["J_anchor(진단)"] if not np.isnan(x)]
        L.append(f"| {label} | {np.mean(v):.4f} |" if v else f"| {label} | — |")

    (OUT / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps({"agg": agg, "raw": raw}, indent=2,
                                                 ensure_ascii=False, default=float))
    print("\n".join(L))
    print(f"\n-> {OUT / 'RESULTS.md'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
r"""D-B36 — **하나의 상태 벡터 + 구조화된 스키마**를 dev-3 에서 평가한다.

D-B33/34 평가기(`inputpath_dynamics_eval.py`)의 지표·판정 규칙을 그대로 재사용하고 셀만 바꾼다.

핵심 대조 = **blocks130 vs plain130** — 같은 130차원 단일 벡터에서 *head 가 자기 블록만 읽게
강제했는지*만 다르다. 차이가 나오면 용량이 아니라 **구획 강제** 때문이다.

🔒 지표 = D-B33 부칙 1·2 + D-B34 정정 3 승계(사후 교체 금지). 판정 = 규칙 3종 중 ≥2 ·
seed 3 중 ≥2 · 문서 3편 중 ≥2 방향 일치.
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

OUT = REPO / "outputs/experiments/schema_vector_ablation"
CELLS = [
    ("구조화 벡터 (블록 26×5)", "M4", "_per_slot_slot_sd130_blocks"),
    ("단일 벡터 130 (대조)",     "M4", "_per_slot_slot_sd130"),
    ("M4 단일 벡터 128 (기존)",  "M4", "_per_slot_slot"),
    ("M5 슬롯 5개 (기존)",       "M5", "_per_slot_slot"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dev3 = Corpus("dev3")
    OUT.mkdir(parents=True, exist_ok=True)

    raw, agg = {}, {}
    for label, tag, ns in CELLS:
        got = {}
        for s in seeds:
            try:
                m, _ = load(tag, s, dev, ns)
            except FileNotFoundError:
                print(f"[skip] {label} seed{s}", flush=True)
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

    ref = "단일 벡터 130 (대조)"      # 🔒 기준 = 같은 용량, 블록만 없음
    L = ["# D-B36 — 하나의 상태 벡터 + 구조화된 스키마 · dev-3\n",
         "> 🔒 핵심 대조 = **구조화 벡터 vs 단일 벡터 130** (같은 용량, head 가 자기 블록만 읽는지만 다름).",
         "> 지표 = D-B33 부칙 1·2 + D-B34 정정 3 승계. 임계 선택 없음, 전 셀 동일 규칙 3종.",
         "> ⚠️ dev-3 = 3편 연습장 — 절대 Score 인용 금지, Δ·방향만. 시험지 미개봉.\n",
         "## 1. 규칙별 pooled Score (mean ± std over 3 seed)\n",
         "| 셀 | n | " + " | ".join(DEPLOYABLE) + " | AUC | *[oracle] gold밀도* |",
         "|---|---|" + "---|" * (len(DEPLOYABLE) + 2)]
    for label in agg:
        a = agg[label]
        cols = [f"{np.mean(a['by_rule'][r]):.4f} ± {np.std(a['by_rule'][r]):.4f}" for r in DEPLOYABLE]
        cols += [f"{np.mean(a['auc']):.4f}", f"*{np.mean(a['by_rule'][DIAG[0]]):.4f}*"]
        L.append(f"| {label} | {a['n_seed']} | " + " | ".join(cols) + " |")

    L += [f"\n## 2. 기준(**{ref}**) 대비 Δ + 🔒 사전 판정\n",
          "| 셀 | " + " | ".join(f"Δ {r}" for r in DEPLOYABLE) + " | Δ AUC | 판정 |",
          "|---|" + "---|" * (len(DEPLOYABLE) + 2)]
    for label in agg:
        if label == ref:
            continue
        a, b = agg[label], agg[ref]
        cols = [f"{np.mean(a['by_rule'][r]) - np.mean(b['by_rule'][r]):+.4f}" for r in DEPLOYABLE]
        cols.append(f"{np.mean(a['auc']) - np.mean(b['auc']):+.4f}")
        L.append(f"| {label} | " + " | ".join(cols) + f" | {verdict(a, b)} |")

    L += ["\n## 3. 문서별 (deployable 3종 평균)\n",
          "| 셀 | " + " | ".join(agg[ref]["by_doc"]) + " |", "|---|" + "---|" * len(agg[ref]["by_doc"])]
    for label in agg:
        L.append(f"| {label} | " + " | ".join(
            f"{agg[label]['by_doc'][d]:.4f}" for d in agg[label]["by_doc"]) + " |")

    L += ["\n## 4. 예측 경계 수 (gold 66)\n", "| 셀 | " + " | ".join(DEPLOYABLE) + " |",
          "|---|" + "---|" * len(DEPLOYABLE)]
    for label in agg:
        L.append(f"| {label} | " + " | ".join(
            f"{np.mean(agg[label]['n_pred'][r]):.0f}" for r in DEPLOYABLE) + " |")

    L += ["\n## 5. [진단] J_anchor — 판정 제외\n", "| 셀 | J_anchor |", "|---|---|"]
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

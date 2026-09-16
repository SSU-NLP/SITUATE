#!/usr/bin/env python3
"""청크 크기 통계 + 동결 회귀선으로 DCG@20 예측 (하류 재검증 1차 진단).

## 왜 이걸 먼저 보는가
D-B44 에서 GutenQA 성능의 **91%가 평균 청크 단어수 하나로 설명**됐다 —
`DCG@20 = 0.0221 × 평균단어 + 46.03`, r = +0.914 (p = 0.00022), **잔차 RMSE 0.148**.
임계 규칙 6종에 임의 조절값 3종이 섞였는데도 그 정도로 맞았다.
따라서 **평균 단어수를 재는 순간 최종 DCG 가 ±0.15 안에서 정해진다.**
검색 평가(셀당 11분 GPU)를 돌리기 전에 이 예측을 먼저 기록해 둔다.

## 🔴 회귀 계수는 재적합하지 않는다
`0.0221 / 46.03` 은 W0 실측에서 유도된 **동결값**이다. 새 데이터로 다시 맞추면 기준선이
데이터를 따라 움직여 "회귀선 대비 이득"이라는 개념 자체가 순환이 된다 (PREREG §3).

## 판정 재료
체크포인트 이득 = `잔차(신) − 잔차(구)`. 원 DCG 상승만으로는 "청크가 굵어졌다"와 구분되지 않는다.

    uv run python scripts/gutenqa/13_chunk_size_diag.py \
        --out-base outputs/experiments/2026-07-31_gutenqa_ckpt_sweep
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OLD = REPO_ROOT / "outputs/experiments/2026-07-29_gutenqa_chunking"

# 🔒 D-B44 동결 회귀선 (W0 10셀에서 적합). 재적합 금지.
SLOPE, INTERCEPT, RESID_RMSE = 0.0221, 46.03, 0.148
BOUNDS = {"MIN": 70, "MAX": 374}

# 궤적 재현 10규칙 — PREREG §3 의 격자
RULES = ["rate0.15", "adaptive_online_nf_c1", "ewma5_c1", "ewma20_c1", "ewma50_c1",
         "rate0.15_b1", "adaptive_online_nf_c1_b1", "ewma20_c1_b1", "cd_adaptive", "cd_ewma20"]


def load_chunks(p: Path) -> list[list[str]]:
    out = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line)["chunks"])
    return out


def cell_stats(p: Path) -> dict | None:
    if not p.exists():
        return None
    books = load_chunks(p)
    w = np.array([len(c.split()) for bk in books for c in bk], dtype=float)
    if w.size == 0:
        return None
    mean = float(w.mean())
    return {"n_chunks": int(w.size), "chunks_per_book": round(w.size / len(books), 1),
            "mean_words": round(mean, 1), "p50": round(float(np.percentile(w, 50)), 1),
            "p95": round(float(np.percentile(w, 95)), 1),
            "cv": round(float(w.std() / mean), 3),
            "under_min": round(float((w < BOUNDS["MIN"]).mean()), 4),
            "over_max": round(float((w > BOUNDS["MAX"]).mean()), 4),
            "pred_dcg": round(SLOPE * mean + INTERCEPT, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-base", required=True, help="신 실험 디렉터리 (chunks/ 보유)")
    ap.add_argument("--gens", nargs="+", default=["i2", "w3"])
    ap.add_argument("--rules", nargs="+", default=RULES)
    a = ap.parse_args()

    base = Path(a.out_base)
    print(f"🔒 동결 회귀선: DCG@20 = {SLOPE} × 평균단어 + {INTERCEPT}  (잔차 RMSE {RESID_RMSE})")
    print(f"🔒 길이 바운드: MIN {BOUNDS['MIN']} · MAX {BOUNDS['MAX']}\n")

    rows, out = [], {"regression": {"slope": SLOPE, "intercept": INTERCEPT,
                                    "resid_rmse": RESID_RMSE, "refit": False}, "cells": {}}
    for rule in a.rules:
        w0 = cell_stats(OLD / "chunks" / f"m5__{rule}.jsonl")
        if w0 is None:
            w0 = cell_stats(OLD / "data/chunks" / f"m5__{rule}.jsonl")
        for gen in a.gens:
            st = cell_stats(base / "chunks" / f"m5{gen}__{rule}.jsonl")
            if st is None:
                continue
            d_words = st["mean_words"] - w0["mean_words"] if w0 else None
            d_pred = st["pred_dcg"] - w0["pred_dcg"] if w0 else None
            rows.append((rule, gen, st, w0, d_words, d_pred))
            out["cells"][f"m5{gen}__{rule}"] = {**st, "w0_mean_words": w0["mean_words"] if w0 else None,
                                                "d_mean_words": d_words, "d_pred_dcg": d_pred}

    hdr = f"{'규칙':28s} {'세대':4s} {'청크':>8s} {'평균단어':>8s} {'ΔW0':>7s} {'CV':>5s} {'<MIN':>6s} {'>MAX':>6s} {'예측DCG':>8s} {'Δ예측':>7s}"
    print(hdr); print("─" * len(hdr))
    for rule, gen, st, w0, dw, dp in rows:
        print(f"{rule:28s} {gen:4s} {st['n_chunks']:>8,} {st['mean_words']:>8.1f} "
              f"{(f'{dw:+.1f}' if dw is not None else '—'):>7s} {st['cv']:>5.2f} "
              f"{st['under_min']*100:>5.1f}% {st['over_max']*100:>5.1f}% "
              f"{st['pred_dcg']:>8.2f} {(f'{dp:+.2f}' if dp is not None else '—'):>7s}")

    print(f"\n※ Δ예측은 **청크 크기만으로** 설명되는 몫이다. 체크포인트의 진짜 기여는")
    print(f"  실측 DCG 가 나온 뒤 `실측 − 예측`(잔차)을 W0 과 비교해야 보인다.")
    print(f"  |Δ예측| < {RESID_RMSE} 면 회귀선의 분해능 아래라 크기 차이로도 못 읽는다.")

    p = base / "chunk_size_diag.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()

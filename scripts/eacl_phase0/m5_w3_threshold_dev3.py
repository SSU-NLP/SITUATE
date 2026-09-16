#!/usr/bin/env python3
r"""M5 **W3** × 임계 규칙 비교 (적응임계 vs EWMA) — dev-3 deployable Score.

## 무엇을
W3 = 학습 조건 라벨 **ALPHA 현행 × pos_weight=one**(D-B44 채택본, 현 `configs/m5_default.json`).
가중치 = `outputs/experiments/m4m5_phase1_anchor/M5_per_slot_slot_pwone_seed{0,1,2}.pt`
(seed0 = `models/m5_default/M5.pt` 와 md5 동일).

**모델은 그대로 두고 임계 규칙만 바꿔 가며** dev-3 Score 를 뽑는다:

| 규칙 | mode | 성격 |
|---|---|---|
| 적응임계 (현행 채택) | `online_nf` | **누적** μ+cσ. 문서 전체 이력을 쓴다 |
| EWMA | `ewma`, span S | **국소** μ+cσ, α=2/(S+1). 현재 값은 임계 계산에 넣지 않는다 |

두 규칙은 **서로 배타적인 임계 규칙**이므로 "둘 다"는 결합이 아니라 **나란히 비교**로 읽는다.

## 하지 않는 것
- 재학습 없음. 체크포인트·인코더·신호(S5 5축 합) 전부 동결값 그대로.
- c 는 **1.0 고정**(D-B39 채택값, 출처 = DTS 공식 test). 여기서 c 를 재탐색하면 selection-on-dev.
- 봉인 시험지 미개봉. dev-3 만 본다.
- 🔴 **dev-3 절대 Score 를 성능으로 인용하지 않는다 — Δ·방향만**(config caveats 승계).

## 실행
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/eacl_phase0/m5_w3_threshold_dev3.py
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
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from hi_ontop.hi_ontop_v2 import adaptive_boundaries  # noqa: E402
from train_anchor_m4m5 import Corpus, axis_probs  # noqa: E402

OUT = REPO / "outputs/experiments/2026-07-31_m5_w3_threshold"
NS_W3 = "_per_slot_slot_pwone"          # 🔒 D-B44 규약: 조건이 네임스페이스에 드러난다
SEEDS = (0, 1, 2)
C_MAIN = 1.0                            # 🔒 D-B39 채택값. 재탐색 금지.


def cuts(s: np.ndarray, mode: str, span: int = 5) -> np.ndarray:
    """동결 규약: 0번 sentinel 을 앞에 붙인 시퀀스를 넘긴다. 재구현 금지 — 기존 함수 호출."""
    seq = np.concatenate([[0.0], s]).tolist()
    return np.asarray(adaptive_boundaries(seq, c=C_MAIN, mode=mode, warmup=span),
                      np.int8)[:len(s)]


def score_of(P_list, G_list, mode: str, span: int = 5):
    ys, ps, n = [], [], 0
    for P, g in zip(P_list, G_list):
        b = cuts(P.sum(1), mode, span)[:len(g)]
        ys.append(np.concatenate([g, [0]]).tolist())
        ps.append(np.concatenate([b, [0]]).tolist())
        n += int(b.sum())
    return score_dialogues(ys, ps)["score"], n


def main() -> None:
    ap = argparse.ArgumentParser(description="M5 W3 × 임계 규칙 (적응임계 vs EWMA), dev-3")
    ap.add_argument("--ewma-spans", type=int, nargs="*", default=[10, 20, 30, 50],
                    help="EWMA span 목록. alpha=2/(span+1)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device={dev}", flush=True)

    dev3 = Corpus("dev3")
    dev3_g = [dev3.gold[int(dev3.to[i]):int(dev3.to[i + 1])].astype(np.int8)
              for i in range(dev3.n_docs)]
    n_gold = int(sum(int(g.sum()) for g in dev3_g))
    print(f"dev-3: {dev3.n_docs}편 · gold 경계 {n_gold}", flush=True)

    rules = [("adaptive (online_nf)", "online_nf", 5)]
    rules += [(f"ewma span={s}", "ewma", s) for s in a.ewma_spans]

    res: dict[str, list] = {}
    for sd in SEEDS:
        ck = REPO / f"outputs/experiments/m4m5_phase1_anchor/M5{NS_W3}_seed{sd}.pt"
        if not ck.exists():
            print(f"  ⚠️ 없음: {ck.name} — 건너뜀", flush=True)
            continue
        m = load("M5", sd, dev, ns=NS_W3)[0]
        with torch.no_grad():
            Pd = axis_probs(m, dev3, list(range(dev3.n_docs)), dev, bs=64, with_anchor=False)
        for label, mode, span in rules:
            sc, npred = score_of(Pd, dev3_g, mode, span)
            res.setdefault(label, []).append({"seed": sd, "score": float(sc), "n_pred": npred})
            print(f"  [seed{sd}] {label:22s} Score {sc:.4f}  예측 {npred}", flush=True)
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()

    if not res:
        print("🔴 체크포인트를 하나도 못 찾았다 — 중단", flush=True)
        return

    base = res.get("adaptive (online_nf)", [])
    base_mu = float(np.mean([r["score"] for r in base])) if base else float("nan")
    L = ["# M5 W3 × 임계 규칙 — dev-3 deployable Score", "",
         "## 실험 setup", "",
         "- **모델:** M5 **W3** = ALPHA 현행 × pos_weight=one (D-B44 채택본).",
         f"  가중치 `outputs/experiments/m4m5_phase1_anchor/M5{NS_W3}_seed{{0,1,2}}.pt`",
         "  (seed0 = `models/m5_default/M5.pt`). **재학습 없음 — 임계 규칙만 바꿨다.**",
         "- **신호:** s = Σ_k P(reset_k), **S5 5축 전부**(축 제거 없음).",
         f"- **c = {C_MAIN} 고정** (D-B39 채택값, 출처 = DTS 공식 test). 여기서 재탐색하지 않는다.",
         "- **지표:** deployable Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD), `dts_scoring.score_dialogues`.",
         "  경계 개수를 방법에 주지 않는다(matched@gold 아님).",
         f"- **데이터:** dev-3 {dev3.n_docs}편 · gold 경계 **{n_gold}**. 봉인 시험지 미개봉.", "",
         "## 결과 (seed 0/1/2 평균 ±SD)", "",
         "| 임계 규칙 | Score | ±SD | 예측 경계 | 과분절 배수 | Δ vs 적응임계 |",
         "|---|---|---|---|---|---|"]
    for label, mode, span in rules:
        rows = res.get(label)
        if not rows:
            continue
        ss = [r["score"] for r in rows]
        nn = [r["n_pred"] for r in rows]
        mu, sd_ = float(np.mean(ss)), (float(np.std(ss, ddof=1)) if len(ss) > 1 else 0.0)
        npm = float(np.mean(nn))
        d = "—" if label.startswith("adaptive") else f"{mu - base_mu:+.4f}"
        L.append(f"| {label} | **{mu:.4f}** | ±{sd_:.4f} | {npm:.0f} | {npm / max(n_gold,1):.2f}× | {d} |")
    L += ["", "## 읽는 법 — 🔴 절대값 인용 금지", "",
          "`configs/m5_default.json` 의 caveat 을 그대로 승계한다: **dev-3 절대 Score 를 성능으로",
          "인용하지 않는다. Δ·방향만.** dev-3 는 연습장 3편이고 pieman 에는 선택압이 9회 누적됐다.",
          "여기서 EWMA 가 이겨도 그것은 *dev-3 에서 더 높다* 는 뜻이지 채택 근거가 아니다 —",
          "현행 c=1.0·`online_nf` 의 출처는 **DTS 공식 test** 이고(D-B39), 그 지위를 dev-3 수치로",
          "뒤집으면 selection-on-dev 가 된다.", "",
          "**과분절 배수**를 같이 본다. 현행 규칙의 알려진 결함이 **약 2.2배 과분절**",
          "(dev-3 145 예측 vs gold 66)이고 경계 개수 제어는 최대 미결 과제다(D-B34/B37/B38).",
          "EWMA 가 Score 를 올리더라도 그것이 *개수를 맞춰서* 인지 *위치를 잘 잡아서* 인지는",
          "이 표만으로 갈리지 않는다.", ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "m5_w3_threshold.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

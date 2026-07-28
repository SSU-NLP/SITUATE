#!/usr/bin/env python3
"""학습된 M4/M5 를 DTS 대화 벤치마크에 적용 — **도메인 전이 테스트**.

## 무엇을
M4/M5 는 **서사 픽션 앵커 10,598편**으로만 학습됐다(D-B15, D-M8 서사 단일 코퍼스).
그 모델을 **대화** 벤치마크(tiage/dialseg711/superseg/ami)에 그대로 적용해 Score 를 낸다.
입력은 얼린 mpnet 문장 임베딩뿐이므로 재학습·재라벨 없이 바로 돌릴 수 있다.

## 경계 규칙 (dev-3 평가와 **동일**, 데이터에 맞춰 바꾸지 않는다)
[주] **per-doc percentile** — 연속 점수 `s = Σ_k P(reset_k)` 의 문서 내 상위 r 분위수.
r = **앵커 코퍼스 경계율 0.0649 고정**(학습 데이터에서 온 밀도 사전). DTS gold 밀도를 쓰지 않는다.
[보조] M3 충실판(축별 이진화 후 count ≥ t\\*=3) 도 병기.

🔴 지위: **탐색적.** DTS 는 D-M5 에서 **scope 밖** 선언 — 성능 주장에 쓰지 않는다.
이 표의 용도는 *"서사에서 학습한 축이 대화로 전이되는가"* 하나다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from hi_ontop.factored_slot import AXES, FactoredSlot  # noqa: E402
from train_anchor_m4m5 import EXP, T_STAR, Corpus, axis_probs  # noqa: E402

ANCHOR_RATE = 0.0649          # 앵커 코퍼스 경계율(D-B15) — 밀도 사전, DTS gold 미사용
SETS = ("dev3", "dts_tiage", "dts_dialseg711", "dts_superseg", "dts_ami")
# (표기, 체크포인트 접미사, seed)
MODELS = [("M5 (slot-native)", "_per_slot_slot", 0), ("M4 (slot-native)", "_per_slot_slot", 0),
          ("M5 (dynamics)", "_per_slot", 0), ("M4 (dynamics)", "_per_slot", 0),
          ("M5 (anchor-only)", "", 1), ("M4 (anchor-only)", "", 1)]


def load_model(name: str, ns: str, seed: int, dev: str):
    tag = name.split()[0]                       # "M5" / "M4"
    ck = torch.load(EXP / f"{tag}{ns}_seed{seed}.pt", map_location=dev, weights_only=False)
    m = FactoredSlot(slot_dim=ck["args"]["slot_dim"], mono=ck["mono"],
                     pos_weight=ck["pos_weight"], dynamics=ck.get("dynamics"),
                     pred_weight=ck["args"].get("pred_weight", 1.0),
                     pred_space=ck["args"].get("pred_space", "obs")).to(dev)
    m.load_state_dict(ck["state"])
    return m, ck


def evaluate(model, c: Corpus, dev: str, thr_axis: np.ndarray) -> tuple[float, float, int]:
    idx = list(range(c.n_docs))
    probs = axis_probs(model, c, idx, dev, bs=8, with_anchor=False)
    ys, p_pri, p_m3 = [], [], []
    for i, pr in zip(idx, probs):
        g = c.gold[int(c.to[i]):int(c.to[i + 1])].astype(np.int8)
        s = pr.sum(1)
        b_pri = (s >= np.quantile(s, 1.0 - ANCHOR_RATE)).astype(np.int8)
        b_m3 = ((pr >= thr_axis[None, :]).sum(1) >= T_STAR).astype(np.int8)
        ys.append(np.concatenate([g, [0]]).tolist())
        p_pri.append(np.concatenate([b_pri, [0]]).tolist())
        p_m3.append(np.concatenate([b_m3, [0]]).tolist())
    return (score_dialogues(ys, p_pri)["score"], score_dialogues(ys, p_m3)["score"],
            int(sum(sum(x) for x in p_pri)))


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    corp = {s: Corpus(s) for s in SETS}

    # 축 임계는 학습 때와 동일하게 **phase1 val** 에서 온 값을 재사용(체크포인트에 저장됨).
    rows = {}
    for name, ns, seed in MODELS:
        m, ck = load_model(name, ns, seed, dev)
        thr = np.asarray(ck["axis_thresholds"])
        rows[name] = {}
        for s in SETS:
            pri, m3, npred = evaluate(m, corp[s], dev, thr)
            rows[name][s] = (pri, m3, npred)
            print(f"  {name:20s} {s:16s} 주 {pri:.4f} · M3충실 {m3:.4f} · 경계 {npred}", flush=True)

    hdr = ["dev-3 (서사)", "tiage", "dialseg711", "superseg", "ami"]
    L = ["# M4/M5 도메인 전이 — 서사 학습 모델을 DTS 대화에 적용\n",
         "> 🔴 **탐색적.** M4/M5 는 **서사 픽션 앵커 10,598편으로만** 학습됐다(D-M8 서사 단일).",
         "> DTS 는 D-M5 에서 **scope 밖** 선언 — 성능 주장이 아니라 **전이 여부** 확인용.",
         "> 경계 규칙은 dev-3 평가와 **동일**(문서 내 상위 6.49%, 앵커 밀도 사전). DTS gold 밀도 미사용.\n",
         "\n## [주] per-doc percentile Score\n",
         "| 모델 | " + " | ".join(hdr) + " |", "|---|" + "---|" * len(hdr)]
    for name, _, _ in MODELS:
        L.append(f"| {name} | " + " | ".join(f"{rows[name][s][0]:.4f}" for s in SETS) + " |")
    base = {s: np.concatenate([corp[s].gold]).mean() for s in SETS}
    L.append("| *gold 밀도* | " + " | ".join(f"*{base[s]:.4f}*" for s in SETS) + " |")
    # 하한: 경계 없음
    none_row = []
    for s in SETS:
        c = corp[s]
        ys = [np.concatenate([c.gold[int(c.to[i]):int(c.to[i+1])], [0]]).tolist()
              for i in range(c.n_docs)]
        ps = [[0] * len(y) for y in ys]
        none_row.append(score_dialogues(ys, ps)["score"])
    L.append("| **경계 없음 (하한)** | " + " | ".join(f"**{v:.4f}**" for v in none_row) + " |")

    L += ["\n## [보조] M3 충실판 (축별 이진화 + count≥3) Score\n",
          "| 모델 | " + " | ".join(hdr) + " |", "|---|" + "---|" * len(hdr)]
    for name, _, _ in MODELS:
        L.append(f"| {name} | " + " | ".join(f"{rows[name][s][1]:.4f}" for s in SETS) + " |")

    L += ["\n> **경계 없음** 하한을 못 넘으면 그 데이터에서 그 방법은 의미가 없다"
          "(Score 는 F1 0.5 + (1−Pk) 0.25 + (1−WD) 0.25 이라 무예측이 벌점 0 을 받는다).",
          "> gold 밀도가 데이터마다 다르므로 **열 안에서만** 비교한다.",
          "\n⚠️ 앵커 k=1 · 탐색적 · dev-3 = 연습장(D-B16). seed 수 부족"
          "(slot-native 1 · dynamics 2 · anchor-only 1) — CLAUDE.md 3-run 요건 미충족."]
    p = REPO / "outputs/experiments/m4m5_phase1_anchor/M4M5_DTS_TRANSFER.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()

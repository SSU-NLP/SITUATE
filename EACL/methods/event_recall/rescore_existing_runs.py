#!/usr/bin/env python3
"""이미 저장된 LLM 분절 예측을 **논문 지표(Hamming distance)로** 재채점한다 — API 비용 0.

## 왜 이 스크립트가 필요했나
"EventRecall·LLM-Direct 는 이미 재현한 거 아니냐"는 질문에 답하기 위해서다.
우리 저장소에는 `outputs/eacl_phase0/dev_b3ref/{doc}_runs.json` 에 gpt-4.1 · 20회 반복의
문장 경계 예측이 남아 있다(`dev_b2b3_topline.py` 산출). 이걸 논문 지표로 다시 재면
새 API 호출 없이 대조가 되는지 확인할 수 있다.

## 대조 대상 (Michelmann et al.)
Pieman 에서 GPT 분절과 인간 주석의 **Hamming distance = 0.223(1차 행동런) / 0.191(2차)**.
논문의 인간 주석은 참가자 평균 **26.41개(1차) / 24.48개(2차)** 경계.

## 🔴 실행 결과 — 대조가 성립하지 않는다 (핵심)
Hamming distance 는 경계가 희소할 때 **"아무 경계도 안 찍기"에 지배된다.** 무경계 예측의
Hamming 을 같이 재면 그게 드러난다:

| 문서 | 전이 | gold 경계 | 우리 예측 | **우리 Hamming** | **무경계 하한** | 판정 |
|---|---:|---:|---:|---:|---:|---|
| pieman | 95 | 18 | 3.5 | 0.163 | 0.189 | 하한을 겨우 0.026 넘음 |
| tunnel | 503 | 18 | 9.7 | 0.044 | **0.036** | 🔴 **무경계보다 나쁨** |
| monkey | 407 | 30 | 14.1 | 0.083 | **0.074** | 🔴 **무경계보다 나쁨** |

따라서 "우리 0.163 < 논문 0.223 이니 더 잘했다"는 **말할 수 없다.** 두 수치의 무경계 하한이
다르기 때문이다 — 논문 gold 는 경계가 ~26개(밀도 높음)라 하한이 훨씬 높고, 우리 gold 는
저자 필터링 합의라 18개(밀도 낮음)다. **gold 집계 방식이 다르면 Hamming 은 비교 불가다.**

## 결론
저장된 예측이 있어도 **논문 수치와의 대조는 성립하지 않는다.** 필요한 것은
① 논문과 같은 gold 집계(개별 행동런 주석) ② 논문과 같은 모델. 둘 다 지금 없다.
이것이 이 프로젝트가 Hamming 대신 DTS Score 를 쓰는 이유이기도 하다.

    uv run python EACL/methods/event_recall/rescore_existing_runs.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
BASE = REPO / "outputs/eacl_phase0"
DOCS = ("pieman", "tunnel", "monkey")
PAPER_PIEMAN_HAMMING = (0.223, 0.191)   # Michelmann et al. — 행동런 1차 / 2차


def main() -> int:
    print("저장된 예측 = gpt-4.1 · temp 0 · 20회 반복 (B3 프롬프트, dev_b2b3_topline.py 산출)\n")
    rows = []
    for doc in DOCS:
        master = [json.loads(l) for l in (BASE / f"{doc}_michelmann_master.jsonl").open()]
        n = len(master)
        gold = np.array([r["boundary"] for r in master])

        d = json.loads((BASE / f"dev_b3ref/{doc}_runs.json").read_text())
        runs = d["runs"]

        def to_vec(idx):
            v = np.zeros(n, dtype=int)
            for t in idx:
                if 1 <= t <= n:
                    v[t - 1] = 1
            return v

        hd = [float(np.mean(to_vec(r) != gold)) for r in runs]
        cnt = Counter(t for r in runs for t in r)
        n_target = int(round(float(np.mean([len(r) for r in runs]))))
        consensus = [t for t, _ in cnt.most_common(n_target)]
        hd_cons = float(np.mean(to_vec(consensus) != gold))
        floor = float(np.mean(gold != 0))          # 무경계 예측의 Hamming

        rows.append({"doc": doc, "n_transitions": n, "n_gold": int(gold.sum()),
                     "mean_pred": float(np.mean([len(r) for r in runs])),
                     "hamming_mean": float(np.mean(hd)), "hamming_consensus": hd_cons,
                     "hamming_no_boundary_floor": floor,
                     "beats_floor": bool(np.mean(hd) < floor)})
        verdict = "하한 상회" if np.mean(hd) < floor else "🔴 무경계보다 나쁨"
        print(f"{doc:<8} 전이 {n:>3} · gold {int(gold.sum()):>2} · 예측 {np.mean([len(r) for r in runs]):>5.1f}"
              f" | Hamming {np.mean(hd):.3f} (합의 {hd_cons:.3f}) vs 무경계 하한 {floor:.3f}  {verdict}")

    print(f"\n논문(Michelmann) Pieman Hamming = {PAPER_PIEMAN_HAMMING[0]} / {PAPER_PIEMAN_HAMMING[1]}"
          " (인간 주석 ~26.4 / 24.5개 기준)")
    print("🔴 우리 gold 는 저자 필터링 합의 18개라 무경계 하한이 다르다 → **직접 대조 불가**.")

    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    (out / "rescore_existing_runs.json").write_text(
        json.dumps({"rows": rows, "paper_pieman_hamming": PAPER_PIEMAN_HAMMING},
                   ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

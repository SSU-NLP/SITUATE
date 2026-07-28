#!/usr/bin/env python3
"""복합 사건 정의 진단 — "동일 참여자 ∧ 같은 주제 ∧ 논리적 순서 ∧ 3~5발화 인접" (사용자 제안).

🔴 지위: **탐색적 진단**(dev-3 = 연습장, D-M13③). 확증 근거 아님. **새 라벨 0 · API 0.**

## 제안된 사건 정의
> 동일 참여자가 같은 주제를 논의하고, 논리적 순서(결정+행동+완료)를 이루며,
> 3~5 발화 이내로 시간적으로 인접하고 주제적으로 관련된 구간 = **하나의 사건**

⇒ 경계 = 이 복합 조건이 **깨지는** 지점.

## 기존 재료로의 매핑 (전부 이미 추출돼 있음)
| 제안 요소 | 사용 신호 |
|---|---|
| 동일 참여자 | `protagonist_present` 리셋 (dev-3 192회) |
| 같은 주제 | `space`(무대) · `intentionality`(목표) 리셋 |
| 논리적 순서(D+A+C) | `chainrole` 라벨 (D-B22 추출분) |
| **3~5 발화 인접** | **최소 구간 길이 제약** ← **한 번도 안 써본 것** |

**핵심:** 지금까지 모든 조작화가 전이마다 **독립 판정**했고 **길이 제약이 전혀 없었다**.
설계 §4.4 는 *"minimum-segment-length / cooldown 휴리스틱은 sticky-CRP κ 로 흡수"* 라 했으나
그 κ 는 미구현이다(D-B19 에서 스위칭 층 종결). 여기서는 κ 대신 **직접적인 최소 길이 제약**을
후처리로 얹어 그 성분만 분리 측정한다.

Score 는 공식 채점기(`hi_ontop.dts_scoring.score_dialogues`) 직접 호출. 메인 = Score.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
import llm_s5_states as s5  # noqa: E402
from axis_mixed_logit import AXES, build  # noqa: E402
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402

DEV3 = ("pieman", "tunnel", "monkey")
CHAIN = REPO / "outputs/runs/2026-07-28_chainrole_dev3"


def min_gap(bnd: np.ndarray, k: int) -> np.ndarray:
    """최소 구간 길이 제약 — 직전 경계 이후 k 전이 안에는 새 경계를 두지 않는다."""
    if k <= 0:
        return bnd.copy()
    out = np.zeros_like(bnd)
    last = -10**9
    for i, b in enumerate(bnd):
        if b and i - last > k:
            out[i] = 1
            last = i
    return out


def score(per_doc: dict[str, tuple[np.ndarray, np.ndarray]]) -> float:
    ys, ps = [], []
    for g, b in per_doc.values():
        ys.append(np.concatenate([g, [0]]).tolist())
        ps.append(np.concatenate([b, [0]]).tolist())
    return score_dialogues(ys, ps)["score"]


def main() -> None:
    df = build()
    chain = {}
    for d in DEV3:
        p = CHAIN / d / "chainrole_states.jsonl"
        st = {int(x["item_id"]): x["intentional_chain_role"] for x in s5.read(p)}
        n = max(st) + 1
        chain[d] = [st[i] for i in range(n)]

    docs = {}
    for d in DEV3:
        sub = df[df.doc == d]
        docs[d] = dict(gold=sub.boundary.to_numpy(),
                       ax={a: sub[a].to_numpy() for a in AXES},
                       n=len(sub))

    L = ["# 복합 사건 정의 진단 — 참여자 ∧ 주제 ∧ 논리순서 ∧ 3~5발화 인접\n",
         "> 🔴 **탐색적**(dev-3 = 연습장). 확증 근거 아님. 새 라벨 0 · API 0.",
         "> 핵심 신규 요소 = **최소 구간 길이 제약**(지금까지 모든 조작화가 전이별 독립 판정이었다).\n",
         "\n## 1. 기존 규칙 + 최소 길이 제약 (k = 직전 경계 이후 금지 구간)\n",
         "| 규칙 | k=0 | k=2 | k=3 | k=4 | k=5 | k=8 |", "|---|---|---|---|---|---|---|"]

    def eval_rule(name: str, fire: dict[str, np.ndarray]) -> list[float]:
        row = []
        for k in (0, 2, 3, 4, 5, 8):
            pd_ = {d: (docs[d]["gold"], min_gap(fire[d], k)) for d in DEV3}
            row.append(score(pd_))
        L.append(f"| {name} | " + " | ".join(f"{v:.4f}" for v in row) + " |")
        return row

    # 동결 규칙: 5축 합 >= 3
    eval_rule("**동결 count≥3**", {d: (np.sum([docs[d]["ax"][a] for a in AXES], 0) >= 3).astype(int)
                                   for d in DEV3})
    # 참여자만
    eval_rule("참여자(protagonist)만", {d: docs[d]["ax"]["protagonist_present"].copy() for d in DEV3})
    # 참여자 ∨ 주제(space∨intent)
    eval_rule("참여자 ∨ 주제(space∨intent)",
              {d: ((docs[d]["ax"]["protagonist_present"] | docs[d]["ax"]["space"] |
                    docs[d]["ax"]["intentionality"])).astype(int) for d in DEV3})
    # 위 + 논리순서(chainrole 이 outside_chain 으로 빠지는 전이)
    def chain_break(d):
        c = chain[d]
        n = docs[d]["n"]
        return np.array([int(c[i + 1] == "outside_chain" and c[i] != "outside_chain")
                         for i in range(min(n, len(c) - 1))])
    eval_rule("참여자 ∨ 주제 ∨ 논리순서이탈",
              {d: (np.pad(chain_break(d), (0, max(0, docs[d]["n"] - len(chain_break(d)))))
                   | docs[d]["ax"]["protagonist_present"] | docs[d]["ax"]["space"]
                   | docs[d]["ax"]["intentionality"]).astype(int) for d in DEV3})

    # ── 2. 참조 ──────────────────────────────────────────────────────────
    L += ["\n> k = 최소 구간 길이. **k=0 이 지금까지의 모든 실험**(길이 제약 없음).",
          "\n## 2. gold 실제 구간 길이 (제안된 '3~5 발화' 와 대조)\n",
          "| 문서 | 경계 | 평균 간격 | 중앙값 | 최소 | 최대 |", "|---|---|---|---|---|---|"]
    for d in DEV3:
        gi = np.flatnonzero(docs[d]["gold"] == 1)
        gaps = np.diff(np.concatenate([[-1], gi]))
        L.append(f"| {d} | {len(gi)} | {gaps.mean():.1f} | {np.median(gaps):.0f} | "
                 f"{gaps.min()} | {gaps.max()} |")
    L += ["\n⚠️ dev-3 = 연습장, pieman 은 선택압 9회 누적(D-M5). 확증 주장 불가.",
          "⚠️ '3~5 발화'는 대화(발화) 단위 제안이나 dev-3 는 **서사 문장** 단위다 — 층위가 다를 수 있다."]
    p = REPO / "outputs/experiments/m4m5_phase1_anchor/COMPOSITE_EVENT.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()

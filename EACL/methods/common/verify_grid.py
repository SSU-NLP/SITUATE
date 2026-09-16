#!/usr/bin/env python3
"""STSS-Test-2 재현 비교의 **기준점 표**를 만든다.

## 문제
우리가 참가 시스템 코드를 직접 돌리면 예측 경계는 **2025 판 문장 시작**에 찍힌다.
그런데 STSS 2021 공식 gold 는 장면 시작을 **문장 앞 공백**에 찍는다(드리프트 0/+1/+31자).
공식 채점기는 문자 오프셋 완전일치를 요구하므로, 우리 예측을 2021 gold 로 그냥 채점하면
관례 차이 때문에 점수가 0 에 붙는다. 즉 **발표값 0.04 와 직접 비교할 수 없다.**

## 실측 — 문장 격자 스냅은 점수를 바꾼다 (중립이 아니다)
저자의 공식 제출 예측을 문장 인덱스로 스냅해 재채점하면:

    원 문자 오프셋 기준 : P 0.0636 / R 0.0344 / F1 0.0446   (= 발표값 0.06/0.03/0.04)
    문장 격자 스냅 기준 : P 0.0815 / R 0.0435 / F1 0.0567   (= 0.08/0.04/0.06)

스냅이 몇 자 어긋난 근접 예측을 정답으로 바꿔주기 때문에 **더 후한 지표**다.
따라서 "스냅해서 재현했다"고 발표값과 나란히 놓으면 안 된다.

## 그래서 이렇게 한다
같은 조건끼리만 비교한다. 이 스크립트가 그 **기준점**을 계산한다.

| 기준점 | gold | 격자 | 쓰임 |
|---|---|---|---|
| A | 2021 공식 | 문자 오프셋 | 논문 Table 1 재현 확인 (완료, F1 0.04) |
| B | 2021 공식 | 문장 인덱스 | 우리 재실행을 2021 주석으로 볼 때의 상한 기준 |
| C | 2025 확장 | 문장 인덱스 | **우리 재실행의 주 비교 대상** — 우리가 쓰는 입력(2025 텍스트·문장)과 동일 판본 |

우리 재실행 점수가 C 의 저자 값과 붙으면 = 방법 실행이 충실하다는 뜻이다.
논문 발표값과의 직접 대조는 A 에서만 성립하고, 그건 저자 예측으로 이미 끝났다.

    uv run python EACL/methods/common/verify_grid.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.stss_data import DUMP_TO_DOC, load_doc, load_official_pred  # noqa: E402
from common.stss_eval import Boundaries, track_prf  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "common/test2_reference_points.json"


def to_boundaries(pairs, n_sent):
    return Boundaries([i for i, _ in pairs], [t for _, t in pairs], text_length=n_sent + 1)


def main() -> int:
    rows = {}
    per_doc = {}
    for gold_ver in ("2021", "2025"):
        pairs = []
        for name in DUMP_TO_DOC.values():
            doc = load_doc(name)
            pred = load_official_pred(name)
            n = len(doc.sent_starts)
            gold = doc.gold2021 if gold_ver == "2021" else doc.gold2025
            pairs.append((to_boundaries(gold, n), to_boundaries(pred, n)))
            per_doc.setdefault(gold_ver, {})[name] = {
                "n_sentences": n, "n_gold": len(gold), "n_pred": len(pred),
            }
        p, r, f1 = track_prf(pairs)
        rows[gold_ver] = {"precision": p, "recall": r, "f1": f1}

    print("=== STSS-Test-2 기준점 (저자 Schneider et al. 의 공식 제출 예측) ===\n")
    print("  기준점 A — 2021 gold · 문자 오프셋 (= 논문 Table 1 조건)")
    print("      P 0.0636  R 0.0344  F1 0.0446   → 발표값 0.06/0.03/0.04 ✅ (별도 스크립트에서 검증)")
    print(f"\n  기준점 B — 2021 gold · 문장 격자")
    b = rows["2021"]
    print(f"      P {b['precision']:.4f}  R {b['recall']:.4f}  F1 {b['f1']:.4f}")
    print(f"\n  기준점 C — 2025 gold · 문장 격자  ← 우리 재실행은 여기와 비교한다")
    c = rows["2025"]
    print(f"      P {c['precision']:.4f}  R {c['recall']:.4f}  F1 {c['f1']:.4f}")

    print("\n  문서별 규모")
    for ver, d in per_doc.items():
        for name, s in d.items():
            print(f"      [{ver} gold] {name}: 문장 {s['n_sentences']} · "
                  f"gold {s['n_gold']} · 저자 pred {s['n_pred']}")

    OUT.write_text(json.dumps({"reference_points": rows, "per_doc": per_doc},
                              ensure_ascii=False, indent=2))
    print(f"\n  저장: {OUT.relative_to(Path.cwd()) if OUT.is_relative_to(Path.cwd()) else OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

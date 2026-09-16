#!/usr/bin/env python3
"""재현 검사 ①: 공식 STSS 표(Zehe et al. 2021a, Table 1)의 Schneider et al. 행을 재계산한다.

**무엇을 검증하는가.** 우리 채점기(`methods/common/stss_eval.py`)가 공식 `eval.py` 와
같은 값을 내는지. embedding-delta 레포에는 저자들의 **공식 제출 예측 + 공식 gold** 가
`training/stss_results/{t1,t2}/` 에 그대로 들어 있으므로, 저작권 텍스트 없이도
공식 표의 수치를 재계산할 수 있다.

**기대값 (Zehe et al. 2021a, Table 1 — Schneider et al. 2021 행)**
    Track 1: P 0.01 / R 0.02 / F1 0.02
    Track 2: P 0.06 / R 0.03 / F1 0.04

일치하면 = 채점기 구현 검증 완료 + 이후 모든 재현 수치의 기준선 확보.
불일치하면 = 우리 채점기 또는 덤프 해석이 틀렸다는 뜻이므로 다른 재현 수치를 믿을 수 없다.

    uv run python EACL/methods/embedding_delta/check_official_submission.py
"""
from __future__ import annotations

import sys
from pathlib import Path

METHODS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHODS))

from common.stss_eval import parse_official_dump, pooled_exact_prf, exact_prf, track_prf  # noqa: E402

DUMPS = METHODS / "_upstream/embedding-delta/training/stss_results"
PAPER = {  # Zehe et al. 2021a Table 1, Schneider et al. 2021 행
    "t1": (0.01, 0.02, 0.02),
    "t2": (0.06, 0.03, 0.04),
}


def main() -> int:
    ok = True
    for track in ("t1", "t2"):
        files = sorted((DUMPS / track).iterdir())
        pairs = []
        print(f"\n=== Track {track[-1]} ({len(files)}개 문서) ===")
        for f in files:
            d = parse_official_dump(f)
            p, r, f1 = exact_prf(d["gold"], d["pred"])
            print(f"  {f.name}: gold {len(d['gold'].offsets):3d}개 · "
                  f"pred {len(d['pred'].offsets):3d}개 · P {p:.3f} R {r:.3f} F1 {f1:.3f}")
            pairs.append((d["gold"], d["pred"]))
        p, r, f1 = track_prf(pairs)
        qp, qr, qf = pooled_exact_prf(pairs)
        pp, pr, pf = PAPER[track]
        # 발표값이 소수 2자리 반올림이므로 "반올림해서 같은가"로 판정한다
        match = all(round(a, 2) == b for a, b in ((p, pp), (r, pr), (f1, pf)))
        ok &= match
        print(f"  → 공식 집계(문서별 micro 의 평균):  P {p:.4f}  R {r:.4f}  F1 {f1:.4f}"
              f"  → 반올림 {round(p,2):.2f}/{round(r,2):.2f}/{round(f1,2):.2f}")
        print(f"    (참고) 트랙 pooled micro       :  P {qp:.4f}  R {qr:.4f}  F1 {qf:.4f}")
        print(f"    논문 Table 1 발표값            :  P {pp:.2f}    R {pr:.2f}    F1 {pf:.2f}"
              f"   {'✅ 일치' if match else '🔴 불일치'}")
    print("\n판정:", "✅ 채점기 재현 성공" if ok else "🔴 재현 실패 — 채점기/덤프 해석 점검 필요")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

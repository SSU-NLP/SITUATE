#!/usr/bin/env python3
"""우리가 이식한 Zehe et al. (2025) 지표가 **저자 원본 함수와 비트 단위로 같은지** 검사한다.

저자 코드 `_upstream/scene-segmentation/ssc/evaluation.py` 를 직접 import 해서
같은 입력에 두 구현을 태우고 값을 대조한다. `ssc.model` 은 torch 등 무거운 의존성을
끌어오므로, 이 검사에 필요한 `Label` 상수만 스텁으로 주입해 evaluation 모듈만 로드한다.

랜덤 입력 2,000 케이스에서 하나라도 어긋나면 실패로 본다.
지표 구현이 다르면 이후의 모든 "재현/미재현" 판정이 무의미해지므로 선행 관문이다.

    uv run python EACL/methods/common/verify_metric_port.py
"""
from __future__ import annotations

import importlib.util
import random
import sys
import types
from pathlib import Path

METHODS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHODS))

from common.stss_eval import NOBORDER, iou_zehe2025, relaxed_f1_zehe2025  # noqa: E402

UPSTREAM_EVAL = METHODS / "_upstream/scene-segmentation/ssc/evaluation.py"
LABELS = ["S2S", "S2NS", "NS2S"]


def load_upstream():
    """`ssc.model` 을 스텁으로 대체하고 저자 evaluation 모듈만 로드한다."""
    ssc = types.ModuleType("ssc")
    model = types.ModuleType("ssc.model")

    class Label:
        BORDER = "BORDER"
        S2NS = "S2NS"
        S2S = "S2S"
        NS2S = "NS2S"
        NOBORDER = NOBORDER

    model.Label = Label
    sys.modules["ssc"] = ssc
    sys.modules["ssc.model"] = model

    spec = importlib.util.spec_from_file_location("upstream_eval", UPSTREAM_EVAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    up = load_upstream()
    rng = random.Random(20260803)
    bad_f1 = bad_iou = 0
    cases = 2000

    for _ in range(cases):
        n = rng.randint(5, 120)
        rate = rng.choice([0.02, 0.05, 0.15, 0.4])
        t = [rng.choice(LABELS) if rng.random() < rate else NOBORDER for _ in range(n)]
        p = [rng.choice(LABELS) if rng.random() < rate else NOBORDER for _ in range(n)]
        tol = rng.choice([0, 1, 2, 3, 5])

        ours = relaxed_f1_zehe2025(t, p, tol)
        theirs = up.evaluate_predictions(t, p, tol)
        if any(abs(ours[k] - theirs[k]) > 1e-12 for k in ("precision", "recall", "f1")):
            bad_f1 += 1
            if bad_f1 == 1:
                print(f"  🔴 완화F1 불일치 예시: tol={tol} 우리={ours} 저자={theirs}")

        if abs(iou_zehe2025(t, p) - up.evaluate_predictions_iou(t, p)) > 1e-12:
            bad_iou += 1

    print(f"랜덤 {cases}케이스 대조")
    print(f"  완화 F1 (tolerance 0/1/2/3/5) : {'✅ 전부 일치' if not bad_f1 else f'🔴 {bad_f1}건 불일치'}")
    print(f"  IoU                          : {'✅ 전부 일치' if not bad_iou else f'🔴 {bad_iou}건 불일치'}")
    return 0 if not (bad_f1 or bad_iou) else 1


if __name__ == "__main__":
    raise SystemExit(main())

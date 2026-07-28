#!/usr/bin/env python3
"""LLM 없는 최고버전 — Hi-OnTop 임베딩 화제분절 (사용자 직접-입력).

우리 방법론의 **비-LLM main 모델**. DTS(대화 화제분절)에서는 de-neut/CR 이 아니라
δ_eff causal-window 신호가 우세하다(``context/methodology/hi-ontop-deneut.md`` §5 ★★).
따라서 여기서는 :class:`hi_ontop.hi_ontop.HiOnTop` (δ_eff) 를 main 으로 쓴다.

- 인코더: ``multi-qa-mpnet-base-dot-v1`` (768-D, L2 정규화) — 로컬, LLM 0콜.
- 온라인·0-look-ahead: 발화가 도착하는 대로 turn 당 O(1) 로 경계 판정.
- 임계 모드 2가지:
  - ``adaptive`` (기본, **calibration-free**): 대화 자체 통계 μ+cσ 로 자기보정
    (:class:`hi_ontop.hi_ontop_v2.HiOnTopV2`). 임의의 사용자 입력에 권장 — δ* 를
    코퍼스별로 튜닝할 필요가 없다.
  - ``fixed``: 고정 δ*(TIAGE-train 보정 0.5594). 보고 수치 재현용.

usage:
  uv run python scripts/segment_manual.py                    # 대화형 입력
  uv run python scripts/segment_manual.py --file conv.txt    # 파일 입력
  echo "..." | uv run python scripts/segment_manual.py       # 파이프 입력
  uv run python scripts/segment_manual.py --threshold fixed --delta-star 0.5594
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hi_ontop.embedding import make_encoder                      # noqa: E402
from hi_ontop.hi_ontop import HiOnTop                            # noqa: E402
from hi_ontop.hi_ontop_v2 import HiOnTopV2, _otsu_threshold      # noqa: E402
from segment_manual_common import print_segments, read_turns     # noqa: E402

# 기본 인코더 = 다국어(한국어 포함). 방법론 원본 영어 인코더는
# multi-qa-mpnet-base-dot-v1 이며 --encoder 로 지정 가능. 다국어 인코더는 δ* 보정값이
# 다르므로 임계는 calibration-free 적응모드(기본) 사용을 권장.
DEFAULT_ENCODER = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def main() -> int:
    ap = argparse.ArgumentParser(description="Hi-OnTop 임베딩 화제분절 (LLM 없음)")
    ap.add_argument("--file", help="발화 파일 경로(줄당 1턴). 없으면 stdin/대화형.")
    ap.add_argument("--encoder", default=DEFAULT_ENCODER,
                    help=f"인코더 모델명. 기본={DEFAULT_ENCODER} (다국어·한국어 지원). "
                         "영어 mpnet 원본은 'sentence-transformers/multi-qa-mpnet-base-dot-v1'.")
    ap.add_argument("--threshold", choices=["adaptive", "otsu", "fixed"], default="adaptive",
                    help="adaptive=μ+cσ 자기보정(c=1.0 고정, calibration-free, online) / "
                         "otsu=parameter-free(c·δ* 둘 다 없음, offline 전체 히스토그램) / "
                         "fixed=고정 δ*(보고 수치 재현용)")
    ap.add_argument("--delta-star", type=float, default=0.5594,
                    help="fixed 모드 경계 임계 δ* (기본 0.5594, TIAGE-train 보정)")
    ap.add_argument("--c", type=float, default=1.0,
                    help="adaptive 모드 임계 = μ + c·σ (기본 1.0, 클수록 경계 적음)")
    ap.add_argument("--warmup", type=int, default=5,
                    help="adaptive 모드 warmup 턴 수(그 전엔 δ* fallback, 기본 5)")
    ap.add_argument("--ctx-window", type=int, default=2, help="causal 맥락 발화 수 m (기본 2)")
    ap.add_argument("--ctx-decay", type=float, default=0.7, help="맥락 기하감쇠 ρ (기본 0.7)")
    ap.add_argument("--ctx-blend-a", type=float, default=0.5,
                    help="δ_eff = a·δ_prev + (1−a)·δ_ctx 의 a (기본 0.5)")
    args = ap.parse_args()

    turns = read_turns(args.file)
    if len(turns) < 2:
        print("발화가 2턴 이상 필요합니다.", file=sys.stderr)
        return 1

    print(f"[인코더 로딩] {args.encoder} ...", file=sys.stderr)
    enc = make_encoder(model=args.encoder)
    embs = np.asarray(enc.encode([t["text"] for t in turns]))

    def _band(g: float) -> str:
        return "strong" if g >= 1.3 else "normal" if g >= 1.0 else "weak" if g >= 0.7 else "·"

    boundaries: list[bool] = []
    extra: list[str] = []

    if args.threshold == "otsu":
        # parameter-free: δ_eff 시퀀스를 먼저 뽑고(임계 미적용용 delta_star=2.0),
        # Otsu 컷으로 경계 판정. c 도 δ* 도 없음.
        probe = HiOnTop(dim=enc.dim, delta_star=2.0, ctx_window=args.ctx_window,
                        ctx_decay=args.ctx_decay, ctx_blend_a=args.ctx_blend_a)
        for s in embs:
            probe.assign(s)
        deff = [float(h["delta_eff"]) for h in probe.history()]
        thr = _otsu_threshold(np.asarray(deff[1:], dtype=np.float64))
        for i, d in enumerate(deff):
            is_b = i > 0 and d >= thr
            boundaries.append(is_b)
            g = d / thr if thr > 0 else 0.0
            extra.append(f"δ={d:.3f} g={g:.2f} {_band(g)}")
        mode_desc = f"otsu parameter-free (thr={thr:.3f}, c·δ* 없음)"
    else:
        if args.threshold == "adaptive":
            seg = HiOnTopV2(dim=enc.dim, c=args.c, warmup=args.warmup,
                            ctx_window=args.ctx_window, ctx_decay=args.ctx_decay,
                            ctx_blend_a=args.ctx_blend_a)
            mode_desc = f"adaptive μ+{args.c}σ (calibration-free, warmup={args.warmup})"
        else:
            seg = HiOnTop(dim=enc.dim, delta_star=args.delta_star,
                          ctx_window=args.ctx_window, ctx_decay=args.ctx_decay,
                          ctx_blend_a=args.ctx_blend_a)
            mode_desc = f"fixed δ*={args.delta_star}"
        for s in embs:
            _, is_b = seg.assign(s)
            boundaries.append(bool(is_b))
            g = seg.last_graded_score
            extra.append(f"δ={seg.last_delta_eff:.3f} g={g:.2f} {_band(g)}")

    print(f"\n[방법] Hi-OnTop δ_eff · 임계={mode_desc}", file=sys.stderr)
    print_segments(turns, boundaries, extra)
    print("범례: δ=δ_eff(놀람) · g=graded score(δ_eff/임계) · "
          "band(weak<1.0≤normal<1.3≤strong) · ▶=화제 경계")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""오프라인 ↔ 온라인 **Δ 측정** — 하네스 C 에서 같은 채점기로 잰다.

## 이 스크립트가 답하는 질문
`EACL/main_result.md` §8 이 한계로 적어 둔 것:

> **"스트리밍화 손실을 방법별로 분리하지 않았다. CSM 한 종에서만 측정했다."**

여기서는 **모든 방법**에 대해 `Δ = online − offline` 을 같은 데이터·같은 채점기로 낸다.

## 왜 Δ 가 의미 있나 — 바뀌는 변수가 하나뿐이다
온라인 변환은 **prefix-recompute** 다: 턴 t 의 판정을 `u_1..u_t` 만으로 내리고 최전선
전이만 채택한다. 모델·가중치·하이퍼·데이터·채점기가 전부 오프라인과 동일하므로
**Δ 는 곧 '미래 문맥의 기여'** 다. 스트리밍 근사(running threshold, delay=k)를 쓰면
근사 오차가 섞여 이 해석이 무너진다.

이것이 Causal-SSC 논증의 측정 도구다 — *"하락폭이 크면 온라인 조건이 어렵다는 증거,
작으면 SotA 가 불필요한 미래 문맥을 쓴다는 발견"*.

## 선행 조건
온라인 경로가 정말 미래를 안 보는지는 `verify_causality.py` 가 별도로 검증한다
(입력 감사 + 미래 교란 불변 + 대조군). 그 검증 없이 나온 Δ 는 믿을 수 없다.

    PYTHONPATH=benchmarks/superdialseg/src NLTK_DATA=/root/nltk_data \\
      .venv-superdialseg/bin/python EACL/methods/common/run_delta.py --method texttiling
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

METHODS = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(METHODS))

from common.superdialseg_harness import (DATASETS, METRICS, PAPER_TABLE3,  # noqa: E402
                                         evaluate, evaluate_online)

OUT = REPO / "outputs/experiments/2026-08-03_online_offline_delta"


def make_segmenter(method: str):
    """방법별 팩토리. 데이터셋마다 새로 만들어야 한다(evaluation 누적기 초기화)."""
    if method == "texttiling":
        from super_dialseg.models.texttiling.modeling_texttiling import TexttilingSegmenter
        return lambda: TexttilingSegmenter()
    if method == "greedyseg":
        from super_dialseg.models.greedyseg.modeling_greedyseg import GreedySegmenter
        return lambda: GreedySegmenter()
    raise ValueError(f"미지원 방법: {method}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="texttiling")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-dialogues", type=int, default=None,
                    help="온라인은 문서당 O(n²) 라 표본을 줄일 수 있다. 줄이면 결과에 명시된다")
    args = ap.parse_args()

    factory = make_segmenter(args.method)
    rows = {}
    for ds in args.datasets.split(","):
        print(f"\n===== {ds} =====")
        off = evaluate(factory(), ds, batch_size=1, num_workers=args.num_workers)
        on = evaluate_online(factory(), ds, batch_size=1, num_workers=args.num_workers,
                             max_dialogues=args.max_dialogues)
        rows[ds] = {"offline": off, "online": on}
        print(f"  offline total_score {off['total_score']:.4f} · "
              f"online {on['total_score']:.4f} · Δ {on['total_score'] - off['total_score']:+.4f}"
              f"  (온라인 문서 {on['_n_dialogues']} · 최전선 miss {on['_n_frontier_miss']})")

    print("\n" + "=" * 84)
    print(f"{args.method} — Δ = online − offline (하네스 C)")
    print("=" * 84)
    print(f"{'데이터셋':<12}{'지표':<14}{'offline':>10}{'online':>10}{'Δ':>10}{'논문(off)':>11}")
    for ds, r in rows.items():
        for m in METRICS:
            off_v, on_v = r["offline"][m], r["online"][m]
            paper = PAPER_TABLE3.get(args.method, {}).get(ds, {}).get(m)
            ps = f"{paper:.3f}" if paper is not None else "—"
            print(f"{ds:<12}{m:<14}{off_v:>10.4f}{on_v:>10.4f}{on_v - off_v:>+10.4f}{ps:>11}")

    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"delta_{args.method}.json"
    f.write_text(json.dumps({"config": vars(args), "rows": rows,
                             "paper_offline": PAPER_TABLE3.get(args.method, {})},
                            ensure_ascii=False, indent=2))
    print(f"\n저장: {f}")
    print("🔴 Δ 해석 전제: 온라인 경로의 인과성은 verify_causality.py 로 별도 검증돼야 한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

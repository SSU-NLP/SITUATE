#!/usr/bin/env python3
"""**lookahead 곡선** — 미래를 얼마나 볼 수 있느냐에 따라 점수가 어떻게 변하는가.

## 왜 단일 Δ 가 아니라 곡선인가
`Δ = online − offline` 하나로는 **무엇이 변했는지 특정되지 않는다.** 온라인 변형에는
최소 두 축이 섞여 있다 — ① 미래 문맥을 얼마나 쓰는가 ② 그 조건에 맞게 재설계했는가.
`2026-08-03_online_offline_delta/REPORT.md` §4 가 그 혼동을 실증했다: prefix-recompute
TextTiling 은 Δ −0.075 를 냈지만 F1 이 0.20→0.01 로 **퇴화**한 결과였고, 같은 방법의
스트리밍 판은 훨씬 높았다.

그래서 **미래 사용량을 축으로 하는 세 점**을 잰다:

| 조건 | 미래 사용 | 판정 확정 | 구현 |
|---|---|---|---|
| `offline` | 무제한 | 문서 종료 후 | 저자 분절기 그대로 |
| `bounded` | k 블록 / delay=d | 턴 t+d | 스트리밍 구현(`streaming_adapters`) |
| `zero` | 없음 | 턴 t 즉시 | prefix-recompute(`evaluate_online`) |

Causal-SSC 논증(*"하락폭이 크면 온라인이 어렵다 / 작으면 SotA 가 불필요한 미래를 쓴다"*)은
이 곡선의 **기울기**로 세워야 한다. offline→bounded 가 완만하고 bounded→zero 가 가파르면
"약간의 lookahead 만 있으면 된다"는 뜻이고, offline→bounded 부터 가파르면 "미래 문맥이
본질적"이라는 뜻이다.

## 🔴 하이퍼를 섞지 말 것
스트리밍 조건의 기본 하이퍼는 **오프라인과 동일**(w=10·k=6)하게 둔다. 기존 러너는 짧은
대화 때문에 w=5·k=3 으로 낮췄는데, 그 상태로 비교하면 하이퍼 변경이 곡선에 섞인다.
낮춘 조건이 필요하면 `--stream-w/--stream-k` 로 명시하고 **별도 조건으로** 보고한다.

    PYTHONPATH="benchmarks/superdialseg/src:src" NLTK_DATA=/root/nltk_data \\
      .venv-superdialseg/bin/python EACL/methods/common/run_lookahead_curve.py \\
        --method texttiling --datasets tiage,dialseg711
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

OUT = REPO / "outputs/experiments/2026-08-03_lookahead_curve"


def offline_factory(method: str):
    if method == "texttiling":
        from super_dialseg.models.texttiling.modeling_texttiling import TexttilingSegmenter
        return lambda: TexttilingSegmenter()
    raise ValueError(f"미지원: {method}")


def bounded_factory(method: str, w: int, k: int):
    if method == "texttiling":
        from common.streaming_adapters import StreamingTextTilingAdapter
        return lambda: StreamingTextTilingAdapter(w=w, k=k)
    raise ValueError(f"미지원: {method}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="texttiling")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--stream-w", type=int, default=10, help="기본 = 오프라인과 동일")
    ap.add_argument("--stream-k", type=int, default=6, help="기본 = 오프라인과 동일")
    ap.add_argument("--max-dialogues", type=int, default=None,
                    help="zero 조건은 O(n²) 라 표본을 줄일 수 있다. 줄이면 결과에 기록된다")
    args = ap.parse_args()

    off_f = offline_factory(args.method)
    bnd_f = bounded_factory(args.method, args.stream_w, args.stream_k)

    rows = {}
    for ds in args.datasets.split(","):
        print(f"\n===== {ds} =====")
        r = {}
        r["offline"] = evaluate(off_f(), ds, batch_size=1, num_workers=args.num_workers)
        print(f"  offline  {r['offline']['total_score']:.4f}")
        # bounded 는 인과성이 push 루프 안에 있으므로 오프라인 평가 경로를 그대로 쓴다
        r["bounded"] = evaluate(bnd_f(), ds, batch_size=1, num_workers=args.num_workers)
        print(f"  bounded  {r['bounded']['total_score']:.4f}  (w={args.stream_w} k={args.stream_k})")
        r["zero"] = evaluate_online(off_f(), ds, batch_size=1, num_workers=args.num_workers,
                                    max_dialogues=args.max_dialogues)
        print(f"  zero     {r['zero']['total_score']:.4f}  "
              f"(문서 {r['zero']['_n_dialogues']} · miss {r['zero']['_n_frontier_miss']})")
        rows[ds] = r

    print("\n" + "=" * 92)
    print(f"{args.method} — lookahead 곡선 (하네스 C)")
    print("=" * 92)
    print(f"{'데이터셋':<12}{'지표':<14}{'offline':>10}{'bounded':>10}{'zero':>10}"
          f"{'Δ(b−o)':>10}{'Δ(z−b)':>10}")
    for ds, r in rows.items():
        for m in METRICS:
            o, b, z = r["offline"][m], r["bounded"][m], r["zero"][m]
            print(f"{ds:<12}{m:<14}{o:>10.4f}{b:>10.4f}{z:>10.4f}{b - o:>+10.4f}{z - b:>+10.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"curve_{args.method}_w{args.stream_w}k{args.stream_k}.json"
    f.write_text(json.dumps({"config": vars(args), "rows": rows,
                             "paper_offline": PAPER_TABLE3.get(args.method, {})},
                            ensure_ascii=False, indent=2))
    print(f"\n저장: {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

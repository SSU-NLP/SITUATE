#!/usr/bin/env python3
"""고정간격 · 무작위 하한을 **NES 무대(서사)** 에서 돌려 예측 덤프를 만든다.

## 무엇을 하는가
문서별 예측 경계 수를 **Hi-OnTop(M5) 과 정확히 일치**시킨 채 위치만 무의미하게 놓는다.
⇒ `M5 − 이 하한` 이 곧 **위치 선택의 기여**다. 밀도가 통제되지 않은 하한
(예: SuperDialseg 의 Random/Even, 경계 수를 무작위로 뽑는다)은 이 역할을 못 한다.

## 왜 밀도 일치가 필요한가
분절 점수는 경계 밀도에 크게 좌우된다. 어떤 방법이 좋아 보일 때 그게 *"경계를 잘 놓아서"* 인지
*"개수가 우연히 맞아서"* 인지 구분되지 않으면 주장이 서지 않는다. `main_result.md` 가
BOR(예측/gold 경계비)을 병기하는 것도 같은 이유다.

## 인과성
둘 다 **정의상 온라인**이다 — 위치·난수만 쓰고 내용을 안 본다. 오프라인/온라인 구분이 없다.

## 산출
`outputs/experiments/2026-07-29_baseline_compare/preds/{FixedInterval,RandomRate}_{ds}.json`
= 기존 덤프 규약과 동일. 이후 `scripts/eacl_phase0/preds_table.py` 가 6지표를 재계산한다.

    .venv/bin/python EACL/methods/trivial/run_nes.py --datasets dev3,test3
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))

PRED = REPO / "outputs/experiments/2026-07-29_baseline_compare/preds"
REF_METHOD = "M5"          # 밀도를 맞출 기준 = Hi-OnTop


def ref_counts(ds: str) -> dict[str, int]:
    """기준 방법의 문서별 예측 경계 수. 없으면 M5-adaptive 로 대체한다."""
    for name in (REF_METHOD, f"{REF_METHOD}-adaptive"):
        f = PRED / f"{name}_{ds}.json"
        if f.exists():
            d = json.loads(f.read_text())
            print(f"  밀도 기준 = {name}_{ds}.json")
            return {k: int(sum(v)) for k, v in d.items()}
    raise FileNotFoundError(f"{ds}: M5 계열 예측 덤프가 없어 밀도를 맞출 수 없다")


def fixed_interval(n: int, k: int) -> list[int]:
    """n 개 전이에 k 개 경계를 **균등 간격**으로. 결정적."""
    out = [0] * n
    if k <= 0 or n <= 0:
        return out
    for j in range(1, k + 1):
        i = min(n - 1, max(0, round(j * n / (k + 1)) - 1))
        out[i] = 1
    return out


def random_positions(n: int, k: int, rng: random.Random) -> list[int]:
    """n 개 전이 중 k 개를 **비복원 무작위** 로. 개수가 정확히 k 다."""
    out = [0] * n
    if k <= 0 or n <= 0:
        return out
    for i in rng.sample(range(n), min(k, n)):
        out[i] = 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="dev3,test3")
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--reps", type=int, default=20,
                    help="무작위 하한의 반복 수 — 단일 시드 보고는 위험하다")
    args = ap.parse_args()

    from baseline_compare import load_dataset   # noqa: PLC0415

    summary = {}
    for ds in args.datasets.split(","):
        print(f"\n=== {ds} ===")
        counts = ref_counts(ds)
        data = load_dataset(ds)

        fixed, rand_reps = {}, []
        for doc, _sents, gold in data:
            n = len(gold)
            k = counts.get(doc)
            if k is None:
                print(f"  ⚠️ {doc}: 기준 밀도 없음 — 건너뜀")
                continue
            fixed[doc] = fixed_interval(n, k)
            print(f"  {doc}: 전이 {n} · gold {sum(gold)} · 기준(M5) 경계 {k}")

        for r in range(args.reps):
            rng = random.Random(f"{args.seed}|{ds}|{r}")
            rand_reps.append({doc: random_positions(len(gold), counts[doc], rng)
                              for doc, _s, gold in data if doc in counts})

        PRED.mkdir(parents=True, exist_ok=True)
        (PRED / f"FixedInterval_{ds}.json").write_text(json.dumps(fixed))
        # 무작위는 대표 1개를 덤프하고(표에 넣기 위해) 전 반복은 별도 저장
        (PRED / f"RandomRate_{ds}.json").write_text(json.dumps(rand_reps[0]))
        outdir = REPO / "outputs/experiments/2026-08-03_trivial_nes"
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / f"random_reps_{ds}.json").write_text(
            json.dumps({"seed": args.seed, "reps": args.reps, "runs": rand_reps}))
        summary[ds] = {"n_docs": len(fixed), "ref_counts": counts}
        print(f"  → FixedInterval_{ds}.json · RandomRate_{ds}.json (+ 반복 {args.reps}개 별도)")

    print("\n다음: scripts/eacl_phase0/preds_table.py 로 6지표 재계산")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

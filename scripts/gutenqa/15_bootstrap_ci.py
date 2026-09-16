#!/usr/bin/env python3
"""GutenQA 셀 쌍의 DCG@20 차이에 대한 **쌍체 부트스트랩** 신뢰구간.

## 왜 신규 스크립트인가
원장(D-B43/B44/B46)의 CI 값들은 세션 내 일회성 인라인 계산으로 나왔고 `scripts/` 에 코드가 없다.
즉 **재현 경로가 없는 수치**였다. 하류 재검증에서 같은 판정을 반복해야 하므로 러너로 승격한다.

## 왜 쌍체(paired)인가
두 셀이 **같은 3,000 질문**을 푼다. 질문 난이도 분산이 방법 간 차이보다 훨씬 크므로,
독립 부트스트랩을 쓰면 CI 가 부풀어 실제로 있는 차이를 못 본다.
질문 단위로 같은 인덱스를 함께 뽑아 **질문 효과를 상쇄**한다.

## 판정
CI 가 0을 포함하면 "구분 안 됨"으로 적는다. 점추정 부호만 보고 주장하지 않는다.

    uv run python scripts/gutenqa/15_bootstrap_ci.py --pairs m5i2__cd_adaptive:m5__cd_adaptive
    uv run python scripts/gutenqa/15_bootstrap_ci.py --auto-ckpt      # I2/W3 vs W0 10규칙 전량
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OLD = REPO_ROOT / "outputs/experiments/2026-07-29_gutenqa_chunking"

RULES = ["rate0.15", "adaptive_online_nf_c1", "ewma5_c1", "ewma20_c1", "ewma50_c1",
         "rate0.15_b1", "adaptive_online_nf_c1_b1", "ewma20_c1_b1", "cd_adaptive", "cd_ewma20"]
SLOPE, INTERCEPT = 0.0221, 46.03      # 🔒 D-B44 동결 회귀선. 재적합 금지.


def load(name: str, root: Path, sub: str = "retrieval") -> tuple[dict[int, float], float] | None:
    """🔴 `sub` 를 반드시 받는다. 예전엔 retrieval/ 를 먼저 보고 없을 때만 retrieval_ada/ 로
    폴백했는데, 두 곳에 같은 이름이 있으면 --ada 를 줘도 mpnet 을 읽으면서 헤더에는
    'retrieval_ada' 라고 찍혔다. 인코더를 조용히 바꿔치기하는 종류의 버그다."""
    p = root / sub / f"{name}.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text())
    by_q = {r["qid"]: float(r["dcg@20"]) for r in j["records"]}
    return by_q, float(j.get("chunk_stats", {}).get("mean_words_per_chunk", float("nan")))


def paired_ci(a: dict[int, float], b: dict[int, float], n_boot: int, seed: int) -> tuple:
    qs = sorted(set(a) & set(b))
    da = np.array([a[q] for q in qs]); db = np.array([b[q] for q in qs])
    d = (da - db) * 100
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(qs), size=(n_boot, len(qs)))
    boot = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), len(qs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=[], metavar="A:B", help="셀A − 셀B")
    ap.add_argument("--auto-ckpt", action="store_true", help="I2·W3 vs W0 를 10규칙 전량 비교")
    ap.add_argument("--ada", action="store_true", help="retrieval_ada 를 쓴다")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    pairs = [tuple(p.split(":")) for p in a.pairs]
    if a.auto_ckpt:
        for r in RULES:
            pairs += [(f"m5i2__{r}", f"m5__{r}"), (f"m5w3__{r}", f"m5__{r}")]

    sub = "retrieval_ada" if a.ada else "retrieval"
    print(f"쌍체 부트스트랩 {a.n_boot:,}회 · seed {a.seed} · 출처 {sub}/")
    print("🔒 회귀선 동결값 사용 (재적합 없음): DCG@20 = 0.0221×평균단어 + 46.03\n")
    hdr = f"{'비교':52s} {'Δ DCG@20':>9s} {'95% CI':>18s} {'Δ잔차':>7s}  판정"
    print(hdr); print("─" * len(hdr))

    out, n_sig = [], 0
    for x, y in pairs:
        lx, ly = load(x, OLD, sub), load(y, OLD, sub)
        if lx is None or ly is None:
            print(f"{x+' − '+y:52s} {'—':>9s} {'(셀 없음)':>18s}")
            continue
        (ax, wx), (by, wy) = lx, ly
        m, lo, hi, n = paired_ci(ax, by, a.n_boot, a.seed)
        # 잔차 차이 = 청크 크기로 설명되는 몫을 뺀 나머지
        rx = np.mean(list(ax.values())) * 100 - (SLOPE * wx + INTERCEPT)
        ry = np.mean(list(by.values())) * 100 - (SLOPE * wy + INTERCEPT)
        sig = lo > 0 or hi < 0
        n_sig += sig
        verdict = ("유의 ↑" if lo > 0 else "유의 ↓") if sig else "구분 안 됨"
        print(f"{x+' − '+y:52s} {m:>+9.2f} [{lo:>+7.2f},{hi:>+7.2f}] {rx-ry:>+7.2f}  {verdict}")
        out.append({"a": x, "b": y, "delta": round(m, 3), "ci95": [round(lo, 3), round(hi, 3)],
                    "d_residual": round(rx - ry, 3), "n_questions": n, "significant": bool(sig)})

    print(f"\nCI 가 0을 배제한 쌍: {n_sig}/{len(out)}")
    if a.out:
        Path(a.out).write_text(json.dumps(
            {"n_boot": a.n_boot, "seed": a.seed, "source": sub,
             "regression": {"slope": SLOPE, "intercept": INTERCEPT, "refit": False},
             "pairs": out}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()

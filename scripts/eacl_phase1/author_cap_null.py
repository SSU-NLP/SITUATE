#!/usr/bin/env python3
"""작가당 상한의 구성 왜곡을 **크기 맞춘 무작위 표본**과 비교한다.

🔴 왜 이 형태인가 (codex 2026-08-02):
   상한 코퍼스는 전체 풀의 **결정적 부분집합**이고 편은 작가 내에서 종속이므로, 일반적인
   독립 2표본 검정(카이제곱·KS의 p값)은 **부적절**하다. "전체와 같다"는 증명도 불가능하다
   (표본이 커지면 어떤 차이도 유의해진다).

   대신 검정 가능한 질문은 이것이다:
     **"상한이 만든 구성 변화가, 같은 크기의 무작위 표본에서 저절로 생기는 변화보다 큰가?"**

   귀무분포 = 전체 풀에서 **작가 단위로** 무작위 추출해 상한 코퍼스와 **같은 편 수**를 만든 것.
   작가 단위로 뽑는 이유 = 편이 작가 내에서 종속이므로 편 단위 추출은 분산을 과소평가한다.

   결과 해석:
     관측 변화 ≤ 귀무 95퍼센타일  ⇒ 상한의 왜곡이 **표집 잡음 범위 안**. 실질적 왜곡 없음
     관측 변화 >  귀무 95퍼센타일  ⇒ 상한 자체가 구성을 바꿈. 그 초과분이 실제 대가
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts/eacl_phase0"), str(REPO / "scripts/eacl_phase1")]
from author_cap_sensitivity import cap  # noqa: E402

EXP = REPO / "outputs/experiments/2026-08-02_author_cap"
DRAWS, SEED = 400, 20260802


def max_shift(idx: np.ndarray, code: np.ndarray, full_p: np.ndarray, K: int) -> float:
    """부분집합의 범주 비율이 전체 비율에서 최대 몇 %p 벗어나나."""
    c = np.bincount(code[idx], minlength=K) / len(idx)
    return float(np.max(np.abs(c - full_p)) * 100)


def main() -> None:
    pool = json.loads((EXP / "pool.json").read_text())
    cats = sorted({p["locc1"] for p in pool})
    cmap = {c: i for i, c in enumerate(cats)}
    code = np.array([cmap[p["locc1"]] for p in pool])
    K = len(cats)
    full_p = np.bincount(code, minlength=K) / len(pool)

    by_auth = defaultdict(list)
    for i, p in enumerate(pool):
        by_auth[p["author"]].append(i)
    auths = list(by_auth)
    sizes = np.array([len(by_auth[a]) for a in auths])
    rng = np.random.default_rng(SEED)

    rows = []
    print(f"전체 풀 {len(pool):,}편 · 작가 {len(auths):,}명 · 범주 {K}개 · 귀무 {DRAWS}회\n")
    print(f"{'N':>3s} {'편수':>7s} {'관측왜곡':>8s} {'귀무중앙':>8s} {'귀무95%':>8s} "
          f"{'초과분':>7s} {'백분위':>7s}  판정")
    for n in range(1, 49):
        sub = cap(pool, n)
        target = len(sub)
        sub_idx = np.array([i for i, p in enumerate(pool)
                            if False]) if False else None
        # 상한 부분집합의 인덱스를 다시 구한다(cap 은 dict 리스트를 돌려주므로 순서로 매칭)
        c2, keep = Counter(), []
        for i, p in enumerate(pool):
            if c2[p["author"]] < n:
                c2[p["author"]] += 1
                keep.append(i)
        obs = max_shift(np.array(keep), code, full_p, K)

        null = np.empty(DRAWS)
        for d in range(DRAWS):
            order = rng.permutation(len(auths))
            cum = np.cumsum(sizes[order])
            take = int(np.searchsorted(cum, target) + 1)
            idx = np.concatenate([by_auth[auths[j]] for j in order[:take]])
            null[d] = max_shift(idx, code, full_p, K)
        q50, q95 = float(np.median(null)), float(np.percentile(null, 95))
        pct = float((null < obs).mean() * 100)
        rows.append({"cap": n, "n_pieces": target, "obs": obs, "null_med": q50,
                     "null_p95": q95, "excess": obs - q95, "pctile": pct,
                     "within_noise": bool(obs <= q95)})
        if n <= 20 or n % 4 == 0:
            print(f"{n:3d} {target:7,} {obs:7.2f}p {q50:7.2f}p {q95:7.2f}p "
                  f"{obs-q95:+6.2f}p {pct:6.1f}%  " + ("표집잡음 내" if obs <= q95 else "초과"))

    (EXP / "null_test.json").write_text(json.dumps(
        {"draws": DRAWS, "n_categories": K, "pool": len(pool), "rows": rows},
        indent=1, ensure_ascii=False), encoding="utf-8")
    inside = [r["cap"] for r in rows if r["within_noise"]]
    print(f"\n🔒 표집 잡음 범위 안인 상한: N ≥ {min(inside)}" if inside else "\n🔒 전 구간 초과")
    print(f"-> {EXP/'null_test.json'}")


if __name__ == "__main__":
    main()

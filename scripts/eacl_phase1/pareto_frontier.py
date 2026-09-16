#!/usr/bin/env python3
"""저자 다양성 ↔ 구성 충실도 파레토 경계.

🔴 두 목표에 순서를 매기지 않는다. 어느 조합이 **달성 가능한가**를 그리고, 그 경계 위에서
   고른다. 목표:
     y  저자 다양성   = 유효 저자 수(1/Σpᵢ²).  높을수록 좋음
     x  구성 왜곡     = 전체 풀 대비 문헌분류 최대 비율차(%p). 낮을수록 좋음

두 방법군:
  A. 단순 작가상한        — 작가당 N편. 구성은 통제하지 않는다(현행)
  B. 층화 + 분류내 상한   — 분류별 할당을 전체 비율로 두고, 각 분류 **안에서** 작가당 N편
                            라운드로빈. 두 목표를 각각 통제한다.

코퍼스 크기는 목표가 아니므로(사용자 판단) 축에 넣지 않고 주석으로만 표시한다.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "outputs/experiments/2026-08-02_author_cap"
TARGET = 10000

pool = json.loads((EXP / "pool.json").read_text())
FULL = len(pool)
cats = Counter(p["locc1"] for p in pool)
full_p = {k: v / FULL for k, v in cats.items()}
BY: dict = defaultdict(lambda: defaultdict(list))
for p in pool:
    BY[p["locc1"]][p["author"]].append(p)


def eff(sub: list) -> float:
    c = Counter(p["author"] for p in sub)
    q = np.array(list(c.values())) / len(sub)
    return float(1 / np.sum(q ** 2))


def shift(sub: list) -> float:
    c = Counter(p["locc1"] for p in sub)
    return max(abs(c[k] / len(sub) - full_p[k]) for k in cats) * 100


def strat_cap(target: int, n: int) -> list:
    out = []
    for cat in cats:
        quota = int(round(target * full_p[cat]))
        by = BY[cat]
        auth = sorted(by, key=lambda a: -len(by[a]))
        picked, r = [], 0
        while len(picked) < quota and r < n:
            added = False
            for a in auth:
                if len(picked) >= quota:
                    break
                if r < len(by[a]):
                    picked.append(by[a][r])
                    added = True
            if not added:
                break
            r += 1
        out += picked
    return out


def plain_cap(n: int) -> list:
    c, out = Counter(), []
    for p in pool:
        if c[p["author"]] < n:
            c[p["author"]] += 1
            out.append(p)
    return out


def main() -> None:
    rows = []
    for n in range(1, 65):
        for name, sub in (("plain", plain_cap(n)), ("strat", strat_cap(TARGET, n))):
            if len(sub) < 500:
                continue
            rows.append({"method": name, "N": n, "pieces": len(sub),
                         "eff": eff(sub), "shift": shift(sub)})
    rows.append({"method": "nocap", "N": None, "pieces": FULL, "eff": eff(pool), "shift": 0.0})
    (EXP / "pareto.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

    print(f"{'방법':10s} {'N':>3s} {'편수':>7s} {'유효저자':>8s} {'왜곡':>7s}")
    for r in rows:
        if r["method"] == "strat" and r["N"] in (4, 8, 12, 16, 24, 48, 64):
            print(f"{'층화+상한':10s} {r['N']:3d} {r['pieces']:7,} {r['eff']:8.0f} {r['shift']:6.2f}p")
    for r in rows:
        if r["method"] == "plain" and r["N"] in (8, 12, 16, 24, 48):
            print(f"{'단순상한':10s} {r['N']:3d} {r['pieces']:7,} {r['eff']:8.0f} {r['shift']:6.2f}p")
    print(f"{'상한없음':10s} {'-':>3s} {FULL:7,} {eff(pool):8.0f} {0.0:6.2f}p")
    print(f"\n-> {EXP/'pareto.json'}")


if __name__ == "__main__":
    main()

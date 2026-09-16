#!/usr/bin/env python3
"""작가당 상한 N 의 다축 영향 — 최종 지표표.

모든 다양성 축을 **유효 개수**(inverse Simpson, 1/Σpᵢ²)로 통일해 서로 비교되게 한다.
전체 풀에서 구간/군집을 **고정**한 뒤 각 코퍼스가 거기에 얼마나 고르게 퍼졌는지 센다.

축 (codex 2026-08-02 권고 반영):
  ① 유효 저자 수        상한이 직접 조작하는 단위
  ② 저자 시대 다양성    출생연도 십분위. 다작 작가가 특정 시대에 몰릴 수 있다
  ③ 문학 전통 다양성    LoCC 상위분류. 충실도가 아니라 **다양성**으로 읽는다
  ④ 문체 다양성         기능어 88개 상대빈도 → 전체 풀 k-means 군집. 저자판별의 표준 신호
  ⑤ 서술 시점 균형      1인칭/3인칭 (대사 제외 서술부 대명사)
  ⑥ 5축 표지어 다양성   인물·장소·시간·인과·의도 표지어 밀도. **모델이 배우는 신호와 직결**
  ⑦ 유명도             다운로드 수. **다양성이 아니라 수준**을 본다(LLM 암기 우려 → 낮을수록 좋음)
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "outputs/experiments/2026-08-02_author_cap"
CAPS = (1, 2, 4, 8, 12, 16, 24, 32, 48)
K_STYLE, K_MARK = 20, 20
SEED = 20260802
MARK = ["표지_인물", "표지_장소", "표지_시간", "표지_인과", "표지_의도"]

pool = json.loads((EXP / "pool_style.json").read_text())
FULL = len(pool)


def inv_simpson(c: Counter) -> float:
    q = np.array(list(c.values()), dtype=float)
    q /= q.sum()
    return float(1 / np.sum(q ** 2))


def cap(n: int) -> list[int]:
    c, out = Counter(), []
    for i, p in enumerate(pool):
        if c[p["author"]] < n:
            c[p["author"]] += 1
            out.append(i)
    return out


# 전체 풀에서 구간·군집을 고정한다 ─────────────────────────────────────────
def decile_labels(vals: list) -> np.ndarray:
    v = np.array([x if x is not None else np.nan for x in vals], dtype=float)
    ok = ~np.isnan(v)
    e = np.quantile(v[ok], np.linspace(0, 1, 11))[1:-1]
    lab = np.full(len(v), -1)
    lab[ok] = np.digitize(v[ok], e)
    return lab


from sklearn.cluster import KMeans  # noqa: E402

FW = np.array([p["fw"] for p in pool])
STYLE_LAB = KMeans(K_STYLE, n_init=4, random_state=SEED).fit_predict(FW)
MK = np.array([[p[k] for k in MARK] for p in pool])
MK = (MK - MK.mean(0)) / (MK.std(0) + 1e-9)
MARK_LAB = KMeans(K_MARK, n_init=4, random_state=SEED).fit_predict(MK)
ERA_LAB = decile_labels([p["birth"] for p in pool])
LOCC = [p["locc1"] for p in pool]
REG = [p["register"] for p in pool]
DL = np.array([p["dl"] for p in pool], dtype=float)
DL_HI = float(np.percentile(DL, 95))


def row(idx: list[int]) -> dict:
    sub = [pool[i] for i in idx]
    era = ERA_LAB[idx]
    return {
        "pieces": len(idx),
        "eff_author": inv_simpson(Counter(p["author"] for p in sub)),
        "eff_era": inv_simpson(Counter(era[era >= 0].tolist())),
        "eff_locc": inv_simpson(Counter(LOCC[i] for i in idx)),
        "eff_style": inv_simpson(Counter(STYLE_LAB[idx].tolist())),
        "eff_reg": inv_simpson(Counter(REG[i] for i in idx)),
        "eff_mark": inv_simpson(Counter(MARK_LAB[idx].tolist())),
        "dl_med": float(np.median(DL[idx])),
        "dl_hi_pct": float((DL[idx] >= DL_HI).mean() * 100),
    }


def main() -> None:
    rows = [{"cap": n, **row(cap(n))} for n in CAPS]
    rows.append({"cap": None, **row(list(range(FULL)))})
    (EXP / "axes_table.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    hdr = (f"{'N':>4s} {'편수':>7s} {'저자':>7s} {'시대':>6s} {'전통':>6s} {'문체':>6s} "
           f"{'시점':>6s} {'5축':>6s} {'DL중앙':>7s} {'상위5%':>7s}")
    print(f"기준: 시대·문체·5축은 전체 풀에서 고정한 구간/군집 기준 (시대 최대 10, 문체·5축 최대 {K_STYLE})")
    print(f"유명도: 다운로드 상위 5% 경계 = {DL_HI:,.0f}회\n")
    print(hdr)
    for r in rows:
        n = "전체" if r["cap"] is None else str(r["cap"])
        print(f"{n:>4s} {r['pieces']:7,} {r['eff_author']:7.0f} {r['eff_era']:6.2f} "
              f"{r['eff_locc']:6.2f} {r['eff_style']:6.2f} {r['eff_reg']:6.3f} "
              f"{r['eff_mark']:6.2f} {r['dl_med']:7.0f} {r['dl_hi_pct']:6.1f}%")
    print(f"\n-> {EXP/'axes_table.json'}")


if __name__ == "__main__":
    main()

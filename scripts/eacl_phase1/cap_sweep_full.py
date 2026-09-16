#!/usr/bin/env python3
"""작가당 상한 N=1..max 전수 계산 + 축별 무릎점.

전제: 다운로드 상위 10% 문서 제외(LLM 사전학습 중복 위험). 그 뒤 남은 풀에서 N 을 1씩 올린다.
무릎 = 정규화한 (N, 지표) 곡선에서 양 끝점을 잇는 현으로부터 수직거리가 최대인 점(Kneedle).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "outputs/experiments/2026-08-02_author_cap"
DL_PCT = 90

pool_all = json.loads((EXP / "pool_style.json").read_text())
EMB_ALL = np.load(EXP / "piece_emb.npy")
import re
PLACE = re.compile(r"\b(various|anonymous|unknown|unidentified)\b", re.I)
EDIT = re.compile(r"\[(editor|compiler|contributor)\]", re.I)


def anthology(a: str) -> bool:
    """저자 미상 선집 — 저자 필드 첫 항목이 총칭이거나 편집자 크레딧."""
    first = a.split(";")[0]
    return bool(PLACE.search(first) or EDIT.search(first))


dl = np.array([x["dl"] for x in pool_all], float)
cut = np.percentile(dl, DL_PCT)
keep = [i for i, x in enumerate(pool_all)
        if x["dl"] < cut and not anthology(x["author"])]
pool = [pool_all[i] for i in keep]
EMB = EMB_ALL[keep]
MAXN = Counter(x["author"] for x in pool).most_common(1)[0][1]
print(f"다운로드 {DL_PCT}분위 {cut:,.0f}회 미만 + 선집 제외 → {len(pool):,}편 · 저자 "
      f"{len({x['author'] for x in pool}):,}명 · 최다 저자 {MAXN}편", flush=True)


def inv(c: Counter) -> float:
    q = np.array(list(c.values()), float)
    q /= q.sum()
    return float(1 / np.sum(q ** 2))


# 범주는 이 풀에서 한 번 정의해 고정
from sklearn.cluster import KMeans  # noqa: E402

birth = np.array([x["birth"] if x["birth"] else np.nan for x in pool], float)
ok = ~np.isnan(birth)
edges = np.quantile(birth[ok], np.linspace(0, 1, 11))[1:-1]
ERA = np.where(ok, np.digitize(birth, edges), -1)
LOCC = [x["locc1"] for x in pool]
STYLE = KMeans(20, n_init=4, random_state=20260802).fit_predict(
    np.array([x["fw"] for x in pool]))
EMBLAB = KMeans(20, n_init=4, random_state=20260802).fit_predict(EMB)
AUTH = [x["author"] for x in pool]

rows = []
for n in range(1, MAXN + 1):
    c, idx = Counter(), []
    for i, a in enumerate(AUTH):
        if c[a] < n:
            c[a] += 1
            idx.append(i)
    era = ERA[idx]
    rows.append({
        "cap": n, "pieces": len(idx),
        "author": inv(Counter(AUTH[i] for i in idx)),
        "era": inv(Counter(era[era >= 0].tolist())),
        "locc": inv(Counter(LOCC[i] for i in idx)),
        "style": inv(Counter(STYLE[idx].tolist())),
        "emb": inv(Counter(EMBLAB[idx].tolist())),
    })
    if n % 100 == 0:
        print(f"  N={n}/{MAXN}", flush=True)

N = np.array([r["cap"] for r in rows], float)


def knee(y: np.ndarray) -> int:
    xn = (N - N.min()) / (N.max() - N.min())
    yn = (y - y.min()) / (y.max() - y.min())
    p0, p1 = np.array([xn[0], yn[0]]), np.array([xn[-1], yn[-1]])
    v = (p1 - p0) / np.linalg.norm(p1 - p0)
    d = np.abs(v[0] * (yn - p0[1]) - v[1] * (xn - p0[0]))
    return int(N[int(np.argmax(d))])


SEL = [r for r in rows if r["cap"] <= 50]
NS = np.array([r["cap"] for r in SEL], float)


def knee50(y):
    xn = (NS - NS.min()) / (NS.max() - NS.min())
    yn = (y - y.min()) / (y.max() - y.min())
    p0, p1 = np.array([xn[0], yn[0]]), np.array([xn[-1], yn[-1]])
    v = (p1 - p0) / np.linalg.norm(p1 - p0)
    return int(NS[int(np.argmax(np.abs(v[0] * (yn - p0[1]) - v[1] * (xn - p0[0]))))])


KNEES = {k: knee50(np.array([r[k] for r in SEL])) for k in ("author", "era", "locc", "style", "emb")}
(EXP / "cap_sweep_full.json").write_text(json.dumps(
    {"dl_cut": float(cut), "pieces": len(pool), "max_n": MAXN,
     "knees": KNEES, "rows": rows}, indent=1), encoding="utf-8")
print("\n무릎점:", KNEES)
print(f"-> {EXP/'cap_sweep_full.json'}")

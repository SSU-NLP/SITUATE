#!/usr/bin/env python3
"""작가당 상한이 코퍼스 분포를 왜곡하는가 — 상한 없는 전체 풀 대비 통계 검정.

🔴 **판정 기준은 결과를 보기 전에 아래 PRE_DECLARED 로 고정한다.**

무엇을: `pg_cache` 의 전권을 상한 없이 분해·필터해 **전체 편 풀**을 만든 뒤,
상한 N 을 사후 적용한 부분집합이 전체 풀과 같은 분포인지 지표별로 잰다.
왜: 현행 상한 16 은 목표 편수를 맞추려 조정된 값이며(10→14→16), 그 과정에서 SF 비중이
0.222→0.180 으로 움직였다. 상한이 대표성을 해치지 않는 범위를 찾는다.

표본이 크면 p 값은 사소한 차이도 유의하게 만든다 ⇒ **효과크기를 주 기준**으로 둔다.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts/eacl_phase0"), str(REPO / "scripts/eacl_phase1")]
from build_narrative_pack import split_sentences  # noqa: E402  동결 분리기
from build_pg_shorts import split_volume, stage0_piece  # noqa: E402  동결 파이프라인
from fetch_repcorpus import density, register  # noqa: E402

CACHE = REPO / "data/eacl_phase1/pg_cache"
CAT = REPO / "data/eacl_phase1/pgcat.csv"
OUT = REPO / "outputs/experiments/2026-08-02_author_cap"
MIN_SENT, MAX_SENT, MIN_MED = 30, 900, 5.0

# 🔒 사전 선언 (결과 보기 전 고정)
PRE_DECLARED = {
    "지표": ["is_sf", "locc1", "issued_year", "n_sent", "dialogue_density", "register"],
    "효과크기 기준": {"범주형 Cramér V": 0.05, "연속형 KS D": 0.05},
    "판정": "여섯 지표 **전부** 기준 미만이면 통과. p 값은 보조로만 기록(표본이 커서 무의미).",
    "채택 규칙": "통과하는 상한 중 **가장 작은 값**을 채택한다 — 작가 편중 방어를 최대로 하면서 "
                 "전체 풀과 구별되지 않는 지점.",
    "상한 격자": [2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48],
}


def cramers_v(a: list, b: list) -> float:
    keys = sorted(set(a) | set(b))
    t = np.array([[a.count(k) for k in keys], [b.count(k) for k in keys]], dtype=float)
    t = t[:, t.sum(0) > 0]
    chi2 = stats.chi2_contingency(t).statistic
    n = t.sum()
    return float(np.sqrt(chi2 / (n * (min(t.shape) - 1))))


def build_full_pool() -> list[dict]:
    """상한·목표 없이 캐시 전권을 분해 → 필터 통과한 전체 편."""
    import csv
    meta = {int(r["Text#"]): r for r in csv.DictReader(CAT.open(encoding="utf-8"))}
    pool, seen = [], set()
    files = sorted(CACHE.glob("*.txt"))
    for i, f in enumerate(files):
        gid = int(f.stem)
        m = meta.get(gid)
        if m is None:
            continue
        pieces, reason = split_volume(f.read_text(encoding="utf-8", errors="ignore"))
        if reason == "chaptered_work":
            continue
        for pi, (title, body) in enumerate(pieces):
            key = (m.get("Authors", ""), title.upper())
            if title and key in seen:
                continue
            sents, _ = stage0_piece(split_sentences(body))
            if not (MIN_SENT <= len(sents) <= MAX_SENT):
                continue
            if statistics.median(len(x.split()) for x in sents) < MIN_MED:
                continue
            seen.add(key)
            subj = (m.get("Subjects") or "").lower()
            iss = (m.get("Issued") or "")[:4]
            loc = (m.get("LoCC") or "").strip()
            pool.append({"gid": gid, "author": m.get("Authors", ""),
                         "is_sf": "science fiction" in subj,
                         "locc1": loc.split(",")[0].strip()[:2] if loc else "NA",
                         "issued_year": int(iss) if iss.isdigit() else None,
                         "n_sent": len(sents),
                         "dialogue_density": round(density(sents), 4),
                         "register": register(sents)[0]})
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{len(files)}권 · 누적 {len(pool):,}편", flush=True)
    return pool


def apply_cap(pool: list[dict], n: int) -> list[dict]:
    c, out = Counter(), []
    for p in pool:                      # 빌더와 같은 순서(권 순)로 앞에서부터
        if c[p["author"]] >= n:
            continue
        c[p["author"]] += 1
        out.append(p)
    return out


def compare(sub: list[dict], full: list[dict]) -> dict:
    r = {}
    for k in ("is_sf", "locc1", "register"):
        r[k] = {"effect": cramers_v([str(x[k]) for x in sub], [str(x[k]) for x in full]),
                "kind": "V"}
    for k in ("issued_year", "n_sent", "dialogue_density"):
        a = [x[k] for x in sub if x[k] is not None]
        b = [x[k] for x in full if x[k] is not None]
        ks = stats.ks_2samp(a, b)
        r[k] = {"effect": float(ks.statistic), "p": float(ks.pvalue), "kind": "KS"}
    return r


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[1] 상한 없는 전체 편 풀 구축", flush=True)
    pool = build_full_pool()
    print(f"    전체 풀 {len(pool):,}편 · 작가 {len({p['author'] for p in pool}):,}명\n", flush=True)

    rows = []
    print(f"{'상한':>5s} {'편수':>8s} {'작가':>6s} "
          + " ".join(f"{k[:9]:>10s}" for k in PRE_DECLARED["지표"]) + "  판정")
    for n in PRE_DECLARED["상한 격자"]:
        sub = apply_cap(pool, n)
        cmp = compare(sub, pool)
        ok = all(cmp[k]["effect"] < (0.05) for k in PRE_DECLARED["지표"])
        rows.append({"cap": n, "n_pieces": len(sub),
                     "n_authors": len({p["author"] for p in sub}),
                     "metrics": cmp, "pass": ok})
        print(f"{n:5d} {len(sub):8,} {len({p['author'] for p in sub}):6,} "
              + " ".join(f"{cmp[k]['effect']:10.4f}" for k in PRE_DECLARED["지표"])
              + ("   ✅" if ok else "   ❌"), flush=True)

    passing = [r["cap"] for r in rows if r["pass"]]
    pick = min(passing) if passing else None
    print(f"\n🔒 통과 상한: {passing}")
    print(f"🔒 채택(최소 통과값): {pick}")
    (OUT / "results.json").write_text(json.dumps(
        {"pre_declared": PRE_DECLARED, "pool_size": len(pool), "rows": rows,
         "passing": passing, "picked": pick}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> {OUT/'results.json'}")


if __name__ == "__main__":
    main()

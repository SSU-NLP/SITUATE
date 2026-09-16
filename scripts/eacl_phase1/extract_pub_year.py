#!/usr/bin/env python3
"""권별 출판연도 추출 — 표제지·판권 영역에서.

Project Gutenberg 카탈로그에는 원 출판연도 필드가 없다(`Issued` 는 PG 등록일).
그러나 본문 표제지에는 저작권 표시·잡지 호수·의회 등록 문구 형태로 연도가 남아 있다:

    Copyright, 1915, by Doubleday, Page & Company
    Planet Stories November 1952.
    Entered, according to Act of Congress, in the year 1868, by
    1869.

⚠️ 이것은 **그 판본의 연도**이지 초판 연도가 아닐 수 있다(재판·선집 수록). 복수 연도가 나오면
   가장 이른 값을 취해 초출에 가깝게 한다. PG 등록 연도(1971 이후)는 범위에서 배제된다.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "data/eacl_phase1/pg_cache"
OUT = REPO / "outputs/experiments/2026-08-02_author_cap/pub_year.json"
HEAD_LINES = 120           # 표제지·판권 영역
YEAR = re.compile(r"\b(1[6-9]\d{2})\b")
LO, HI = 1600, 1970
CUE = re.compile(r"copyright|entered|according to act|published|printed|"
                 r"stories|magazine|weekly|monthly|quarterly|vol\.|no\.|issue", re.I)


def year_of(text: str) -> tuple[int | None, str]:
    lines = text.splitlines()
    for i, l in enumerate(lines):
        if "START OF THE PROJECT GUTENBERG" in l.upper():
            lines = lines[i + 1:]
            break
    head = lines[:HEAD_LINES]
    cued, bare = [], []
    for l in head:
        ys = [int(y) for y in YEAR.findall(l) if LO <= int(y) <= HI]
        if not ys:
            continue
        (cued if CUE.search(l) else bare).extend(ys)
    if cued:
        return min(cued), "cue"
    if bare:
        return min(bare), "bare"
    return None, "none"


def main() -> None:
    out = {}
    files = sorted(CACHE.glob("*.txt"), key=lambda f: int(f.stem))
    for i, f in enumerate(files):
        y, src = year_of(f.read_text(encoding="utf-8", errors="ignore"))
        out[f.stem] = {"year": y, "src": src}
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)
    OUT.write_text(json.dumps(out), encoding="utf-8")
    got = [v["year"] for v in out.values() if v["year"]]
    from collections import Counter
    print(f"권 {len(out):,} · 연도 추출 {len(got):,} ({len(got)/len(out)*100:.1f}%) · "
          f"출처 {dict(Counter(v['src'] for v in out.values()))}")
    import numpy as np
    print(f"연도 분포: 중앙 {np.median(got):.0f} · 5~95% {np.percentile(got,5):.0f}~{np.percentile(got,95):.0f}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

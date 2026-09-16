#!/usr/bin/env python3
"""작가당 상한 검정을 위해 **후보 전권**을 캐시에 채운다.

기존 캐시(3,214권)는 이전 빌드가 실제로 내려받은 권만 담고 있다. 그 빌드는 작가당 상한 16 에
걸린 권을 **다운로드 자체를 건너뛰었고**(권 단위 조기 스킵) 격리 권도 받지 않았다.
⇒ 캐시를 '상한 없는 전체'로 쓰면 순환이다. 빠진 권을 받아 진짜 전체 풀을 만든다.
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts/eacl_phase0"), str(REPO / "scripts/eacl_phase1")]
from build_pg_shorts import fetch, CACHE, CAT

tag = [r for r in csv.DictReader(CAT.open(encoding="utf-8"))
       if r.get("Language") == "en" and r.get("Type") == "Text"
       and "short stories" in (r.get("Subjects", "") or "").lower()]
have = {int(f.stem) for f in CACHE.glob("*.txt")}
todo = [int(r["Text#"]) for r in tag if int(r["Text#"]) not in have]
print(f"후보 {len(tag):,}권 · 캐시 {len(have):,}권 · 받을 것 {len(todo):,}권", flush=True)
ok = fail = 0
for i, gid in enumerate(todo):
    if fetch(gid, 0.5):
        ok += 1
    else:
        fail += 1
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(todo)} · 성공 {ok} 실패 {fail}", flush=True)
print(f"[완료] 성공 {ok} · 실패 {fail} · 캐시 총 {len(list(CACHE.glob('*.txt'))):,}권")

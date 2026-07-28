#!/usr/bin/env python3
"""causality cf-window 서술 측정 집계 (RESULT §4).

판정(P1·P2)은 `PRE_REGISTRATION.md` 기준으로 이미 확정됐다. 이 스크립트는 판정을 바꾸지 않는
서술 통계만 낸다: 발현율 · 관계 유형 분포 · 1단계 false 중 enabling 비율 · W 민감도 ·
경계횡단 링크밀도.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LINK = frozenset({"physical", "psychological", "motivational"})
RELATIONS = ("none", "physical", "psychological", "motivational", "enabling")


def read(p: Path) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def reset_at(rel: dict[str, str], w: int) -> int:
    cand = [d for d in range(1, w + 1) if str(d) in rel]
    return int(not any(rel[str(d)] in LINK for d in cand))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf", type=Path, required=True)
    ap.add_argument("--master-dir", type=Path, help="gold 있는 경우 (dev-3) 경계횡단 밀도용")
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    rows = read(a.cf)
    print(f"\n{'=' * 70}\n### {a.name or a.cf.stem}  (n={len(rows)} 전이)\n{'=' * 70}")

    # --- 발현율 (판정 창 W=3) + 문서별
    print("\n[발현율 · 판정 창 W=3]  P3 대역 [0.15, 0.40]")
    bydoc = collections.defaultdict(list)
    for r in rows:
        bydoc[r["doc_id"]].append(r)
    for d, rr in sorted(bydoc.items()):
        f = sum(r["causality_cf"] for r in rr) / len(rr)
        print(f"  {d:<24} n={len(rr):>4}  {f:.3f}")
    pooled = sum(r["causality_cf"] for r in rows) / len(rows)
    print(f"  {'POOLED':<24} n={len(rows):>4}  {pooled:.3f}   "
          f"{'✅ 대역 안' if 0.15 <= pooled <= 0.40 else '❌ 대역 밖'}")

    # --- W 민감도 (서술 전용 — 판정은 W=3)
    print("\n[W 민감도 · 서술 전용, 판정 불변]")
    for w in (1, 2, 3, 4, 5):
        f = sum(reset_at(r["relations"], w) for r in rows) / len(rows)
        print(f"  W={w}  발현율={f:.3f}{'   <- 사전등록 판정 창' if w == 3 else ''}")

    # --- 관계 유형 분포
    labs = collections.Counter(l for r in rows for l in r["relations"].values())
    tot = sum(labs.values())
    print(f"\n[관계 유형 분포 · 창 안 전체 쌍 {tot}]")
    for lab in RELATIONS:
        print(f"  {lab:<15} {labs[lab]:>6}  {labs[lab] / tot:.3f}")
    dep = tot - labs["none"]
    print(f"  1단계 false(반사실 필요성 인정) = {dep}/{tot} = {dep / tot:.3f}")
    if dep:
        print(f"    그중 enabling = {labs['enabling']}/{dep} = {labs['enabling'] / dep:.3f}"
              f"   <- 두 기준이 실제로 갈리는 비율")
    links = sum(labs[l] for l in LINK)
    if links:
        comp = {l: round(labs[l] / links, 3) for l in sorted(LINK)}
        print(f"  링크 내부 구성 {comp}"
              f"   {'⚠ 한 종류 ≥0.95 = 4종 스킴이 이진으로 퇴화' if max(comp.values()) >= 0.95 else ''}")

    # --- 경계횡단 링크밀도 (서술 통계, 기각조건 아님 — D-B9 게이트는 P2 로 조작화)
    if a.master_dir:
        gold = {}
        for m in sorted(a.master_dir.glob("*.jsonl")):
            for r in read(m):
                gold[(r["doc_id"], int(r["transition"]))] = int(r.get("boundary", 0))
        cross, within = [], []
        for r in rows:
            doc, i = r["doc_id"], int(r["transition"])
            for d_s, lab in r["relations"].items():
                d = int(d_s)
                # 창 [i-d, i] 이 gold 경계를 가로지르는가
                crosses = any(gold.get((doc, t)) for t in range(i - d + 1, i + 1))
                (cross if crosses else within).append(int(lab in LINK))
        if cross and within:
            print(f"\n[경계횡단 링크밀도 · 서술 통계, 기각조건 아님]")
            print(f"  경계 가로지르는 쌍 : {sum(cross)}/{len(cross)} = {sum(cross) / len(cross):.3f}")
            print(f"  경계 안 쌍         : {sum(within)}/{len(within)} = "
                  f"{sum(within) / len(within):.3f}")
            print("  (D-B9 지연halo 게이트는 이 밀도차가 아니라 P2 probe 판별타당도로 조작화됨)")


if __name__ == "__main__":
    main()

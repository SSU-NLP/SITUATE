#!/usr/bin/env python3
"""카나리아 sanity 체크 — 50편 라벨링 결과가 dev-3 대비 이상치인가.

전체 8,000편(≈20시간)에 착수하기 전 확인:
  ① 파싱 실패율 / no_valid 재시도
  ② 축별 발현율(reset rate) — dev-3 대비 이상치인가
  ③ Stage-1 목록 크기(cast/locations/goals) — dev-3 대비 이상치인가
  ④ carry 전 명시율 (space/intentionality 희소도)

**판정 아님 — sanity 확인용.** 기준은 "dev-3 범위에서 크게 벗어나는가"이며, 벗어나면 전체 착수 전에
원인을 본다(장르·문체 차이일 수도, 파이프라인 결함일 수도 있으므로 자동 통과/실패를 선언하지 않는다).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
import llm_s5_states as s5  # noqa: E402

AXES = ["protagonist_present", "space", "time", "causality", "intentionality"]
DEV3 = ("pieman", "tunnel", "monkey")


def dev3_ref() -> dict:
    """dev-3 (동결 라벨러 q36_35b v2) 참조값."""
    out = {"reset": {a: [] for a in AXES}, "lists": {"cast": [], "locations": [], "goals": []},
           "explicit": {"space": [], "intentionality": []}}
    for d in DEV3:
        base = REPO / f"outputs/eacl_phase0/{d}_michelmann_s5_q36_35b_v2"
        M = s5.read(base / "reset_matrix.jsonl")
        for a in AXES:
            out["reset"][a].append(sum(int(r[a]) for r in M) / len(M))
        L = json.loads((base / d / "story_lists.json").read_text())["lists"]
        for k in out["lists"]:
            out["lists"][k].append(len(L[k]))
        for ax, fld, null in (("space", "location", "unspecified"),
                              ("intentionality", "active_goal", "none")):
            st_ = s5.read(base / d / f"{ax}_states.jsonl")
            out["explicit"][ax].append(sum(1 for x in st_ if x[fld] != null) / len(st_))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s5-dir", type=Path, required=True)
    ap.add_argument("--log", type=Path, default=None)
    a = ap.parse_args()

    ref = dev3_ref()
    docs = sorted(p.name for p in a.s5_dir.iterdir() if p.is_dir())
    print(f"=== 카나리아: {len(docs)} 편 · {a.s5_dir} ===\n")

    # ① 파싱/재시도
    if a.log and a.log.exists():
        txt = a.log.read_text(errors="ignore")
        print(f"① 로그 진단: no_valid={txt.count('no valid outputs')} · "
              f"Traceback={txt.count('Traceback')} · retry 흔적={txt.count('retry')}")

    # ② 축별 발현율
    M = s5.read(a.s5_dir / "reset_matrix.jsonl")
    print(f"\n② 축별 발현율 (n={len(M)} 전이)\n")
    print(f"{'축':<22}{'카나리아':>10}{'dev-3 범위':>20}{'':>4}")
    for ax in AXES:
        v = sum(int(r[ax]) for r in M) / len(M)
        lo, hi = min(ref["reset"][ax]), max(ref["reset"][ax])
        flag = "" if lo * 0.5 <= v <= hi * 1.5 else "  <-- 이상치"
        print(f"{ax:<22}{v:>10.3f}{f'{lo:.3f}~{hi:.3f}':>20}{flag}")

    # ③ Stage-1 목록 크기
    print(f"\n③ Stage-1 목록 크기 (편당)\n")
    sizes = {k: [] for k in ("cast", "locations", "goals")}
    nsent = []
    for d in docs:
        p = a.s5_dir / d / "story_lists.json"
        if not p.exists():
            continue
        L = json.loads(p.read_text())["lists"]
        for k in sizes:
            sizes[k].append(len(L[k]))
        sp = a.s5_dir / d / "space_states.jsonl"
        if sp.exists():
            nsent.append(len(s5.read(sp)))
    print(f"{'목록':<14}{'카나리아 중앙(범위)':>24}{'dev-3':>18}")
    for k in sizes:
        if not sizes[k]:
            continue
        cur = f"{st.median(sizes[k]):.0f} ({min(sizes[k])}~{max(sizes[k])})"
        rlo, rhi = min(ref["lists"][k]), max(ref["lists"][k])
        print(f"{k:<14}{cur:>24}{f'{rlo}~{rhi}':>18}")
    if nsent:
        print(f"\n  편 길이: 중앙 {st.median(nsent):.0f} 문장 (범위 {min(nsent)}~{max(nsent)})"
              f"   [dev-3: 96 / 408 / 504]")

    # ④ carry 전 명시율
    print(f"\n④ carry 전 명시율 (희소도)\n")
    print(f"{'축':<18}{'카나리아':>10}{'dev-3 범위':>20}")
    for ax, fld, null in (("space", "location", "unspecified"),
                          ("intentionality", "active_goal", "none")):
        tot = ex = 0
        for d in docs:
            p = a.s5_dir / d / f"{ax}_states.jsonl"
            if not p.exists():
                continue
            rows = s5.read(p)
            tot += len(rows)
            ex += sum(1 for x in rows if x[fld] != null)
        if tot:
            lo, hi = min(ref["explicit"][ax]), max(ref["explicit"][ax])
            print(f"{ax:<18}{ex/tot:>10.3f}{f'{lo:.3f}~{hi:.3f}':>20}")

    print("\n※ 판정 아님. 이상치가 보이면 전체 착수 전에 원인을 확인한다"
          "(장르·문체 차이 vs 파이프라인 결함).")


if __name__ == "__main__":
    main()

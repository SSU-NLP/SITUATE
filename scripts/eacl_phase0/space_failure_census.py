#!/usr/bin/env python3
"""dev-3 space 실패 전수 census — v2 vs 일반-A(Agen). 샘플 아니라 전량.

3편(pieman/tunnel/monkey) 1008문장 전체에서 space 실패 후보를 전량 열거:
  OVER  = v2가 위치 부여했는데 문장에 무대 근거 없음(과잉귀속) → H 대상.
  UNDER = v2가 unspecified인데 문장에 무대 단서 있음(누락) → A(목록) or Stage-2.
각 케이스: v2라벨·Agen라벨·목록멤버십. 일반-A가 고쳤나 전수 판정 + 근본원인 분류.
산출: outputs/experiments/2026-07-24_qwen_human_verify/SPACE_FAILURE_CENSUS.md
"""
from __future__ import annotations
import json, sys, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "eacl_phase0"))
from findings_diagnostics import (CONVEYANCE, SETTING, SPATIAL_PREP, MOVEMENT,  # noqa: E402
                                  loc_tokens, has_place_cue, sequence, read as fread)

DOCS = ["pieman", "tunnel", "monkey"]
OUT = ROOT / "outputs/experiments/2026-07-24_qwen_human_verify"
V2 = "q36_35b_v2"
AGEN = "fix_space_Agen"


def sents(doc):
    return sequence(fread(ROOT / f"outputs/eacl_phase0/{doc}_michelmann_master.jsonl"))


def sp(suf, doc):
    return {int(x["item_id"]): x["location"]
            for x in fread(ROOT / f"outputs/eacl_phase0/{doc}_michelmann_s5_{suf}/{doc}/space_states.jsonl")}


def lst(suf, doc):
    return json.load(open(ROOT / f"outputs/eacl_phase0/{doc}_michelmann_s5_{suf}/{doc}/story_lists.json"))["lists"]["locations"]


def has_stage_cue(s):  # list-독립 무대 단서 (under 탐지용)
    return bool(SETTING.search(s) or CONVEYANCE.search(s) or MOVEMENT.search(s) or SPATIAL_PREP.search(s))


def build():
    L = ["# dev-3 space 실패 전수 census (v2 vs 일반-A)", "",
         "> 3편 1008문장 전량. OVER=위치부여+무대근거없음(H대상) · UNDER=unspecified+무대단서(A/Stage-2).",
         "> 각 케이스 v2→Agen + 목록멤버십. 일반-A가 고쳤나 전수.", ""]
    tot = {"over": 0, "under": 0}
    fix = {"over_fixed": 0, "under_fixed": 0,
           "under_by_list": 0, "under_by_stage2": 0,   # under 고침: 목록추가 vs 목록엔이미있는데 Agen이 이제 고름
           "over_by_list": 0, "over_by_stage2": 0}      # over 고침: 목록제외 vs 목록엔있는데 Agen이 안고름
    cases = {"over": [], "under": []}
    for doc in DOCS:
        ss = sents(doc); v2 = sp(V2, doc); ag = sp(AGEN, doc)
        v2L = set(lst(V2, doc)); agL = set(lst(AGEN, doc))
        v2tok = loc_tokens(v2L)
        for i in range(len(ss)):
            if i not in v2 or i >= len(ss): continue
            sent = ss[i]; v2loc = v2[i]; agloc = ag.get(i)
            # OVER: v2 위치부여 + 문장에 무대 근거(cue) 전혀 없음
            if v2loc != "unspecified" and not has_place_cue(sent, v2tok):
                tot["over"] += 1
                fixed = (agloc == "unspecified")
                if fixed:
                    fix["over_fixed"] += 1
                    # v2가 고른 무대가 Agen 목록에서 빠졌나(목록제외) vs 목록엔 있는데 Agen이 안고름(Stage-2)
                    if v2loc not in agL: fix["over_by_list"] += 1
                    else: fix["over_by_stage2"] += 1
                cases["over"].append((doc, i, sent[:95], v2loc, agloc, v2loc in agL, fixed))
            # UNDER: v2 unspecified + 문장에 무대 단서
            elif v2loc == "unspecified" and has_stage_cue(sent):
                tot["under"] += 1
                fixed = (agloc is not None and agloc != "unspecified")
                if fixed:
                    fix["under_fixed"] += 1
                    if agloc not in v2L: fix["under_by_list"] += 1   # Agen이 새 무대를 목록에 넣어 고름
                    else: fix["under_by_stage2"] += 1                # v2 목록에도 있었는데 Agen이 이제 고름
                cases["under"].append((doc, i, sent[:95], agloc, fixed))

    L += ["## 요약 (전수)", "",
          f"- **OVER(과잉귀속) 후보 총 {tot['over']}건** → 일반-A가 고침 **{fix['over_fixed']}건** "
          f"(목록제외 {fix['over_by_list']} / Stage-2 {fix['over_by_stage2']}). 미해결 {tot['over']-fix['over_fixed']}건=H 필요.",
          f"- **UNDER(누락) 후보 총 {tot['under']}건** → 일반-A가 고침 **{fix['under_fixed']}건** "
          f"(목록추가=A효과 {fix['under_by_list']} / 목록엔있었음=Stage-2 {fix['under_by_stage2']}). 미해결 {tot['under']-fix['under_fixed']}건.", "",
          "## 판정 근거", "",
          f"- **일반-A(목록)가 실제로 고친 건**: under 중 목록추가 {fix['under_by_list']}건 + over 중 목록제외 {fix['over_by_list']}건.",
          f"- **목록으론 못 고치는 것(Stage-2/H)**: over 미해결 {tot['over']-fix['over_fixed']}건 + under 미해결 {tot['under']-fix['under_fixed']}건.", ""]

    # 전량 나열 (샘플 아님)
    L += ["## OVER 전량 (v2위치→Agen, agL=Agen목록有)", ""]
    for doc, i, s, v2loc, agloc, inagL, fixed in cases["over"]:
        m = "✅고침" if fixed else "❌미해결(H)"
        L.append(f"- `{doc}#{i}` {m}: v2=`{v2loc}`→ag=`{agloc}` (agL有={inagL})  “{s}”")
    L += ["", "## UNDER 전량 (v2=unspec, 무대단서有 → Agen)", ""]
    for doc, i, s, agloc, fixed in cases["under"]:
        m = "✅고침" if fixed else "❌미해결"
        L.append(f"- `{doc}#{i}` {m}: ag=`{agloc}`  “{s}”")
    return "\n".join(L) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rep = build()
    (OUT / "SPACE_FAILURE_CENSUS.md").write_text(rep, encoding="utf-8")
    # 요약만 콘솔
    print("\n".join(rep.split("\n")[:24]))
    print(f"\n[written] {OUT/'SPACE_FAILURE_CENSUS.md'}")


if __name__ == "__main__":
    main()

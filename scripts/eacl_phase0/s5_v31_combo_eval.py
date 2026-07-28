#!/usr/bin/env python3
"""v3.1 = H+E 조합 평가 (dev-3, 동결 라벨러 q36_35b no-think, API 0 재조합).

전제(검증됨): `space_H` 는 `build_space_field` 만, `time_E/conseq/first` 는 `build_time_field` 만
바꾸고 **Stage-1 을 건드리지 않는다**(Stage-1 을 바꾸는 fix = space_A / space_Agen / intent_2).
축은 독립 호출이므로 **H+E 조합 = fix_space_H 의 space + fix_time 의 time + v2 의 나머지 3축**.
따라서 새 추출 없이 캐시 재조합으로 계산된다(COMPONENT_ABLATION.md 와 동일 기법).

판정은 **봉인된 기존 규칙 그대로** (새 기준 만들지 않음):
  · H 타깃 = no-cue 문장 non-unspecified 율 개선 (단독 5.9% → 3.6%)
  · E 타깃 = 문중 marker 율 개선 (단독 41.2% → 22.5%, repcorpus 12편 — dev-3 는 표본부족)
  · 안전장치 = 평균 ΔScore ≥ −0.09
  → 셋 다 통과해야 채택. 하나라도 미달이면 v2 유지.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_s5_states as s5  # noqa: E402
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402

DEV3 = ("pieman", "tunnel", "monkey")
FROZEN = ["protagonist_present", "space", "time", "causality", "intentionality"]
T_STAR = 3
# H 타깃: 장소 단서가 없는 문장 (findings_diagnostics 와 동일한 regex 스크리닝)
PLACE_CUE = re.compile(
    r"\b(in|at|on|into|onto|inside|outside|toward|towards|through|across|down|up|from|"
    r"beside|behind|near|under|over|within|aboard|upon)\b", re.I)


def states(doc, tag, axis):
    p = REPO / f"outputs/eacl_phase0/{doc}_michelmann_s5_{tag}/{doc}/{axis}_states.jsonl"
    return {int(x["item_id"]): x for x in s5.read(p)}


def matrix(doc, space_tag, time_tag):
    """축별 출처를 섞어 reset_matrix 를 재조립. 나머지 3축은 항상 v2."""
    v2 = f"q36_35b_v2"
    st = {
        "protagonist": states(doc, v2, "protagonist"),
        "causality": states(doc, v2, "causality"),
        "intentionality": s5.carry_forward(states(doc, v2, "intentionality"), "active_goal", "none"),
        "space": s5.carry_forward(states(doc, space_tag, "space"), "location", "unspecified"),
        "time": states(doc, time_tag, "time"),
    }
    master = s5.read(REPO / f"outputs/eacl_phase0/{doc}_michelmann_master.jsonl")
    gold = {int(r["transition"]): int(r["boundary"]) for r in master}
    rows = []
    for i in sorted(gold):
        if any(i not in st[a] or i - 1 not in st[a] for a in st):
            continue
        rows.append({
            "transition": i, "boundary": gold[i],
            "protagonist_present": s5.reset_protagonist_present(st["protagonist"][i - 1], st["protagonist"][i]),
            "space": s5.reset_space(st["space"][i - 1], st["space"][i]),
            "time": s5.reset_time(st["time"][i - 1], st["time"][i]),
            "causality": s5.reset_causality(st["causality"][i - 1], st["causality"][i]),
            "intentionality": s5.reset_intentionality(st["intentionality"][i - 1], st["intentionality"][i]),
        })
    return rows


def score(rows):
    n = len(rows) + 1
    y = [0] * n
    for r in rows:
        if r["boundary"]:
            y[int(r["transition"]) - 1] = 1
    p = [0] * n
    for r in rows:
        if sum(int(r[a]) for a in FROZEN) >= T_STAR:
            p[int(r["transition"]) - 1] = 1
    return score_dialogues([y], [p])["score"]


def h_target(doc, tag):
    """no-cue 문장의 non-unspecified 율."""
    master = s5.read(REPO / f"outputs/eacl_phase0/{doc}_michelmann_master.jsonl")
    sents = s5.sequence(master)
    sp = states(doc, tag, "space")
    nu = tot = 0
    for i, s in enumerate(sents):
        if i not in sp or PLACE_CUE.search(s):
            continue
        tot += 1
        nu += int(sp[i]["location"] != "unspecified")
    return nu, tot


def main():
    print("=" * 76)
    print("v3.1 = H+E 조합 평가 — dev-3, 동결 q36_35b no-think, API 0 (캐시 재조합)")
    print("=" * 76)

    print("\n## 안전장치: 평균 ΔScore ≥ −0.09  (t*=3, 나머지 3축 v2 고정)\n")
    print(f"{'doc':<9}{'v2':>9}{'H단독':>9}{'E단독':>9}{'H+E':>9}{'Δ(H+E−v2)':>12}")
    tot = {}
    for d in DEV3:
        s_v2 = score(matrix(d, "q36_35b_v2", "q36_35b_v2"))
        s_h = score(matrix(d, "fix_space_H", "q36_35b_v2"))
        s_e = score(matrix(d, "q36_35b_v2", "fix_time"))
        s_he = score(matrix(d, "fix_space_H", "fix_time"))
        tot[d] = (s_v2, s_h, s_e, s_he)
        print(f"{d:<9}{s_v2:>9.4f}{s_h:>9.4f}{s_e:>9.4f}{s_he:>9.4f}{s_he - s_v2:>+12.4f}")
    mv2 = sum(v[0] for v in tot.values()) / 3
    mhe = sum(v[3] for v in tot.values()) / 3
    d_he = mhe - mv2
    print(f"{'평균':<9}{mv2:>9.4f}{sum(v[1] for v in tot.values())/3:>9.4f}"
          f"{sum(v[2] for v in tot.values())/3:>9.4f}{mhe:>9.4f}{d_he:>+12.4f}")
    ok_score = d_he >= -0.09
    print(f"\n  => 평균 ΔScore = {d_he:+.4f}   [안전장치 ≥ −0.09] : {'✅ 통과' if ok_score else '❌ 미달'}")

    print("\n## H 타깃: no-cue 문장 non-unspecified 율 (↓ 의도, 단독 5.9% → 3.6%)\n")
    print(f"{'doc':<9}{'v2':>14}{'H+E':>14}")
    nv = tv = nh = th = 0
    for d in DEV3:
        a, at = h_target(d, "q36_35b_v2")
        b, bt = h_target(d, "fix_space_H")   # 조합의 space = H 단독과 동일 (Stage-1 불변)
        nv += a; tv += at; nh += b; th += bt
        print(f"{d:<9}{f'{a}/{at}={a/at:.3f}':>14}{f'{b}/{bt}={b/bt:.3f}':>14}")
    print(f"{'전체':<9}{f'{nv}/{tv}={nv/tv:.3f}':>14}{f'{nh}/{th}={nh/th:.3f}':>14}")
    ok_h = nh / th < nv / tv
    print(f"\n  => {nv/tv:.1%} → {nh/th:.1%}  [의도방향 개선] : {'✅ 통과' if ok_h else '❌ 미달'}")

    print("\n## E 타깃: 문중 marker 율 — repcorpus 12편 (dev-3 은 non-absent 13개로 표본부족)")
    print("   조합의 time 축 = E 단독과 동일(Stage-1 불변·축 독립) → 봉인된 단독 측정치를 그대로 승계")
    print("   v2 41.2%(14/34) → fix 22.5%(9/40)   [의도방향 개선] : ✅ 통과  (PERFIX_REPORT.md)")
    ok_e = True

    print("\n" + "=" * 76)
    verdict = ok_h and ok_e and ok_score
    print(f"판정 = {'✅ 채택 (v3.1 = H+E)' if verdict else '❌ v2 유지'}"
          f"   [H {'✅' if ok_h else '❌'} · E {'✅' if ok_e else '❌'} · Score {'✅' if ok_score else '❌'}]")
    print("=" * 76)


if __name__ == "__main__":
    main()

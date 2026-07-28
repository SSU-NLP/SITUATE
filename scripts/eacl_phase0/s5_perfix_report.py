#!/usr/bin/env python3
"""④-fine: 개별 fix 기여 — 타깃 지표 primary + Score 안전장치 (사전선언 채택규칙).

채택 = (타깃 지표 의도방향 개선) AND (Score v2 대비 −0.09 이내). Score 상승은 채택근거 아님(σ0.06<0.09).
base labeler = Qwen3.6-35B-A3B(no-think). v2 = q36_35b_v2. 각 fix = 영향축만 재추출.
스펙 plans/optimized-growing-crystal.md · 결정 context/06-decision-log.md(2026-07-24).
산출: outputs/experiments/2026-07-23_s5_redesign/PERFIX_REPORT.md
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "eacl_phase0"))
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
import llm_s5_states as S  # noqa: E402
from findings_diagnostics import CONVEYANCE, loc_tokens, has_place_cue, sequence  # noqa: E402
from time_marker_position_diag import opens_with_temporal, CONJ, LEAD  # noqa: E402
import re  # noqa: E402

DOCS = ["pieman", "tunnel", "monkey"]
FROZEN = ["protagonist", "space", "time", "causality", "intentionality"]
OUT = ROOT / "outputs/experiments/2026-07-23_s5_redesign"
V2 = "q36_35b_v2"   # base labeler v2


def rj(p) -> list[dict]:
    p = Path(p)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def st(suf, doc, axis) -> dict[int, dict]:
    return {int(s["item_id"]): s for s in rj(ROOT / f"outputs/eacl_phase0/{doc}_michelmann_s5_{suf}" / doc / f"{axis}_states.jsonl")}


def sents(doc) -> list[str]:
    return sequence(rj(ROOT / f"outputs/eacl_phase0/{doc}_michelmann_master.jsonl"))


# ---------- Score 안전장치 (v2 other-axes + fix axis 재조합) ----------
def reset_col(states: dict[int, dict], axis: str, n: int) -> dict[int, int]:
    s = states
    if axis in ("space", "intentionality"):
        fld, null = ("location", "unspecified") if axis == "space" else ("active_goal", "none")
        s = S.carry_forward({i: dict(s[i]) for i in s}, fld, null)
    out = {}
    for i in range(1, n):
        if i not in s or i - 1 not in s:
            out[i] = 0; continue
        if axis == "space": out[i] = S.reset_space(s[i - 1], s[i])
        elif axis == "intentionality": out[i] = S.reset_intentionality(s[i - 1], s[i])
        elif axis == "time": out[i] = S.reset_time(s[i - 1], s[i])
        elif axis == "causality": out[i] = S.reset_causality(s[i - 1], s[i])
        elif axis == "protagonist":
            out[i] = int(S.reset_protagonist_charset(s[i - 1], s[i]) or S.reset_perspective(s[i - 1], s[i]))
    return out


def gold_of(doc):
    rows = [r for r in rj(ROOT / f"outputs/eacl_phase0/{doc}_michelmann_s5_{V2}/reset_matrix.jsonl")
            if str(r["doc_id"]) == doc]
    return {int(r["transition"]): int(r["boundary"]) for r in rows}, len(rows) + 1


def score_with(fix_axis: str, fix_suf: str, doc: str, t=3):
    """v2 다른 축 + fix_axis(fix_suf) 재조합 deployable Score."""
    gold, n = gold_of(doc)
    cols = {}
    for ax in FROZEN:
        suf = fix_suf if ax == fix_axis else V2
        cols[ax] = reset_col(st(suf, doc, ax), ax, n)
    gy = [0] * n
    for tr, b in gold.items():
        if b: gy[tr - 1] = 1
    cnt = {tr: sum(cols[a].get(tr, 0) for a in cols) for tr in gold}
    bt = [tr for tr in gold if cnt[tr] >= t]
    pred = [0] * n
    for tr in bt: pred[tr - 1] = 1
    return score_dialogues([gy], [pred])["score"]


def score_v2(doc, t=3):
    return score_with("__none__", V2, doc, t)  # all v2


# ---------- 타깃 지표 ----------
def target_H(suf):  # dev-3 no-cue 문장 non-unspecified율 (+per-doc)
    nc_n = nc_nu = 0
    per = []
    for doc in DOCS:
        lists = json.loads((ROOT / f"outputs/eacl_phase0/{doc}_michelmann_s5_{suf}" / doc / "story_lists.json").read_text())["lists"]
        ltoks = loc_tokens(lists["locations"])
        s = st(suf, doc, "space"); ss = sents(doc)
        dn = dnu = 0
        for i in range(len(s)):
            if i >= len(ss) or i not in s: continue
            if not has_place_cue(ss[i], ltoks):
                dn += 1; dnu += int(s[i]["location"] != "unspecified")
        nc_n += dn; nc_nu += dnu
        per.append((doc, dnu, dn))
    return (nc_nu / nc_n if nc_n else None), nc_n, per


_CONSEQ_START = re.compile(r"^(so|and yet|therefore|but)\b", re.IGNORECASE)
_FIRST_START = re.compile(r"^first\b", re.IGNORECASE)


def _lead(s):
    x = LEAD.sub("", s.strip()); x = CONJ.sub("", x); return LEAD.sub("", x)


REP_DOCS = ["fam_london_fire", "fam_bierce_owlcreek", "fam_gilman_wallpaper", "fam_doyle_speckled",
            "fam_jewett_heron", "fam_twain_frog", "obs_37511", "obs_28922", "obs_74162",
            "obs_25744", "obs_74140", "obs_77176"]


def rep_sents(doc):
    return sequence(rj(ROOT / f"outputs/eacl_phase0/repcorpus_docmasters/{doc}.jsonl"))


def target_time(rep_suf):  # repcorpus per-marker 문중율 + So/AndYet→seq + First
    per = {"sequence": [0, 0], "jump": [0, 0], "simultaneity": [0, 0]}  # [mid, total]
    conseq_seq = first_total = first_caught = 0
    for doc in REP_DOCS:
        s = {int(x["item_id"]): x for x in rj(ROOT / f"outputs/eacl_phase0/{doc}_s5_{rep_suf}" / doc / "time_states.jsonl")}
        ss = rep_sents(doc)
        for i in s:
            if i >= len(ss): continue
            lab = s[i]["temporal_shift_marker"]; sent = ss[i]
            if lab in per:
                per[lab][1] += 1
                if not opens_with_temporal(sent): per[lab][0] += 1
            if lab == "sequence" and _CONSEQ_START.match(sent.strip()): conseq_seq += 1
            if _FIRST_START.match(_lead(sent)):
                first_total += 1
                if lab == "sequence": first_caught += 1
    nonabs_mid = sum(per[k][0] for k in per); nonabs_tot = sum(per[k][1] for k in per)
    return {"per": per, "mid": nonabs_mid, "tot": nonabs_tot,
            "mid_rate": (nonabs_mid / nonabs_tot if nonabs_tot else None),
            "jump_mid": (per["jump"][0] / per["jump"][1] if per["jump"][1] else None),
            "conseq_seq": conseq_seq, "first_caught": first_caught, "first_total": first_total}


def target_A(suf, doc="obs_77176"):  # 탈것/공중 목록누락 문장 수 (unspecified인데 vehicle term, 목록밖)
    d = ROOT / f"outputs/eacl_phase0/{doc}_michelmann_s5_{suf}"
    lists = json.loads((d / doc / "story_lists.json").read_text())["lists"]
    ltoks = loc_tokens(lists["locations"])
    ss = sequence(rj(ROOT / "outputs/eacl_phase0/obs_77176_only_master.jsonl"))
    s = st(suf, doc, "space")
    miss = 0; unspec = 0
    veh_in_list = any(re.search(r"cockpit|aircraft|airspace|sky|plane|air|deck|cabin", " ".join(lists["locations"]), re.I) for _ in [0])
    for i in s:
        if s[i]["location"] == "unspecified":
            unspec += 1
            if i < len(ss):
                m = [w.lower() for w in CONVEYANCE.findall(ss[i])]
                if [w for w in m if not any(w[:4] in t or t in w for t in ltoks)]: miss += 1
    return {"conveyance_miss": miss, "unspec": unspec, "n_loc": len(lists["locations"]),
            "veh_in_list": veh_in_list, "loc": lists["locations"]}


def target_intent2(suf):  # dev-3 상시목표(span≥0.6) 수 + 총 목표수
    perv = ngoals = 0
    per_doc = []
    for doc in DOCS:
        s = st(suf, doc, "intentionality"); n = len(s)
        by = {}
        for i in sorted(s):
            g = s[i]["active_goal"]
            if g != "none": by.setdefault(g, []).append(i)
        p = sum(1 for g, it in by.items() if (max(it) - min(it) + 1) / n >= 0.6)
        perv += p; ngoals += len(by)
        per_doc.append((doc, len(by), p))
    return {"pervasive": perv, "n_goals": ngoals, "per_doc": per_doc}


def build():
    L = ["# ④-fine: 개별 fix 기여 (타깃 지표 primary, Score 안전장치)", "",
         "> base = Qwen3.6-35B-A3B(no-think). 채택 = (타깃↑ 의도방향) ∧ (Score −0.09 이내). **Score 상승은 채택근거 아님**",
         "> (σ0.06<0.09). 각 fix = 영향축만 재추출, v2 다른 축 재조합. 단일 라벨러 caveat.", "",
         "## Score 안전장치 (dev-3 deployable, v2=q36_35b_v2)", "",
         "| fix | pieman Δ | tunnel Δ | monkey Δ | 평균 Δ | −0.09 이내? |", "|---|---|---|---|---|---|"]
    variants = {"space-H": ("space", "fix_space_H"), "time(E+conseq+first)": ("time", "fix_time"),
                "space-A": ("space", "fix_space_A"), "intent-②": ("intentionality", "fix_intent_2")}
    for name, (ax, suf) in variants.items():
        ds, ok = [], True
        cells = []
        for doc in DOCS:
            try:
                d = score_with(ax, suf, doc) - score_v2(doc)
                ds.append(d); cells.append(f"{d:+.3f}")
            except Exception:
                cells.append("—"); ok = False
        avg = sum(ds) / len(ds) if ds else None
        gate = "✅" if (avg is not None and avg >= -0.09) else ("❌" if avg is not None else "—")
        L.append(f"| {name} | " + " | ".join(cells) + f" | {avg:+.3f} | {gate} |" if avg is not None
                 else f"| {name} | " + " | ".join(cells) + " | — | — |")

    # 타깃 지표
    L += ["", "## 타깃 지표 (주 판정) — v2 → fix", ""]
    # H
    hv2, hn, hpv2 = target_H(V2); hf, _, hpf = target_H("fix_space_H")
    perline = " / ".join(f"{d} {a2}→{b2} / {a}/{b}" for (d, a, b), (_, a2, b2) in zip(hpv2, hpf))
    L += ["### space-H: no-cue 문장 non-unspecified율 (↓ 의도) — dev-3",
          f"- **전체: v2 {hv2:.1%}({sum(x[1] for x in hpv2)}/{hn}) → fix {hf:.1%}({sum(x[1] for x in hpf)}/{hn})**. 개선={'✅' if hf < hv2 else '❌'}",
          f"- per-doc (v2 nu → fix nu / n): " + " ; ".join(f"{d}: {a}→{a2}/{b}" for (d, a, b), (_, a2, _2) in zip(hpv2, hpf)), ""]
    # time (repcorpus, proper sample)
    tv2 = target_time("reptime_v2"); tf = target_time("reptime_fix")
    def permid(t, k):
        m, tt = t["per"][k]
        return f"{m}/{tt} ({m/tt:.0%})" if tt else "0/0"
    L += ["### time (E+conseq+first) — repcorpus 12편 (dev-3은 non-absent 13개로 표본부족→repcorpus로 제대로)",
          "| marker | v2 문중 | fix 문중 |", "|---|---|---|",
          f"| sequence | {permid(tv2,'sequence')} | {permid(tf,'sequence')} |",
          f"| **jump** | {permid(tv2,'jump')} | {permid(tf,'jump')} |",
          f"| simultaneity | {permid(tv2,'simultaneity')} | {permid(tf,'simultaneity')} |",
          "",
          f"- **전체 non-absent 문중율: v2 {tv2['mid_rate']:.1%}({tv2['mid']}/{tv2['tot']}) → fix {tf['mid_rate']:.1%}({tf['mid']}/{tf['tot']})**. 개선={'✅' if (tf['mid_rate'] or 1) < (tv2['mid_rate'] or 0) else '❌'}",
          f"- jump 문중(E 핵심타깃): v2 {tv2['jump_mid']:.0%} → fix {tf['jump_mid']:.0%}. 개선={'✅' if (tf['jump_mid'] or 1) < (tv2['jump_mid'] or 0) else '❌'}",
          f"- So/And yet→sequence(conseq): v2 {tv2['conseq_seq']} → fix {tf['conseq_seq']}. 개선={'✅' if tf['conseq_seq'] <= tv2['conseq_seq'] else '❌'}",
          f"- First→sequence 검출(first): v2 {tv2['first_caught']}/{tv2['first_total']} → fix {tf['first_caught']}/{tf['first_total']}.", ""]
    # A
    av2 = target_A("fix_obs_v2"); af = target_A("fix_space_A")
    L += ["### space-A: obs_77176 탈것/공중 목록누락 문장 수 (↓ 의도)",
          f"- v2 conveyance-miss **{av2['conveyance_miss']}** (loc {av2['n_loc']}개, vehicle 목록포함={av2['veh_in_list']}) →",
          f"  fix **{af['conveyance_miss']}** (loc {af['n_loc']}개, vehicle 목록포함={af['veh_in_list']}). 개선={'✅' if af['conveyance_miss'] < av2['conveyance_miss'] else '❌'}",
          f"- fix locations: {af['loc']}", ""]
    # intent-②
    iv2 = target_intent2(V2); iff = target_intent2("fix_intent_2")
    L += ["### intent-②: 상시목표(span≥0.6) 수 + 총 목표수 (↓ 의도)",
          f"- 상시목표: v2 **{iv2['pervasive']}** → fix **{iff['pervasive']}**. 총 목표: v2 {iv2['n_goals']} → fix {iff['n_goals']}. 개선={'✅' if iff['pervasive'] < iv2['pervasive'] else '❌'}",
          f"- 문서별(n목표/상시): v2 {iv2['per_doc']} / fix {iff['per_doc']}", ""]

    L += ["## 채택 판정 (타깃개선 ∧ Score −0.09이내)", "", "_(위 두 표 종합 — 서술)_", ""]
    return "\n".join(L) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rep = build()
    (OUT / "PERFIX_REPORT.md").write_text(rep, encoding="utf-8")
    print(rep)


if __name__ == "__main__":
    main()

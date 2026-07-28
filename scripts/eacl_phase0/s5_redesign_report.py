#!/usr/bin/env python3
"""④ 재설계 실험 리포트 — v2 baseline vs v3(프롬프트 재설계) × reasoning on/off.

Gate C(주판정): deployable Score(5축 t*=3) 의 ΔScore ≥ −0.03 (v2-ON 대비). 배포 후보 = v3-OFF
(프롬프트 재설계 + no-reasoning). v3-ON = 프롬프트효과 vs no-reasoning효과 귀속용 arm.

타깃 진단(fix가 실제로 작동했나, dev-3 캐시 gold-free):
  - B②: raw active_goal 목표 수 + 상시(span≥0.6)목표 수 (v3 Stage-1 제한 → 줄어야 함).
  - H : space no-cue 문장 non-unspecified율 (v3 강화 → 줄어야 함).
  - E : non-absent time 라벨의 문중(mid-sentence) 비율 (v3 대칭화 → jump 문중 줄어야 함).
  - A : 탈것/공중 목록누락 문장 수 (dev-3 엔 항공서사 없어 ~0 예상 — regression 확인용).

configs = {label: out_dir}. 각 dir 은 reset_matrix.jsonl + <doc>/<axis>_states.jsonl 보유.
산출: outputs/experiments/2026-07-23_s5_redesign/REPORT.md
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "eacl_phase0"))
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from findings_diagnostics import (CONVEYANCE, loc_tokens, has_place_cue, runs_of,  # noqa: E402
                                  sequence, read as fread)
from time_marker_position_diag import opens_with_temporal  # noqa: E402

MASTER_TMPL = "outputs/eacl_phase0/{doc}_michelmann_master.jsonl"
FROZEN = ["protagonist", "space", "time", "causality", "intentionality"]
T_STAR = 3
DOCS = ["pieman", "tunnel", "monkey"]
OUT = ROOT / "outputs/experiments/2026-07-23_s5_redesign"

CONFIGS = {  # label -> out_dir (relative to ROOT)
    "v2-ON (frozen baseline)": "outputs/eacl_phase0/{doc}_michelmann_s5_eismv2",
    "v3-ON (redesign, reasoning)": "outputs/eacl_phase0/{doc}_michelmann_s5_eismv3",
    "v3-OFF (redesign, no-reason)": "outputs/eacl_phase0/{doc}_michelmann_s5_eismv3_noreason",
}


def rj(p) -> list[dict]:
    p = Path(p)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def sentences(doc: str) -> list[str]:
    rows = rj(ROOT / MASTER_TMPL.format(doc=doc))
    return sequence(rows)


def deployable_score(rm_rows: list[dict], axes: list[str], t: int) -> tuple[float, int, int]:
    rows = sorted(rm_rows, key=lambda r: int(r["transition"]))
    n = len(rows) + 1
    gold = [0] * n
    for r in rows:
        if int(r["boundary"]):
            gold[int(r["transition"]) - 1] = 1
    cnt = {int(r["transition"]): sum(int(r[a]) for a in axes) for r in rows}
    bt = [i for i in cnt if cnt[i] >= t]
    pred = [0] * n
    for i in bt:
        pred[i - 1] = 1
    return score_dialogues([gold], [pred])["score"], sum(gold), len(bt)


def score_config(dir_tmpl: str, doc: str):
    rm = [r for r in rj(ROOT / dir_tmpl.format(doc=doc) / "reset_matrix.jsonl")
          if str(r["doc_id"]) == doc]
    if not rm:
        return None
    return deployable_score(rm, FROZEN, T_STAR)


# ---- target diagnostics (per config × doc, gold-free) ----
def diag_goals(dir_tmpl: str, doc: str):
    st = rj(ROOT / dir_tmpl.format(doc=doc) / doc / "intentionality_states.jsonl")
    if not st:
        return None
    n = len(st)
    raw = {int(s["item_id"]): s["active_goal"] for s in st}
    by_goal = {}
    for i in sorted(raw):
        g = raw[i]
        if g != "none":
            by_goal.setdefault(g, []).append(i)
    perv = sum(1 for g, items in by_goal.items()
               if (max(items) - min(items) + 1) / n >= 0.6)
    return {"n_goals": len(by_goal), "pervasive": perv,
            "goal_fire_rate": sum(1 for g in raw.values() if g != "none") / n}


def diag_H(dir_tmpl: str, doc: str, sents: list[str]):
    d = ROOT / dir_tmpl.format(doc=doc)
    lists = json.loads((d / doc / "story_lists.json").read_text())["lists"]
    ltoks = loc_tokens(lists["locations"])
    st = {int(s["item_id"]): s for s in rj(d / doc / "space_states.jsonl")}
    if not st:
        return None
    nocue_n = nocue_nu = 0
    for i in range(len(st)):
        if i >= len(sents) or i not in st:
            continue
        if not has_place_cue(sents[i], ltoks):
            nocue_n += 1
            nocue_nu += int(st[i]["location"] != "unspecified")
    return {"nocue_n": nocue_n, "nocue_nonunspec": (nocue_nu / nocue_n if nocue_n else None)}


def diag_E(dir_tmpl: str, doc: str, sents: list[str]):
    st = {int(s["item_id"]): s for s in rj(ROOT / dir_tmpl.format(doc=doc) / doc / "time_states.jsonl")}
    if not st:
        return None
    res = {"sequence": [0, 0], "jump": [0, 0], "simultaneity": [0, 0]}  # [mid, total]
    for i, s in st.items():
        lab = s["temporal_shift_marker"]
        if lab in res and i < len(sents):
            res[lab][1] += 1
            if not opens_with_temporal(sents[i]):
                res[lab][0] += 1
    return res


def diag_A(dir_tmpl: str, doc: str, sents: list[str]):
    d = ROOT / dir_tmpl.format(doc=doc)
    lists = json.loads((d / doc / "story_lists.json").read_text())["lists"]
    ltoks = loc_tokens(lists["locations"])
    st = rj(d / doc / "space_states.jsonl")
    miss = 0
    for s in st:
        i = int(s["item_id"])
        if s["location"] == "unspecified" and i < len(sents):
            m = [w.lower() for w in CONVEYANCE.findall(sents[i])]
            if [w for w in m if not any(w[:4] in t or t in w for t in ltoks)]:
                miss += 1
    return miss


def build() -> str:
    sents = {d: sentences(d) for d in DOCS}
    L = ["# ④ S5 재설계 실험 — v2 baseline vs v3 (프롬프트 재설계) × reasoning on/off", "",
         "> ②진단 통과분(A·H·B②·E) + B①(경보강)을 v3 프롬프트에 적용. Gate C(주판정)=deployable Score",
         "> ΔScore≥−0.03(v2-ON 대비). 배포후보=v3-OFF(재설계+no-reasoning, 4.5×빠름). v3-ON=귀속용 arm.",
         "> dev-3(pieman/tunnel/monkey, gold 有). reset/schema/protagonist/causality = v2와 동일(최소변경).", "",
         "## 실험 setup", "",
         "- **데이터:** dev-3 michelmann master (pieman 96·tunnel 504·monkey 408 문장; gold 18/18/30).",
         "- **라벨러:** deepseek-v4-pro. v2=frozen p0s5-eism-v2(reasoning ON). v3=p0s5-eism-v3(A/H space,",
         "  E time, B② goals, B① intent). arm: ON(reasoning) / OFF(no-reasoning).",
         "- **Score:** `hi_ontop.dts_scoring.score_dialogues`, deployable 5축 t\\*=3(=v2 동결 작동점).",
         "- **Gate C:** ΔScore(cfg − v2ON) ≥ −0.03 per doc. 타깃 진단=gold-free regex(②진단 동급 스크리닝).", ""]

    # ---- Gate C Score
    L += ["## Gate C — deployable Score (5축 t\\*=3)", "",
          "| doc | " + " | ".join(CONFIGS) + " |", "|---|" + "---|" * len(CONFIGS)]
    base = {}
    scores = {lab: {} for lab in CONFIGS}
    for doc in DOCS:
        cells = []
        for lab, tmpl in CONFIGS.items():
            sc = score_config(tmpl, doc)
            if sc is None:
                cells.append("_(pending)_")
                continue
            scores[lab][doc] = sc[0]
            if lab.startswith("v2"):
                base[doc] = sc[0]
            cells.append(f"{sc[0]:.3f} (g{sc[1]}/p{sc[2]})")
        L.append(f"| {doc} | " + " | ".join(cells) + " |")
    L += ["", "**ΔScore vs v2-ON + Gate C(≥−0.03):**", "",
          "| doc | v3-ON Δ | v3-OFF Δ | v3-ON gate | v3-OFF gate |", "|---|---|---|---|---|"]
    for doc in DOCS:
        if doc not in base:
            L.append(f"| {doc} | _(pending)_ | | | |")
            continue
        don = scores["v3-ON (redesign, reasoning)"].get(doc)
        doff = scores["v3-OFF (redesign, no-reason)"].get(doc)
        def cell(x):
            return "_pend_" if x is None else f"{x - base[doc]:+.3f}"
        def gate(x):
            return "_pend_" if x is None else ("✅" if x - base[doc] >= -0.03 else "❌")
        L.append(f"| {doc} | {cell(don)} | {cell(doff)} | {gate(don)} | {gate(doff)} |")
    L.append("")

    # ---- target diagnostics
    L += ["## 타깃 진단 (fix 실제 작동?)", "",
          "### B②: 목표 수 · 상시(span≥0.6)목표 수 (v3 Stage-1 제한 → 줄어야)", "",
          "| doc | " + " | ".join(f"{c} (n목표/상시/발화율)" for c in CONFIGS) + " |",
          "|---|" + "---|" * len(CONFIGS)]
    for doc in DOCS:
        cells = []
        for tmpl in CONFIGS.values():
            g = diag_goals(tmpl, doc)
            cells.append("_pend_" if g is None else f"{g['n_goals']}/{g['pervasive']}/{g['goal_fire_rate']:.0%}")
        L.append(f"| {doc} | " + " | ".join(cells) + " |")

    L += ["", "### H: space no-cue 문장 non-unspecified율 (v3 → 줄어야)", "",
          "| doc | " + " | ".join(CONFIGS) + " |", "|---|" + "---|" * len(CONFIGS)]
    for doc in DOCS:
        cells = []
        for tmpl in CONFIGS.values():
            h = diag_H(tmpl, doc, sents[doc])
            cells.append("_pend_" if h is None else
                         (f"{h['nocue_nonunspec']:.1%} (n{h['nocue_n']})" if h['nocue_nonunspec'] is not None else "—"))
        L.append(f"| {doc} | " + " | ".join(cells) + " |")

    L += ["", "### E: non-absent time 라벨 문중(mid) 비율 (v3 → jump 문중 줄어야)", "",
          "| doc | config | sequence mid | jump mid | simult mid |", "|---|---|---|---|---|"]
    for doc in DOCS:
        for lab, tmpl in CONFIGS.items():
            e = diag_E(tmpl, doc, sents[doc])
            if e is None:
                L.append(f"| {doc} | {lab} | _pend_ | | |")
                continue
            def frac(k):
                m, t = e[k]
                return f"{m}/{t} ({m/t:.0%})" if t else "0/0"
            L.append(f"| {doc} | {lab} | {frac('sequence')} | {frac('jump')} | {frac('simultaneity')} |")

    L += ["", "### A: 탈것/공중 목록누락 문장 수 (dev-3 항공서사 없음 → ~0, regression 확인)", "",
          "| doc | " + " | ".join(CONFIGS) + " |", "|---|" + "---|" * len(CONFIGS)]
    for doc in DOCS:
        cells = [str(diag_A(tmpl, doc, sents[doc])) if (ROOT / tmpl.format(doc=doc) / doc / "space_states.jsonl").exists()
                 else "_pend_" for tmpl in CONFIGS.values()]
        L.append(f"| {doc} | " + " | ".join(cells) + " |")

    L += ["", "## 해석 / 판정", "", "_(데이터 수집 후 서술)_", "",
          "## 한계", "",
          "- A: dev-3 에 항공/탈것 서사 없어 A fix 효과는 여기서 안 드러남(regression 확인만). A 실증은 repcorpus obs_77176.",
          "- v3-OFF 는 프롬프트+no-reasoning 동시 변경 → v3-ON arm 으로 귀속 분리. Gate C 판정은 배포후보(v3-OFF)로.",
          "- 타깃 진단 = gold-free regex 스크리닝(②진단 동급). 방향·규모용, 정밀도 아님.", ""]
    return "\n".join(L) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rep = build()
    (OUT / "REPORT.md").write_text(rep, encoding="utf-8")
    print(rep)
    print(f"[written] {OUT/'REPORT.md'}")


if __name__ == "__main__":
    main()

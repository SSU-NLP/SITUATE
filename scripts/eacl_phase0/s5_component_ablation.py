#!/usr/bin/env python3
"""④ 구성요소별 attribution — v3 통짜 대신 축별 fix 기여 분해 (API 0, 캐시 재조합).

배경: ④가 v2 vs v3(전 fix 묶음)만 봐서 어느 fix가 Score를 깎았는지 모름. 각 fix는 축-국소:
  H/A→space, E→time, B②/B①→intentionality (protagonist·causality 불변).
v2-ON·v3-ON 양쪽 축별 states 를 재조합해 부분 config 의 deployable Score 를 매긴다. 같은 reasoning-ON
설정이라 프롬프트 효과만 격리(단 v2/v3는 다른 run = 축간 nondeterminism 잔존, caveat).

config = 각 축 reset 을 v2 or v3 에서 선택. protagonist·causality 는 항상 v2(불변 축; 재추출 drift 제거).
산출: outputs/experiments/2026-07-23_s5_redesign/COMPONENT_ABLATION.md
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

DOCS = ["pieman", "tunnel", "monkey"]
T_STAR = 3
OUT = ROOT / "outputs/experiments/2026-07-23_s5_redesign"
V2 = "outputs/eacl_phase0/{doc}_michelmann_s5_eismv2"
V3 = "outputs/eacl_phase0/{doc}_michelmann_s5_eismv3"   # reasoning-ON redesign


def rj(p) -> list[dict]:
    p = Path(p)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def axis_states(dir_tmpl: str, doc: str, axis: str) -> dict[int, dict]:
    return {int(s["item_id"]): s for s in rj(ROOT / dir_tmpl.format(doc=doc) / doc / f"{axis}_states.jsonl")}


def reset_col(dir_tmpl: str, doc: str, axis: str, n: int) -> dict[int, int]:
    """축별 reset dict {transition: 0/1}, llm_s5_states 규칙 그대로(carry 포함)."""
    st = axis_states(dir_tmpl, doc, axis)
    out = {}
    if axis == "space":
        st = S.carry_forward({i: dict(st[i]) for i in st}, "location", "unspecified")
        for i in range(1, n):
            out[i] = S.reset_space(st[i - 1], st[i])
    elif axis == "intentionality":
        st = S.carry_forward({i: dict(st[i]) for i in st}, "active_goal", "none")
        for i in range(1, n):
            out[i] = S.reset_intentionality(st[i - 1], st[i])
    elif axis == "time":
        for i in range(1, n):
            out[i] = S.reset_time(st[i - 1], st[i])
    elif axis == "causality":
        for i in range(1, n):
            out[i] = S.reset_causality(st[i - 1], st[i])
    elif axis == "protagonist":
        for i in range(1, n):
            pc = S.reset_protagonist_charset(st[i - 1], st[i])
            pp = S.reset_perspective(st[i - 1], st[i])
            out[i] = int(pc or pp)
    return out


def gold_of(doc: str) -> tuple[dict[int, int], int]:
    rows = rj(ROOT / V2.format(doc=doc) / "reset_matrix.jsonl")
    rows = [r for r in rows if str(r["doc_id"]) == doc]
    g = {int(r["transition"]): int(r["boundary"]) for r in rows}
    return g, len(rows) + 1


def score(cols: dict[str, dict[int, int]], gold: dict[int, int], n: int, t: int):
    gy = [0] * n
    for tr, b in gold.items():
        if b:
            gy[tr - 1] = 1
    trans = sorted(gold)
    cnt = {tr: sum(cols[a].get(tr, 0) for a in cols) for tr in trans}
    bt = [tr for tr in trans if cnt[tr] >= t]
    pred = [0] * n
    for tr in bt:
        pred[tr - 1] = 1
    return score_dialogues([gy], [pred])["score"], sum(gy), len(bt)


AXES = ["protagonist", "space", "time", "causality", "intentionality"]
# config: 각 축을 어느 버전에서. protagonist·causality 는 항상 v2(불변축).
CONFIGS = {
    "v2 (baseline)":            {"space": "v2", "time": "v2", "intentionality": "v2"},
    "+H/A (space→v3)":          {"space": "v3", "time": "v2", "intentionality": "v2"},
    "+E (time→v3)":             {"space": "v2", "time": "v3", "intentionality": "v2"},
    "+B②/B① (intent→v3)":       {"space": "v2", "time": "v2", "intentionality": "v3"},
    "full v3 (3축 모두→v3)":     {"space": "v3", "time": "v3", "intentionality": "v3"},
    "v3 leave-out H (space→v2)": {"space": "v2", "time": "v3", "intentionality": "v3"},
    "v3 leave-out E (time→v2)":  {"space": "v3", "time": "v2", "intentionality": "v3"},
    "v3 leave-out B② (int→v2)":  {"space": "v3", "time": "v3", "intentionality": "v2"},
}


def build() -> str:
    L = ["# ④ 구성요소별 attribution (v3 축별 fix 기여, reasoning-ON, API 0)", "",
         "> v2-ON·v3-ON 축별 states 재조합. protagonist·causality 는 항상 v2(불변축). deployable Score(t\\*=3).",
         "> ⚠️ v2/v3 다른 run → 축간 nondeterminism 잔존(정밀 attribution 아닌 방향). H/A=space, E=time, B②/B①=intent.", ""]
    per = {}
    for doc in DOCS:
        gold, n = gold_of(doc)
        # 축별 reset 미리 계산 (v2, v3)
        rc = {(v, a): reset_col(tmpl, doc, a, n)
              for v, tmpl in [("v2", V2), ("v3", V3)] for a in ["space", "time", "intentionality"]}
        fixed = {a: reset_col(V2, doc, a, n) for a in ["protagonist", "causality"]}
        per[doc] = {}
        for name, sel in CONFIGS.items():
            cols = {"protagonist": fixed["protagonist"], "causality": fixed["causality"],
                    "space": rc[(sel["space"], "space")], "time": rc[(sel["time"], "time")],
                    "intentionality": rc[(sel["intentionality"], "intentionality")]}
            per[doc][name] = score(cols, gold, n, T_STAR)

    L += ["## deployable Score (t\\*=3) — 축별 fix on/off", "",
          "| config | " + " | ".join(DOCS) + " | 평균 |", "|---|" + "---|" * (len(DOCS) + 1)]
    base = {d: per[d]["v2 (baseline)"][0] for d in DOCS}
    for name in CONFIGS:
        cells = []
        deltas = []
        for d in DOCS:
            s, g, p = per[d][name]
            dlt = s - base[d]
            deltas.append(dlt)
            tag = "" if name.startswith("v2 ") else f" ({dlt:+.3f})"
            cells.append(f"{s:.3f}{tag} p{p}")
        avg = sum(deltas) / len(deltas)
        avgcell = f"{sum(per[d][name][0] for d in DOCS)/3:.3f}" + ("" if name.startswith("v2 ") else f" ({avg:+.3f})")
        L.append(f"| {name} | " + " | ".join(cells) + f" | {avgcell} |")

    L += ["", "## 읽는 법", "",
          "- **+X 단독행** = v2에 그 축 fix 하나만 → 그 fix의 순기여(Δ).",
          "- **leave-out 행** = full v3에서 그 축만 v2로 되돌림 → 그 fix를 뺀 효과(full 대비).",
          "- pieman 붕괴(예측수 p) 원인 축 = +단독행에서 Score 급락 or 예측수 급변하는 축.", ""]
    return "\n".join(L) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rep = build()
    (OUT / "COMPONENT_ABLATION.md").write_text(rep, encoding="utf-8")
    print(rep)
    print(f"[written] {OUT/'COMPONENT_ABLATION.md'}")


if __name__ == "__main__":
    main()

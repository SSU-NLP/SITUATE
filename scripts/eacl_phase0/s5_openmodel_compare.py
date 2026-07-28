#!/usr/bin/env python3
"""④ 3-labeler 종합 비교 — deepseek/GLM/Qwen × v2/v3, dev-3.

(1) Score 표: 각 (labeler,prompt) deployable Score + within-model v2→v3 Δ (프롬프트 효과 격리).
(2) 각 labeler 구성요소 ablation (v2↔v3 축 스왑): H(space)/E(time)/B②(intent) 기여 → B②가
    모든 라벨러에서 pieman 깨는지 확인. protagonist·causality = 그 라벨러 v2 고정(불변축).
전부 캐시 재조합, API 0. 산출: outputs/experiments/2026-07-23_s5_redesign/OPENMODEL_COMPARE.md
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

# labeler -> (v2 dir suffix, v3 dir suffix)
LABELERS = {
    "deepseek": ("eismv2", "eismv3"),
    "GLM": ("glm_v2", "glm_v3"),
    "Qwen32B": ("qwen32b_v2", "qwen32b_v3"),
    "Qwen3-Next-80B": ("qwennext_v2", "qwennext_v3"),
    "deepseek-chat": ("dschat_v2", "dschat_v3"),
}
CHANGED = ["space", "time", "intentionality"]   # v2↔v3 다른 축
FIXED = ["protagonist", "causality"]            # 불변축


def rj(p) -> list[dict]:
    p = Path(p)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def states(suf: str, doc: str, axis: str) -> dict[int, dict]:
    return {int(s["item_id"]): s for s in rj(ROOT / f"outputs/eacl_phase0/{doc}_michelmann_s5_{suf}" / doc / f"{axis}_states.jsonl")}


def has(suf: str, doc: str) -> bool:
    return (ROOT / f"outputs/eacl_phase0/{doc}_michelmann_s5_{suf}" / doc / "intentionality_states.jsonl").exists()


def gold_of(doc: str):
    rows = [r for r in rj(ROOT / f"outputs/eacl_phase0/{doc}_michelmann_s5_eismv2/reset_matrix.jsonl")
            if str(r["doc_id"]) == doc]
    return {int(r["transition"]): int(r["boundary"]) for r in rows}, len(rows) + 1


def reset_col(suf: str, doc: str, axis: str, n: int) -> dict[int, int]:
    st = states(suf, doc, axis)
    if not st:
        return {}
    out = {}
    if axis in ("space", "intentionality"):
        fld, null = ("location", "unspecified") if axis == "space" else ("active_goal", "none")
        st = S.carry_forward({i: dict(st[i]) for i in st}, fld, null)
    for i in range(1, n):
        if i not in st or (i - 1) not in st:   # 결측 item → reset 판정 불가, 0 처리
            out[i] = 0
            continue
        if axis == "space":
            out[i] = S.reset_space(st[i - 1], st[i])
        elif axis == "intentionality":
            out[i] = S.reset_intentionality(st[i - 1], st[i])
        elif axis == "time":
            out[i] = S.reset_time(st[i - 1], st[i])
        elif axis == "causality":
            out[i] = S.reset_causality(st[i - 1], st[i])
        elif axis == "protagonist":
            out[i] = int(S.reset_protagonist_charset(st[i - 1], st[i]) or S.reset_perspective(st[i - 1], st[i]))
    return out


def score(cols: dict[str, dict[int, int]], gold: dict[int, int], n: int, t=T_STAR):
    gy = [0] * n
    for tr, b in gold.items():
        if b: gy[tr - 1] = 1
    trans = sorted(gold)
    cnt = {tr: sum(cols[a].get(tr, 0) for a in cols) for tr in trans}
    bt = [tr for tr in trans if cnt[tr] >= t]
    pred = [0] * n
    for tr in bt: pred[tr - 1] = 1
    return score_dialogues([gy], [pred])["score"], len(bt), sum(gy)


def full_score(suf: str, doc: str):
    if not has(suf, doc):
        return None
    gold, n = gold_of(doc)
    cols = {a: reset_col(suf, doc, a, n) for a in FIXED + CHANGED}
    return score(cols, gold, n)


def build() -> str:
    L = ["# ④ 3-labeler 종합 비교 (deepseek/GLM/Qwen32B × v2/v3, dev-3)", "",
         "> deployable Score(t\\*=3) + 축별 ablation. 오픈모델은 no-thinking, deepseek v2/v3=reasoning-ON.",
         "> within-model v2→v3 = 프롬프트 순효과(라벨러 고정). ⚠️ 각 1 run(배치 비결정 잔존).", ""]

    # (1) Score 표
    L += ["## 1. deployable Score — labeler × prompt", "",
          "| labeler | pieman v2→v3 | tunnel v2→v3 | monkey v2→v3 | 평균 v2→v3 |",
          "|---|---|---|---|---|"]
    for lab, (s2, s3) in LABELERS.items():
        cells = []
        avg2 = avg3 = cnt = 0
        for d in DOCS:
            r2, r3 = full_score(s2, d), full_score(s3, d)
            if r2 and r3:
                cells.append(f"{r2[0]:.3f}→{r3[0]:.3f} ({r3[0]-r2[0]:+.3f})")
                avg2 += r2[0]; avg3 += r3[0]; cnt += 1
            else:
                cells.append("_pend_")
        avgc = f"{avg2/cnt:.3f}→{avg3/cnt:.3f} ({(avg3-avg2)/cnt:+.3f})" if cnt == 3 else "_pend_"
        L.append(f"| {lab} | " + " | ".join(cells) + f" | {avgc} |")

    # (2) 축별 ablation per labeler
    L += ["", "## 2. 구성요소 ablation — v2에 축 fix 하나씩 (Δ vs 그 labeler v2)", "",
          "protagonist·causality = 그 labeler v2 고정. H=space, E=time, B②=intent.", "",
          "| labeler | config | pieman | tunnel | monkey | 평균 Δ |",
          "|---|---|---|---|---|---|"]
    ABL = {
        "v2": {"space": "v2", "time": "v2", "intentionality": "v2"},
        "+H (space)": {"space": "v3", "time": "v2", "intentionality": "v2"},
        "+E (time)": {"space": "v2", "time": "v3", "intentionality": "v2"},
        "+B② (intent)": {"space": "v2", "time": "v2", "intentionality": "v3"},
        "full v3": {"space": "v3", "time": "v3", "intentionality": "v3"},
    }
    for lab, (s2, s3) in LABELERS.items():
        if not all(has(s2, d) and has(s3, d) for d in DOCS):
            L.append(f"| {lab} | _(pending)_ | | | | |")
            continue
        sufmap = {"v2": s2, "v3": s3}
        base = {}
        # precompute reset cols
        pre = {}
        for d in DOCS:
            _, n = gold_of(d)
            pre[d] = (gold_of(d)[0], n,
                      {("v2", a): reset_col(s2, d, a, n) for a in CHANGED},
                      {("v3", a): reset_col(s3, d, a, n) for a in CHANGED},
                      {a: reset_col(s2, d, a, n) for a in FIXED})
        for cfgname, sel in ABL.items():
            cells, deltas = [], []
            for d in DOCS:
                gold, n, v2c, v3c, fx = pre[d]
                cols = {**fx,
                        "space": (v3c if sel["space"] == "v3" else v2c)[(sel["space"], "space")],
                        "time": (v3c if sel["time"] == "v3" else v2c)[(sel["time"], "time")],
                        "intentionality": (v3c if sel["intentionality"] == "v3" else v2c)[(sel["intentionality"], "intentionality")]}
                s, p, g = score(cols, gold, n)
                if cfgname == "v2":
                    base[d] = s
                dlt = s - base[d]
                deltas.append(dlt)
                cells.append(f"{s:.3f}" + ("" if cfgname == "v2" else f"({dlt:+.3f})") + f" p{p}")
            avgd = sum(deltas) / 3
            L.append(f"| {lab} | {cfgname} | " + " | ".join(cells) + f" | {avgd:+.3f} |")

    L += ["", "## 판정 (데이터 후 서술)", "", "_(Qwen 완료 후 종합)_", ""]
    return "\n".join(L) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rep = build()
    (OUT / "OPENMODEL_COMPARE.md").write_text(rep, encoding="utf-8")
    print(rep)
    print(f"[written] {OUT/'OPENMODEL_COMPARE.md'}")


if __name__ == "__main__":
    main()

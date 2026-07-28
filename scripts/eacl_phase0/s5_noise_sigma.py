#!/usr/bin/env python3
"""Score noise σ 직접 측정 — 같은 config(q36_35b_v2 dev-3)를 5회 독립 재추출 → Score mean±σ.

각 run = fresh stage1 + stage2 (stage1 비결정 + vLLM 배치 비결정 포함 = 진짜 run-to-run 노이즈).
deployable Score(5축 t*=3). 이 σ가 fix Δ 판정 문턱의 근거. 산출: PERFIX_REPORT 옆 NOISE_SIGMA.md
"""
from __future__ import annotations
import json, statistics, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "eacl_phase0"))
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
import llm_s5_states as S  # noqa: E402

DOCS = ["pieman", "tunnel", "monkey"]
AX = ["protagonist", "space", "time", "causality", "intentionality"]
RUNS = [1, 2, 3, 4, 5]
OUT = ROOT / "outputs/experiments/2026-07-23_s5_redesign"


def rj(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()] if Path(p).exists() else []


def gold_of(doc):
    rows = [r for r in rj(ROOT / f"outputs/eacl_phase0/{doc}_michelmann_s5_q36_35b_v2/reset_matrix.jsonl")
            if str(r["doc_id"]) == doc]
    return {int(r["transition"]): int(r["boundary"]) for r in rows}, len(rows) + 1


def rc(suf, doc, ax, n):
    st = {int(x["item_id"]): x for x in rj(ROOT / f"outputs/eacl_phase0/{doc}_michelmann_s5_{suf}/{doc}/{ax}_states.jsonl")}
    if ax in ("space", "intentionality"):
        fld, nu = ("location", "unspecified") if ax == "space" else ("active_goal", "none")
        st = S.carry_forward({i: dict(st[i]) for i in st}, fld, nu)
    o = {}
    for i in range(1, n):
        if i not in st or i - 1 not in st: o[i] = 0; continue
        if ax == "space": o[i] = S.reset_space(st[i - 1], st[i])
        elif ax == "intentionality": o[i] = S.reset_intentionality(st[i - 1], st[i])
        elif ax == "time": o[i] = S.reset_time(st[i - 1], st[i])
        elif ax == "causality": o[i] = S.reset_causality(st[i - 1], st[i])
        elif ax == "protagonist": o[i] = int(S.reset_protagonist_charset(st[i - 1], st[i]) or S.reset_perspective(st[i - 1], st[i]))
    return o


def score(suf, doc, t=3):
    g, n = gold_of(doc)
    cols = {a: rc(suf, doc, a, n) for a in AX}
    gy = [0] * n
    for tr, b in g.items():
        if b: gy[tr - 1] = 1
    cnt = {tr: sum(cols[a].get(tr, 0) for a in cols) for tr in g}
    bt = [tr for tr in g if cnt[tr] >= t]
    pr = [0] * n
    for tr in bt: pr[tr - 1] = 1
    return score_dialogues([gy], [pr])["score"], len(bt)


def main():
    L = ["# Score noise σ 직접 측정 (q36_35b_v2 dev-3 × 5 독립 run)", "",
         "> 각 run = fresh stage1+stage2 (stage1 비결정 + vLLM 배치 비결정). deployable Score 5축 t\\*=3.", ""]
    L += ["| doc | " + " | ".join(f"r{k}" for k in RUNS) + " | mean | **σ** | range |", "|---|" + "---|" * (len(RUNS) + 3)]
    all_scores = {d: [] for d in DOCS}
    per_run_avg = {k: [] for k in RUNS}
    for doc in DOCS:
        cells = []
        for k in RUNS:
            s, p = score(f"q36_35b_v2_r{k}", doc)
            all_scores[doc].append(s); per_run_avg[k].append(s)
            cells.append(f"{s:.3f}")
        m = statistics.mean(all_scores[doc]); sd = statistics.pstdev(all_scores[doc])
        rng = max(all_scores[doc]) - min(all_scores[doc])
        L.append(f"| {doc} | " + " | ".join(cells) + f" | {m:.3f} | **{sd:.3f}** | {rng:.3f} |")
    # 전체(문서평균) run별
    run_avgs = [statistics.mean(per_run_avg[k]) for k in RUNS]
    m = statistics.mean(run_avgs); sd = statistics.pstdev(run_avgs)
    L.append(f"| **평균(3문서)** | " + " | ".join(f"{v:.3f}" for v in run_avgs) +
             f" | {m:.3f} | **{sd:.3f}** | {max(run_avgs)-min(run_avgs):.3f} |")
    # per-doc σ 평균
    doc_sds = [statistics.pstdev(all_scores[d]) for d in DOCS]
    L += ["", f"## 판정", "",
          f"- **문서별 Score σ: pieman {doc_sds[0]:.3f} / tunnel {doc_sds[1]:.3f} / monkey {doc_sds[2]:.3f}** (per-doc mean σ={statistics.mean(doc_sds):.3f}).",
          f"- **문서평균 Score σ = {sd:.3f}** (3문서 평균을 run별로 낸 것).",
          "- 이 σ가 fix Δ 판정의 실측 기준. 기존 '추정 σ≈0.06'과 대조.", ""]
    (OUT / "NOISE_SIGMA.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

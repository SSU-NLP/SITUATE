#!/usr/bin/env python3
"""잡음이 병목인가, 프롬프트 편향이 병목인가 — dev-3 반복추출 5개로 판별 (API 0).

물음
----
D-B19 이후 "다음 최우선 = 1만편 k 재설정" 이라 했으나, k 재설정은 **동결 프롬프트를 그대로 두고
여러 번 뽑아 합의**하는 것이므로 **분산만 줄이고 편향은 못 줄인다.** 프롬프트 자체가 사람의 사건
경계와 어긋나 있으면 더 정밀하게 틀릴 뿐이다.

dev-3 에는 동결 라벨러(q36_35b, v2)로 뽑은 **반복추출 r1~r5** 가 이미 있다. 그래서 직접 묻는다:

    **5개 draw 의 합의 라벨이 개별 draw 보다 사람 gold 를 더 잘 맞추는가?**

**사전선언 판정 (측정 전 확정, 2026-07-28):**
  · 합의 Score − 개별 draw 평균 Score = **ΔS**
  · **ΔS ≥ +0.06** → 잡음이 실제 비용. **k 재설정 정당** (0.06 = D-B11 후속 노이즈 floor).
  · **0 ≤ ΔS < 0.06** → 회색. 잡음은 있으나 해석 가능 폭 아래 → k 우선순위 강등.
  · **ΔS < 0** → 잡음이 병목이 **아니다**. **프롬프트 편향이 병목** → k 재설정 기각,
    프롬프트/조작화 재설계로 전환.

또한 **개별 draw Score 의 분산**과 **축별 draw 간 일치도**를 함께 보고해 어느 축이 불안정한지 짚는다.

⚠️ 합의는 **라벨 수준**(축별 5draw 다수결)에서 만든다. "개별 Score 의 평균"과는 다른 것이며,
잡음이 무작위라면 라벨 다수결이 개별보다 좋아야 한다 — 그게 이 검정의 핵심.
🔒 사람 gold 는 **채점에만** 쓴다. 합의 구성에는 gold 를 전혀 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
import llm_s5_states as s5  # noqa: E402
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402

DEV3 = ("pieman", "tunnel", "monkey")
FROZEN = ["protagonist_present", "space", "time", "causality", "intentionality"]
T_STAR = 3
BASE = REPO / "outputs/eacl_phase0"
FLOOR = 0.06   # D-B11 후속 노이즈 floor


def axis_matrix(doc: str, rep: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """한 draw 의 (축별 리셋 [T,5], gold [T], 전이 인덱스). s5_v31_combo_eval 과 동일 규약."""
    d = BASE / f"{doc}_michelmann_s5_q36_35b_v2_{rep}" / doc

    def st(axis):
        return {int(x["item_id"]): x for x in s5.read(d / f"{axis}_states.jsonl")}

    S = {"protagonist": st("protagonist"), "causality": st("causality"),
         "intentionality": s5.carry_forward(st("intentionality"), "active_goal", "none"),
         "space": s5.carry_forward(st("space"), "location", "unspecified"),
         "time": st("time")}
    master = s5.read(BASE / f"{doc}_michelmann_master.jsonl")
    gold = {int(r["transition"]): int(r["boundary"]) for r in master}
    rows, g, idx = [], [], []
    for i in sorted(gold):
        if any(i not in S[a] or i - 1 not in S[a] for a in S):
            continue
        rows.append([
            s5.reset_protagonist_present(S["protagonist"][i - 1], S["protagonist"][i]),
            s5.reset_space(S["space"][i - 1], S["space"][i]),
            s5.reset_time(S["time"][i - 1], S["time"][i]),
            s5.reset_causality(S["causality"][i - 1], S["causality"][i]),
            s5.reset_intentionality(S["intentionality"][i - 1], S["intentionality"][i]),
        ])
        g.append(gold[i])
        idx.append(i)
    return np.asarray(rows, dtype=np.int8), np.asarray(g, dtype=np.int8), idx


def to_labels(bnd: np.ndarray) -> list[int]:
    """끝-turn 규약: 전이 수 + 1 문장, 마지막 0."""
    return np.concatenate([bnd, [0]]).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", default="r1,r2,r3,r4,r5")
    args = ap.parse_args()
    reps = [r.strip() for r in args.reps.split(",")]

    per_doc: dict[str, dict] = {}
    for doc in DEV3:
        mats, golds = [], None
        for r in reps:
            M, g, _ = axis_matrix(doc, r)
            mats.append(M)
            golds = g
        n = min(len(m) for m in mats)
        mats = [m[:n] for m in mats]
        per_doc[doc] = {"mats": mats, "gold": golds[:n]}

    # ── 개별 draw Score ────────────────────────────────────────────────────
    single = {}
    for j, r in enumerate(reps):
        ys = [to_labels(per_doc[d]["gold"]) for d in DEV3]
        ps = [to_labels((per_doc[d]["mats"][j].sum(1) >= T_STAR).astype(np.int8)) for d in DEV3]
        single[r] = score_dialogues(ys, ps)["score"]
    s_vals = np.array(list(single.values()))

    # ── 합의(축별 다수결) Score ────────────────────────────────────────────
    ys = [to_labels(per_doc[d]["gold"]) for d in DEV3]
    cons_ax = []
    for d in DEV3:
        stack = np.stack(per_doc[d]["mats"])                    # [R, T, 5]
        maj = (stack.mean(0) >= 0.5).astype(np.int8)            # 축별 다수결
        cons_ax.append((maj.sum(1) >= T_STAR).astype(np.int8))
    s_cons = score_dialogues(ys, [to_labels(b) for b in cons_ax])["score"]

    # 참고: 경계 수준 다수결(축이 아니라 최종 경계를 투표)
    cons_bnd = []
    for d in DEV3:
        stack = np.stack(per_doc[d]["mats"])
        b = np.stack([(stack[j].sum(1) >= T_STAR) for j in range(len(reps))])
        cons_bnd.append((b.mean(0) >= 0.5).astype(np.int8))
    s_cons_b = score_dialogues(ys, [to_labels(b) for b in cons_bnd])["score"]

    delta = s_cons - s_vals.mean()
    verdict = ("잡음이 실제 비용 → **k 재설정 정당**" if delta >= FLOOR else
               "회색 — 잡음은 있으나 해석 가능 폭 아래 → **k 우선순위 강등**" if delta >= 0 else
               "잡음이 병목이 **아니다** → **프롬프트 편향이 병목**, k 재설정 기각")

    # ── 축별 draw 간 일치도 ────────────────────────────────────────────────
    ax_agree, ax_rate = [], []
    for k in range(5):
        agr, rates = [], []
        for d in DEV3:
            stack = np.stack(per_doc[d]["mats"])[:, :, k]
            for a, b in itertools.combinations(range(len(reps)), 2):
                agr.append((stack[a] == stack[b]).mean())
            rates.append(stack.mean())
        ax_agree.append(float(np.mean(agr)))
        ax_rate.append(float(np.mean(rates)))

    L = [f"# 잡음 vs 프롬프트 편향 — dev-3 반복추출 {len(reps)}개 (동결 v2, q36_35b)\n",
         "> 사람 gold 는 **채점에만** 사용. 합의 구성에 gold 미사용.",
         f"> **사전선언**: ΔS ≥ {FLOOR} 통과 · 0~{FLOOR} 회색 · < 0 프롬프트 편향 판정\n",
         "\n## 개별 draw vs 합의 (dev-3 pooled Score, deployable t\\*=3)\n",
         "| | Score |", "|---|---|"]
    for r, v in single.items():
        L.append(f"| draw {r} | {v:.4f} |")
    L += [f"| **개별 평균** | **{s_vals.mean():.4f}** (σ {s_vals.std():.4f}) |",
          f"| 개별 최고 / 최저 | {s_vals.max():.4f} / {s_vals.min():.4f} |",
          f"| **합의 — 축별 다수결** | **{s_cons:.4f}** |",
          f"| 합의 — 경계 다수결(참고) | {s_cons_b:.4f} |",
          f"\n**ΔS = 합의 − 개별평균 = {delta:+.4f}** → **판정: {verdict}**",
          "\n## 축별 draw 간 일치도 (5draw 쌍별 평균)\n",
          "| 축 | 발화율 | draw 간 일치율 |", "|---|---|---|"]
    for k, a in enumerate(FROZEN):
        L.append(f"| {a} | {ax_rate[k]:.3f} | {ax_agree[k]:.4f} |")
    L.append("\n⚠️ 앵커 k=1 · 탐색적 — 성능 주장 아님(D-B16). dev-3 = 연습장.")
    out = REPO / "outputs/experiments/m4m5_phase1_anchor/CONSENSUS_VS_SINGLE.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    (out.with_suffix(".json")).write_text(json.dumps(
        {"single": single, "consensus_axis": s_cons, "consensus_boundary": s_cons_b,
         "delta": delta, "axis_agreement": ax_agree, "axis_rate": ax_rate}, indent=2))
    print("\n".join(L))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

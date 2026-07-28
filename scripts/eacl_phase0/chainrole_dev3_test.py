#!/usr/bin/env python3
"""D-B22 §D 본검정 — 의도 사슬 역할 리셋의 dev-3 확증 평가 (1회만).

기준은 **D-B22 §D 에 결과 보기 전 고정**됐고 아래 상수가 그 사본이다. 사후 수정 금지.
리셋 규칙 = **D-B22-A2 ε-전이**(`llm_s5_states.chainrole_resets`, 승인된 단일 예외).

    ① lift 95% CI 하한 (이야기 단위 cluster bootstrap) >= 1.10
    ② 발화율 0.025 ~ 0.150
    ③ ΔScore >= -0.010          (5축 count>=t* 의 causality 열을 교체했을 때)
    채택 = 3/3.
"""
from __future__ import annotations

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
BASE = REPO / "outputs/eacl_phase0"
CHAIN = REPO / "outputs/runs/2026-07-28_chainrole_dev3"
T_STAR = 3
# ── D-B22 §D 사전등록 상수 (수정 금지) ──────────────────────────────────────
LIFT_CI_MIN, RATE_LO, RATE_HI, DSCORE_FLOOR = 1.10, 0.025, 0.150, -0.010
N_BOOT, CI = 10_000, 95


def load(doc: str) -> dict:
    d = BASE / f"{doc}_michelmann_s5_q36_35b_v2" / doc

    def st(axis):
        return {int(x["item_id"]): x for x in s5.read(d / f"{axis}_states.jsonl")}

    S = {"protagonist": st("protagonist"), "causality": st("causality"),
         "intentionality": s5.carry_forward(st("intentionality"), "active_goal", "none"),
         "space": s5.carry_forward(st("space"), "location", "unspecified"),
         "time": st("time")}
    cr = {int(x["item_id"]): x for x in s5.read(CHAIN / doc / "chainrole_states.jsonl")}
    master = s5.read(BASE / f"{doc}_michelmann_master.jsonl")
    gold = {int(r["transition"]): int(r["boundary"]) for r in master}
    n = max(cr) + 1
    chain = s5.chainrole_resets([cr[i] for i in range(n)])   # A2 ε-전이, 길이 n-1
    rows, g, ch = [], [], []
    for i in sorted(gold):
        if any(i not in S[a] or i - 1 not in S[a] for a in S) or i - 1 >= len(chain):
            continue
        rows.append([
            s5.reset_protagonist_present(S["protagonist"][i - 1], S["protagonist"][i]),
            s5.reset_space(S["space"][i - 1], S["space"][i]),
            s5.reset_time(S["time"][i - 1], S["time"][i]),
            s5.reset_causality(S["causality"][i - 1], S["causality"][i]),
            s5.reset_intentionality(S["intentionality"][i - 1], S["intentionality"][i]),
        ])
        ch.append(chain[i - 1])
        g.append(gold[i])
    return {"R": np.asarray(rows, np.int8), "gold": np.asarray(g, np.int8),
            "chain": np.asarray(ch, np.int8), "doc": doc}


def lift(fire, gold) -> float:
    b, nb = gold == 1, gold == 0
    if b.sum() == 0 or nb.sum() == 0 or fire[nb].mean() == 0:
        return float("nan")
    return float(fire[b].mean() / fire[nb].mean())


def main() -> None:
    docs = [load(d) for d in DEV3]
    fire = np.concatenate([d["chain"] for d in docs])
    gold = np.concatenate([d["gold"] for d in docs])
    rate = float(fire.mean())
    obs = lift(fire, gold)

    # ① 이야기 단위 cluster bootstrap — 이야기를 통째로 복원추출
    rng = np.random.default_rng(20260728)
    vals = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(docs), len(docs))
        f = np.concatenate([docs[j]["chain"] for j in pick])
        g = np.concatenate([docs[j]["gold"] for j in pick])
        v = lift(f, g)
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [(100 - CI) / 2, 100 - (100 - CI) / 2])

    # ③ 5축 count>=t* 에서 causality 열 교체
    def score(use_chain: bool) -> float:
        ys, ps = [], []
        for d in docs:
            R = d["R"].copy()
            if use_chain:
                R[:, 3] = d["chain"]
            b = (R.sum(1) >= T_STAR).astype(np.int8)
            ys.append(np.concatenate([d["gold"], [0]]).tolist())
            ps.append(np.concatenate([b, [0]]).tolist())
        return score_dialogues(ys, ps)["score"]

    s_old, s_new = score(False), score(True)
    d_score = s_new - s_old
    c1, c2, c3 = lo >= LIFT_CI_MIN, RATE_LO <= rate <= RATE_HI, d_score >= DSCORE_FLOOR
    adopt = c1 and c2 and c3

    L = ["# D-B22 §D 본검정 — 의도 사슬 역할 리셋 (dev-3, 1회)\n",
         "> 리셋 규칙 = **D-B22-A2 ε-전이**(승인된 단일 예외). 기준은 §D 에 결과 보기 전 고정, 무수정.",
         "> gold 는 채점에만 사용. 이 평가는 **1회만** 수행한다.\n",
         "\n## 사전등록 3조건 (3/3 충족해야 채택)\n",
         "| # | 조건 | 기준 | 관측 | 판정 |", "|---|---|---|---|---|",
         f"| ① | lift 95% CI 하한 (이야기 cluster bootstrap) | ≥ {LIFT_CI_MIN} | "
         f"**{lo:.3f}** (점추정 {obs:.3f}, 상한 {hi:.3f}) | {'✅' if c1 else '❌'} |",
         f"| ② | 발화율 | [{RATE_LO}, {RATE_HI}] | **{rate:.4f}** | {'✅' if c2 else '❌'} |",
         f"| ③ | ΔScore (causality 열 교체) | ≥ {DSCORE_FLOOR} | **{d_score:+.4f}** "
         f"({s_old:.4f} → {s_new:.4f}) | {'✅' if c3 else '❌'} |",
         f"\n**판정 = {'✅ 채택' if adopt else '❌ 실패 — D-B22 폐기'}**",
         "\n## 문서별 (서술통계)\n", "| 문서 | 전이 | gold | 리셋 | 발화율 | lift |", "|---|---|---|---|---|---|"]
    for d in docs:
        L.append(f"| {d['doc']} | {len(d['gold'])} | {int(d['gold'].sum())} | "
                 f"{int(d['chain'].sum())} | {d['chain'].mean():.4f} | "
                 f"{lift(d['chain'], d['gold']):.3f} |")
    L += [f"\n> 부트스트랩 = **이야기 단위** cluster, {N_BOOT:,}회, percentile {CI}% CI.",
          "> ⚠️ codex 명시 한계: **이야기가 3편뿐이라 cluster bootstrap CI 안정성이 낮다.**",
          "\n⚠️ 앵커 k=1 · 탐색적 — dev-3 = 연습장(D-B16). 절대 Score 는 성능이 아니다."]
    out = REPO / "outputs/experiments/m4m5_phase1_anchor/CHAINROLE_DEV3_TEST.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(
        {"lift": obs, "ci_lo": float(lo), "ci_hi": float(hi), "rate": rate,
         "score_old": s_old, "score_new": s_new, "d_score": d_score,
         "conditions": {"lift": bool(c1), "rate": bool(c2), "dscore": bool(c3)},
         "adopt": bool(adopt)}, indent=2))
    print("\n".join(L))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

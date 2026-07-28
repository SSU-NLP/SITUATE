#!/usr/bin/env python3
"""사전등록 aspect 전이 기반 확증적 리셋 진단 (ledger D-B21). **API 0** — 기존 라벨 재판정.

🔴 명칭 주의: 이것을 "결정→행동→완료 검증"이라 부르지 않는다. 기존 `aspect`(사건의 시작·진행·완료)와
목표지향 사슬에서의 기능은 **관련되나 동일하지 않다**(codex 2026-07-28). 이 진단의 성공이 그
조작화를 입증하지도, 실패가 반증하지도 않는다.

전이표·기준·중단조건은 **D-B21 에 결과 보기 전 고정**돼 있으며 아래 상수가 그 사본이다.
사후 수정 금지.
"""
from __future__ import annotations

import argparse
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
T_STAR = 3

# ── D-B21 사전등록 상수 (수정 금지) ─────────────────────────────────────────
# aspect_reset = (직전 == completive) AND (현재 ∈ {inceptive, ongoing})
RESET_TRANSITIONS = {("completive", "inceptive"), ("completive", "ongoing")}
RATE_LO, RATE_HI = 0.15, 0.40      # ② 발화율 sanity
DSCORE_FLOOR = -0.01               # ③ 안전장치
N_BOOT, CI = 10_000, 95            # 문서 내 층화 재표집, percentile 95% CI


def load_doc(doc: str, tag: str) -> dict:
    """한 문서의 5축 상태 + gold. tag 예: q36_35b_v2_r1"""
    d = BASE / f"{doc}_michelmann_s5_{tag}" / doc

    def st(axis):
        return {int(x["item_id"]): x for x in s5.read(d / f"{axis}_states.jsonl")}

    S = {"protagonist": st("protagonist"), "causality": st("causality"),
         "intentionality": s5.carry_forward(st("intentionality"), "active_goal", "none"),
         "space": s5.carry_forward(st("space"), "location", "unspecified"),
         "time": st("time")}
    master = s5.read(BASE / f"{doc}_michelmann_master.jsonl")
    gold = {int(r["transition"]): int(r["boundary"]) for r in master}
    rows, g, asp_new, asp_old = [], [], [], []
    for i in sorted(gold):
        if any(i not in S[a] or i - 1 not in S[a] for a in S):
            continue
        rows.append([
            s5.reset_protagonist_present(S["protagonist"][i - 1], S["protagonist"][i]),
            s5.reset_space(S["space"][i - 1], S["space"][i]),
            s5.reset_time(S["time"][i - 1], S["time"][i]),
            s5.reset_causality(S["causality"][i - 1], S["causality"][i]),   # 동결 규칙
            s5.reset_intentionality(S["intentionality"][i - 1], S["intentionality"][i]),
        ])
        prev_a, cur_a = S["causality"][i - 1]["aspect"], S["causality"][i]["aspect"]
        asp_new.append(int((prev_a, cur_a) in RESET_TRANSITIONS))
        asp_old.append((prev_a, cur_a))
        g.append(gold[i])
    return {"R": np.asarray(rows, np.int8), "gold": np.asarray(g, np.int8),
            "aspect_reset": np.asarray(asp_new, np.int8), "pairs": asp_old}


def lift(fire: np.ndarray, gold: np.ndarray) -> float:
    """경계에서의 발화율 / 비경계에서의 발화율 (micro)."""
    b, nb = gold == 1, gold == 0
    if b.sum() == 0 or nb.sum() == 0 or fire[nb].mean() == 0:
        return float("nan")
    return float(fire[b].mean() / fire[nb].mean())


def boot_ci(docs: list[dict], rng: np.random.Generator) -> tuple[float, float, float]:
    """문서 내 층화 재표집 → 세 문서 합쳐 micro lift. (점추정, CI하한, CI상한)"""
    obs = lift(np.concatenate([d["aspect_reset"] for d in docs]),
               np.concatenate([d["gold"] for d in docs]))
    vals = []
    for _ in range(N_BOOT):
        f, g = [], []
        for d in docs:                                   # 문서별 규모·밀도 보존
            idx = rng.integers(0, len(d["gold"]), len(d["gold"]))
            f.append(d["aspect_reset"][idx])
            g.append(d["gold"][idx])
        v = lift(np.concatenate(f), np.concatenate(g))
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [(100 - CI) / 2, 100 - (100 - CI) / 2])
    return obs, float(lo), float(hi)


def score_with(docs: list[dict], use_new: bool) -> float:
    """5축 count>=t* 의 공식 Score. use_new 면 causality 열을 aspect 전이 규칙으로 교체."""
    ys, ps = [], []
    for d in docs:
        R = d["R"].copy()
        if use_new:
            R[:, 3] = d["aspect_reset"]
        b = (R.sum(1) >= T_STAR).astype(np.int8)
        ys.append(np.concatenate([d["gold"], [0]]).tolist())
        ps.append(np.concatenate([b, [0]]).tolist())
    return score_dialogues(ys, ps)["score"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="q36_35b_v2_r1", help="추출 태그(동결 라벨러)")
    ap.add_argument("--seed", type=int, default=20260728)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    docs = [load_doc(d, args.tag) for d in DEV3]

    fire = np.concatenate([d["aspect_reset"] for d in docs])
    gold = np.concatenate([d["gold"] for d in docs])
    old_fire = np.concatenate([d["R"][:, 3] for d in docs])
    rate, old_rate = float(fire.mean()), float(old_fire.mean())
    obs, lo, hi = boot_ci(docs, rng)
    old_lift = lift(old_fire, gold)
    s_old, s_new = score_with(docs, False), score_with(docs, True)
    d_score = s_new - s_old

    c1 = lo > 1.0
    c2 = RATE_LO <= rate <= RATE_HI
    c3 = d_score >= DSCORE_FLOOR
    adopt = c1 and c2 and c3

    L = [f"# 사전등록 aspect 전이 기반 확증적 리셋 진단 (D-B21) — tag `{args.tag}`\n",
         "> 🔴 이것을 \"결정→행동→완료 검증\"이라 부르지 않는다(D-B21 명칭 규정).",
         "> 전이표·기준·중단조건은 **결과 보기 전** D-B21 에 고정됨. 사후 수정 없음.",
         f"> 규칙: `aspect_reset = (직전 == completive) AND (현재 ∈ {{inceptive, ongoing}})`,"
         " `event_type ≠ 직전` 항 제거.\n",
         "\n## 사전등록 3조건 (3/3 충족해야 채택)\n",
         "| # | 조건 | 기준 | 관측 | 판정 |", "|---|---|---|---|---|",
         f"| ① | lift 95% bootstrap CI 하한 | > 1.0 | **{lo:.3f}** "
         f"(점추정 {obs:.3f}, 상한 {hi:.3f}) | {'✅' if c1 else '❌'} |",
         f"| ② | 발화율 | [{RATE_LO}, {RATE_HI}] | **{rate:.3f}** | {'✅' if c2 else '❌'} |",
         f"| ③ | ΔScore (5축 count≥3) | ≥ {DSCORE_FLOOR} | **{d_score:+.4f}** "
         f"({s_old:.4f} → {s_new:.4f}) | {'✅' if c3 else '❌'} |",
         f"\n**판정 = {'✅ 채택' if adopt else '❌ 실패 — D-B21 중단조건 발동'}**",
         "\n## 대조 — 동결 규칙\n",
         "| | 발화율 | lift |", "|---|---|---|",
         f"| 동결 `event_type≠직전 OR aspect==inceptive` | {old_rate:.3f} | {old_lift:.3f} |",
         f"| **신규 aspect 전이** | **{rate:.3f}** | **{obs:.3f}** |",
         f"| *참조: D-B13 #5 반사실창* | *0.309* | *CI하한 0.758 (FAIL)* |",
         "\n## 전이 분포 (서술통계)\n", "| 전이 | 건수 | 비율 |", "|---|---|---|"]
    from collections import Counter
    cnt = Counter(p for d in docs for p in d["pairs"])
    tot = sum(cnt.values())
    for k, v in sorted(cnt.items(), key=lambda x: -x[1]):
        mark = " **(RESET)**" if k in RESET_TRANSITIONS else ""
        L.append(f"| {k[0]} → {k[1]}{mark} | {v} | {v/tot:.3f} |")
    L += [f"\n> gold 66/{len(gold)} (밀도 {gold.mean():.3f}). 부트스트랩 = 문서 내 층화, "
          f"{N_BOOT:,}회, percentile {CI}% CI.",
          "\n⚠️ 검정력: gold 66개뿐이라 낮다. CI 하한 실패는 \"효과 없음\"이 아니라 **채택 근거 "
          "불충분**이며, 검정력 부족을 이유로 임계값을 사후 완화하지 않는다(D-B21).",
          "\n⚠️ 앵커 k=1 · 탐색적 — dev-3 = 연습장(D-B16)."]
    out = REPO / "outputs/experiments/m4m5_phase1_anchor/ASPECT_TRANSITION_DIAG.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(
        {"lift": obs, "ci_lo": lo, "ci_hi": hi, "rate": rate, "old_rate": old_rate,
         "old_lift": old_lift, "score_old": s_old, "score_new": s_new, "d_score": d_score,
         "conditions": {"lift_ci": c1, "rate": c2, "dscore": c3}, "adopt": adopt}, indent=2))
    print("\n".join(L))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

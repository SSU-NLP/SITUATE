#!/usr/bin/env python3
"""DTS 벤치마크에 5축 혼합효과 로지스틱 + permutation null (dev-3 와 동일 분석).

## 왜
dev-3(서사)에서 축별 오즈비를 냈다(protagonist 3.34\\*, space 2.32, time 2.95, causality 1.44,
intentionality 1.42). 같은 분석을 **DTS 대화 벤치마크**에 돌려 축이 도메인을 넘는지 본다.

**프로젝트의 기존 주장(경계조건 ③, D-M5/D-M7):** *tiage 의 "화제 전환"은 우리 타깃인 "상황 리셋"과
**구성개념이 다르다**.* 따라서 축이 약하게 나오는 것이 **예측된 결과**이며, 본 분석은 그 주장에
**통계적 근거**를 붙인다(지금까지는 Score 비교뿐이었다).

## 데이터 (이미 추출됨 — 새 라벨 0 · API 0)
`outputs/eacl_phase0/dts_s5_q36_35b_v31/{tiage,dialseg711,superseg,ami}/reset_matrix.jsonl`
동결 v3.1 · Qwen3.6-35B-A3B no-thinking. 대화(doc_id)를 **클러스터**로 다룬다.

🔴 지위: **탐색적**. DTS 는 D-M5 에서 **scope 밖**으로 선언된 데이터이며 성능 주장에 쓰지 않는다.
ami 는 대화 2편뿐이라 클러스터가 부족 — 참고로만 싣는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
AXES = ["protagonist_present", "space", "time", "causality", "intentionality"]
SETS = ("tiage", "dialseg711", "superseg", "ami")
BASE = REPO / "outputs/eacl_phase0/dts_s5_q36_35b_v31"
N_PERM = 5_000
SEED = 20260728


def load(name: str) -> pd.DataFrame:
    import json
    rows = [json.loads(l) for l in open(BASE / name / "reset_matrix.jsonl") if l.strip()]
    return pd.DataFrame([{**{a: int(r[a]) for a in AXES},
                          "boundary": int(r["boundary"]), "doc": r["doc_id"]} for r in rows])


def analyze(df: pd.DataFrame, n_perm: int) -> tuple[np.ndarray, list[float], object]:
    X = sm.add_constant(df[AXES].astype(float))
    y = df["boundary"].to_numpy().astype(float)
    groups = df["doc"].to_numpy()
    fit = sm.GEE(y, X, groups=groups, family=sm.families.Binomial(),
                 cov_struct=sm.cov_struct.Exchangeable()).fit()
    beta = fit.params[AXES].to_numpy()
    rng = np.random.default_rng(SEED)
    idx = [np.flatnonzero(groups == d) for d in pd.unique(groups)]
    perm = []
    for _ in range(n_perm):
        ys = y.copy()
        for ii in idx:
            ys[ii] = rng.permutation(ys[ii])       # 대화 내 셔플 = 대화별 경계 수 보존
        try:
            perm.append(sm.GEE(ys, X, groups=groups, family=sm.families.Binomial(),
                               cov_struct=sm.cov_struct.Exchangeable()
                               ).fit(maxiter=25).params[AXES].to_numpy())
        except Exception:
            pass
    perm = np.asarray(perm)
    p = [(np.sum(np.abs(perm[:, j]) >= abs(beta[j])) + 1) / (len(perm) + 1)
         for j in range(len(AXES))]
    return beta, p, fit


def main() -> None:
    L = ["# DTS 벤치마크 5축 효과 — dev-3 와 동일 분석 (GEE + permutation null)\n",
         "> 🔴 **탐색적.** DTS 는 D-M5 에서 **scope 밖**으로 선언된 데이터 — 성능 주장에 쓰지 않는다.",
         "> 프로젝트 기존 주장(경계조건 ③): *tiage 의 '화제 전환'은 타깃인 '상황 리셋'과 구성개념이"
         " 다르다.* 축이 약한 것이 **예측된 결과**이며 여기서 통계적 근거를 붙인다.",
         "> 새 라벨 0 · API 0 (동결 v3.1 기추출분 재사용). 대화(doc_id) = 클러스터.\n",
         "\n| 데이터 | 전이 | 경계 | 밀도 | 대화 |", "|---|---|---|---|---|"]
    dfs = {}
    for s in SETS:
        d = load(s)
        dfs[s] = d
        L.append(f"| {s} | {len(d)} | {int(d.boundary.sum())} | {d.boundary.mean():.4f} | "
                 f"{d.doc.nunique()} |")
    L.append("| *참조: dev-3(서사)* | *1005* | *66* | *0.0657* | *3* |")

    for s in SETS:
        d = dfs[s]
        note = " ⚠️ 대화 2편 — 클러스터 부족, 참고용" if s == "ami" else ""
        L.append(f"\n## {s}{note}\n")
        if d.boundary.sum() < 5 or d.doc.nunique() < 2:
            L.append("> 표본 부족으로 생략.")
            continue
        beta, p, fit = analyze(d, N_PERM)
        ci = fit.conf_int()
        L += ["| 축 | 오즈비 exp(β) | 95% CI | permutation p |", "|---|---|---|---|"]
        for j, a in enumerate(AXES):
            lo, hi = ci.loc[a]
            star = " **\\***" if p[j] < 0.05 else ""
            L.append(f"| {a}{star} | **{np.exp(beta[j]):.2f}** | "
                     f"[{np.exp(lo):.2f}, {np.exp(hi):.2f}] | {p[j]:.4f} |")
        sig = [a for a, pv in zip(AXES, p) if pv < 0.05]
        L.append(f"\n**p < 0.05: {', '.join(sig) if sig else '**없음**'}**")

    L += ["\n## 참조 — dev-3(서사 픽션) 같은 분석\n",
          "| 축 | 오즈비 | permutation p |", "|---|---|---|",
          "| protagonist_present **\\*** | **3.34** | 0.0005 |",
          "| space | 2.32 | 0.0606 |", "| time | 2.95 | 0.2837 |",
          "| causality | 1.44 | 0.1113 |", "| intentionality | 1.42 | 0.3460 |",
          "\n⚠️ DTS 는 scope 밖(D-M5). 이 표는 **구성개념 불일치 주장의 증거**로만 읽고"
          " 성능 비교로 읽지 않는다."]
    p_ = REPO / "outputs/experiments/m4m5_phase1_anchor/DTS_AXIS_LOGIT.md"
    p_.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n-> {p_}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""5축 리셋 → 경계 혼합효과 로지스틱 + permutation null (dev-3 전체 1,005 전이).

## 왜 이걸 하나
직전 coarse/fine lift 진단은 **경계 18개만 꺼내 세는** 방식이라 982 전이를 버렸고, 표본이 작아
숫자가 널뛰었다(time 축 부호가 문서 간 +5.70 / −26.94 로 반전). **같은 데이터를 전부 쓰면**
축별 효과를 안정적으로 추정할 수 있다.

## 무엇을 하나
- 단위 = 문장 전이 1개. 결과변수 = gold 경계(0/1). 설명변수 = 5축 리셋(0/1).
- 문서(3편)를 **클러스터**로 다뤄 문서 간 이질성(길이·장르·gold 밀도 0.19/0.036/0.074)을 흡수:
  ① **GEE**(exchangeable, cluster-robust SE) = 주 모형 — 모집단 평균 효과, 소표본 클러스터에 견고
  ② **Bayes mixed GLM**(문서 랜덤절편) = 교차 확인
- **permutation null**: 경계 라벨을 **문서 내에서만** 셔플(문서별 경계 수·밀도 보존) → 계수의
  귀무분포 → 경험적 양측 p. 클러스터 3개뿐이라 정규근사 p 보다 이쪽이 정직하다.

🔴 지위: **탐색적**(dev-3 = 연습장, D-M13③). 확증 근거가 아니다. causality 결론(D-B13 약축)을
바꾸려는 게 아니라, "약한 것 같다(널뜀)" → "약하다(추정치+구간)" 로 **서술 형태를 바꾸는** 것이다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
import llm_s5_states as s5  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM  # noqa: E402

BASE = REPO / "outputs/eacl_phase0"
DEV3 = ("pieman", "tunnel", "monkey")
AXES = ["protagonist_present", "space", "time", "causality", "intentionality"]
N_PERM = 10_000
SEED = 20260728


def build() -> pd.DataFrame:
    rows = []
    for doc in DEV3:
        # 🔒 **동결 v3.1 재조합**(D-B14). dev-3 에는 v3.1 직접 추출본이 없고, v3.1 = H+E 는
        # Stage-1 불변·축 독립이므로 `fix_space_H` 의 space + `fix_time` 의 time + v2 의 나머지
        # 3축 재조합으로 계산한다(`s5_v31_combo_eval.py` 와 동일 규약).
        # ⚠️ 이전 판은 v2 전체를 썼다 — DTS(v3.1)와 조건이 어긋나 정정했다.
        def S(tag, axis):
            return {int(x["item_id"]): x
                    for x in s5.read(BASE / f"{doc}_michelmann_s5_{tag}" / doc / f"{axis}_states.jsonl")}
        st = {"protagonist": S("q36_35b_v2", "protagonist"),
              "causality": S("q36_35b_v2", "causality"),
              "time": S("fix_time", "time"),
              "space": s5.carry_forward(S("fix_space_H", "space"), "location", "unspecified"),
              "intentionality": s5.carry_forward(S("q36_35b_v2", "intentionality"),
                                                 "active_goal", "none")}
        for r in sorted(s5.read(BASE / f"{doc}_michelmann_master.jsonl"),
                        key=lambda r: int(r["transition"])):
            i = int(r["transition"])
            if any(i not in st[a] or i - 1 not in st[a] for a in st):
                continue
            rows.append(dict(
                doc=doc, boundary=int(r["boundary"]),
                protagonist_present=s5.reset_protagonist_present(st["protagonist"][i-1], st["protagonist"][i]),
                space=s5.reset_space(st["space"][i-1], st["space"][i]),
                time=s5.reset_time(st["time"][i-1], st["time"][i]),
                causality=s5.reset_causality(st["causality"][i-1], st["causality"][i]),
                intentionality=s5.reset_intentionality(st["intentionality"][i-1], st["intentionality"][i])))
    return pd.DataFrame(rows)


def fit_gee(df: pd.DataFrame) -> np.ndarray:
    X = sm.add_constant(df[AXES].astype(float))
    m = sm.GEE(df["boundary"].astype(float), X, groups=df["doc"],
               family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable())
    return m.fit()


def main() -> None:
    df = build()
    n, nb = len(df), int(df["boundary"].sum())
    print(f"[data] {n} 전이 · gold 경계 {nb} ({nb/n:.4f})", flush=True)
    print(df.groupby("doc").agg(n=("boundary", "size"), gold=("boundary", "sum")).to_string(), flush=True)

    res = fit_gee(df)
    beta = res.params[AXES].to_numpy()

    # ── permutation null: 문서 내 경계 라벨 셔플 (문서별 경계 수 보존) ────
    rng = np.random.default_rng(SEED)
    perm = np.zeros((N_PERM, len(AXES)))
    idx_by_doc = [np.flatnonzero((df["doc"] == d).to_numpy()) for d in DEV3]
    y = df["boundary"].to_numpy().astype(float)
    X = sm.add_constant(df[AXES].astype(float))
    groups = df["doc"].to_numpy()
    for k in range(N_PERM):
        ys = y.copy()
        for ii in idx_by_doc:
            ys[ii] = rng.permutation(ys[ii])
        try:
            r = sm.GEE(ys, X, groups=groups, family=sm.families.Binomial(),
                       cov_struct=sm.cov_struct.Exchangeable()).fit(maxiter=30)
            perm[k] = r.params[AXES].to_numpy()
        except Exception:
            perm[k] = np.nan
        if (k + 1) % 2000 == 0:
            print(f"  [perm] {k+1}/{N_PERM}", flush=True)
    ok = ~np.isnan(perm).any(1)
    perm = perm[ok]
    pvals = [(np.sum(np.abs(perm[:, j]) >= abs(beta[j])) + 1) / (len(perm) + 1)
             for j in range(len(AXES))]

    # ── 교차 확인: 문서 랜덤절편 Bayes mixed GLM ──────────────────────────
    try:
        vc = {"doc": "0 + C(doc)"}
        mg = BinomialBayesMixedGLM.from_formula(
            "boundary ~ " + " + ".join(AXES), vc, df).fit_vb()
        mixed = {a: float(mg.fe_mean[list(mg.model.exog_names).index(a)]) for a in AXES}
    except Exception as e:
        mixed = {a: float("nan") for a in AXES}
        print(f"[warn] mixed GLM 실패: {e}", flush=True)

    ci = res.conf_int()
    L = ["# 5축 리셋 → 경계: 혼합효과 로지스틱 + permutation null (dev-3 전체, **동결 v3.1**)\n",
         "> 🔴 **탐색적**(dev-3 = 연습장, D-M13③). 확증 근거 아님.",
         "> 직전 coarse/fine lift 진단은 경계 18개만 써 982 전이를 버렸고 부호가 널뛰었다"
         "(time +5.70 / −26.94). **여기서는 1,005 전이 전부를 쓴다.**",
         f"> 주 모형 = GEE(exchangeable, **문서 클러스터-robust SE**) · null = 문서 내 라벨 셔플 "
         f"{len(perm):,}회 · 교차확인 = 문서 랜덤절편 Bayes mixed GLM.\n",
         f"\n데이터: **{n} 전이 · gold 경계 {nb}** (밀도 {nb/n:.4f}) · "
         + " · ".join(f"{d} {int((df.doc==d).sum())}/{int(df[df.doc==d].boundary.sum())}" for d in DEV3),
         "\n## 축별 효과 (경계 로그오즈)\n",
         "| 축 | 계수 β | **오즈비 exp(β)** | 95% CI (robust) | **permutation p** | mixed-GLM β |",
         "|---|---|---|---|---|---|"]
    for j, a in enumerate(AXES):
        lo, hi = ci.loc[a]
        star = " **\\***" if pvals[j] < 0.05 else ""
        L.append(f"| {a}{star} | {beta[j]:+.3f} | **{np.exp(beta[j]):.2f}** | "
                 f"[{np.exp(lo):.2f}, {np.exp(hi):.2f}] | **{pvals[j]:.4f}** | {mixed[a]:+.3f} |")
    sig = [a for a, p in zip(AXES, pvals) if p < 0.05]
    L += [f"\n**permutation p < 0.05: {', '.join(sig) if sig else '없음'}**",
          "\n> 오즈비 = 그 축이 리셋된 전이가 안 된 전이보다 경계일 오즈가 몇 배인가."
          " 1.0 = 무관.",
          "> permutation null 은 **문서 내에서만** 라벨을 섞어 문서별 경계 수·밀도를 보존한다."
          " 클러스터가 3개뿐이라 정규근사 p 보다 이쪽이 정직하다.",
          "\n⚠️ 한계: 문서 3편(클러스터 3) — 문서 간 분산 추정이 불안정하다. dev-3 는 연습장이며"
          " pieman 은 선택압 9회 누적(D-M5). **확증적 주장에 쓸 수 없다.**"]
    p = REPO / "outputs/experiments/m4m5_phase1_anchor/AXIS_MIXED_LOGIT.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()

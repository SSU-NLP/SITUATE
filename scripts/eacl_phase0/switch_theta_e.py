#!/usr/bin/env python3
"""SEM 식 (2) 의 **사건별 dynamics `θ_e`** 를 넣은 스위칭 재검정 — D-B26/D-B27.

## 왜 다시 하나 (D-B26)
D-B18/B19 의 스위칭 실험에는 **`θ_e` 가 없었다**. SEM 식 (2)

    Pr(x_{n+1} | x_{1:n}, e) = N( x_{n+1} ; f(x_{1:n}; **θ_e**), diag(β) )

는 dynamics 파라미터가 **사건 e 마다 다르다**. SEM 에서 경계가 생기는 원리 = *"**지금 사건의**
dynamics 가 예측을 못 하게 됐다"*. 그런데 이전 구현은 continue 쪽에 **전역 `f_k` 하나**만 썼다
(문서 전체 동일). 그러면 *"이 사건의 모델이 무너진다"* 는 신호가 **원리적으로 발생할 수 없다**.
따라서 D-B19 의 "스위칭 층 종결" 판정은 **무효**였고, 여기서 1회만 재검정한다.

## 무엇을 (D-B27 사전 선언 = 최소형)
사건마다 GRU 를 만들지 않는다. **공유 frozen 슬롯 상태 + 학습된 전역 `f_k` 위의 사건별 head**:

| 항 | 형태 | SEM 대응 |
|---|---|---|
| 사건별 dynamics | `r_t = z_t − f_k(z_{t-1})` 의 **사건별 평균 이동** `m_{e,k}` | 식 (2) `θ_e` |
| 사전 | `m_{e,k} ~ N(μ0_k, τ²_k)` **켤레** (τ² = 사건 **간** 분산, 경험적 베이즈) | — |
| 갱신 | 충분통계 `(n, Σr)` 만 → **O(1)/전이**, 재학습 없음 | 온라인 |
| continue 우도 | **사후예측** `N(f_k(z_{t-1}) + m̂_e, σ²_k + v_e)` | 식 (2) |
| new 우도 | **사전예측** = onset 분포 | 식 (3) `f_0` 적분 |
| 결합 | sticky-CRP(식 1) × 슬롯별 우도비 곱(식 5) → 전진 local MAP(식 8) | 식 (4)(8) |

`n` 이 작으면 사후가 사전으로 **수축**해 전역 dynamics 로 되돌아간다 — 과적합 방지가 켤레 사전에
내장돼 있다. `τ²_k ≈ 0` 이면 *"사건마다 dynamics 가 다르지 않다"* 는 **그 자체가 결론**이다.

## 판정 (D-B27, 결과 보기 전 고정)
> 스위칭(λ sweep 최고) − 기준선(적응임계 규칙군 최고) **≥ +0.010** (dev-3 pooled Score)

양쪽 다 최고값을 허용하는 **관대한 검정**. 미달이면 스위칭 층 **최종 종결**(재시도 없음).
λ sweep 이 `inert` 면 그 자체로 실패(2026-05-22 v4.1.x 재발 방지).

## 이번에도 안 하는 것 (사전 명시, D-B27)
§4.8 particle smoothing · protagonist point process(데이터에 `characters_present` 없음) ·
ψₑ 가 MoG 혼합가중을 결정하는 형태(최소형은 평균 이동만) ·
**schema reuse(§4.5 A3)** 는 보조 진단행으로만 병기하고 **종료 기준에 쓰지 않는다**(다중비교 방지).

🔒 D-B16: 캘리브레이션(σ²_k, τ²_k, onset)은 **phase1 val 에서만**. dev-3 gold 미사용.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from hi_ontop.factored_slot import FactoredSlot  # noqa: E402
from hi_ontop.hi_ontop_v2 import adaptive_boundaries  # noqa: E402
from hi_ontop.scrp import sticky_crp_unnormed  # noqa: E402
from switch_scorer import slot_states_and_pred  # noqa: E402
from train_anchor_m4m5 import EXP, T_STAR, Corpus, axis_probs  # noqa: E402

MAX_K = 256
ANCHOR_RATE = 0.0649
EPS = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 캘리브레이션 — phase1 val 에서만 (dev-3 미사용)
# ─────────────────────────────────────────────────────────────────────────────
def calibrate_theta_e(model, c: Corpus, va: list[int], dev: str, bs: int) -> dict:
    """σ²_k(사건 **내** 분산) · τ²_k(사건 **간** 분산) · onset 분포를 추정.

    앵커 코퍼스에는 gold 세그먼트가 없으므로 **앵커가 경계라 한 전이**(`count ≥ t*=3`)로 사건을
    나눈다(§4.7 A1 이 "세그먼트 첫 문장"을 쓰는 것과 같은 대체).

    τ²_k 가 핵심이다 — *사건마다 dynamics 가 실제로 다른가* 를 수치로 만든 것.
      · 사건 내 잔차 분산 = σ²_k   (같은 사건 안에서의 흔들림)
      · 사건 평균의 분산  = τ²_k   (사건 사이의 차이)
    τ²/σ² 가 0 에 가까우면 사건별 head 는 정보를 못 준다.
    """
    packs = slot_states_and_pred(model, c, va, dev, bs, anchor=True)
    n_slots, d = model.n_slots, model.slot_dim
    within: list[list[np.ndarray]] = [[] for _ in range(n_slots)]   # 사건평균 제거 후 잔차
    ev_means: list[list[np.ndarray]] = [[] for _ in range(n_slots)]  # 사건별 평균 (n≥MIN_N)
    onset: list[list[np.ndarray]] = [[] for _ in range(n_slots)]
    MIN_N = 3                                    # 사건 평균 추정에 최소 3전이
    for i, (S, Sh) in zip(va, packs):
        R = c.resets[int(c.to[i]):int(c.to[i + 1])]
        bnd = R.sum(1) >= T_STAR                 # [T-1]
        r = S[1:] - Sh                           # [T-1, n_slots, d] 전역 f_k 잔차
        # 경계로 사건 분할: 전이 t 가 경계면 t 부터 새 사건
        seg_id = np.concatenate([[0], np.cumsum(bnd[:-1])]) if len(bnd) > 1 else np.zeros(1, int)
        for k in range(n_slots):
            rk = r[:, k, :]
            for e in np.unique(seg_id):
                m = seg_id == e
                blk = rk[m]
                if len(blk) >= MIN_N:
                    mu = blk.mean(0)
                    ev_means[k].append(mu)
                    within[k].append(blk - mu)   # 사건평균 제거 → 순수 사건 내 변동
            if bnd.any():
                onset[k].append(S[1:][bnd, k, :])

    def _cat(x, fb):
        return np.concatenate(x) if x else fb

    sig2 = np.stack([_cat(within[k], np.ones((1, d))).var(0) + EPS for k in range(n_slots)])
    mus = [np.stack(ev_means[k]) if ev_means[k] else np.zeros((1, d)) for k in range(n_slots)]
    mu0 = np.stack([m.mean(0) for m in mus])
    tau2 = np.stack([m.var(0) + EPS for m in mus])
    mu_o = np.stack([_cat(onset[k], np.zeros((1, d))).mean(0) for k in range(n_slots)])
    v_o = np.stack([_cat(onset[k], np.ones((1, d))).var(0) + EPS for k in range(n_slots)])
    return {"sig2": sig2, "mu0": mu0, "tau2": tau2, "mu_onset": mu_o, "var_onset": v_o,
            "n_events": [len(m) for m in mus],
            # 진단: 사건 간/사건 내 분산비. 0 에 가까우면 θ_e 가 정보 없음.
            "tau_over_sig": (tau2.mean(1) / sig2.mean(1)).tolist()}


# ─────────────────────────────────────────────────────────────────────────────
# 전진 local MAP (식 8) — θ_e 사후예측 vs onset 사전예측
# ─────────────────────────────────────────────────────────────────────────────
def segment_doc(S: np.ndarray, Sh: np.ndarray, cal: dict, alpha: float, lmda: float,
                beta: float, reuse: bool = False) -> np.ndarray:
    """전이마다 continue(현 사건의 θ_e) vs new. 반환 = 경계 0/1 [T-1].

    log-ratio = [log P(new) − log P(cur)]                <- sticky-CRP 사전 (식 1)
              + Σ_k [ log p_new,k − log p_cont,k(θ_e) ]  <- 슬롯별 우도비 (식 5), ad-hoc 가중 없음

    `reuse=True` 면 **기존 사건 전체**에 대해 argmax (식 8 충실형). D-B27 상 **보조 진단**이며
    주 판정에 쓰지 않는다.
    """
    sig2, mu0, tau2 = cal["sig2"], cal["mu0"], cal["tau2"]
    mu_o, v_o = cal["mu_onset"], cal["var_onset"]
    T1, n_slots, d = Sh.shape[0], S.shape[1], S.shape[2]
    logdet_o = np.log(v_o).sum(1)                                  # [n_slots]

    # new(=onset) 사전예측 로그밀도: 사건과 무관하므로 벡터화해 미리 계산
    lp_new = np.zeros(T1)
    for k in range(n_slots):
        dn = S[1:, k] - mu_o[k]
        lp_new += -0.5 * ((dn * dn / v_o[k]).sum(1) + logdet_o[k])

    # 사건별 충분통계: n[e], sum_r[e,k,:]  → 사후 m̂, v_e 를 O(1) 로 복원
    n_ev = np.zeros(MAX_K)
    sum_r = np.zeros((MAX_K, n_slots, d))
    counts = np.zeros(MAX_K)
    cur = 0
    counts[cur] = 1.0
    bnd = np.zeros(T1, dtype=np.int8)

    def lp_cont(e: int, t: int) -> float:
        """사건 e 의 사후예측으로 본 전이 t 의 로그밀도 (전 슬롯 합)."""
        tot = 0.0
        for k in range(n_slots):
            prec = 1.0 / tau2[k] + n_ev[e] / sig2[k]               # 사후 정밀도
            m_hat = (mu0[k] / tau2[k] + sum_r[e, k] / sig2[k]) / prec
            v_pred = sig2[k] + 1.0 / prec                           # 사후예측 분산
            res = (S[t + 1, k] - Sh[t, k]) - m_hat
            tot += -0.5 * ((res * res / v_pred).sum() + np.log(v_pred).sum())
        return float(tot)

    for t in range(T1):
        prior = sticky_crp_unnormed(counts, cur, alpha, lmda, beta)
        n_vis = int(np.count_nonzero(counts))
        p_new = prior[n_vis] if n_vis < MAX_K else 1e-12
        best_e, best_v = cur, np.log(max(prior[cur], 1e-12)) + lp_cont(cur, t)
        if reuse:
            for e in range(n_vis):
                if e == cur:
                    continue
                v = np.log(max(prior[e], 1e-12)) + lp_cont(e, t)
                if v > best_v:
                    best_e, best_v = e, v
        v_new = np.log(max(p_new, 1e-12)) + lp_new[t]

        if v_new > best_v and n_vis < MAX_K - 1:                   # 새 사건
            bnd[t] = 1
            cur = n_vis
            counts[cur] = 1.0
        else:
            if best_e != cur:                                       # 기존 사건 재사용
                bnd[t] = 1
                cur = best_e
            counts[cur] += 1.0
            n_ev[cur] += 1.0
            sum_r[cur] += S[t + 1] - Sh[t]
    return bnd


# ─────────────────────────────────────────────────────────────────────────────
def eval_switch(packs, dev3: Corpus, cal, alpha, lmda, beta, reuse=False) -> dict:
    ys, ps, npred = [], [], 0
    for i, (S, Sh) in enumerate(packs):
        g = dev3.gold[int(dev3.to[i]):int(dev3.to[i + 1])].astype(np.int8)
        b = segment_doc(S, Sh, cal, alpha, lmda, beta, reuse)
        ys.append(np.concatenate([g, [0]]).tolist())
        ps.append(np.concatenate([b, [0]]).tolist())
        npred += int(b.sum())
    return {"score": score_dialogues(ys, ps)["score"], "n_pred": npred}


def eval_baseline(model, dev3: Corpus, dev: str) -> dict:
    """D-B27 기준선 — **같은 체크포인트**의 Σ_k P(reset_k) + 적응임계 규칙군 전체."""
    probs = axis_probs(model, dev3, list(range(dev3.n_docs)), dev, bs=8, with_anchor=False)
    ys = [np.concatenate([dev3.gold[int(dev3.to[i]):int(dev3.to[i + 1])], [0]]).tolist()
          for i in range(dev3.n_docs)]
    out: dict[str, tuple[float, int]] = {}
    acc: dict[str, list] = {}
    npred: dict[str, int] = {}
    for pr in probs:
        s = pr.sum(1)
        seq = np.concatenate([[0.0], s]).tolist()
        rules = {"고정 6.49%": (s >= np.quantile(s, 1 - ANCHOR_RATE)).astype(np.int8),
                 "otsu": np.asarray(adaptive_boundaries(seq, mode="otsu"), np.int8)[:len(s)]}
        for c_ in (0.5, 1.0, 1.5, 2.0):
            for mode, tag in (("offline", "offline"), ("online_nf", "online")):
                rules[f"{tag} μ+{c_}σ"] = np.asarray(
                    adaptive_boundaries(seq, c=c_, mode=mode), np.int8)[:len(s)]
        for k, b in rules.items():
            acc.setdefault(k, []).append(np.concatenate([b, [0]]).tolist())
            npred[k] = npred.get(k, 0) + int(b.sum())
    for k, v in acc.items():
        out[k] = (score_dialogues(ys, v)["score"], npred[k])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="M5")
    ap.add_argument("--ns", default="_per_slot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="phase1")
    ap.add_argument("--cal-docs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--crp-beta", type=float, default=1.0)
    ap.add_argument("--lmda-sweep", default="0,1,3,10,30,100")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    c, dev3 = Corpus(args.split), Corpus("dev3")
    ck = torch.load(EXP / f"{args.model}{args.ns}_seed{args.seed}.pt",
                    map_location=dev, weights_only=False)
    if ck.get("dynamics") is None:
        raise SystemExit("dynamics 없는 체크포인트 — 전역 f_k 가 없으면 잔차를 못 만든다.")
    model = FactoredSlot(slot_dim=ck["args"]["slot_dim"], mono=ck["mono"],
                         pos_weight=ck["pos_weight"], dynamics=ck["dynamics"],
                         pred_space=ck["args"].get("pred_space", "obs")).to(dev)
    model.load_state_dict(ck["state"])

    rng = np.random.default_rng(20260728)                  # 학습과 동일 분할
    perm = rng.permutation(c.n_docs)
    va_all = perm[:max(1, int(c.n_docs * 0.05))].tolist()
    va = np.random.default_rng(args.seed).choice(
        va_all, min(args.cal_docs, len(va_all)), replace=False).tolist()
    print(f"[cal] phase1 val {len(va)}편에서 σ²_k·τ²_k·onset 추정 (dev-3 미사용)", flush=True)
    cal = calibrate_theta_e(model, c, va, dev, args.batch)
    tos = cal["tau_over_sig"]
    print(f"[cal] 사건 수 {cal['n_events']}\n[cal] τ²/σ² (사건간/사건내) = "
          f"{[round(x, 4) for x in tos]}", flush=True)

    print("[base] 같은 체크포인트로 기준선(적응임계 규칙군) 계산", flush=True)
    base = eval_baseline(model, dev3, dev)
    base_best = max(base.items(), key=lambda kv: kv[1][0])

    packs = slot_states_and_pred(model, dev3, list(range(dev3.n_docs)), dev, 4, anchor=False)
    rows = []
    for lm in [float(x) for x in args.lmda_sweep.split(",")]:
        r = eval_switch(packs, dev3, cal, args.alpha, lm, args.crp_beta, reuse=False)
        rows.append((lm, r))
        print(f"  λ={lm:>6.1f}  Score {r['score']:.4f}  예측경계 {r['n_pred']:>4d}", flush=True)
    sw_best = max(rows, key=lambda x: x[1]["score"])
    inert = (max(r["score"] for _, r in rows) - min(r["score"] for _, r in rows) < 1e-9
             and len({r["n_pred"] for _, r in rows}) == 1)

    # 보조 진단 — schema reuse (식 8 충실형). 주 판정 제외(D-B27).
    r_reuse = eval_switch(packs, dev3, cal, args.alpha, sw_best[0], args.crp_beta, reuse=True)
    print(f"  [보조] reuse λ={sw_best[0]:g}  Score {r_reuse['score']:.4f}  "
          f"경계 {r_reuse['n_pred']}", flush=True)

    delta = sw_best[1]["score"] - base_best[1][0]
    passed = (delta >= 0.010) and not inert

    L = [f"# θ_e 스위칭 재검정 — {args.model}{args.ns} seed {args.seed} (D-B26/D-B27)\n",
         "> **D-B19 무효화 후 재검정.** 이전 구현엔 SEM 식(2)의 사건별 dynamics `θ_e` 가 없어"
         " *\"이 사건의 모델이 무너진다\"* 는 신호가 원리적으로 생길 수 없었다(D-B26).",
         "> 최소형 = 공유 frozen 슬롯 + 전역 `f_k` 위의 **사건별 잔차 평균 이동**,"
         " 켤레 사전 `N(μ0, τ²)`, 충분통계만 갱신(O(1)). **재학습 없음.**",
         "> 캘리브레이션(σ²·τ²·onset) = phase1 val 전용. dev-3 gold 미사용(D-B16).",
         f"> α={args.alpha} · CRP β={args.crp_beta} · `scrp.sticky_crp_unnormed`(SEM 식1) 재사용.\n",
         "\n## 0. 사건마다 dynamics 가 실제로 다른가 — τ²/σ²\n",
         "| 축 | protagonist | space | time | causality | intentionality |", "|---|---|---|---|---|---|",
         "| τ²/σ² | " + " | ".join(f"{x:.4f}" for x in tos) + " |",
         f"\n캘리브레이션 사건 수: {cal['n_events']}",
         "\n> τ²(사건 **간** 잔차평균 분산) / σ²(사건 **내** 분산). **0 에 가까우면 사건별 head 가"
         " 줄 정보가 없다** — 그 자체로 결론이다.",
         "\n## 1. 기준선 — 같은 체크포인트, 적응 임계 규칙군 (D-B27)\n",
         "| 규칙 | dev-3 pooled Score | 예측 경계 |", "|---|---|---|"]
    for k, (sc, npd) in sorted(base.items(), key=lambda kv: -kv[1][0]):
        L.append(f"| {'**' + k + '**' if k == base_best[0] else k} | {sc:.4f} | {npd} |")
    L += [f"\n**기준선 최고 = {base_best[0]} · {base_best[1][0]:.4f}**",
          "\n## 2. 스위칭 — λ 민감도 (inert 게이트 포함)\n",
          "| λ | dev-3 pooled Score | 예측 경계 |", "|---|---|---|"]
    for lm, r in rows:
        L.append(f"| {lm:g} | {r['score']:.4f} | {r['n_pred']} |")
    L += [f"\n**스위칭 최고 = λ={sw_best[0]:g} · {sw_best[1]['score']:.4f}**",
          f"\n게이트: {'🔴 INERT — λ 를 바꿔도 출력 불변 (v4.1.x 재발)' if inert else '✅ 사전이 결정을 바꾼다 (inert 아님)'}",
          "\n## 3. 판정 — D-B27 사전 선언 기준\n",
          "| 항목 | 값 |", "|---|---|",
          f"| 기준선(규칙군 최고) | {base_best[1][0]:.4f} ({base_best[0]}) |",
          f"| 스위칭(λ sweep 최고) | {sw_best[1]['score']:.4f} (λ={sw_best[0]:g}) |",
          f"| **ΔScore** | **{delta:+.4f}** |", "| 기준 | ≥ +0.010 |",
          f"| inert | {'🔴 예' if inert else '아니오'} |",
          f"\n# {'✅ 통과 — 스위칭 층 유지' if passed else '🔴 미달 → **스위칭 층 최종 종결**(D-B27, 재시도 없음)'}",
          "\n> 양쪽 모두 **최고값**을 허용한 관대한 검정이다. 그래도 못 넘으면 종결이 안전하다.",
          "\n## 4. 보조 진단 — schema reuse (§4.5 A3, 식 8 충실형)\n",
          f"기존 사건 전체에 argmax: Score **{r_reuse['score']:.4f}** (경계 {r_reuse['n_pred']}),"
          f" λ={sw_best[0]:g}. 🔴 **D-B27 상 주 판정에서 제외**(다중비교 방지) — 진단용.",
          "\n## 5. 이번에도 안 한 것 (사전 명시, D-B27)\n",
          "- §4.8 particle smoothing — 전진 local MAP(식 8)만",
          "- protagonist point process — 데이터에 `characters_present` 없음(도달 불가)",
          "- ψₑ 가 MoG 혼합가중·공분산을 결정하는 형태 — 최소형은 **평균 이동**만",
          "\n⚠️ 앵커 k=1 (S_self 0.659 = 상한) · dev-3 = 연습장 · **탐색적**, 성능 주장 아님."]
    out = EXP / f"SWITCH_THETA_E_{args.model}{args.ns}_seed{args.seed}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    (EXP / f"switch_theta_e_{args.model}{args.ns}_seed{args.seed}.json").write_text(json.dumps(
        {"tau_over_sig": tos, "n_events": cal["n_events"], "baseline": base,
         "sweep": [(lm, r) for lm, r in rows], "reuse": r_reuse,
         "delta": delta, "inert": bool(inert), "passed": bool(passed)},
        indent=2, ensure_ascii=False, default=float))
    print("\n".join(L))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

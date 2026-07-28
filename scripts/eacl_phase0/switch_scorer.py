#!/usr/bin/env python3
"""§4.4 sticky-CRP + §4.5 슬롯별 우도비 스위칭 — 정수 임계(t*=3) 대체 실험.

배경 (왜 이걸 하나)
------------------
D-B17 에서 축 분리는 통과했으나(factoring gain +0.150) 분절은 개선되지 않았다. 진단된 병목 =
**최종 경계가 5축 합의 정수 임계 하나로 압축된다**는 것(D-B16 결과: 경계 불일치의 **77.7%가
count 2↔3**). 설계 본체 `EACL/factored-slot-eism-v0_1.md` 는 애초에 정수 임계를 쓰지 않는다:

  §4.5  P(switch)/P(cont) ∝ [P(new)/κ] · Π_k p(target_k | new) / p(target_k | current)
        *"각 인자가 슬롯별 우도비다. 비율이므로 차원·스케일 불일치가 대체로 상쇄된다 —
          **ad-hoc per-slot weights 없음**, normalization 은 emergent."*
  §4.4  sticky-CRP 스위칭 사전: P(e_t|e_{t-1}) ∝ κ·1[e_t=e_{t-1}] + CRP(...)

⚠️ 선행 사고 회피 (2026-05-22 감사)
-----------------------------------
v4.1.x 에서 sticky-CRP 는 **전부 inert** 였다(38,242 turn byte-identical). 원인은 도메인도 HP 도
아니고 `_fresh_baseline_for_prev` 의 **prior-cancel 설계** — 같은 prior 항이 repeat-vs-fresh argmax
양변에 들어가 대수적으로 상쇄됐다. **메커니즘이 검정된 적이 없다.**
여기서는 비대칭 형태 `log α − log(C_prev + λ)` 를 써서 구조적으로 상쇄가 불가능하게 하고,
**λ 민감도 sweep 을 필수 산출로 박아** 사전(prior)이 실제로 결정을 바꾸는지 매번 확인한다
(`--lmda-sweep`). λ 를 바꿔도 출력이 안 변하면 그 자체가 inert 재발 신호다.

계승 근거 (`context/00-sem-paper.md` §5 결정표 — 이미 내려진 결정)
------------------------------------------------------------------
식 1 sticky-CRP **유지**(α/λ 반전) · 식 4 Bayes posterior **유지** · 식 5 likelihood 분해 **유지** ·
식 6 log-lik = −PE² **유지**(metric 은 사건 모델에 따름) · 식 8~9 local MAP **유지** ·
식 7·10~15·18~23 **폐기**. 구현은 `src/hi_ontop/scrp.py`(SEM 식 1 그대로) 재사용.

무엇을 안 하나 (정직 기록)
-------------------------
- **schema reuse 항(§4.5 A3) 미구현.** reuse 는 에피소드별 스키마 ψ_e 가 있어야 의미가 있는데
  §4.3(스키마 조건부 dynamics)이 미구현이다. 따라서 continue-vs-new **2지 선택**만 한다.
- **particle smoothing(§4.7) 미구현** — 전진 local MAP(식 8)만.
- **슬롯별 이종 분포족(§4.2) 미구현** — 전 슬롯 등방 가우시안(식 6 = −PE²). σ_k² 만 축별로 다르다.
- 재학습하지 않는다. 이미 학습된 체크포인트의 슬롯 상태·forward model 을 그대로 쓴다.

🔒 D-B16: 캘리브레이션(σ_k², onset, counts 초기화)은 **phase1 val 에서만** 한다. dev-3 gold 미사용.
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
from hi_ontop.factored_slot import AXES, FactoredSlot  # noqa: E402
from hi_ontop.scrp import sticky_crp_unnormed  # noqa: E402
from train_anchor_m4m5 import EXP, T_STAR, Corpus, batches, make_batch  # noqa: E402

MAX_K = 256  # 문서당 에피소드 상한 (비모수지만 배열 상한은 필요)

# §4.2 슬롯별 이종 분포족 — 축마다 성분 수를 다르게 준다 (원문):
#   "Place / Time — small MoG (or categorical). Low-variance; **2–3 components suffice**."
#   "Causality / Goal — **larger MoG** or VQ-categorical. Must be **multimodal** because
#    within-episode causal content jumps discretely."
#   "Protagonist — **point process** (not Gaussian) ... per-tracked-entity Bernoulli(persist)
#    + CRP-style novel-entity rate λ."
# ⚠️ 도달 불가 2건(정직 기록):
#   · protagonist point process = **활성 개체 집합**이 타깃인데 본 데이터셋은 전역 enum 3종만
#     저장해 `characters_present` 가 없다 → MoG 로 대체.
#   · causality/goal 의 ψₑ 파라미터화(혼합 가중·평균을 스키마가 결정) = §4.3 미구현 → 무조건부 MoG.
N_COMPONENTS = {0: 5, 1: 3, 2: 3, 3: 8, 4: 8}   # protagonist/space/time/causality/intentionality


@torch.no_grad()
def slot_states_and_pred(model, c: Corpus, idx: list[int], dev: str, bs: int, anchor: bool):
    """문서별 (S [T,n_slots,d], Ŝ_cont [T-1,n_slots,d]) — 학습된 f_k 의 continue 예측."""
    model.eval()
    out: dict[int, tuple] = {}
    for chunk in batches(idx, c, bs):
        z, rs, st, sm, tm, order = make_batch(c, chunk, dev, anchor)
        S = model.encode_seq_batched(z)                    # [B,T,n_slots,d]
        Sh, _ = model.predict_next(S)                      # [B,T-1,n_slots,d]
        for b, i in enumerate(order):
            n = int(c.so[i + 1] - c.so[i])
            out[i] = (S[b, :n].float().cpu().numpy(), Sh[b, :n - 1].float().cpu().numpy())
    return [out[i] for i in idx]


def calibrate(model, c: Corpus, va: list[int], dev: str, bs: int,
              density: str = "diag", fit_cap: int = 20000, seed: int = 0) -> dict:
    """§4.7 A1 — amortized onset prior q_onset^(k) 를 phase1 val 에서 적합 (dev-3 미사용).

    A1 원문: *"learned from the **first sentences of every segment**"*. 앵커 코퍼스에는 gold
    세그먼트가 없으므로 **앵커가 경계라 한 전이(count≥t\*) 직후 문장**을 세그먼트 첫 문장으로 쓴다.

    §4.6 *"Auto-normalization via likelihood (not point error) — Predict a **distribution** per slot;
    surprise = −log p"* 에 따라 **점 프로토타입이 아니라 분포**를 적합한다. SEM 식 2 의 diag
    covariance 를 따라 **대각 공분산**:
      continue : r = obs − f_k(prev)  ~  N(μ_cont,k, diag σ²_cont,k)
      new      : obs                  ~  N(μ_onset,k, diag σ²_onset,k)

    🔴 이전 구현(D-B18)의 버그: 우도비에서 정규화 상수를 "소거된다"며 버렸다. 그건 두 분포의
    분산이 같을 때만 참이다. onset 분포는 넓고(에피소드 시작은 제각각) continue 는 좁으므로
    `−(1/2)Σ_j log σ²_j` 항이 결정적이다(d=128 → 수십 단위). 그 항을 빼면 "중심점에 가깝다"가
    공짜로 이겨 ll 이 상시 양수가 된다.
    """
    packs = slot_states_and_pred(model, c, va, dev, bs, anchor=True)
    n_slots, d = model.n_slots, model.slot_dim
    res = [[] for _ in range(n_slots)]      # continue 잔차
    ons = [[] for _ in range(n_slots)]      # 세그먼트 첫 문장의 슬롯 상태
    for i, (S, Sh) in zip(va, packs):
        R = c.resets[int(c.to[i]):int(c.to[i + 1])]
        bnd = R.sum(1) >= T_STAR
        r = S[1:] - Sh                                   # [T-1, n_slots, d]
        for k in range(n_slots):
            res[k].append(r[:, k, :])
            if bnd.any():
                ons[k].append(S[1:][bnd, k, :])
    eps = 1e-6
    mu_c = np.stack([np.concatenate(res[k]).mean(0) for k in range(n_slots)])
    v_c = np.stack([np.concatenate(res[k]).var(0) + eps for k in range(n_slots)])
    mu_o = np.stack([np.concatenate(ons[k]).mean(0) if ons[k] else np.zeros(d)
                     for k in range(n_slots)])
    v_o = np.stack([np.concatenate(ons[k]).var(0) + eps if ons[k] else np.ones(d)
                    for k in range(n_slots)])
    cal = {"kind": density,
           "mu_cont": mu_c, "var_cont": v_c, "mu_onset": mu_o, "var_onset": v_o,
           "logdet_cont": np.log(v_c).sum(1), "logdet_onset": np.log(v_o).sum(1)}
    if density == "mog":
        # §4.2 — 축별 성분 수가 다른 대각 MoG. 재학습 없이 캘리브레이션 단계에서만 적합.
        from sklearn.mixture import GaussianMixture
        rs = np.random.default_rng(seed)

        def _fit(X, n):
            if len(X) > fit_cap:
                X = X[rs.choice(len(X), fit_cap, replace=False)]
            n = min(n, max(1, len(X) // 50))
            g = GaussianMixture(n_components=n, covariance_type="diag",
                                reg_covar=1e-6, max_iter=200, random_state=seed)
            g.fit(X)
            return g
        cal["gm_cont"] = [_fit(np.concatenate(res[k]), N_COMPONENTS[k]) for k in range(n_slots)]
        cal["gm_onset"] = [_fit(np.concatenate(ons[k]), N_COMPONENTS[k]) if ons[k] else None
                           for k in range(n_slots)]
        cal["n_comp"] = [g.n_components for g in cal["gm_cont"]]
    return cal


def slot_logdensities(S: np.ndarray, Sh: np.ndarray, cal: dict,
                      use_normalizer: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """전이×슬롯 로그밀도 (lp_new, lp_cont), 각 [T-1, n_slots]. 순차 루프 밖에서 벡터화."""
    T1, n_slots = Sh.shape[0], S.shape[1]
    lp_new = np.zeros((T1, n_slots))
    lp_con = np.zeros((T1, n_slots))
    for k in range(n_slots):
        obs = S[1:, k]                      # [T-1, d]
        res = obs - Sh[:, k]                # continue 잔차
        if cal["kind"] == "mog":
            lp_new[:, k] = (cal["gm_onset"][k].score_samples(obs)
                            if cal["gm_onset"][k] is not None else -1e9)
            lp_con[:, k] = cal["gm_cont"][k].score_samples(res)
        else:
            dn = obs - cal["mu_onset"][k]
            dc = res - cal["mu_cont"][k]
            lp_new[:, k] = -0.5 * (dn * dn / cal["var_onset"][k]).sum(1)
            lp_con[:, k] = -0.5 * (dc * dc / cal["var_cont"][k]).sum(1)
            if use_normalizer:
                lp_new[:, k] -= 0.5 * cal["logdet_onset"][k]
                lp_con[:, k] -= 0.5 * cal["logdet_cont"][k]
    return lp_new, lp_con


def segment_doc(S: np.ndarray, Sh: np.ndarray, cal: dict, alpha: float, lmda: float,
                beta: float, use_normalizer: bool = True) -> np.ndarray:
    """전진 local MAP (식 8) — 전이마다 continue vs new. 반환 = 경계 0/1 [T-1].

    log-ratio = [log α − log(C_prev + λ)]        <- §4.4 sticky-CRP (비대칭 = 상쇄 불가)
              + Σ_k [ log p_new,k − log p_cont,k ]   <- §4.5 슬롯별 우도비의 곱 (ad-hoc 가중 없음)
    """
    lp_new, lp_con = slot_logdensities(S, Sh, cal, use_normalizer)
    ll_all = (lp_new - lp_con).sum(1)
    T1 = Sh.shape[0]
    counts = np.zeros(MAX_K, dtype=np.float64)
    cur = 0
    counts[cur] = 1.0
    bnd = np.zeros(T1, dtype=np.int8)
    for t_ in range(T1):
        prior = sticky_crp_unnormed(counts, cur, alpha, lmda, beta)
        n_visited = int(np.count_nonzero(counts))
        p_new = prior[n_visited] if n_visited < MAX_K else 1e-12
        log_ratio = np.log(max(p_new, 1e-12)) - np.log(max(prior[cur], 1e-12)) + ll_all[t_]
        if log_ratio > 0 and n_visited < MAX_K - 1:
            bnd[t_] = 1
            cur = n_visited
            counts[cur] = 1.0
        else:
            counts[cur] += 1.0
    return bnd


def eval_dev3(model, dev3: Corpus, cal: dict, dev: str, alpha: float, lmda: float,
              beta: float, use_normalizer: bool = True) -> dict:
    idx = list(range(dev3.n_docs))
    packs = slot_states_and_pred(model, dev3, idx, dev, 4, anchor=False)
    ys, ps, per_doc = [], [], {}
    for i, (S, Sh) in zip(idx, packs):
        g = dev3.gold[int(dev3.to[i]):int(dev3.to[i + 1])].astype(np.int8)
        b = segment_doc(S, Sh, cal, alpha, lmda, beta, use_normalizer)
        y = np.concatenate([g, [0]]).tolist()
        p = np.concatenate([b, [0]]).tolist()
        ys.append(y)
        ps.append(p)
        per_doc[dev3.doc_ids[i]] = {"score": score_dialogues([y], [p])["score"],
                                    "n_pred": int(b.sum()), "n_gold": int(g.sum())}
    return {"pooled_score": score_dialogues(ys, ps)["score"],
            "mean_doc_score": float(np.mean([v["score"] for v in per_doc.values()])),
            "n_pred_total": int(sum(v["n_pred"] for v in per_doc.values())),
            "per_doc": per_doc}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="M5")
    ap.add_argument("--ns", default="_per_slot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="phase1")
    ap.add_argument("--cal-docs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=1.0)   # 00-sem-paper §5: α/λ 반전
    ap.add_argument("--lmda", type=float, default=10.0)
    ap.add_argument("--crp-beta", type=float, default=1.0, help="1.0 = SEM 식1 그대로")
    ap.add_argument("--density", default="diag", choices=["diag", "mog"],
                    help="diag = 단일 대각 가우시안 · mog = §4.2 축별 이종 MoG")
    ap.add_argument("--lmda-sweep", default="0,1,3,10,30,100",
                    help="필수 산출 — 사전이 inert 하지 않은지 확인 (2026-05-22 재발 방지)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    c, dev3 = Corpus(args.split), Corpus("dev3")
    ck = torch.load(EXP / f"{args.model}{args.ns}_seed{args.seed}.pt",
                    map_location=dev, weights_only=False)
    if ck.get("dynamics") is None:
        raise SystemExit("dynamics 없는 체크포인트 — f_k 가 없어 우도비를 계산할 수 없다.")
    model = FactoredSlot(slot_dim=ck["args"]["slot_dim"], mono=ck["mono"],
                         pos_weight=ck["pos_weight"], dynamics=ck["dynamics"],
                         pred_space=ck["args"].get("pred_space", "obs")).to(dev)
    model.load_state_dict(ck["state"])

    rng = np.random.default_rng(20260728)               # 학습과 동일 분할
    perm = rng.permutation(c.n_docs)
    va_all = perm[:max(1, int(c.n_docs * 0.05))].tolist()
    va = np.random.default_rng(args.seed).choice(
        va_all, min(args.cal_docs, len(va_all)), replace=False).tolist()
    print(f"[cal] phase1 val {len(va)}편에서 σ_k²·onset 추정 (dev-3 미사용)", flush=True)
    cal = calibrate(model, c, va, dev, args.batch, args.density, seed=args.seed)
    print(f"[cal] logdet cont = {np.round(cal['logdet_cont'],1).tolist()}\n      logdet onset = {np.round(cal['logdet_onset'],1).tolist()}", flush=True)

    rows = []
    for lm in [float(x) for x in args.lmda_sweep.split(",")]:
        r = eval_dev3(model, dev3, cal, dev, args.alpha, lm, args.crp_beta, True)
        rows.append((lm, r))
        print(f"  λ={lm:>6.1f}  Score {r['pooled_score']:.4f}  예측경계 {r['n_pred_total']:>4d}",
              flush=True)

    scores = [r["pooled_score"] for _, r in rows]
    preds = [r["n_pred_total"] for _, r in rows]
    inert = (max(scores) - min(scores) < 1e-9) and (max(preds) - min(preds) == 0)

    L = [f"# §4.4 sticky-CRP + §4.5 우도비 스위칭 — {args.model}{args.ns} seed {args.seed}\n",
         "> 정수 임계(t\\*=3) 대체 실험. **재학습 없음** — 학습된 슬롯 상태·f_k 재사용.",
         "> 캘리브레이션(σ_k²·onset)은 phase1 val 에서만. dev-3 gold 미사용(D-B16).",
         f"> α={args.alpha} · CRP β={args.crp_beta} · `scrp.sticky_crp_unnormed`(SEM 식1) 재사용.\n",
         "\n## λ 민감도 — 사전이 inert 하지 않은지 (2026-05-22 재발 방지 게이트)\n",
         "| λ | dev-3 pooled Score | 예측 경계 수 |", "|---|---|---|"]
    for lm, r in rows:
        L.append(f"| {lm:g} | {r['pooled_score']:.4f} | {r['n_pred_total']} |")
    L.append(f"\n**게이트: {'🔴 INERT — λ 를 바꿔도 출력 불변. v4.1.x 재발.' if inert else '✅ 사전이 결정을 바꾼다 (inert 아님).'}**")
    best = max(rows, key=lambda x: x[1]["pooled_score"])
    L += [f"\n## 문서별 (λ={best[0]:g}, 최고 Score 행)\n",
          "| doc | Score | 예측 | gold |", "|---|---|---|---|"]
    for d, v in best[1]["per_doc"].items():
        L.append(f"| {d} | {v['score']:.4f} | {v['n_pred']} | {v['n_gold']} |")
    L += ["\n## 미구현 (정직 기록)\n",
          "- §4.5 schema reuse 항 — §4.3(스키마 조건부 dynamics) 미구현이라 의미 없음. continue-vs-new 2지만.",
          "- §4.7 particle smoothing — 전진 local MAP(식 8)만.",
          "- §4.2 슬롯별 이종 분포족 — 전 슬롯 등방 가우시안(식 6). σ_k² 만 축별.",
          "\n⚠️ 앵커 k=1 · S_self 0.659(상한) · 탐색적 — 성능 주장 아님(D-B16)."]
    sfx = "" if args.density == "diag" else f"_{args.density}"
    out = EXP / f"SWITCH_{args.model}{args.ns}{sfx}_seed{args.seed}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    (EXP / f"switch_{args.model}{args.ns}{sfx}_seed{args.seed}.json").write_text(
        json.dumps({"lmda_sweep": [(lm, r) for lm, r in rows], "logdet_cont": cal["logdet_cont"].tolist(), "logdet_onset": cal["logdet_onset"].tolist(),
                    "inert": bool(inert)}, indent=2, default=float))
    print("\n".join(L[4:]))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

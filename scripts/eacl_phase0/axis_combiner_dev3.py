#!/usr/bin/env python3
r"""실험 ② 5축 결합 방식 — "현행 M5 가 5축을 낭비하고 있는가".

사전등록 `outputs/experiments/2026-07-31_teacher_and_combiner/PREREG.md` (실행 전 작성, 미수정).

가설: 현행 신호 `s = Σ_k P(리셋_k)` 는 **균등 합**이라 5차원이 그 자리에서 스칼라로 붕괴한다.
축별 발화율이 0.0213(시간) ~ 0.6950(인과)로 30배 이상 다른데도 같은 무게로 더한다.
⇒ 5축이 예측에 기여하려면 **결합이 학습되어야 한다**.

- **M5 체크포인트 동결. 재학습 없음.** 축별 확률까지는 현행 그대로 뽑고 결합 함수만 바꾼다.
- 결합 계수는 **앵커 val 529편**(의사-gold = 동시 리셋 ≥ 3)에서 맞춘다. 🔴 dev-3 미사용.
- 평가·임계 규칙은 M8 실험과 동일(`train_direct_boundary.score_signals`).

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/eacl_phase0/axis_combiner_dev3.py
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
import train_direct_boundary as m8  # noqa: E402
from db29_final_eval import load  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from train_anchor_m4m5 import AXES, Corpus, axis_probs  # noqa: E402

OUT = REPO / "outputs/experiments/2026-07-31_teacher_and_combiner"
NS_W3 = "_per_slot_slot_pwone"   # 🔒 현행 채택본(W3)
T_STAR = 3                       # 🔒 동결
SEEDS = (0, 1, 2)
COMBINERS = ["sum", "logistic", "zscore_sum", "logistic_pair"]


def pair_feats(P: np.ndarray) -> np.ndarray:
    """[N,5] -> [N,15]: 축 확률 5 + 축쌍 곱 10 (동시성 패턴)."""
    cols = [P] + [(P[:, i] * P[:, j])[:, None] for i in range(5) for j in range(i + 1, 5)]
    return np.concatenate(cols, 1)


def fit_combiner(kind: str, Pv: np.ndarray, yv: np.ndarray):
    """앵커 val 에서만 맞춘다. 반환 = [N,5] -> [N] 신호 함수."""
    if kind == "sum":
        return lambda P: P.sum(1)
    if kind == "zscore_sum":
        mu, sd = Pv.mean(0), Pv.std(0) + 1e-8
        return lambda P: ((P - mu) / sd).sum(1)
    if kind == "logistic":
        lr = LogisticRegression(max_iter=2000).fit(Pv, yv)
        return lambda P: lr.decision_function(P)
    if kind == "logistic_pair":
        lr = LogisticRegression(max_iter=2000).fit(pair_feats(Pv), yv)
        return lambda P: lr.decision_function(pair_feats(P))
    raise SystemExit(f"🔴 알 수 없는 결합: {kind}")


def main() -> None:
    ap = argparse.ArgumentParser(description="5축 결합 방식 비교 (M5 동결, 결합만 교체)")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument("--combiners", nargs="*", default=COMBINERS)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    c, dev3 = Corpus("phase1"), Corpus("dev3")
    _, va_docs = m8.doc_split(c.n_docs)          # 🔒 M4/M5 와 동일 분할
    G = [dev3.gold[int(dev3.to[i]):int(dev3.to[i + 1])].astype(np.int8) for i in range(dev3.n_docs)]
    yv_docs = [(c.resets[int(c.to[i]):int(c.to[i + 1])].sum(1) >= T_STAR).astype(np.int8)
               for i in va_docs]
    yv = np.concatenate(yv_docs)
    print(f"[data] 앵커 val {len(va_docs)}편 · {len(yv):,} 전이 · 의사-gold 경계율 {yv.mean():.4f} · "
          f"dev-3 gold {sum(int(g.sum()) for g in G)}", flush=True)

    results: dict = {"_meta": {"prereg": "PREREG.md 실험②", "ns": NS_W3, "t_star": T_STAR,
                               "fit_data": f"앵커 val {len(va_docs)}편 (dev-3 미사용)"}}
    per_comb: dict[str, list] = {k: [] for k in a.combiners}
    coefs: dict[str, dict] = {}
    for sd in a.seeds:
        m = load("M5", sd, dev, ns=NS_W3)[0]
        with torch.no_grad():
            Pv = np.concatenate(axis_probs(m, c, va_docs, dev, bs=48, with_anchor=False))
            Pd = axis_probs(m, dev3, list(range(dev3.n_docs)), dev, bs=4, with_anchor=False)
        print(f"[seed{sd}] 축별 val 평균확률 " +
              " ".join(f"{ax[:5]} {Pv[:, k].mean():.3f}" for k, ax in enumerate(AXES)), flush=True)
        for kind in a.combiners:
            f = fit_combiner(kind, Pv, yv)
            r = m8.score_signals([f(P) for P in Pd], G)
            r["seed"] = sd
            per_comb[kind].append(r)
            print(f"  [{kind:14s} seed{sd}] 적응 {r['adaptive']['score']:.4f} "
                  f"({r['adaptive']['n_pred']}) · 분위수 {r['percentile']['score']:.4f} "
                  f"· AUC {r['auc_pooled']:.4f}", flush=True)
            if kind == "logistic" and sd == a.seeds[0]:
                lr = LogisticRegression(max_iter=2000).fit(Pv, yv)
                coefs["logistic"] = dict(zip(AXES, [float(x) for x in lr.coef_[0]]))
                print(f"    축 가중치 {coefs['logistic']}", flush=True)
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()

    for kind, runs in per_comb.items():
        results[kind] = {"runs": runs, **m8.agg(runs)}
    results["_meta"]["logistic_coef"] = coefs.get("logistic")
    (OUT / "axis_combiner.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print("\n=== 판정 (사전등록 §② · Δ = 결합 − sum, 실질성 한계 0.06) ===")
    for kind in a.combiners:
        if kind == "sum":
            continue
        for rule in ("adaptive", "percentile"):
            d = results[kind][rule]["mean"] - results["sum"][rule]["mean"]
            print(f"  {kind:14s} {rule:11s} {results[kind][rule]['mean']:.4f} · Δ {d:+.4f}"
                  f"{'  ← 실질적' if d > 0.06 else ''}")
    print(f"\n-> {OUT/'axis_combiner.json'}")


if __name__ == "__main__":
    main()

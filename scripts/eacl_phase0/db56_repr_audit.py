#!/usr/bin/env python3
r"""D-B56 §3 — **무점수 표현 감사**. 스위칭 층 재개 게이트.

## 무엇을
D-B28 🔒 종결의 실패 진단(*"128차원 우도가 스칼라 사전을 압도"*, *"λ inert"*)이
**pw-one 계열 체크포인트에서 실제로 달라졌는지**만 본다. 성능을 재는 것이 아니다.

측정 (전부 **phase1 val**, 캘리브레이션에 안 쓴 문서에서):
| 게이트 | 산출 |
|---|---|
| ① 우도비 분산 W0 대비 ≥50% 감소 | `ll_var` (Σ_k[logp_new−logp_cont] 의 분산) |
| ② λ sweep 예측 경계 수 max/min ≥ 1.25 | `lmda_sweep_val` |
| ③ 우도비 AUC ≥ 0.7006 | `auc_ll` (+ 동일 val 의 `auc_resetsum` 동반) |

🔴 **게이트 ③의 알려진 결함(D-B56 §3 에 실행 전 선언).** 0.7006 은 D-B19 에서 **dev-3**
리셋확률 합의 AUC 다. 여기는 **phase1 val 의사-gold**(count ≥ t\*=3)라 모수가 다르다.
⇒ 같은 val 에서 `auc_resetsum` 을 **함께 재서** 동일 기준으로 비교한다. 그 값이 0.7006 과
크게 다르면 ③은 **판정 불가**로 처리하고 ①②로만 간다.

## 하지 않는 것
- **dev-3 를 열지 않는다.** 이 스크립트는 dev-3 를 아예 로드하지 않는다.
- 재학습·재구현 없음 — `switch_scorer.py` 의 함수를 그대로 import 한다.
- 새 density family·차원 사영·ψₑ·schema reuse·ALPHA 변형 없음 (D-B56 §5 고정 조건).

## 실행
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/eacl_phase0/db56_repr_audit.py \
        --ns _per_slot_slot_pwone --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
from hi_ontop.factored_slot import FactoredSlot  # noqa: E402
from switch_scorer import (EXP, T_STAR, calibrate, segment_doc,  # noqa: E402
                           slot_logdensities, slot_states_and_pred)
from train_anchor_m4m5 import Corpus, axis_probs  # noqa: E402

OUT = REPO / "outputs/experiments/2026-07-31_sem_switching_reopen/audit"


def main() -> None:
    ap = argparse.ArgumentParser(description="D-B56 표현 감사 (dev-3 미개봉)")
    ap.add_argument("--ns", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="M5")
    ap.add_argument("--split", default="phase1")
    ap.add_argument("--cal-docs", type=int, default=200)
    ap.add_argument("--audit-docs", type=int, default=120, help="감사용(캘리브레이션과 배타)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--crp-beta", type=float, default=1.0)
    ap.add_argument("--lmda-sweep", default="0,1,3,10,30,100")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    c = Corpus(a.split)                       # 🔒 dev-3 는 로드조차 하지 않는다
    ck = torch.load(EXP / f"{a.model}{a.ns}_seed{a.seed}.pt", map_location=dev,
                    weights_only=False)
    if ck.get("dynamics") is None:
        raise SystemExit(f"dynamics 없는 체크포인트: {a.model}{a.ns}_seed{a.seed}")
    model = FactoredSlot(slot_dim=ck["args"]["slot_dim"], mono=ck["mono"],
                         pos_weight=ck["pos_weight"], dynamics=ck["dynamics"],
                         pred_space=ck["args"].get("pred_space", "obs")).to(dev)
    model.load_state_dict(ck["state"])
    model.eval()

    # 🔒 switch_scorer.py:250 과 동일한 분할 규약 (학습과 같은 rng seed 20260728)
    perm = np.random.default_rng(20260728).permutation(c.n_docs)
    va_all = perm[:max(1, int(c.n_docs * 0.05))].tolist()
    va_cal = np.random.default_rng(a.seed).choice(
        va_all, min(a.cal_docs, len(va_all)), replace=False).tolist()
    rest = [i for i in va_all if i not in set(va_cal)]
    va_audit = rest[:a.audit_docs] if rest else va_cal[:a.audit_docs]
    print(f"[split] val {len(va_all)}편 → 캘리브 {len(va_cal)} · 감사 {len(va_audit)}"
          f"{' ⚠️ 겹침(rest 부족)' if not rest else ''}", flush=True)

    cal = calibrate(model, c, va_cal, dev, a.batch, "diag", seed=a.seed)
    packs = slot_states_and_pred(model, c, va_audit, dev, a.batch, anchor=True)
    with torch.no_grad():
        P = axis_probs(model, c, va_audit, dev, bs=64, with_anchor=True)

    ll_parts, y_parts, rs_parts = [], [], []
    for i, (S, Sh), Pi in zip(va_audit, packs, P):
        R = c.resets[int(c.to[i]):int(c.to[i + 1])]
        y = (R.sum(1) >= T_STAR).astype(np.int8)          # 의사-gold (앵커, 사람 gold 아님)
        lp_new, lp_con = slot_logdensities(S, Sh, cal, True)
        ll = (lp_new - lp_con).sum(1)
        n = min(len(ll), len(y), len(Pi))
        ll_parts.append(ll[:n]); y_parts.append(y[:n]); rs_parts.append(Pi[:n].sum(1))
    ll = np.concatenate(ll_parts); y = np.concatenate(y_parts); rs = np.concatenate(rs_parts)

    res = {"ns": a.ns, "seed": a.seed, "n_trans_audit": int(len(y)),
           "pos_rate": float(y.mean()),
           "ll_var": float(ll.var()), "ll_std": float(ll.std()), "ll_mean": float(ll.mean()),
           "ll_sep_gold_minus_bg": float(ll[y == 1].mean() - ll[y == 0].mean())
           if y.any() and (y == 0).any() else float("nan"),
           "auc_ll": float(roc_auc_score(y, ll)) if y.any() and (y == 0).any() else float("nan"),
           "auc_resetsum": float(roc_auc_score(y, rs)) if y.any() and (y == 0).any()
           else float("nan"),
           "logdet_cont": cal["logdet_cont"].tolist(),
           "logdet_onset": cal["logdet_onset"].tolist()}

    # 게이트 ② — λ sweep 을 **val 에서** (dev-3 아님)
    sweep = {}
    for lm in [float(x) for x in a.lmda_sweep.split(",")]:
        nb = sum(int(segment_doc(S, Sh, cal, a.alpha, lm, a.crp_beta, True).sum())
                 for (S, Sh) in packs)
        sweep[f"{lm:g}"] = nb
        print(f"  λ={lm:>6.1f}  val 예측 경계 {nb}", flush=True)
    res["lmda_sweep_val"] = sweep
    v = list(sweep.values())
    res["lmda_ratio"] = float(max(v) / max(min(v), 1))
    res["n_gold_val"] = int(y.sum())

    print(f"[audit] ll_var {res['ll_var']:.3g} · ll_std {res['ll_std']:.3g} · "
          f"AUC(ll) {res['auc_ll']:.4f} · AUC(Σp) {res['auc_resetsum']:.4f} · "
          f"λ ratio {res['lmda_ratio']:.3f}", flush=True)
    p = OUT / f"audit_{a.model}{a.ns}_seed{a.seed}.json"
    p.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> {p}", flush=True)


if __name__ == "__main__":
    main()

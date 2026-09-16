#!/usr/bin/env python3
"""예측 덤프 → 공통 지표표 (Score · BOR · purity · coverage · F(p,c) · L1).

🔒 **지표 정의를 재구현하지 않는다.** 전부 기존 동결 구현을 import 한다:
  - Score      = `hi_ontop.dts_scoring.score_dialogues` (CLAUDE.md 강제, deployable)
  - purity/coverage = `full_comparison_dev3.purity_coverage`
                 purity   = 예측 구간이 gold 구간 하나에 얼마나 순수하게 담기나 (**과소분절**에 벌점)
                 coverage = gold 구간이 예측 구간 하나에 얼마나 덮이나 (**과분절**에 벌점)
  - L1         = `full_comparison_dev3.lift_l1` = P(경계|예측) / P(경계) (표준 lift)
  - BOR        = 예측 경계 수 / gold 경계 수 (1.0 = 밀도 일치, >1 과분절)
  - F(p,c)     = purity 와 coverage 의 조화평균

입력 = `outputs/experiments/2026-07-29_baseline_compare/preds/{방법}_{데이터셋}.json`
(baseline_compare 가 전이별 0/1 을 덤프한 **원자료**. 집계는 언제든 재계산된다.)

    .venv/bin/python scripts/eacl_phase0/preds_table.py --datasets dev3,dts_full_tiage
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
from full_comparison_dev3 import lift_l1, purity_coverage      # noqa: E402  🔒 동결 정의
from hi_ontop.dts_scoring import score_dialogues               # noqa: E402

PRED = REPO / "outputs/experiments/2026-07-29_baseline_compare/preds"
OUT = REPO / "outputs/experiments/2026-07-29_baseline_compare"
ORDER = ["M5", "M5-adaptive", "M4", "M4-adaptive", "M1-CosDrift", "M1-CosDrift-adaptive",
         "TextTiling", "GraphSeg", "GreedySeg",
         "CSM", "CSM-devbest", "CSM-offverified", "RoBERTa", "Membox-TopicLoom",
         # LLM 직접질의 — 예측 덤프가 있는 것만 자동 편입(없으면 조용히 건너뛴다)
         "B3event-online-r10_gpt-4.1", "B3event-online-r10_gpt-5",
         "B3event-online-r10_gpt-oss-120b", "B3event-binary-stateful_gpt-5",
         "SeCom-binary_gpt-4o", "SeCom-binary-stateful_gpt-4o",
         "SeCom-binary-stateful_gpt-5-mini",
         # 밀도 하한 (2026-08-03) — 문서별 경계 수를 M5 와 일치시킨 통제군.
         # `M5 − 이 하한` = 위치 선택의 기여. 생성 = EACL/methods/trivial/run_nes.py
         "FixedInterval", "RandomRate",
         # EM-LLM SM 변형 (2026-08-03) — 이동 윈도우 적응 임계 μ+γσ, γ=1.1 고정, SM+ 제외.
         # 생성 = EACL/methods/emllm/run_sm_dev3.py
         "EM-LLM-SM"]


def score_of(P, G) -> dict:
    ys = [np.concatenate([np.asarray(g, np.int8), [0]]).tolist() for g in G]
    ps = [np.concatenate([np.asarray(p[:len(g)], np.int8), [0]]).tolist() for p, g in zip(P, G)]
    return score_dialogues(ys, ps)


def floors(G) -> dict[str, float]:
    z = [np.zeros(len(g), np.int8) for g in G]
    o = [np.ones(len(g), np.int8) for g in G]
    return {"무경계 하한": score_of(z, G)["score"], "전부경계 하한": score_of(o, G)["score"]}


def rows_for(ds: str) -> tuple[list[dict], int, dict]:
    from baseline_compare import load_dataset                  # noqa: PLC0415
    data = load_dataset(ds)
    docs = [d for d, _, _ in data]
    G = [g for _, _, g in data]
    n_gold = int(sum(int(g.sum()) for g in G))
    res = []
    for name in ORDER:
        fp = PRED / f"{name}_{ds}.json"
        # 시험지 gold 변형(`test3_goldauthor`)은 **예측이 같다** — gold 만 다르다.
        # 예측을 다시 만들지 않고 `test3` 덤프를 그대로 재채점한다.
        if not fp.exists() and ds.startswith("test3"):
            fp = PRED / f"{name}_test3.json"
        if not fp.exists():
            continue
        d = json.loads(fp.read_text())
        if any(x not in d for x in docs):
            continue
        P = [np.asarray(d[x], np.int8) for x in docs]
        s = score_of(P, G)
        pur, cov = purity_coverage(P, G)
        n_pred = int(sum(int(np.asarray(p[:len(g)]).sum()) for p, g in zip(P, G)))
        res.append({"method": name, "score": s["score"], "pk": s["pk"], "wd": s["wd"],
                    "f1": s["f1"], "n_pred": n_pred, "bor": n_pred / max(n_gold, 1),
                    "purity": pur, "coverage": cov,
                    "f_pc": 2 * pur * cov / (pur + cov) if (pur + cov) else 0.0,
                    "L1": lift_l1(P, G)})
    res.sort(key=lambda r: -r["score"])
    return res, n_gold, floors(G)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="dev3,dts_full_tiage,dts_full_dialseg711,dts_full_superseg")
    args = ap.parse_args()
    allr = {}
    for ds in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        res, n_gold, fl = rows_for(ds)
        if not res:
            print(f"[{ds}] 예측 없음 — 건너뜀")
            continue
        allr[ds] = {"n_gold": n_gold, "floors": fl, "rows": res}
        print(f"\n[{ds}] gold {n_gold} · " + " · ".join(f"{k} {v:.4f}" for k, v in fl.items()))
        print(f"{'방법':18s} {'Score':>7s} {'BOR':>6s} {'purity':>7s} {'cover':>6s} "
              f"{'F(p,c)':>7s} {'L1':>5s}")
        for r in res:
            print(f"{r['method']:18s} {r['score']:7.4f} {r['bor']:6.2f} {r['purity']:7.3f} "
                  f"{r['coverage']:6.3f} {r['f_pc']:7.3f} {r['L1']:5.2f}")
    (OUT / "preds_table.json").write_text(json.dumps(allr, indent=1, ensure_ascii=False,
                                                     default=float))
    print(f"\n-> {OUT / 'preds_table.json'}")


if __name__ == "__main__":
    main()

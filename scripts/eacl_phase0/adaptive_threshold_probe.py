"""M3 coordinated-count 경계 규칙: 고정 t*=3 vs 적응 임계 비교 (탐색적).

동기(사용자 2026-07-14): 고정 count≥t*는 문서 길이에 무관해 긴 문서에서 과예측(D-M5-b
작동점 결함). "개수가 평균대비 팍 튀는 지점을 적응적으로 찾자"는 제안 시험.

규칙 3종 (메인=DTS Score, deployable):
  A. fixed      : count ≥ 3 (동결 t*)
  B. global σ   : count ≥ mean_doc + c·std_doc         (문서 분포에 적응)
  C. local spike: count_t − rollmean(±W) ≥ margin       (국소 평균 대비 스파이크)

HP(c / W,margin)는 **Pieman(dev)에서만 argmax-Score 캘리브** → Tunnel/Monkey는 그 값
그대로 적용(탐색적, test 봉인 취지 유지). LLM·인코더 불필요(count 스트림만).

usage: uv run python scripts/eacl_phase0/adaptive_threshold_probe.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO / "src"))
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402

AX = ["protagonist_perspective", "space", "time", "causality", "intentionality"]
DOCS = ["pieman", "tunnel", "monkey"]


def load(doc):
    rm = [json.loads(l) for l in open(REPO / f"outputs/eacl_phase0/{doc}_michelmann_s5/reset_matrix.jsonl")]
    rm.sort(key=lambda r: r["transition"])
    cnt = np.array([sum(int(r[a]) for a in AX) for r in rm], float)
    tx = [r["transition"] for r in rm]
    gold = [r["transition"] for r in rm if r["boundary"]]
    return tx, cnt, gold


def endturn(n, ts):
    y = [0] * n
    for t in ts:
        y[t - 1] = 1
    return y


def score(tx, gold, pred):
    n = len(tx) + 1
    return score_dialogues([endturn(n, gold)], [endturn(n, pred)])["score"]


def pred_fixed(tx, cnt, t=3):
    return [tx[i] for i in range(len(cnt)) if cnt[i] >= t]


def pred_global(tx, cnt, c):
    thr = cnt.mean() + c * cnt.std()
    return [tx[i] for i in range(len(cnt)) if cnt[i] >= thr]


def pred_local(tx, cnt, W, margin):
    n = len(cnt)
    pred = []
    for i in range(n):
        lo, hi = max(0, i - W), min(n, i + W + 1)
        local = np.concatenate([cnt[lo:i], cnt[i + 1:hi]])
        lm = local.mean() if len(local) else 0.0
        if cnt[i] - lm >= margin and cnt[i] > 0:
            pred.append(tx[i])
    return pred


def main():
    data = {d: load(d) for d in DOCS}
    # --- calibrate on Pieman (dev) ---
    ptx, pcnt, pgold = data["pieman"]
    best_c = max(np.arange(0.0, 2.01, 0.1),
                 key=lambda c: score(ptx, pgold, pred_global(ptx, pcnt, c)))
    best_wm = max(((W, m) for W in (3, 5, 8, 12) for m in np.arange(0.5, 3.01, 0.5)),
                  key=lambda wm: score(ptx, pgold, pred_local(ptx, pcnt, wm[0], wm[1])))
    print(f"[dev=Pieman 캘리브] global c*={best_c:.1f} | local W*={best_wm[0]} margin*={best_wm[1]:.1f}")
    print(f"\n{'doc':8s} {'N/K':>9s} | {'A.fixed t*=3':>12s} {'B.global σ':>11s} {'C.local spike':>13s}")
    for d in DOCS:
        tx, cnt, gold = data[d]
        n, K = len(tx), len(gold)
        pa = pred_fixed(tx, cnt); pb = pred_global(tx, cnt, best_c); pc = pred_local(tx, cnt, *best_wm)
        sa, sb, sc = score(tx, gold, pa), score(tx, gold, pb), score(tx, gold, pc)
        tag = "(dev)" if d == "pieman" else "(test*)"
        print(f"{d:8s} {n:>4d}/{K:<4d} | {sa:>7.3f}(N={len(pa):>2d}) {sb:>6.3f}(N={len(pb):>2d}) "
              f"{sc:>6.3f}(N={len(pc):>2d}) {tag}")
    print("\n* Tunnel/Monkey는 dev 캘리브값 적용한 탐색적 결과(개봉된 test, 재봉인 아님).")
    print("  N = 각 규칙이 예측한 경계 개수(deployable, 개수 자가결정).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
r"""dev-3 전체 비교표 — 모든 방법에 **밀도 매칭 무작위 대조(Δ무작위)** 를 붙인다.

오늘의 교훈: **Score 는 밀도에 끌려간다.** 밀도가 다른 방법을 Score 만으로 비교하면 순위가
왜곡된다. 그래서 이 표는 모든 행에 Δ무작위를 넣는다 — 없는 행은 만들지 않는다.

🔒 Δ무작위 정의 (프로젝트 공용, 여기서 확정)
  Δ무작위 = Score(방법) − mean(Score(무작위 200회))
  · 무작위 = **문서별로 그 방법이 찍은 개수를 그대로 유지**한 균등 무작위 배치
  · seed 20260731 고정 · Score = deployable DTS Score, dev-3 3편 **pooled**
  · 백분위 = 무작위 200회 분포에서 방법 Score 가 차지하는 위치

예측 출처
  · 우리 방법(M5 4조건·M4·M8·M1) = 체크포인트/신호에서 **적응 임계(online_nf, c=1.0)** 로 그 자리 산출
  · 외부 baseline·LLM = `2026-07-29_baseline_compare/preds/*_dev3.json` (각자의 count-free 규칙)
  · B1 = `outputs/eacl_phase0/b1_bayes/qwen36_35b` 신호에 동결 readout(newsent_max) + 적응 c=1.0

    .venv/bin/python scripts/eacl_phase0/full_comparison_dev3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
from b1_deployable import b1_newsent_max  # noqa: E402
from db29_final_eval import load as load_m5  # noqa: E402
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from train_anchor_m4m5 import Corpus, axis_probs  # noqa: E402
import train_direct_boundary as m8mod  # noqa: E402

P0 = REPO / "outputs/eacl_phase0"
PRED = REPO / "outputs/experiments/2026-07-29_baseline_compare/preds"
OUT = REPO / "outputs/experiments/2026-07-31_full_comparison"
DOCS = ("pieman", "tunnel", "monkey")
N_RAND, RAND_SEED = 200, 20260731


def gold_vectors():
    G = []
    for d in DOCS:
        rm = sorted((json.loads(l) for l in
                     (P0 / f"{d}_michelmann_master.jsonl").read_text().splitlines() if l.strip()),
                    key=lambda r: r["transition"])
        G.append(np.asarray([int(r["boundary"]) for r in rm], np.int8))
    return G


def score_of(P, G):
    ys = [np.concatenate([g, [0]]).tolist() for g in G]
    ps = [np.concatenate([np.asarray(p[:len(g)], np.int8), [0]]).tolist() for p, g in zip(P, G)]
    return float(score_dialogues(ys, ps)["score"])


def segments(v: np.ndarray) -> list[tuple[int, int]]:
    """경계 벡터(끝-turn 규약: 1 = 그 문장이 세그먼트의 **마지막**) -> 문장 구간 목록."""
    n = len(v) + 1                      # 문장 수 = 전이 수 + 1
    cuts = [i + 1 for i in np.flatnonzero(np.asarray(v, np.int8))] + [n]
    segs, prev = [], 0
    for c in cuts:
        if c > prev:
            segs.append((prev, c))
            prev = c
    return segs


def purity_coverage(P, G) -> tuple[float, float]:
    """purity = 예측 구간이 gold 구간 하나에 얼마나 순수하게 담기나 (과소분절에 벌점)
       coverage = gold 구간이 예측 구간 하나에 얼마나 덮이나 (과분절에 벌점).
       문장 수로 가중해 3편 pooled."""
    num_p = num_c = tot = 0
    for p, g in zip(P, G):
        p = np.asarray(p[:len(g)], np.int8)
        ps, gs = segments(p), segments(np.asarray(g, np.int8))
        n = len(g) + 1
        tot += n
        for a, b in ps:
            num_p += max(min(b, y) - max(a, x) for x, y in gs if min(b, y) > max(a, x))
        for a, b in gs:
            num_c += max(min(b, y) - max(a, x) for x, y in ps if min(b, y) > max(a, x))
    return num_p / tot, num_c / tot


def lift_l1(P, G) -> float:
    """L1 = P(경계|예측) / P(경계).  (정의 통일: 표준 lift)"""
    pp = np.concatenate([np.asarray(p[:len(g)], np.int8) for p, g in zip(P, G)])
    gg = np.concatenate([np.asarray(g, np.int8) for g in G])
    return float(gg[pp == 1].mean() / gg.mean()) if pp.sum() else float("nan")


def delta_random(P, G, n=N_RAND, seed=RAND_SEED):
    rg = np.random.default_rng(seed)
    obs = score_of(P, G)
    draws = []
    for _ in range(n):
        rp = []
        for p, g in zip(P, G):
            v = np.zeros(len(g), np.int8)
            k = int(np.asarray(p[:len(g)]).sum())
            if k:
                v[rg.choice(len(g), size=min(k, len(g)), replace=False)] = 1
            rp.append(v)
        draws.append(score_of(rp, G))
    draws = np.asarray(draws)
    return obs, float(draws.mean()), obs - draws.mean(), float((draws < obs).mean() * 100)


def from_preds(name: str, G):
    """baseline_compare 저장 예측(문서별 0/1 리스트)."""
    p = PRED / f"{name}_dev3.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return [np.asarray(d[doc], np.int8) for doc in DOCS]


def adaptive(sig_list, G):
    return [m8mod.cuts_adaptive(s)[:len(g)] for s, g in zip(sig_list, G)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    G = gold_vectors()
    n_gold = sum(int(g.sum()) for g in G)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dev3 = Corpus("dev3")
    rows: list[tuple[str, str, list]] = []

    # ── 우리 방법 (적응 임계로 그 자리 산출) ──────────────────────────────
    M5_NS = [("M5 (인과 동결 × 의도 I2) — 현행 디폴트", "_phase1_i2_per_slot_slot_pwone"),
             ("M5 (동결 × 동결)", "_per_slot_slot_pwone"),
             ("M5 (F3 × I2)", "_phase1_aspcomp_i2_per_slot_slot_pwone"),
             ("M5 (F3 × 동결)", "_phase1_aspcomp_per_slot_slot_pwone"),
             ("M5 (F4 × 동결)", "_phase1_nonong_per_slot_slot_pwone"),
             ("M5 (F4 × I2)", "_phase1_nonong_i2_per_slot_slot_pwone"),
             ("M4-MonoAnchor", None)]
    for nm, ns in M5_NS:
        tag, seeds = ("M4", (0, 1, 2)) if ns is None else ("M5", (0, 1, 2))
        ns = "_per_slot_slot" if ns is None else ns
        acc = []
        for sd in seeds:
            ck = REPO / f"outputs/experiments/m4m5_phase1_anchor/{tag}{ns}_seed{sd}.pt"
            if not ck.exists():
                continue
            m = load_m5(tag, sd, dev, ns=ns)[0]
            with torch.no_grad():
                Pp = axis_probs(m, dev3, list(range(dev3.n_docs)), dev, bs=4, with_anchor=False)
            acc.append(adaptive([p.sum(1) for p in Pp], G))
            del m
            if dev == "cuda":
                torch.cuda.empty_cache()
        if acc:
            rows.append((nm, "우리", acc))

    # M8 (두 변종, 3시드)
    for var in ("ctx", "pair"):
        acc = []
        for sd in (0, 1, 2):
            ck = REPO / f"outputs/experiments/2026-07-31_m8_direct_boundary/M8_{var}_seed{sd}.pt"
            if not ck.exists():
                continue
            st = torch.load(ck, map_location=dev, weights_only=False)
            mdl = m8mod.DirectBoundaryHead(768 * (5 if var == "ctx" else 4)).to(dev)
            mdl.load_state_dict(st["state"])
            acc.append(adaptive(m8mod.m8_dev3_signal(mdl, var, dev3, dev), G))
            del mdl
        if acc:
            rows.append((f"M8-DirectBoundary `{var}`", "우리·하한", acc))

    # M1-CosDrift (delta_eff 스트림 + 적응 임계)
    sig = []
    for d in DOCS:
        rm = sorted((json.loads(l) for l in
                     (P0 / f"{d}_michelmann_s5_q36_35b_v31/reset_matrix.jsonl").read_text()
                     .splitlines() if l.strip()), key=lambda r: r["transition"])
        sig.append(np.asarray([float(r["idm_score"]) for r in rm]))
    rows.append(("M1-CosDrift", "우리·규칙", [adaptive(sig, G)]))

    # ── B1 (동결 readout + 적응 c=1.0) ────────────────────────────────────
    b1sig = []
    for d in DOCS:
        w = [json.loads(l) for l in
             (P0 / f"b1_bayes/qwen36_35b/{d}_words.jsonl").read_text().splitlines() if l.strip()]
        b1sig.append(np.asarray(b1_newsent_max(w, len(G[DOCS.index(d)]))))
    rows.append(("B1-BayesSurprise (Qwen 35B)", "스칼라", [adaptive(b1sig, G)]))

    # ── 외부 baseline · LLM (저장 예측) ───────────────────────────────────
    EXT = [("B3event-online-r10_gpt-4.1", "GPT-4.1 스트리밍", "LLM"),
           ("B3event-online-r10_gpt-5", "GPT-5 스트리밍", "LLM"),
           ("B3event-online-r10_gpt-oss-120b", "gpt-oss-120b 스트리밍", "LLM"),
           ("SeCom-binary-stateful_gpt-4o", "SeCom-binary-stateful (gpt-4o)", "LLM"),
           ("SeCom-binary_gpt-4o", "SeCom-binary 무상태 (gpt-4o)", "LLM"),
           ("TextTiling", "TextTiling", "규칙"),
           ("CSM", "CSM (우리 학습)", "규칙"),
           ("GraphSeg", "GraphSeg", "규칙"),
           ("GreedySeg", "GreedySeg", "규칙"),
           ("RoBERTa", "RoBERTa (SuperDialseg)", "supervised")]
    for key, nm, kind in EXT:
        P = from_preds(key, G)
        if P is not None:
            rows.append((nm, kind, [P]))

    rows.append(("무경계 하한", "자명", [[np.zeros(len(g), np.int8) for g in G]]))

    # ── 집계 ──────────────────────────────────────────────────────────────
    res = []
    for nm, kind, accs in rows:
        sc, rnd, dlt, pct, npr, pur, cov, lf = [], [], [], [], [], [], [], []
        for P in accs:
            o, r, d, q = delta_random(P, G)
            sc.append(o); rnd.append(r); dlt.append(d); pct.append(q)
            npr.append(sum(int(np.asarray(p[:len(g)]).sum()) for p, g in zip(P, G)))
            a, b = purity_coverage(P, G)
            pur.append(a); cov.append(b); lf.append(lift_l1(P, G))
        res.append({"method": nm, "kind": kind, "n_runs": len(accs),
                    "score": float(np.mean(sc)), "score_sd": float(np.std(sc)),
                    "random": float(np.mean(rnd)), "delta_random": float(np.mean(dlt)),
                    "pct": float(np.mean(pct)), "n_pred": float(np.mean(npr)),
                    "bor": float(np.mean(npr)) / n_gold,
                    "purity": float(np.mean(pur)), "coverage": float(np.mean(cov)),
                    "f_pc": float(2 * np.mean(pur) * np.mean(cov) / (np.mean(pur) + np.mean(cov))),
                    "L1": float(np.nanmean(lf))})
    res.sort(key=lambda r: -r["delta_random"])

    print(f"dev-3 gold {n_gold} · 무작위 {N_RAND}회 seed {RAND_SEED} · 3편 pooled\n")
    print(f"{'방법':32s} {'Score':>7s} {'Δ무작위':>8s} {'BOR':>5s} {'purity':>7s} {'cover':>6s} "
          f"{'F(p,c)':>6s} {'L1':>5s}")
    for r in res:
        print(f"{r['method']:32s} {r['score']:7.4f} {r['delta_random']:+8.4f} {r['bor']:5.2f} "
              f"{r['purity']:7.3f} {r['coverage']:6.3f} {r['f_pc']:6.3f} {r['L1']:5.2f}")
    (OUT / "full_comparison.json").write_text(
        json.dumps({"n_gold": n_gold, "n_rand": N_RAND, "seed": RAND_SEED, "rows": res},
                   indent=2, ensure_ascii=False))
    print(f"\n-> {OUT/'full_comparison.json'}")


if __name__ == "__main__":
    main()

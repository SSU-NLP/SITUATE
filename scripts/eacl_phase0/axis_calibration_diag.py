#!/usr/bin/env python3
r"""학습 전 공짜 진단 4종 — dev-3, API 0.

1. **통일된 lift 표** — 두 정의를 나란히. 지금까지 두 정의가 섞여 인용됐다.
   L1 = P(경계|리셋)/P(경계)              (표준 lift, 오늘 대결에서 쓴 것)
   L2 = P(리셋|경계)/P(리셋|비경계)        (D-B23 이 쓴 우도비)
2. **밀도 매칭 무작위 대조** — 후보와 **같은 문서·같은 개수**로 무작위 분절(시드 10개).
   `Score − Score_random` 이 진짜 기여다. 0 에 가까우면 그 규칙은 밀도만 맞춘 것이다.
3. **F12 lift 불일치 해소** — D-B23 의 1.37/1.31/2.09 와 오늘의 3.35 가 같은 양인지.
   strict/loose × L1/L2 를 전부 낸다.
4. **5축 단독 캘리브레이션** — 작동하는 축이 단독으로 얼마를 받는지. 지금까지 이 표가 없었다.

    .venv/bin/python scripts/eacl_phase0/axis_calibration_diag.py
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
from causality_axis_only_bakeoff import base_labels, hist_labels  # noqa: E402
from causality_bakeoff import DOCS, REPS  # noqa: E402
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402

OUT = REPO / "outputs/experiments/2026-07-31_axis_calibration"
N_RAND = 10          # 밀도 매칭 무작위 시드 수
RAND_SEED = 20260731
AXES5 = ("protagonist_present", "space", "time", "causality", "intentionality")


def lifts(gold: np.ndarray, f: np.ndarray) -> tuple[float, float]:
    """L1 = P(경계|리셋)/P(경계) · L2 = P(리셋|경계)/P(리셋|비경계)."""
    l1 = float(gold[f == 1].mean() / gold.mean()) if f.sum() and gold.mean() else float("nan")
    d = f[gold == 0].mean()
    l2 = float(f[gold == 1].mean() / d) if d > 0 and gold.sum() else float("nan")
    return l1, l2


def score_of(preds: list[np.ndarray], golds: list[np.ndarray]) -> float:
    ys = [np.concatenate([g, [0]]).tolist() for g in golds]
    ps = [np.concatenate([p[:len(g)], [0]]).tolist() for p, g in zip(preds, golds)]
    return float(score_dialogues(ys, ps)["score"])


def random_matched(preds: list[np.ndarray], golds: list[np.ndarray], n_rand: int) -> tuple:
    """문서별 예측 개수를 그대로 유지한 무작위 분절 n_rand 회."""
    rng = np.random.default_rng(RAND_SEED)
    out = []
    for _ in range(n_rand):
        rp = []
        for p in preds:
            k = int(p.sum())
            v = np.zeros(len(p), np.int8)
            if k:
                v[rng.choice(len(p), size=min(k, len(p)), replace=False)] = 1
            rp.append(v)
        out.append(score_of(rp, golds))
    return float(np.mean(out)), float(np.std(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="*", default=list(REPS))
    ap.add_argument("--n-rand", type=int, default=N_RAND)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # 라벨 적재 (반복별)
    per_rep = []
    for rep in a.reps:
        L = []
        for doc in DOCS:
            b = base_labels(doc, rep)
            b.update(hist_labels(doc, b["tr"]))
            b["_zero"] = np.zeros_like(b["gold"])
            b["count3"] = None  # 아래에서 채움
            L.append(b)
        per_rep.append(L)

    # 5축 count>=3 (기준 규칙) 재구성
    for L, rep in zip(per_rep, a.reps):
        for b, doc in zip(L, DOCS):
            d = REPO / f"outputs/eacl_phase0/{doc}_michelmann_s5_{rep}/reset_matrix.jsonl"
            rm = sorted((json.loads(l) for l in d.read_text().splitlines() if l.strip()),
                        key=lambda r: r["transition"])
            M = np.stack([np.asarray([r[a_] for r in rm], np.int8) for a_ in AXES5], 1)
            b["count3"] = (M.sum(1) >= 3).astype(np.int8)
            for a_ in AXES5:
                b[f"axis_{a_}"] = np.asarray([r[a_] for r in rm], np.int8)

    CANDS = [
        ("F0 동결 인과축", "frozen"), ("F3 completive", "comp"), ("F4 ≠ongoing", "nonong"),
        ("F12 사슬 strict", "chain_strict"), ("F11 사슬 loose", "chain_loose"),
        ("F8 문두 표지", "open"), ("F2 inceptive", "inc"), ("F6 aspect 전이", "asp_trans"),
        ("F9 반사실 창", "cf_window"), ("F10 의도 사슬", "chainrole"),
        ("[참조] 5축 count≥3", "count3"),
        *[(f"[5축] {a_}", f"axis_{a_}") for a_ in AXES5],
    ]

    res: dict = {"n_rand": a.n_rand, "reps": a.reps, "candidates": {}}
    print(f"dev-3 gold 66 · 반복 {len(a.reps)} · 무작위 대조 {a.n_rand} 시드\n")
    print(f"{'후보':22s} {'Score':>7s} {'무작위':>7s} {'Δ무작위':>8s} {'예측':>5s} {'발화율':>7s} "
          f"{'L1':>5s} {'L2':>5s}   문서별 L1 (pie/tun/mon)")
    for name, key in CANDS:
        sc, rnd_mu, rnd_sd, npred, fire = [], [], [], [], []
        l1s, l2s, per_doc_l1 = [], [], {d: [] for d in DOCS}
        for L in per_rep:
            if any(key not in b or b[key] is None for b in L):
                break
            golds = [b["gold"] for b in L]
            preds = [b[key] for b in L]
            sc.append(score_of(preds, golds))
            mu, sd = random_matched(preds, golds, a.n_rand)
            rnd_mu.append(mu)
            rnd_sd.append(sd)
            npred.append(sum(int(p.sum()) for p in preds))
            fire.append(float(np.mean([p.mean() for p in preds])))
            g_all = np.concatenate(golds)
            f_all = np.concatenate(preds)
            l1, l2 = lifts(g_all, f_all)
            l1s.append(l1)
            l2s.append(l2)
            for b, d in zip(L, DOCS):
                per_doc_l1[d].append(lifts(b["gold"], b[key])[0])
        else:
            v = {"score": float(np.mean(sc)), "score_sd": float(np.std(sc)),
                 "random": float(np.mean(rnd_mu)), "random_sd": float(np.mean(rnd_sd)),
                 "delta_random": float(np.mean(sc) - np.mean(rnd_mu)),
                 "n_pred": float(np.mean(npred)), "fire": float(np.mean(fire)),
                 "L1_pooled": float(np.nanmean(l1s)), "L2_pooled": float(np.nanmean(l2s)),
                 "L1_per_doc": {d: float(np.nanmean(x)) for d, x in per_doc_l1.items()}}
            res["candidates"][name] = v
            print(f"{name:22s} {v['score']:7.4f} {v['random']:7.4f} {v['delta_random']:+8.4f} "
                  f"{v['n_pred']:5.0f} {v['fire']:7.4f} {v['L1_pooled']:5.2f} {v['L2_pooled']:5.2f}"
                  f"   {v['L1_per_doc']['pieman']:.2f}/{v['L1_per_doc']['tunnel']:.2f}/"
                  f"{v['L1_per_doc']['monkey']:.2f}")
            continue
        print(f"{name:22s} — 라벨 없음")

    # ── F12 lift 불일치 해소 ─────────────────────────────────────────────────
    print("\n=== F12 lift 불일치 해소 (D-B23 공표 1.37/1.31/2.09) ===")
    print(f"  {'규칙':16s} {'정의':34s} pieman tunnel monkey")
    rec = {}
    L = per_rep[0]
    for key, label in (("chain_strict", "strict(양쪽 non-none)"), ("chain_loose", "loose(단순 변화)")):
        for which, dl in ((0, "L1 = P(경계|리셋)/P(경계)"),
                          (1, "L2 = P(리셋|경계)/P(리셋|비경계) ← D-B23")):
            vals = [lifts(b["gold"], b[key])[which] for b in L]
            rec[f"{key}_{'L1' if which == 0 else 'L2'}"] = [float(x) for x in vals]
            print(f"  {label:16s} {dl:34s} " + " ".join(f"{v:6.2f}" for v in vals))
    res["f12_reconciliation"] = rec

    (OUT / "calibration.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n-> {OUT/'calibration.json'}")


if __name__ == "__main__":
    main()

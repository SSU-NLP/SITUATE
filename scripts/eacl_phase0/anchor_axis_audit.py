#!/usr/bin/env python3
r"""앵커 축 감사 — 리뷰 지적 1·4·5번을 한 번의 스캔으로 확정한다.

1. **주인공 축 정의 불일치**: docstring 은 D-B8 로 `protagonist = perspective 단독`이라 하는데
   코드(`llm_s5_states.py:763`)는 `int(pc or pp)` = charset OR perspective 다.
   → 전 행에서 `protagonist == protagonist_perspective` 가 성립하는지, 각 후보 컬럼의 발화율,
     그리고 **어느 컬럼을 쓰느냐에 따라 경계율(count ≥ 3)이 얼마나 달라지는지** 를 낸다.
4. **축 간 상관 행렬(5×5)**: 동시 리셋 ≥3 은 축이 독립 증거라는 전제 위에 있다. 상관이 높으면
   "3개 축"이 3개의 증거가 아니다. 문서에 이 표가 없다.
5. **인과 제외 4축 ≥2 규칙**: 인과가 0.695 로 상시 발화하면 `≥3` 은 사실상 `인과 + 나머지 2` =
   `4축 중 ≥2` 다. 두 규칙의 경계 집합이 얼마나 겹치는지 직접 잰다.

읽기 전용. 앵커를 수정하지 않는다.

    .venv/bin/python scripts/eacl_phase0/anchor_axis_audit.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RM = REPO / "outputs/eacl_phase0/phase1_all_v31/reset_matrix.jsonl"
OUT = REPO / "outputs/experiments/2026-07-31_anchor_axis_audit"
COLS = ["protagonist_charset", "protagonist_perspective", "protagonist", "protagonist_present",
        "space", "time", "causality", "intentionality", "causal_marker"]
PROT_CANDIDATES = ["protagonist", "protagonist_perspective", "protagonist_charset",
                   "protagonist_present"]
REST = ["space", "time", "causality", "intentionality"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset-matrix", type=Path, default=DEFAULT_RM)
    ap.add_argument("--limit", type=int, default=0, help="스모크용 행 수 제한")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows, docs = [], []
    with a.reset_matrix.open() as f:
        for i, line in enumerate(f):
            if a.limit and i >= a.limit:
                break
            r = json.loads(line)
            rows.append([r.get(c, 0) for c in COLS])
            docs.append(r["doc_id"])
    M = np.asarray(rows, np.int8)
    doc_ids = np.asarray(docs)
    print(f"[data] {len(M):,} 전이 · {len(set(docs)):,} 문서 · {a.reset_matrix}", flush=True)
    ix = {c: k for k, c in enumerate(COLS)}

    res: dict = {"n_trans": int(len(M)), "n_docs": int(len(set(docs))),
                 "source": str(a.reset_matrix)}

    # ── 1. 주인공 축 정의 ────────────────────────────────────────────────────
    p, pp, pc = M[:, ix["protagonist"]], M[:, ix["protagonist_perspective"]], \
        M[:, ix["protagonist_charset"]]
    eq_pp = int((p == pp).sum())
    res["protagonist_definition"] = {
        "equals_perspective_rows": eq_pp, "equals_perspective_frac": float(eq_pp / len(M)),
        "equals_charset_or_perspective_frac": float(((pc | pp) == p).mean()),
        "rates": {c: float(M[:, ix[c]].mean()) for c in PROT_CANDIDATES},
    }
    print("\n=== 1. 주인공 축 정의 ===")
    print(f"  protagonist == protagonist_perspective : {eq_pp:,}/{len(M):,} "
          f"= {eq_pp/len(M):.4f}")
    print(f"  protagonist == (charset OR perspective): {((pc|pp)==p).mean():.4f}")
    for c in PROT_CANDIDATES:
        print(f"  발화율 {c:24s} {M[:, ix[c]].mean():.4f}")

    # 주인공 컬럼 선택이 경계율(count≥3)에 미치는 영향
    res["boundary_rate_by_protagonist_col"] = {}
    print("\n  경계율(동시 리셋 ≥3) — 주인공 컬럼만 바꿔가며")
    for c in PROT_CANDIDATES:
        S = M[:, [ix[c]] + [ix[x] for x in REST]].sum(1)
        rates = {f">={t}": float((S >= t).mean()) for t in (2, 3, 4)}
        res["boundary_rate_by_protagonist_col"][c] = rates
        print(f"    {c:24s} " + " · ".join(f"{k} {v:.4f}" for k, v in rates.items()))

    # ── 4. 축 간 상관 (동결 5축 = 학습에 실제로 쓰인 protagonist_present 기준) ──
    used = ["protagonist_present"] + REST          # build_anchor_dataset.py:31 이 쓰는 조합
    A = M[:, [ix[c] for c in used]].astype(np.float64)
    C = np.corrcoef(A.T)
    res["axis_correlation"] = {"axes": used, "matrix": C.tolist(),
                               "marginal_rates": {c: float(A[:, k].mean())
                                                  for k, c in enumerate(used)}}
    print("\n=== 4. 축 간 상관 (φ 계수) — 학습에 쓰인 5축 ===")
    print("            " + " ".join(f"{c[:6]:>7s}" for c in used))
    for k, c in enumerate(used):
        print(f"  {c[:10]:10s} " + " ".join(f"{C[k, j]:7.3f}" for j in range(len(used))))

    # 동시 리셋 수 분포 + 독립 가정 대비
    S = A.sum(1)
    dist = {int(v): float((S == v).mean()) for v in range(6)}
    pk = A.mean(0)
    pb = np.poly1d([1])
    for q in pk:                                   # Poisson-binomial PGF
        pb = pb * np.poly1d([q, 1 - q])
    indep = pb.c[::-1]
    res["count_distribution"] = {"observed": dist,
                                 "independent_poisson_binomial": [float(x) for x in indep]}
    print("\n  동시 리셋 수 분포 (관측 대 독립 가정)")
    for v in range(6):
        print(f"    {v}개  관측 {dist[v]:.4f}  독립 {indep[v]:.4f}  Δ {dist[v]-indep[v]:+.4f}")

    # ── 5. 인과 제외 4축 ≥2 대 5축 ≥3 ───────────────────────────────────────
    b5 = (S >= 3)
    no_causal = [c for c in used if c != "causality"]
    S4 = M[:, [ix[c] for c in no_causal]].sum(1)
    print("\n=== 5. 인과 제외 4축 규칙 대 현행 5축 ≥3 ===")
    res["causality_excluded"] = {}
    for t in (1, 2, 3):
        b4 = (S4 >= t)
        inter = float((b5 & b4).sum())
        j = inter / max(1.0, float((b5 | b4).sum()))
        agree = float((b5 == b4).mean())
        res["causality_excluded"][f"4axis>={t}"] = {
            "rate": float(b4.mean()), "jaccard_vs_5axis_t3": j, "agreement": agree,
            "recall_of_5axis": inter / max(1.0, float(b5.sum()))}
        print(f"  4축 ≥{t}: 경계율 {b4.mean():.4f} · 5축≥3 과 Jaccard {j:.4f} · "
              f"일치율 {agree:.4f} · 5축 경계의 {inter/max(1.0,b5.sum()):.4f} 를 포함")
    print(f"  (현행 5축 ≥3 경계율 {b5.mean():.4f})")
    res["boundary_rate_5axis_t3"] = float(b5.mean())

    # 인과 축의 조건부 기여: 인과가 켜진 전이에서만 경계가 나오는가
    caus = A[:, used.index("causality")].astype(bool)
    res["causality_conditional"] = {
        "P(boundary|causality=1)": float(b5[caus].mean()),
        "P(boundary|causality=0)": float(b5[~caus].mean()),
        "frac_boundaries_with_causality": float(caus[b5].mean())}
    print(f"\n  P(경계|인과=1) {b5[caus].mean():.4f} · P(경계|인과=0) {b5[~caus].mean():.4f} · "
          f"경계 중 인과가 켜진 비율 {caus[b5].mean():.4f}")

    # 문서별 축 발화율 (지적 2번 = 대사 밀도 상관에 쓴다)
    uniq, inv = np.unique(doc_ids, return_inverse=True)
    per_doc = {}
    for k, c in enumerate(used):
        s = np.bincount(inv, weights=A[:, k], minlength=len(uniq))
        n = np.bincount(inv, minlength=len(uniq))
        per_doc[c] = (s / np.maximum(n, 1)).tolist()
    (OUT / "per_doc_axis_rates.json").write_text(json.dumps(
        {"doc_ids": uniq.tolist(), "axes": used, "rates": per_doc}, indent=2))

    (OUT / "audit.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n-> {OUT/'audit.json'} · {OUT/'per_doc_axis_rates.json'}")


if __name__ == "__main__":
    main()

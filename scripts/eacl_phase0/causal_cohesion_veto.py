#!/usr/bin/env python3
r"""인과 응집 **거부권**(A)과 **응집 사슬 종료**(B) — dev-3, 재추출 0.

사전등록 `outputs/experiments/2026-07-31_causal_cohesion_veto/PREREG.md` (실행 전 작성, 미수정).

8건의 인과 조작화는 전부 인과를 **경계 탐지기**로 썼다. 문두 인과 접속사는 방향이 반대다 —
`So he ran.` 은 이 문장이 직전 문장에 **묶여 있다**는 저자의 선언(연결 신호)이다.

- **A**: `P(경계 | 인과표지)` 대 `P(경계 | 무표지)`. 현행 `causal_marker` 는 위치 무관 ⇒ **하한**.
- **A-E**: 문장 텍스트 정규식으로 **문두** 접속사만 잡아 E 처방(time_E 와 같은 교정)을 근사.
- **B**: `reset = 표지(a) ∧ ¬표지(b)` — a 를 쓰는 진짜 전이 함수.

읽기 전용. 동결 설정·체크포인트·시험지 미접근.

    .venv/bin/python scripts/eacl_phase0/causal_cohesion_veto.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
P0 = REPO / "outputs/eacl_phase0"
OUT = REPO / "outputs/experiments/2026-07-31_causal_cohesion_veto"
DOCS = ("pieman", "tunnel", "monkey")
REPS = ("q36_35b_v31", "q36_35b_v31_r2", "q36_35b_v31_r3")   # 🔒 동결 v3.1 3회 반복

# 문두 인과 접속사 (E 처방 근사). 대소문자 무시, 문장 시작에서만.
CAUSE_OPEN = r"(because|since|as)\b"                       # 원인 제시
CONSEQ_OPEN = (r"(so|therefore|thus|hence|consequently|accordingly|then"
               r"|as a result|for this reason|that(?:'s| is) why)\b")
RE_CAUSE = re.compile(r"^\W*" + CAUSE_OPEN, re.I)
RE_CONSEQ = re.compile(r"^\W*" + CONSEQ_OPEN, re.I)
RE_ANY_MID = re.compile(r"\b(because|since|therefore|thus|hence|consequently|as a result)\b", re.I)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rr_ci(k1: int, n1: int, k0: int, n0: int) -> tuple[float, float, float]:
    """위험비와 로그 정규 근사 95% CI."""
    if k1 == 0 or k0 == 0 or n1 == 0 or n0 == 0:
        return (float("nan"), float("nan"), float("nan"))
    p1, p0 = k1 / n1, k0 / n0
    rr = p1 / p0
    se = np.sqrt(1 / k1 - 1 / n1 + 1 / k0 - 1 / n0)
    return (rr, float(rr * np.exp(-1.96 * se)), float(rr * np.exp(1.96 * se)))


def load(doc: str, rep: str):
    """reset_matrix + master 텍스트. 반환 = (gold, cm_llm, open_cause, open_conseq, mid_only)."""
    rm = [json.loads(l) for l in (P0 / f"{doc}_michelmann_s5_{rep}/reset_matrix.jsonl")
          .read_text().splitlines() if l.strip()]
    rm.sort(key=lambda r: r["transition"])
    master = [json.loads(l) for l in (P0 / f"{doc}_michelmann_master.jsonl")
              .read_text().splitlines() if l.strip()]
    master.sort(key=lambda r: r["transition"])
    txt = {int(r["transition"]): r["current"] for r in master}
    gold = np.asarray([r["boundary"] for r in rm], np.int8)
    cm = np.asarray([r["causal_marker"] for r in rm], np.int8)
    cur = [txt[int(r["transition"])] for r in rm]
    oc = np.asarray([1 if RE_CAUSE.match(s) else 0 for s in cur], np.int8)
    oq = np.asarray([1 if RE_CONSEQ.match(s) else 0 for s in cur], np.int8)
    mid = np.asarray([1 if (RE_ANY_MID.search(s) and not (RE_CAUSE.match(s) or RE_CONSEQ.match(s)))
                      else 0 for s in cur], np.int8)
    return gold, cm, oc, oq, mid


def contrast(name: str, gold: np.ndarray, flag: np.ndarray, res: dict, indent: str = "  ") -> None:
    n1, n0 = int(flag.sum()), int((1 - flag).sum())
    k1, k0 = int(gold[flag == 1].sum()), int(gold[flag == 0].sum())
    p1 = k1 / n1 if n1 else float("nan")
    p0 = k0 / n0 if n0 else float("nan")
    rr, lo, hi = rr_ci(k1, n1, k0, n0)
    w1 = wilson(k1, n1)
    res[name] = {"n_flag": n1, "n_bnd_flag": k1, "p_bnd_flag": p1, "wilson_flag": list(w1),
                 "n_noflag": n0, "n_bnd_noflag": k0, "p_bnd_noflag": p0,
                 "risk_ratio": rr, "rr_ci": [lo, hi], "flag_rate": float(flag.mean())}
    print(f"{indent}{name:26s} 표지율 {flag.mean():.4f} · P(경계|표지) {p1:.4f} "
          f"[{w1[0]:.4f},{w1[1]:.4f}] (n={n1}) · P(경계|무표지) {p0:.4f} · "
          f"RR {rr:.3f} [{lo:.3f},{hi:.3f}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="*", default=list(REPS))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    results: dict = {"prereg": "PREREG.md", "reps": a.reps, "per_rep": {}, "per_doc": {}}

    for rep in a.reps:
        G, CM, OC, OQ, MID = [], [], [], [], []
        print(f"\n=== 반복 {rep} ===")
        for doc in DOCS:
            g, cm, oc, oq, mid = load(doc, rep)
            G.append(g); CM.append(cm); OC.append(oc); OQ.append(oq); MID.append(mid)
            d = results["per_doc"].setdefault(f"{rep}/{doc}", {})
            print(f" [{doc}] 전이 {len(g)} · gold {int(g.sum())}")
            contrast("A: LLM 표지(위치무관)", g, cm, d, indent="   ")
            contrast("A-E: 문두 원인/결과", g, ((oc | oq) > 0).astype(np.int8), d, indent="   ")
        g = np.concatenate(G); cm = np.concatenate(CM)
        oc = np.concatenate(OC); oq = np.concatenate(OQ); mid = np.concatenate(MID)
        r = results["per_rep"].setdefault(rep, {})
        print(f" [pooled] 전이 {len(g)} · gold {int(g.sum())} · 기저 경계율 {g.mean():.4f}")
        contrast("A: LLM 표지(위치무관)", g, cm, r)
        contrast("A-E: 문두 원인/결과", g, ((oc | oq) > 0).astype(np.int8), r)
        contrast("A-E1: 문두 결과(so 등)", g, oq, r)
        contrast("A-E2: 문두 원인(because)", g, oc, r)
        contrast("대조: 문중에만 인과어", g, mid, r)

        # ── B: 응집 사슬 종료 = 표지(a) ∧ ¬표지(b) ─────────────────────────
        # reset_matrix 의 전이 t 는 (문장 t−1 -> t). causal_marker[t] = 문장 t 의 표지.
        # 따라서 "a 에 표지" = causal_marker[t−1] 이므로 한 칸 시프트해 AND.
        for label, flag in (("B: LLM 표지", cm), ("B-E: 문두 표지", ((oc | oq) > 0).astype(np.int8))):
            prev = np.concatenate([[0], flag[:-1]])
            b = ((prev == 1) & (flag == 0)).astype(np.int8)
            print(f"\n  --- {label} 기반 B(응집 사슬 종료) ---")
            contrast(f"{label} -> B reset", g, b, r)
            base = g.mean()
            lift = (g[b == 1].mean() / base) if b.sum() else float("nan")
            k1, n1 = int(g[b == 1].sum()), int(b.sum())
            _, lo, hi = rr_ci(k1, n1, int(g[b == 0].sum()), int((1 - b).sum()))
            r[f"{label}_B"] = {"fire_rate": float(b.mean()), "lift": float(lift),
                               "lift_ci": [lo, hi], "n_fire": n1, "n_bnd_fire": k1}
            print(f"    발화율 {b.mean():.4f} · lift {lift:.3f} · RR CI [{lo:.3f},{hi:.3f}]"
                  f"{'  🔴 발화율<0.05 = 검정력 부족' if b.mean() < 0.05 else ''}")

    (OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False,
                                                 default=float))
    print(f"\n-> {OUT/'results.json'}")


if __name__ == "__main__":
    main()

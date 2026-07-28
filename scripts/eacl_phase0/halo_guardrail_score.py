#!/usr/bin/env python3
"""가드레일: causality 열만 halo로 스왑(나머지 4축·threshold 동결) 후 deployable Gate C Score 비교.

per-doc deployable(고정 t*) Score 를 12편 평균. t* = 현행 config argmax(축합≥t) 로 dev 동결,
동일 t* 를 halo config 에 그대로 적용(스왑 외 모든 것 동결). 하락 −0.09 이내면 PASS.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402

FROZEN = ["protagonist_present", "space", "time", "causality", "intentionality"]


def read(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def end_turn(n_sent, bt):
    y = [0] * n_sent
    for i in bt:
        y[i - 1] = 1
    return y


def by_doc(matrix):
    d = {}
    for r in matrix:
        d.setdefault(r["doc_id"], []).append(r)
    for rows in d.values():
        rows.sort(key=lambda r: int(r["transition"]))
    return d


def avg_score(docs, tstar):
    """per-doc deployable Score at fixed t*, averaged."""
    ss = []
    for rows in docs.values():
        n = len(rows) + 1
        count = {int(r["transition"]): sum(int(r[a]) for a in FROZEN) for r in rows}
        gold = end_turn(n, [int(r["transition"]) for r in rows if int(r["boundary"])])
        bt = [i for i in count if count[i] >= tstar]
        ss.append(score_dialogues([gold], [end_turn(n, bt)])["score"])
    return sum(ss) / len(ss)


# ------------------------------------------------------- lift (P5) + bootstrap
def lift_of(rows, field="causality"):
    """lift = P(reset|gold 경계) / P(reset|비경계).  ledger 의 '+0.007' 은 차이값이라 단위가 다르다."""
    b = [int(r[field]) for r in rows if int(r["boundary"])]
    nb = [int(r[field]) for r in rows if not int(r["boundary"])]
    if not b or not nb or sum(nb) == 0:
        return float("nan"), float("nan"), float("nan")
    p1, p0 = sum(b) / len(b), sum(nb) / len(nb)
    return p1 / p0, p1, p0


def lift_ci(rows, field="causality", n_boot=10000, seed=20260725):
    """전이 수준·문서별 층화 부트스트랩.  ⚠ 문서 간 변동 미포함 — 실제 불확실성은 이보다 크다
    (문서 3편이라 doc-level 부트스트랩 불가)."""
    import random
    rng = random.Random(seed)
    by_doc = {}
    for r in rows:
        by_doc.setdefault(r["doc_id"], []).append(r)
    out = []
    for _ in range(n_boot):
        samp = [rng.choice(v) for v in by_doc.values() for _ in range(len(v))]
        l, _p1, _p0 = lift_of(samp, field)
        if l == l:  # not nan
            out.append(l)
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, action="append",
                    help="반복 가능 (dev-3 는 문서별 reset_matrix 3개). 미지정 시 repcorpus stage0.")
    ap.add_argument("--halo", type=Path,
                    default=REPO / "outputs/experiments/2026-07-25_causality_halo_zacks/halo_dev.jsonl")
    ap.add_argument("--swap-file", type=Path, help="스왑할 열 파일 (미지정 시 --halo)")
    ap.add_argument("--swap-field", default="causality_halo", help="스왑 파일 안의 열 이름")
    ap.add_argument("--label", default="halo", help="출력 표기")
    ap.add_argument("--tstar", type=int,
                    help="동결 t* 를 명시 (dev-3 = 3). 미지정 시 현행 config argmax (halo 재현 경로).")
    ap.add_argument("--min-gold-density", type=float, default=0.01,
                    help="gold 밀도가 이 값 미만이면 Score 계산 거부 (hollow PASS 재발 방지)")
    a = ap.parse_args()
    mats = a.matrix or [REPO / "outputs/eacl_phase0/repcorpus_s5_q36_35b_v2_stage0/reset_matrix.jsonl"]
    M = [r for p in mats for r in read(p)]
    swap_path = a.swap_file or a.halo
    swap = {(r["doc_id"], int(r["transition"])): int(r[a.swap_field]) for r in read(swap_path)}

    # ---- gold-density guard: repcorpus stage0 는 gold 10/2222 라 Score 가 무의미했다 (halo hollow PASS)
    dens = sum(int(r["boundary"]) for r in M) / len(M)
    if dens < a.min_gold_density:
        print(f"🔴 REFUSED: gold 밀도 {dens:.4f} < {a.min_gold_density} "
              f"({sum(int(r['boundary']) for r in M)}/{len(M)} 경계).")
        print("   이 문서집합에서 Score 는 무의미하다 (2026-07-25 halo 가드레일이 이 함정에 빠졌다).")
        print("   사람 gold 가 있는 dev-3 matrix 를 --matrix 로 주거나 --min-gold-density 를 낮춰라.")
        return

    cur = by_doc([dict(r) for r in M])
    tstar = a.tstar if a.tstar else max(range(1, 6), key=lambda t: avg_score(cur, t))
    src = "동결값(사전등록)" if a.tstar else "현행 config argmax"
    cur_s = avg_score(cur, tstar)

    Mh = [dict(r) for r in M]
    miss = 0
    for r in Mh:
        k = (r["doc_id"], int(r["transition"]))
        if k in swap:
            r["causality"] = swap[k]
        else:
            miss += 1
    halo_s = avg_score(by_doc(Mh), tstar)  # SAME t*, threshold frozen

    cf = sum(int(r["causality"]) for r in M) / len(M)
    hf = sum(int(r["causality"]) for r in Mh) / len(Mh)
    ndoc = len({r["doc_id"] for r in M})
    print(f"t* = {tstar}  ({src})   (missing swap keys: {miss})   문서 {ndoc}편 / {len(M)} 전이 / "
          f"gold {sum(int(r['boundary']) for r in M)} 경계")
    print(f"causality 발현율: 현행(event_type) {cf:.3f}  →  {a.label} {hf:.3f}")
    print(f"\n=== P4 deployable Gate C Score (per-doc avg, 나머지4축·t* 동결) ===")
    print(f"현행 causality  : {cur_s:.4f}")
    print(f"{a.label:<15} : {halo_s:.4f}")
    print(f"Δ = {halo_s - cur_s:+.4f}   (가드레일 통과 ≥ -0.09)")
    print(f"[P4 guardrail] {'PASS' if (halo_s - cur_s) >= -0.09 else 'FAIL'}")

    # ---- P5 lift + 부트스트랩 CI
    print(f"\n=== P5 판별력 lift = P(reset|경계)/P(reset|비경계) ===")
    for name, rows in (("현행 proxy", M), (a.label, Mh)):
        l, p1, p0 = lift_of(rows)
        lo, hi = lift_ci(rows)
        print(f"  {name:<15} pooled lift={l:.3f}  (P1={p1:.3f} P0={p0:.3f})  "
              f"95% CI [{lo:.3f}, {hi:.3f}]")
        for d in sorted({r['doc_id'] for r in rows}):
            dl, dp1, dp0 = lift_of([r for r in rows if r["doc_id"] == d])
            print(f"      {d:<22} lift={dl:.3f} (P1={dp1:.3f} P0={dp0:.3f})")
    lo, _hi = lift_ci(Mh)
    print(f"[P5] CI 하한 {lo:.3f} > 1.0 ? {'PASS' if lo > 1.0 else 'FAIL'}")
    print("  ⚠ CI = 전이 수준(문서별 층화). 문서 간 변동 미포함 → 실제 불확실성은 이보다 크다.")
    print("  ⚠ MDE: CI하한>1.0 은 관측 lift ≈1.44 를 요구. 참값 1.3 에서 검정력 29%. "
          "FAIL 은 '판별력 없음'이 아니라 'n=66 에서 중간 효과 검출 불가'다.")


if __name__ == "__main__":
    main()

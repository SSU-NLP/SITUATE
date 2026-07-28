#!/usr/bin/env python3
"""intentionality 해상도 진단 (서술 전용 · API 0 · 판정 아님).

동기: Stage-1 goals 목록 크기가 문서 길이와 무상관(r=+0.218, monkey 제외 −0.290)이라 문서별
해상도(문장/목록)가 27배 차이난다(tunnel 63.0 vs obs_37511 2.3). Stage-2 프롬프트로는 안 움직이는
Stage-1 결함 가설. 이를 반증 가능하게 만들고 space와 직접 대조한다.

측정 4종:
  ① intentionality baseline lift = P(reset|경계)/P(reset|비경계) — 전체 + 문서별 (dev-3, 사람 gold)
  ② space locations 목록 크기 vs 문서 길이 상관 ← 승부처
     (비례하면 "장소는 세고 목표는 요약한다" = Stage-1 확정 / 안 비례하는데 space가 작동하면 가설 폐기)
  ③ 해상도(문장/목록) vs 문서별 lift 단조성 = ① 가설의 검정
  ④ 경계에서 명시율 + 리셋÷명시 — D-M13⑤ place 표와 직접 비교
     (place: 경계발화 pieman 3/18·tunnel 6/18·monkey 1/30 · 리셋÷명시 0.56·0.75·0.22)

lift 정의는 `axis_liveness.py` 와 동일(비율). 전부 frozen 캐시 재사용 — LLM 호출 0.
"""
from __future__ import annotations

import glob
import json
import statistics as stats
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEV3 = ("pieman", "tunnel", "monkey")
AXES = {"intentionality": ("active_goal", "none", "goals"),
        "space": ("location", "unspecified", "locations")}


def read(p) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def doc_dirs() -> list[tuple[str, Path, Path | None]]:
    """(doc_id, s5_doc_dir, reset_matrix or None).  repcorpus 는 gold 없음 -> matrix None."""
    out = []
    for p in sorted(glob.glob(str(REPO / "outputs/eacl_phase0/repcorpus_s5_q36_35b_v2_stage0/*/story_lists.json"))):
        out.append((Path(p).parent.name, Path(p).parent, None))
    for d in DEV3:
        base = REPO / f"outputs/eacl_phase0/{d}_michelmann_s5_q36_35b_v2"
        out.append((d, base / d, base / "reset_matrix.jsonl"))
    return out


def lift(rows, col):
    b = [int(r[col]) for r in rows if int(r["boundary"])]
    n = [int(r[col]) for r in rows if not int(r["boundary"])]
    if not b or not n or sum(n) == 0:
        return float("nan"), float("nan"), float("nan")
    p1, p0 = sum(b) / len(b), sum(n) / len(n)
    return p1 / p0, p1, p0


def main() -> None:
    docs = doc_dirs()
    info = {}
    for doc, ddir, mat in docs:
        lists = json.load(open(ddir / "story_lists.json"))["lists"]
        rec = {"n_sent": 0, "matrix": read(mat) if mat and mat.exists() else None}
        for axis, (field, null, listkey) in AXES.items():
            st = {int(x["item_id"]): x[field] for x in read(ddir / f"{axis}_states.jsonl")}
            rec["n_sent"] = max(rec["n_sent"], len(st))
            rec[axis] = {"list": lists[listkey], "n_list": len(lists[listkey]), "states": st,
                         "n_explicit": sum(1 for v in st.values() if v != null)}
        info[doc] = rec

    # ---------------------------------------------------------------- ①
    print("=" * 78)
    print("① intentionality baseline lift (dev-3, 사람 gold) — 아직 한 번도 안 잰 값")
    print("=" * 78)
    print(f"{'doc':<10}{'n':>6}{'경계':>5}{'reset율':>9}{'P(r|경계)':>11}{'P(r|비경계)':>12}{'lift':>8}")
    pool = []
    for d in DEV3:
        m = info[d]["matrix"]
        pool += m
        l, p1, p0 = lift(m, "intentionality")
        nb = sum(int(r["boundary"]) for r in m)
        rr = sum(int(r["intentionality"]) for r in m) / len(m)
        print(f"{d:<10}{len(m):>6}{nb:>5}{rr:>9.3f}{p1:>11.3f}{p0:>12.3f}{l:>8.3f}")
    l, p1, p0 = lift(pool, "intentionality")
    rr = sum(int(r["intentionality"]) for r in pool) / len(pool)
    print(f"{'POOLED':<10}{len(pool):>6}{sum(int(r['boundary']) for r in pool):>5}"
          f"{rr:>9.3f}{p1:>11.3f}{p0:>12.3f}{l:>8.3f}")
    print("\n  (대조) 같은 pool 에서 다른 축의 lift:")
    for col in ("space", "time", "causality", "protagonist_present"):
        if col in pool[0]:
            cl, cp1, cp0 = lift(pool, col)
            print(f"    {col:<22} lift={cl:.3f}  (P1={cp1:.3f} P0={cp0:.3f})")

    # ---------------------------------------------------------------- ②
    print("\n" + "=" * 78)
    print("② 승부처 — 목록 크기가 문서 길이에 비례하는가?  space vs intentionality")
    print("=" * 78)
    print(f"{'doc':<22}{'문장':>6}{'locations':>11}{'goals':>7}{'문장/loc':>10}{'문장/goal':>11}")
    N, L, G = [], [], []
    for doc, _, _ in docs:
        r = info[doc]
        n, nl, ng = r["n_sent"], r["space"]["n_list"], r["intentionality"]["n_list"]
        N.append(n); L.append(nl); G.append(ng)
        print(f"{doc:<22}{n:>6}{nl:>11}{ng:>7}{n / nl:>10.1f}{n / ng:>11.1f}")
    print(f"\n  문서길이 vs locations  Pearson r = {stats.correlation(N, L):+.3f}")
    print(f"  문서길이 vs goals      Pearson r = {stats.correlation(N, G):+.3f}")
    keep = [i for i, d in enumerate(docs) if d[0] != "monkey"]
    print(f"  (monkey 제외)  locations r = {stats.correlation([N[i] for i in keep], [L[i] for i in keep]):+.3f}"
          f"   goals r = {stats.correlation([N[i] for i in keep], [G[i] for i in keep]):+.3f}")
    print(f"  해상도 범위: 문장/loc {min(n/l for n,l in zip(N,L)):.1f}~{max(n/l for n,l in zip(N,L)):.1f}"
          f"  ({max(n/l for n,l in zip(N,L))/min(n/l for n,l in zip(N,L)):.0f}배)"
          f"   ·  문장/goal {min(n/g for n,g in zip(N,G)):.1f}~{max(n/g for n,g in zip(N,G)):.1f}"
          f"  ({max(n/g for n,g in zip(N,G))/min(n/g for n,g in zip(N,G)):.0f}배)")

    # ---------------------------------------------------------------- ③
    print("\n" + "=" * 78)
    print("③ 27배 가설 검정 — 해상도(문장/목록)와 lift 가 단조인가?  ⚠ dev-3 = n=3")
    print("=" * 78)
    for axis in ("intentionality", "space"):
        print(f"  [{axis}]  해상도 높은(=문장/목록 작은) 순")
        rec = []
        for d in DEV3:
            r = info[d]
            res = r["n_sent"] / r[axis]["n_list"]
            l, _, _ = lift(r["matrix"], axis)
            rec.append((res, d, l))
        for res, d, l in sorted(rec):
            print(f"    {d:<10} 문장/목록={res:>6.1f}   lift={l:.3f}")
        order_res = [d for _, d, _ in sorted(rec)]
        order_lift = [d for _, d, _ in sorted(rec, key=lambda x: -x[2])]
        print(f"    해상도 순 {order_res}  vs  lift 내림차순 {order_lift}"
              f"   -> {'단조 일치' if order_res == order_lift else '불일치'}")

    # ---------------------------------------------------------------- ④
    print("\n" + "=" * 78)
    print("④ 경계에서 명시율 + 리셋÷명시  (D-M13⑤ place 표와 직접 비교)")
    print("   place 기준값: 경계발화 pieman 3/18 · tunnel 6/18 · monkey 1/30 / 리셋÷명시 0.56·0.75·0.22")
    print("=" * 78)
    for axis in ("intentionality", "space"):
        print(f"  [{axis}]")
        print(f"    {'doc':<10}{'경계발화':>12}{'경계명시율':>12}{'전체명시율':>12}{'리셋':>6}{'리셋÷명시':>11}")
        for d in DEV3:
            r = info[d]
            m, st = r["matrix"], r[axis]["states"]
            null = AXES[axis][1]
            bnd = [int(x["transition"]) for x in m if int(x["boundary"])]
            hit = sum(1 for t in bnd if st.get(t, null) != null)
            nres = sum(int(x[axis]) for x in m)
            nexp = r[axis]["n_explicit"]
            print(f"    {d:<10}{f'{hit}/{len(bnd)}':>12}{hit / len(bnd):>12.3f}"
                  f"{nexp / r['n_sent']:>12.3f}{nres:>6}{nres / max(nexp, 1):>11.3f}")


if __name__ == "__main__":
    main()

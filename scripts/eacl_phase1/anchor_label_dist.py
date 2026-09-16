#!/usr/bin/env python3
r"""앵커 코퍼스 라벨 분포 대조 — Gemma-4-26B-A4B(신) vs Qwen3.6-35B-A3B(D-B15, 구).

D-B15 §"결과 — 라벨 분포"와 **같은 산식**으로 5축 발화율·count 분포·경계율(t*=3)을 낸다.
🔴 이 코퍼스는 `boundary` 가 전부 0이므로 **정확도가 아니다** — 방법이 스스로 찍은 값의 분포다.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# 🔒 동결 5축 = **D-B10 채택본**. protagonist 성분 = `protagonist_present`.
#    근거: ledger 스냅숏 §동결 설정, D-B10 본문("frozen count의 protagonist 성분 = 이제
#    protagonist_present"), 그리고 학습 데이터 빌더 `build_anchor_dataset.py:31`
#    ("D-B10 동결 5축") + repcorpus_qc / consensus_vs_single / noise_propagate_pieman /
#    s5_v31_combo_eval / canary_check / dts_axis_logit / coarse_fine_lift_diag 전부 동일.
AXES = ("protagonist_present", "space", "time", "causality", "intentionality")
# ⚠️ **`s5_crossmodel.py:17` 만 `protagonist`(=charset OR perspective)를 쓴다 — 유일한 이탈.**
#    D-B47~D-B52 라벨러 비교가 그 스크립트로 채점됐다(→ D-B54). 두 집합을 나란히 낸다.
AXES_XM = ("protagonist", "space", "time", "causality", "intentionality")
TSTAR = 3


def stats(p: Path) -> dict:
    n = 0
    on = Counter()
    cnt, cnt_x = Counter(), Counter()
    docs = set()
    with p.open() as f:
        for line in f:
            r = json.loads(line)
            n += 1
            docs.add(r["doc_id"])
            for ax in set(AXES) | set(AXES_XM):
                on[ax] += int(r[ax])
            cnt[sum(int(r[ax]) for ax in AXES)] += 1
            cnt_x[sum(int(r[ax]) for ax in AXES_XM)] += 1
    return {"n": n, "n_docs": len(docs),
            "rate": {ax: on[ax] / n for ax in (*AXES, "protagonist")},
            "count": {k: cnt[k] / n for k in sorted(cnt)},
            "boundary_rate": sum(v for k, v in cnt.items() if k >= TSTAR) / n,
            "boundary_rate_xm": sum(v for k, v in cnt_x.items() if k >= TSTAR) / n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--out", required=True)
    # 🔴 라벨러명이 제목·표머리에 하드코딩돼 있어 다른 코퍼스에 쓰면 오표기된다(2026-08-01 실측).
    ap.add_argument("--new-label", default="신 라벨러", help="비교 대상(새) 라벨러 이름")
    ap.add_argument("--old-label", default="Qwen3.6-35B-A3B (D-B15)", help="기준 라벨러 이름")
    a = ap.parse_args()
    N, O = stats(Path(a.new)), stats(Path(a.old))

    L = [f"# Phase-1 앵커 라벨 분포 — {a.new_label} vs {a.old_label}", "",
         "🔴 이 코퍼스의 `boundary` 는 전부 0이다. 아래는 **정확도가 아니라 방법이 찍은 값의 분포**다.", "",
         f"| 항목 | 신: {a.new_label} | 구: {a.old_label} | Δ |", "|---|---|---|---|",
         f"| 문서 | {N['n_docs']:,} | {O['n_docs']:,} | {N['n_docs']-O['n_docs']:+,} |",
         f"| 전이 | {N['n']:,} | {O['n']:,} | {N['n']-O['n']:+,} |"]
    for ax in AXES:
        L.append(f"| {ax} | {N['rate'][ax]:.4f} | {O['rate'][ax]:.4f} | {N['rate'][ax]-O['rate'][ax]:+.4f} |")
    L.append(f"| **경계율 (t\\*={TSTAR}, D-B10 동결 5축)** | **{N['boundary_rate']:.4f}** | "
             f"{O['boundary_rate']:.4f} | {N['boundary_rate']-O['boundary_rate']:+.4f} |")
    L += ["", "### 참고 — `s5_crossmodel.py` 열 집합으로 다시 센 값 (D-B54)", "",
          "`s5_crossmodel.py:17` 만 protagonist 성분을 `protagonist`(=charset OR perspective)로 쓴다.",
          "그 열 집합으로 세면 아래처럼 달라진다. **D-B15 표에 적힌 경계율 0.0649 는 위(동결 5축) 값과",
          "일치하며, 어긋난 쪽은 `s5_crossmodel.py` 다.**", "",
          f"| 항목 | 신: {a.new_label} | 구: {a.old_label} |", "|---|---|---|",
          f"| protagonist (charset OR perspective) | {N['rate']['protagonist']:.4f} | "
          f"{O['rate']['protagonist']:.4f} |",
          f"| 경계율 (t\\*={TSTAR}, s5_crossmodel 판) | {N['boundary_rate_xm']:.4f} | "
          f"{O['boundary_rate_xm']:.4f} |"]
    L += ["", "## count 분포", "", "| count | 신 | 구 |", "|---|---|---|"]
    for k in range(6):
        L.append(f"| {k} | {N['count'].get(k,0):.3f} | {O['count'].get(k,0):.3f} |")
    L.append("")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    (Path(a.out).with_suffix(".json")).write_text(
        json.dumps({"new": N, "old": O}, indent=2, default=str), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

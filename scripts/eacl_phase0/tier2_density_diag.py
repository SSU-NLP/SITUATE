#!/usr/bin/env python3
"""D-M11 ⑥ — 밀도 교란 진단: Tier2의 M3>M1이 "정확해서"인가 "작동점이 맞아서"인가?

가설 (codex 2026-07-16). knob(M3 t*=3 · M1 c=1.0 · B1 c=0.75)은 전부 pieman(인간 gold 밀도
0.189)에서 동결됐다. Tier2 B3-gold 밀도 중앙값은 0.042(4.5배 희소). 모든 방법이 참조보다 많이
예측하면 Pk/WD/precision/exact-F1이 구조적으로 악화된다. 이때 M3가 M1보다 높은 이유가
**더 정확한 사건경계 의미를 포착해서가 아니라 B3의 희소한 경계 수에 더 가까운 작동점을 가졌기
때문**일 수 있다 (matched@gold에서 M3≈M1인 것이 이미 이 가능성을 뒷받침).

측정. 방법별 예측 밀도 / 참조 밀도 / 비율(pred/ref)을 문서별로 낸다. API 호출 0 — 기존 예측 재사용.
  - 교란 지지 = M3의 |log(pred/ref)|가 M1보다 작다 (M3가 gold 밀도에 더 가까운 작동점)
  - 교란 기각 = M3와 M1의 밀도 정합이 비슷한데도 M3 Score가 높다

⚠️ deployable 원칙 유지: 이 진단은 **보고 전용**. gold를 보고 threshold를 맞추는 데 쓰지 않는다
(D-M11 ⑥). 결과를 본 뒤 knob 재선택 금지.

usage: uv run python scripts/eacl_phase0/tier2_density_diag.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
from tier2_eval import (  # noqa: E402
    BASE, S5_RM, AX, C_M1, T_M3, C_B1,
    read, adaptive, count_pred, b1_newsent_max,
)

DEV = REPO / "outputs/eacl_phase0"
OUT = REPO / "outputs/experiments/tier2_density_diag"


def dev_rows(doc: str) -> list[dict]:
    return sorted(read(DEV / f"{doc}_michelmann_master.jsonl"), key=lambda r: int(r["transition"]))


def dens(n_pred: int, n_trans: int) -> float:
    return n_pred / n_trans if n_trans else 0.0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    s5: dict[str, dict[int, int]] = {}
    for r in read(S5_RM):
        s5.setdefault(r["doc_id"], {})[int(r["transition"])] = sum(int(r[ax]) for ax in AX)

    rows_out = []

    # ---- Tier2 (B3 silver reference) ----
    manifest = json.loads((REPO / "data/eacl_phase0/tier2/manifest.json").read_text())
    for m in manifest:
        sid = m["story_id"]
        if sid not in s5:
            continue
        rows = sorted(read(BASE / f"{sid}_master.jsonl"), key=lambda r: int(r["transition"]))
        nt = len(rows)
        ref = sum(1 for r in rows if int(r["boundary"]))
        idm = [float(r["idm_score"]) for r in rows]
        cnt = [s5[sid].get(int(r["transition"]), 0) for r in rows]
        b1 = b1_newsent_max(read(BASE / f"b1/{sid}_words.jsonl"), nt)
        rows_out.append({
            "corpus": "tier2", "doc": sid, "n_trans": nt, "ref_n": ref, "ref_dens": dens(ref, nt),
            "M3_n": len(count_pred(cnt, T_M3)), "M1_n": len(adaptive(idm, C_M1)),
            "B1_n": len(adaptive(b1, C_B1)),
        })

    # ---- dev (human gold) — knob 동결 홈 ----
    for d in ["pieman", "tunnel", "monkey"]:
        rows = dev_rows(d)
        nt = len(rows)
        ref = sum(1 for r in rows if int(r["boundary"]))
        idm = [float(r["idm_score"]) for r in rows]
        rm = DEV / f"{d}_michelmann_s5/reset_matrix.jsonl"
        cnt_map = {}
        if rm.exists():
            for r in read(rm):
                if r["doc_id"] in (d, f"{d}_michelmann"):
                    cnt_map[int(r["transition"])] = sum(int(r[ax]) for ax in AX)
        cnt = [cnt_map.get(int(r["transition"]), 0) for r in rows]
        rec = {"corpus": "dev", "doc": d, "n_trans": nt, "ref_n": ref, "ref_dens": dens(ref, nt),
               "M3_n": len(count_pred(cnt, T_M3)) if cnt_map else None,
               "M1_n": len(adaptive(idm, C_M1)), "B1_n": None}
        wp = DEV / f"b1_bayes/{d}_words.jsonl"
        if wp.exists():
            rec["B1_n"] = len(adaptive(b1_newsent_max(read(wp), nt), C_B1))
        rows_out.append(rec)

    for r in rows_out:
        for m in ["M3", "M1", "B1"]:
            n = r[f"{m}_n"]
            r[f"{m}_dens"] = dens(n, r["n_trans"]) if n is not None else None
            r[f"{m}_ratio"] = (n / r["ref_n"]) if (n is not None and r["ref_n"]) else None

    (OUT / "density.json").write_text(json.dumps(rows_out, ensure_ascii=False, indent=1))

    # ---- 보고 ----
    def fmt(v, w=6, p=3):
        return " " * w if v is None else f"{v:{w}.{p}f}"

    lines = []
    for corpus in ["tier2", "dev"]:
        sub = [r for r in rows_out if r["corpus"] == corpus]
        if not sub:
            continue
        lines.append(f"\n=== {corpus} ({'B3 silver reference' if corpus=='tier2' else '인간 gold'}) ===")
        lines.append(f"{'doc':24s} {'전이':>5s} {'ref밀도':>7s} | "
                     f"{'M3밀도':>7s} {'M3비':>6s} | {'M1밀도':>7s} {'M1비':>6s} | {'B1밀도':>7s} {'B1비':>6s}")
        for r in sub:
            lines.append(
                f"{r['doc']:24s} {r['n_trans']:5d} {r['ref_dens']:7.3f} | "
                f"{fmt(r['M3_dens'],7)} {fmt(r['M3_ratio'],6,2)} | "
                f"{fmt(r['M1_dens'],7)} {fmt(r['M1_ratio'],6,2)} | "
                f"{fmt(r['B1_dens'],7)} {fmt(r['B1_ratio'],6,2)}")

    t2 = [r for r in rows_out if r["corpus"] == "tier2"]
    lines.append("\n=== 교란 검정 (Tier2, n=%d) ===" % len(t2))
    lines.append("|log(pred/ref)| = 참조 밀도와의 정합도 (0에 가까울수록 작동점이 gold에 맞음)")
    stat = {}
    for m in ["M3", "M1", "B1"]:
        d = [abs(np.log(r[f"{m}_ratio"])) for r in t2 if r.get(f"{m}_ratio")]
        stat[m] = float(np.mean(d)) if d else float("nan")
        lines.append(f"  {m}: mean |log ratio| = {stat[m]:.3f}   (median {np.median(d):.3f})")
    lines.append("")
    if stat["M3"] < stat["M1"]:
        lines.append(f"  -> M3({stat['M3']:.3f}) < M1({stat['M1']:.3f}): M3의 작동점이 B3-gold 밀도에 **더 가까움**")
        lines.append("     => 교란 가설 지지. M3>M1의 일부는 작동점 정합 덕일 수 있음.")
    else:
        lines.append(f"  -> M3({stat['M3']:.3f}) >= M1({stat['M1']:.3f}): M3의 밀도 정합이 M1보다 낫지 않음")
        lines.append("     => 교란 가설 기각 쪽. M3 우위를 작동점만으로 설명 못 함.")

    txt = "\n".join(lines)
    print(txt)
    (OUT / "density_table.txt").write_text(txt)
    print(f"\n-> {OUT}/density.json · density_table.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

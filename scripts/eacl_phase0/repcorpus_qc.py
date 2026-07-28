#!/usr/bin/env python3
"""①B 대표코퍼스 QC — 집단별(famous/obscure) + 결정성(run1 vs run2). 사전등록 D-B11.

gold 경계 없음(Phase-1도 무라벨) → liveness = base reset rate + count 분포 + density@t*
(boundary lift는 gold 필요라 n/a). frozen 판정기준(over-fire 0.70, dev-3와 동일)을 집단별 적용.
**headline = 두 집단 비교**: 유명만 좋으면 암기 의존 → Phase-1 예측치 = 무명 숫자(D-B11 판정규칙).

지표(집단별, 가능하면 register별):
- provenance(전 행 p0s5-eism-v2·deepseek 단일) · coverage(item_id 완결) · anchor 양성률(축별 base rate)
- 축 liveness(base·count hist·density@t*) · 결정성(run1 vs run2 라벨 일치율, 축별) · retry/504(로그)
- 비용/문장 = 토큰 미계측(한계) → wall/호출수 proxy만.

Usage: uv run python scripts/eacl_phase0/repcorpus_qc.py \
  --manifest outputs/experiments/2026-07-22_gate1b_repcorpus/manifest.json \
  --run1 outputs/eacl_phase0/repcorpus_s5_eismv2_run1 \
  --run2 outputs/eacl_phase0/repcorpus_s5_eismv2_run2 \
  --out outputs/experiments/2026-07-22_gate1b_repcorpus
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]

VERSION = "p0s5-eism-v2"
EXPECT_MODEL = "openrouter/deepseek/deepseek-v4-pro"
FROZEN_AXES = ["protagonist_present", "space", "time", "causality", "intentionality"]
AXIS_FILES = {"protagonist": ["characters_present", "perspective", "protagonist_present"],
              "space": ["location"], "time": ["tense", "temporal_shift_marker"],
              "causality": ["event_type", "aspect", "causal_marker"], "intentionality": ["active_goal"]}
T_STAR = 3
OVERFIRE = 0.70

KNOWN_AXIS_ISSUES = {"causality": "over-fires by construction (relational construct = base-rate "
                     "noise); documented at freeze REPORT §5. NOT new."}


def rj(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def axis_states(run: Path, doc: str, axis: str) -> dict[int, dict]:
    return {int(r["item_id"]): r for r in rj(run / doc / f"{axis}_states.jsonl")}


def provenance(run: Path, docs: list[str]) -> dict:
    vers, mods = set(), set()
    for d in docs:
        for ax in AXIS_FILES:
            for r in rj(run / d / f"{ax}_states.jsonl"):
                vers.add(r.get("prompt_version")); mods.add(r.get("model"))
    return {"versions": sorted(v for v in vers if v), "models": sorted(m for m in mods if m)}


def coverage(run: Path, docs: list[str], nmap: dict[str, int]) -> dict:
    issues = {}
    for d in docs:
        for ax in AXIS_FILES:
            got = {int(r["item_id"]) for r in rj(run / d / f"{ax}_states.jsonl")}
            miss = set(range(nmap[d])) - got
            if miss:
                issues[f"{d}/{ax}"] = f"missing {len(miss)}"
    return {"complete": not issues, "issues": issues}


def liveness(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    axes = {}
    for a in FROZEN_AXES:
        axes[a] = {"base_rate": sum(r[a] for r in rows) / len(rows)}
    counts = [sum(r[a] for a in FROZEN_AXES) for r in rows]
    hist = {k: round(counts.count(k) / len(rows), 3) for k in range(len(FROZEN_AXES) + 1)}
    return {"n": len(rows), "axes": axes, "count_hist": hist,
            "density_at_tstar": round(sum(1 for c in counts if c >= T_STAR) / len(rows), 3)}


def determinism(rm1: dict, rm2: dict, docs: list[str]) -> dict:
    """Reset-column reproducibility between the two runs (frozen-relevant: does the
    reset DECISION reproduce, not the noisy auxiliary fields). temp-0 should be high.

    rm{1,2}: {(doc_id, transition): row}. Compares FROZEN_AXES reset columns + the
    coordinated count + the boundary(count>=t*) decision, on shared transitions of `docs`.
    """
    shared = [k for k in (set(rm1) & set(rm2)) if k[0] in set(docs)]
    if not shared:
        return {"n": 0}
    out = {"n": len(shared), "axes": {}}
    for a in FROZEN_AXES:
        out["axes"][a] = round(sum(1 for k in shared if rm1[k][a] == rm2[k][a]) / len(shared), 4)
    c1 = {k: sum(rm1[k][x] for x in FROZEN_AXES) for k in shared}
    c2 = {k: sum(rm2[k][x] for x in FROZEN_AXES) for k in shared}
    out["count_exact"] = round(sum(1 for k in shared if c1[k] == c2[k]) / len(shared), 4)
    out["boundary_at_tstar"] = round(sum(1 for k in shared if (c1[k] >= T_STAR) == (c2[k] >= T_STAR))
                                     / len(shared), 4)
    return out


def retry_stats(run: Path) -> dict:
    txt = "".join(p.read_text(errors="replace") for p in run.glob("parallel_*.log"))
    return {"http_504": len(re.findall(r"\b504\b", txt)),
            "no_valid_outputs": len(re.findall(r"no valid outputs", txt)),
            "tracebacks": len(re.findall(r"Traceback", txt))}


def verdict(prov: dict, cov: dict, live: dict) -> tuple[bool, list[str], list[tuple]]:
    fails, known = [], []
    if prov["versions"] != [VERSION]:
        fails.append(f"prompt_version mixed/stale: {prov['versions']}")
    if prov["models"] and prov["models"] != [EXPECT_MODEL]:
        fails.append(f"model mismatch: {prov['models']}")
    if not cov["complete"]:
        fails.append(f"coverage gaps: {list(cov['issues'])[:4]}")
    for a, m in live.get("axes", {}).items():
        if m["base_rate"] > OVERFIRE:
            f = f"{a} over-fires (base {m['base_rate']:.2f} > {OVERFIRE})"
            fails.append(f)
            if a in KNOWN_AXIS_ISSUES:
                known.append((f, KNOWN_AXIS_ISSUES[a]))
    return (not fails), fails, known


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--run1", type=Path, required=True)
    ap.add_argument("--run2", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    man = json.loads(args.manifest.read_text())
    meta = {m["doc_id"]: m for m in man}
    nmap = {m["doc_id"]: m["n_sent"] for m in man}
    groups = {"famous": [m["doc_id"] for m in man if m["group"] == "famous"],
              "obscure": [m["doc_id"] for m in man if m["group"] == "obscure"]}
    rmat1 = rj(args.run1 / "reset_matrix.jsonl")
    by_doc = {}
    for r in rmat1:
        by_doc.setdefault(r["doc_id"], []).append(r)
    rm1k = {(r["doc_id"], r["transition"]): r for r in rmat1}
    rm2k = {(r["doc_id"], r["transition"]): r for r in rj(args.run2 / "reset_matrix.jsonl")} if args.run2 else {}

    result: dict[str, Any] = {"groups": {}}
    print("=" * 82)
    print(f"①B REPCORPUS QC — frozen {VERSION}  (per-group; run1={args.run1.name})")
    print("=" * 82)
    for g, docs in groups.items():
        prov = provenance(args.run1, docs)
        cov = coverage(args.run1, docs, nmap)
        rows = [r for d in docs for r in by_doc.get(d, [])]
        live = liveness(rows)
        det = determinism(rm1k, rm2k, docs) if rm2k else {}
        ok, fails, known = verdict(prov, cov, live)
        fp = sum(1 for d in docs if meta[d]["register"] == "first")
        result["groups"][g] = {"docs": docs, "provenance": prov, "coverage": cov,
                               "liveness": live, "determinism": det, "pass": ok, "fails": fails,
                               "n_first_person": fp}
        print(f"\n### {g.upper()} ({len(docs)} docs, 1인칭 {fp}) — {'PASS' if ok else 'FAIL'}")
        print(f"  provenance {prov['versions']} {prov['models']} | coverage complete={cov['complete']}")
        if live.get("axes"):
            print(f"  density@t*={live['density_at_tstar']}")
            dax = det.get("axes", {})
            print(f"  {'axis':<22}{'base_rate':>10}{'reset-determ':>14}")
            for a in FROZEN_AXES:
                flag = "OVERFIRE" if live["axes"][a]["base_rate"] > OVERFIRE else ""
                print(f"  {a:<22}{live['axes'][a]['base_rate']:>10.3f}{str(dax.get(a)):>14}  {flag}")
            if det.get("n"):
                print(f"  [determinism run2∩: {det['n']} transitions | count-exact "
                      f"{det.get('count_exact')} | boundary(count>=t*) {det.get('boundary_at_tstar')}]")
        if fails:
            print(f"  FAIL: {fails}")
        for c, w in known:
            print(f"    ↳ known at freeze (proceed knowingly): {w}")

    # retry stats
    rs = retry_stats(args.run1)
    result["retry_run1"] = rs
    print(f"\nretry/504 (run1): {rs}")
    print("cost/sentence: NOT captured (no token accounting) — limitation.")

    # headline comparison
    print("\n" + "-" * 82)
    fam, obs = result["groups"]["famous"]["liveness"], result["groups"]["obscure"]["liveness"]
    if fam.get("axes") and obs.get("axes"):
        print("HEADLINE (famous vs obscure base_rate; large gap => memorization signal):")
        for a in FROZEN_AXES:
            print(f"  {a:<22} famous {fam['axes'][a]['base_rate']:.3f}  obscure {obs['axes'][a]['base_rate']:.3f}")
    print("판정규칙(D-B11): 유명만 좋으면 암기 의존 → Phase-1 예측치 = 무명 숫자.")

    if args.out:
        (args.out / "qc.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\n[wrote] {args.out/'qc.json'}")


if __name__ == "__main__":
    main()

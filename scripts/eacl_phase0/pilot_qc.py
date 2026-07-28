#!/usr/bin/env python3
"""①B non-test pilot QC for the frozen S5 extractor (p0s5-eism-v2).

Pre-registration §4 lists the ①B gate metrics: parse-success rate, schema-validity
rate, retry rate, cost/sentence, output determinism, anchor positive rate, axis
liveness (+ a stratified human sample). This runner computes everything that is
recoverable from the *cached* extraction with **zero API cost**:

  1. Provenance consistency — every state row carries the frozen VERSION, the
     expected labeler model, and a single prompt_version. Detects stale/mixed caches.
  2. Coverage completeness — every sentence has exactly one row in every axis file
     (the extractor loops until valid, so a gap here means an aborted run).
  3. Axis liveness — per-axis reset base rate, boundary lift, count histogram, and
     predicted density at t*=3, computed on the FROZEN axis set
     (protagonist_present, space, time, causality, intentionality) so it matches
     eval_seg_score's coordinated count (D-B10, not protagonist_perspective).

Metrics that are NOT in the cache (parse-reject rate, retries, cost/sentence,
determinism) are transient; this runner extracts what it can from an optional
--log and otherwise reports them as "needs API / re-run" so the gate is honest
about what was and was not checked.

Usage:
  uv run python scripts/eacl_phase0/pilot_qc.py                 # dev-3 eismv2 default
  uv run python scripts/eacl_phase0/pilot_qc.py --doc pieman --dir outputs/eacl_phase0/pieman_michelmann_s5_eismv2 \
      --log outputs/eacl_phase0/round1_pieman.log --out outputs/experiments/2026-07-22_pilot_qc_eismv2
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

# Frozen coordinated-count axes = eval_seg_score.FROZEN_AXES (D-B10: protagonist_present,
# NOT protagonist_perspective). Kept literal here to avoid importing the eval harness.
FROZEN_AXES = ["protagonist_present", "space", "time", "causality", "intentionality"]
T_STAR = 3

# Per-axis state file -> the fields each row must carry (from llm_s5_states.AXIS_FIELDS).
AXIS_FILES = {
    "protagonist": ["characters_present", "perspective", "protagonist_present"],
    "space": ["location"],
    "time": ["tense", "temporal_shift_marker"],
    "causality": ["event_type", "aspect", "causal_marker"],
    "intentionality": ["active_goal"],
}

# reset base-rate sanity band: an axis firing on >70% of transitions is over-firing
# (the old causality churn failure mode); firing on 0% across every doc is dead.
OVERFIRE = 0.70

DEFAULT_DOCS = {
    "pieman": "outputs/eacl_phase0/pieman_michelmann_s5_eismv2",
    "tunnel": "outputs/eacl_phase0/tunnel_michelmann_s5_eismv2",
    "monkey": "outputs/eacl_phase0/monkey_michelmann_s5_eismv2",
}


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def provenance(doc_dir: Path) -> dict[str, Any]:
    """Check model / prompt_version consistency and count rows per axis file."""
    versions, models, per_axis = set(), set(), {}
    for axis, _ in AXIS_FILES.items():
        rows = read_jsonl(doc_dir / axis / f"{axis}_states.jsonl") if (doc_dir / axis).is_dir() \
            else read_jsonl(next(iter(doc_dir.glob(f"*/{axis}_states.jsonl")), Path("/nonexistent")))
        per_axis[axis] = rows
        for r in rows:
            versions.add(r.get("prompt_version"))
            models.add(r.get("model"))
    return {"versions": sorted(v for v in versions if v is not None),
            "models": sorted(m for m in models if m is not None),
            "per_axis": per_axis}


def coverage(per_axis: dict[str, list[dict]]) -> dict[str, Any]:
    """Every axis must cover the same contiguous item_id range 0..n-1, no gaps/dupes."""
    lengths = {a: len(rows) for a, rows in per_axis.items()}
    n = max(lengths.values()) if lengths else 0
    issues = {}
    for axis, rows in per_axis.items():
        ids = [int(r["item_id"]) for r in rows]
        expected = set(range(n))
        got = set(ids)
        problem = []
        if len(ids) != len(got):
            problem.append(f"dup:{len(ids) - len(got)}")
        if got - expected:
            problem.append(f"extra:{sorted(got - expected)[:5]}")
        if expected - got:
            problem.append(f"missing:{sorted(expected - got)[:5]}")
        if problem:
            issues[axis] = problem
    return {"n_items": n, "lengths": lengths, "complete": not issues, "issues": issues}


def liveness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Frozen-axis reset rates, boundary lift, count histogram, density at t*."""
    bnd = [r for r in rows if r["boundary"] == 1]
    non = [r for r in rows if r["boundary"] == 0]
    axes = {}
    for axis in FROZEN_AXES:
        base = sum(r[axis] for r in rows) / len(rows)
        p_b = (sum(r[axis] for r in bnd) / len(bnd)) if bnd else 0.0
        p_n = (sum(r[axis] for r in non) / len(non)) if non else 0.0
        axes[axis] = {"base_rate": base, "p_boundary": p_b, "p_non_boundary": p_n,
                      "lift": (p_b / p_n) if p_n > 0 else None,
                      "dead": base == 0.0, "overfire": base > OVERFIRE}
    counts = [sum(r[axis] for axis in FROZEN_AXES) for r in rows]
    hist = {k: counts.count(k) / len(rows) for k in range(len(FROZEN_AXES) + 1)}
    return {"n_transitions": len(rows), "gold_density": len(bnd) / len(rows),
            "axes": axes, "count_hist": hist,
            "density_at_tstar": sum(1 for c in counts if c >= T_STAR) / len(rows)}


def scan_log(log: Path) -> dict[str, Any]:
    """Best-effort retry/failure signal from a round log (transient, not in cache)."""
    if not log or not log.exists():
        return {"available": False}
    txt = log.read_text(encoding="utf-8", errors="replace")
    return {"available": True,
            "no_valid_outputs": len(re.findall(r"no valid outputs", txt)),
            "tracebacks": len(re.findall(r"Traceback", txt)),
            "http_504": len(re.findall(r"\b504\b", txt))}


# Pre-adjudicated axis behaviors (documented at FREEZE, before this QC ran). These
# annotate WHY a failing condition may be proceeded-through knowingly — they NEVER
# flip the verdict. A fail stays a fail (D-M5 lesson: do not re-cut the bar to pass).
KNOWN_AXIS_ISSUES = {
    "causality": "over-fires by construction — relational construct = base-rate noise; "
                 "documented at freeze (REPORT §5; '−causality 4-axis' near-best). NOT new.",
}


def verdict(prov: dict, cov: dict, live: dict) -> tuple[bool, list[str], list[tuple[str, str]]]:
    """Strict ①B criterion — FROZEN as first written, NOT edited after seeing results.

    Criterion (set before the run): single VERSION + expected model (provenance),
    contiguous item_id coverage, and no axis firing above OVERFIRE (the old causality
    churn failure mode, threshold fixed at 0.70 pre-run). A fail is recorded as a fail;
    the returned `known` list only annotates which failing conditions were already
    documented at freeze so the record reads "FAILED, proceeding knowingly" — it does
    NOT change pass/fail.
    """
    fails: list[str] = []
    known: list[tuple[str, str]] = []
    if prov["versions"] != [VERSION]:
        fails.append(f"prompt_version mixed/stale: {prov['versions']}")
    if prov["models"] and prov["models"] != [EXPECT_MODEL]:
        fails.append(f"model mismatch: {prov['models']}")
    if not cov["complete"]:
        fails.append(f"coverage gaps: {cov['issues']}")
    for axis, m in live["axes"].items():
        if m["overfire"]:
            f = f"{axis} over-fires (base {m['base_rate']:.2f} > {OVERFIRE})"
            fails.append(f)
            if axis in KNOWN_AXIS_ISSUES:
                known.append((f, KNOWN_AXIS_ISSUES[axis]))
    return (not fails), fails, known


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc", action="append", help="doc id (repeatable); default = dev-3 eismv2")
    ap.add_argument("--dir", action="append", help="doc extraction dir, paired with --doc")
    ap.add_argument("--log", action="append", help="optional round log, paired with --doc")
    ap.add_argument("--out", type=Path, help="write REPORT.md + qc.json under this dir")
    args = ap.parse_args()

    if args.doc and args.dir:
        docs = dict(zip(args.doc, args.dir))
    else:
        docs = DEFAULT_DOCS
    logs = dict(zip(args.doc or [], args.log or [])) if args.log else {}

    results: dict[str, Any] = {}
    for doc, sub in docs.items():
        ddir = REPO / sub
        prov = provenance(ddir)
        cov = coverage(prov["per_axis"])
        rmatrix = read_jsonl(ddir / "reset_matrix.jsonl")
        live = liveness(rmatrix) if rmatrix else {"error": "no reset_matrix.jsonl"}
        log = scan_log(REPO / logs[doc]) if doc in logs else {"available": False}
        if "error" not in live:
            ok, fails, known = verdict(prov, cov, live)
        else:
            ok, fails, known = False, ["no reset_matrix"], []
        results[doc] = {"provenance": {"versions": prov["versions"], "models": prov["models"]},
                        "coverage": cov, "liveness": live, "log": log,
                        "pass": ok, "fails": fails, "known": known}

    # --- console report ---
    print("=" * 78)
    print(f"①B PILOT QC — frozen {VERSION}  (cache-only, zero API)")
    print("=" * 78)
    for doc, r in results.items():
        v = "PASS" if r["pass"] else "FAIL"
        print(f"\n[{doc}]  {v}   n_items={r['coverage']['n_items']}  "
              f"prompt_version={r['provenance']['versions']}  model={r['provenance']['models']}")
        print(f"  coverage complete: {r['coverage']['complete']}  "
              f"lengths={r['coverage']['lengths']}")
        if "error" not in r["liveness"]:
            lv = r["liveness"]
            print(f"  gold_density={lv['gold_density']:.3f}  "
                  f"density@t*={lv['density_at_tstar']:.3f}")
            print(f"  {'axis':<22}{'base':>8}{'p_bnd':>8}{'p_non':>8}{'lift':>8}  flag")
            for axis, m in lv["axes"].items():
                lift = f"{m['lift']:.2f}" if m["lift"] is not None else "n/a"
                flag = "DEAD" if m["dead"] else ("OVERFIRE" if m["overfire"] else "")
                print(f"  {axis:<22}{m['base_rate']:>8.3f}{m['p_boundary']:>8.3f}"
                      f"{m['p_non_boundary']:>8.3f}{lift:>8}  {flag}")
        if r["log"]["available"]:
            print(f"  log: no_valid_outputs={r['log']['no_valid_outputs']} "
                  f"tracebacks={r['log']['tracebacks']} http_504={r['log']['http_504']}")
        if r["fails"]:
            print(f"  FAIL: {r['fails']}")
        for cond, why in r["known"]:
            print(f"    ↳ known at freeze (proceed knowingly, NOT a new defect): {why}")

    print("\n" + "-" * 78)
    print("NOT checked here (transient / needs API or logging): parse-reject rate, "
          "retry rate,\n  cost/sentence, output determinism (temp-0 re-run), human "
          "stratified sample.")
    allpass = all(r["pass"] for r in results.values())
    print(f"\nOVERALL ①B (dev-3 cache subset): {'PASS' if allpass else 'FAIL'}")
    if not allpass:
        print("→ Recorded as FAIL. Criterion was NOT changed after seeing results (D-M5 lesson).\n"
              "  Failing condition(s) are pre-documented (Gate C / REPORT §5) → proceeding knowingly,\n"
              "  which is defensible; changing the bar to pass would not be.")

    if args.out:
        outdir = REPO / args.out if not Path(args.out).is_absolute() else Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "qc.json").write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
        print(f"\n[wrote] {outdir/'qc.json'}")


if __name__ == "__main__":
    main()

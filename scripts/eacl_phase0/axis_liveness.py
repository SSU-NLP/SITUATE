#!/usr/bin/env python3
"""S5 5-axis liveness diagnosis — per-axis reset rate, boundary lift, and RAW COUNTS.

Motivating question (2026-07-20): Monkey(journalism) fails while pieman/tunnel(narrative)
work. Is the cause "the stage stands still" — i.e. does the `place` axis actually go dead?

The headline metric is **lift** = P(reset | boundary) / P(reset | non-boundary); 1.0 means the
axis carries no boundary information. Lift alone is misleading on sparse axes, so this script
always prints the underlying integer counts next to every ratio: `place` fires on only 3/18
boundaries in pieman, so a lift of 6.42 there rests on three events.

Reads frozen `reset_matrix.jsonl` (one row per transition, produced by `llm_s5_states.py`);
no API calls, no model loading.

Usage:
  uv run python scripts/eacl_phase0/axis_liveness.py
  uv run python scripts/eacl_phase0/axis_liveness.py --doc pieman --doc run --json out.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]

# Frozen S5 axis set (D-M5-b / D-B8): time = temporal_shift_marker only,
# protagonist = perspective alone (charset is ablation-only).
AXES = ["protagonist_perspective", "space", "time", "causality", "intentionality"]
T_STAR = 3  # frozen operating point: boundary iff reset count >= t*

# doc_id -> (reset_matrix path template, genre note for the report)
DEFAULT_DOCS = {
    "pieman": ("outputs/eacl_phase0/pieman_michelmann_s5", "narrative / spoken (The Moth)"),
    "tunnel": ("outputs/eacl_phase0/tunnel_michelmann_s5", "narrative / SF short story"),
    "monkey": ("outputs/eacl_phase0/monkey_michelmann_s5", "journalism (This American Life)"),
}


def load(doc: str, subdir: str) -> list[dict[str, Any]]:
    """Load a frozen reset matrix.

    Args:
        doc: Document id, used only for the error message.
        subdir: Repo-relative directory holding `reset_matrix.jsonl`.

    Returns:
        One dict per transition, with 0/1 columns for each axis plus `boundary`.
    """
    path = REPO / subdir / "reset_matrix.jsonl"
    if not path.exists():
        raise SystemExit(f"[{doc}] missing reset matrix: {path}")
    return [json.loads(line) for line in path.open() if line.strip()]


def analyse(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute per-axis rates, boundary lift with raw counts, and operating-point density."""
    bnd = [r for r in rows if r["boundary"] == 1]
    non = [r for r in rows if r["boundary"] == 0]
    out: dict[str, Any] = {
        "n_transitions": len(rows),
        "n_boundary": len(bnd),
        "n_non_boundary": len(non),
        "gold_density": len(bnd) / len(rows),
        "axes": {},
    }
    for axis in AXES:
        hit_b = sum(r[axis] for r in bnd)
        hit_n = sum(r[axis] for r in non)
        p_b = hit_b / len(bnd) if bnd else 0.0
        p_n = hit_n / len(non) if non else 0.0
        out["axes"][axis] = {
            "base_rate": sum(r[axis] for r in rows) / len(rows),
            "hits_at_boundary": hit_b,
            "hits_at_non_boundary": hit_n,
            "p_boundary": p_b,
            "p_non_boundary": p_n,
            "lift": (p_b / p_n) if p_n > 0 else None,
        }
    counts = [sum(r[axis] for axis in AXES) for r in rows]
    out["count_hist"] = {k: counts.count(k) / len(rows) for k in range(len(AXES) + 1)}
    out["predicted_density_at_tstar"] = sum(1 for c in counts if c >= T_STAR) / len(rows)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--doc", action="append", help="doc id (repeatable); default = dev 3")
    ap.add_argument("--dir", action="append", help="explicit subdir, paired with --doc")
    ap.add_argument("--json", type=Path, help="also write the full result as JSON")
    args = ap.parse_args()

    if args.doc and args.dir:
        docs = {d: (s, "") for d, s in zip(args.doc, args.dir)}
    elif args.doc:
        docs = {d: DEFAULT_DOCS[d] for d in args.doc}
    else:
        docs = DEFAULT_DOCS

    res = {doc: analyse(load(doc, sub)) for doc, (sub, _) in docs.items()}
    names = list(res)
    w = 13

    def header(title: str) -> None:
        print("\n" + "=" * (26 + w * len(names)))
        print(title)
        print("=" * (26 + w * len(names)))
        print(f"{'axis':<26}" + "".join(f"{n:>{w}}" for n in names))

    header("Base rate — P(reset), all transitions")
    for axis in AXES:
        print(f"{axis:<26}" + "".join(f"{res[n]['axes'][axis]['base_rate']:>{w}.3f}" for n in names))

    header("Boundary lift  [1.0 = no diagnostic value]")
    for axis in AXES:
        row = f"{axis:<26}"
        for n in names:
            lift = res[n]["axes"][axis]["lift"]
            row += f"{lift:>{w}.2f}" if lift is not None else f"{'n/a':>{w}}"
        print(row)

    header("RAW COUNTS  hits@boundary / n_boundary  ::  hits@non / n_non")
    for axis in AXES:
        row = f"{axis:<26}"
        for n in names:
            a, r = res[n]["axes"][axis], res[n]
            row += f"{a['hits_at_boundary']:>3}/{r['n_boundary']:<3}:{a['hits_at_non_boundary']:>4}/{r['n_non_boundary']:<4}"
        print(row)

    header(f"Operating point (t*={T_STAR}) vs gold")
    print(f"{'predicted density':<26}" + "".join(f"{res[n]['predicted_density_at_tstar']:>{w}.3f}" for n in names))
    print(f"{'gold density':<26}" + "".join(f"{res[n]['gold_density']:>{w}.3f}" for n in names))
    print(f"{'n_transitions':<26}" + "".join(f"{res[n]['n_transitions']:>{w}d}" for n in names))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()

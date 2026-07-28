#!/usr/bin/env python3
"""Merge completed blinded packs into the master without exposing labels early."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SLOTS = ("protagonist", "place", "time", "causality", "goal")
LABELS = (*SLOTS, "boundary")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def key(r: dict) -> tuple[str, int]:
    return str(r["doc_id"]), int(r["transition"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("master", type=Path)
    ap.add_argument("packs_dir", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--boundary-from-master", action="store_true",
                    help="Use concatenation's free boundary; do not read boundary.jsonl")
    args = ap.parse_args()
    master = read_jsonl(args.master)
    maps = {}
    labels = SLOTS if args.boundary_from_master else LABELS
    for label in labels:
        path = args.packs_dir / f"{label}.jsonl"
        rows = read_jsonl(path)
        values = {key(r): r.get(label) for r in rows}
        if len(values) != len(master):
            raise SystemExit(f"{path}: expected {len(master)} unique transitions, got {len(values)}")
        invalid = [k for k, v in values.items() if v not in (0, 1, False, True)]
        if invalid:
            raise SystemExit(f"{path}: incomplete/invalid labels at {invalid[:5]} ({len(invalid)} total)")
        maps[label] = values
    for r in master:
        k = key(r)
        for label in labels:
            if k not in maps[label]:
                raise SystemExit(f"{label}: missing transition {k}")
            r[label] = int(maps[label][k])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in master), encoding="utf-8")
    print(f"merged {len(master)} complete transitions -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

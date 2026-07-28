#!/usr/bin/env python3
"""Split a Phase-0 master pack into six mutually blinded annotation packs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SLOTS = ("protagonist", "place", "time", "causality", "goal")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("master", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--exclude-boundary", action="store_true",
                    help="For concatenated stories: emit only five reset-axis packs")
    args = ap.parse_args()
    rows = read_jsonl(args.master)
    common = ("doc_id", "transition", "previous", "previous_ko_reference",
              "current", "current_ko_reference")
    labels = SLOTS if args.exclude_boundary else (*SLOTS, "boundary")
    for label in labels:
        pack = []
        for r in rows:
            item = {k: r[k] for k in common if k in r}
            item[label] = None
            item["notes"] = ""
            pack.append(item)
        write_jsonl(args.out_dir / f"{label}.jsonl", pack)
    manifest = {
        "source": str(args.master), "n_transitions": len(rows),
        "recommended_order": list(labels),
        "rule": ("Annotate one file at a time. Boundary comes from concatenation and is hidden."
                 if args.exclude_boundary else
                 "Annotate one file at a time. Do not open boundary.jsonl until all five slot files are complete."),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(labels)} blinded packs ({len(rows)} transitions each) -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

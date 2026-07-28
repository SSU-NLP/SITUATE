#!/usr/bin/env python3
"""Make a non-fabricated Phase-0 annotation pack from one-sentence-per-line text."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--doc-id")
    args = ap.parse_args()
    texts = [x.strip() for x in args.input.read_text(encoding="utf-8").splitlines() if x.strip()]
    doc = args.doc_id or args.input.stem
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for i in range(1, len(texts)):
            row = {
                "doc_id": doc, "transition": i, "previous": texts[i - 1], "current": texts[i],
                "boundary": None, "idm_score": None,
                "protagonist": None, "place": None, "time": None,
                "causality": None, "goal": None, "notes": "",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {max(0, len(texts)-1)} transitions -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

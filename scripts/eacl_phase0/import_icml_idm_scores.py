#!/usr/bin/env python3
"""Import the frozen ICML delta_eff trace from HANDOFF_06 into its Phase-0 pack."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PATTERN = re.compile(r"^- `\[\s*(\d+)\].*?δ=([0-9.]+)", re.MULTILINE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("annotations", type=Path)
    ap.add_argument("--handoff", type=Path,
                    default=Path("handoff/HANDOFF_06_icml-diary-baseline_0709.md"))
    args = ap.parse_args()
    # HANDOFF has one canonical bullet trace (§4); the later compact NLL trace does
    # not match this pattern. Keep first occurrence if the prose is ever duplicated.
    scores = {}
    for i, v in PATTERN.findall(args.handoff.read_text(encoding="utf-8")):
        scores.setdefault(int(i), float(v))
    rows = [json.loads(x) for x in args.annotations.read_text(encoding="utf-8").splitlines() if x.strip()]
    missing = []
    for r in rows:
        i = int(r["transition"])
        if i not in scores:
            missing.append(i)
            continue
        r["idm_score"] = scores[i]
        r["idm_score_source"] = str(args.handoff)
        r["idm_encoder"] = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    if missing:
        raise SystemExit(f"handoff is missing transitions: {missing}")
    args.annotations.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"imported {len(rows)} frozen IDM scores -> {args.annotations}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

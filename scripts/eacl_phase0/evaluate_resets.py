#!/usr/bin/env python3
"""EACL Phase-0 kill-switch: coordinated EISM resets vs scalar IDM drift."""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

SLOTS = ("protagonist", "place", "time", "causality", "goal")


def auc(y: list[int], score: list[float]) -> float:
    """Mann-Whitney AUC with exact tie handling, no sklearn dependency."""
    pos = [s for yy, s in zip(y, score) if yy == 1]
    neg = [s for yy, s in zip(y, score) if yy == 0]
    if not pos or not neg:
        return math.nan
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def summarize(rows: list[dict]) -> dict:
    y = [int(r["boundary"]) for r in rows]
    reset = [sum(int(r[s]) for s in SLOTS) for r in rows]
    idm = [float(r["idm_score"]) for r in rows]
    bp = [i for i, v in enumerate(y) if v]
    nb = [i for i, v in enumerate(y) if not v]
    mean = lambda xs: sum(xs) / len(xs) if xs else math.nan
    reset_margin = mean([reset[i] for i in bp]) - mean([reset[i] for i in nb])
    idm_margin = mean([idm[i] for i in bp]) - mean([idm[i] for i in nb])
    return {
        "n": len(rows), "n_boundary": len(bp), "n_internal": len(nb),
        "auc_resets": auc(y, reset), "auc_idm": auc(y, idm),
        "auc_gain": auc(y, reset) - auc(y, idm),
        "reset_margin": reset_margin, "idm_margin": idm_margin,
        "slot_auc": {s: auc(y, [int(r[s]) for r in rows]) for s in SLOTS},
        "mean_resets_boundary": mean([reset[i] for i in bp]),
        "mean_resets_internal": mean([reset[i] for i in nb]),
    }


def bootstrap(rows: list[dict], draws: int, seed: int) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r["doc_id"])].append(r)
    keys = sorted(groups)
    if len(keys) < 2 or draws <= 0:
        return {}
    rng = random.Random(seed)
    vals = []
    for _ in range(draws):
        sample = []
        for k in rng.choices(keys, k=len(keys)):
            sample.extend(groups[k])
        v = summarize(sample)["auc_gain"]
        if not math.isnan(v):
            vals.append(v)
    vals.sort()
    if not vals:
        return {}
    q = lambda p: vals[min(len(vals) - 1, int(p * len(vals)))]
    return {"draws_valid": len(vals), "auc_gain_ci95": [q(.025), q(.975)]}


def load_complete(path: Path) -> tuple[list[dict], int]:
    complete, skipped = [], 0
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        required = ["doc_id", "boundary", "idm_score", *SLOTS]
        if any(r.get(k) is None for k in required):
            skipped += 1
            continue
        if r["boundary"] not in (0, 1, False, True) or any(r[s] not in (0, 1, False, True) for s in SLOTS):
            raise ValueError(f"line {lineno}: boundary/resets must be 0, 1, or null")
        complete.append(r)
    return complete, skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("annotations", type=Path)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    rows, skipped = load_complete(args.annotations)
    if not rows:
        raise SystemExit("No complete rows. Fill boundary, idm_score, and all five reset fields.")
    result = summarize(rows)
    result["skipped_incomplete"] = skipped
    result["bootstrap_by_doc"] = bootstrap(rows, args.bootstrap, args.seed)
    # plan-factored-slot.md Phase-0 primary gate. Margin is reported, not mixed into
    # this boolean because reset-count and IDM margins have different units.
    result["gate_auc_gain_ge_0.10"] = result["auc_gain"] >= .10
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

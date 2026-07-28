#!/usr/bin/env python3
"""Build an S5 master from AMI meeting topic-segmentation (data/ami/topic/*.json).

Theory-check: meeting topic boundaries are same-room / continuous-time discourse shifts, so
place/time situation-model axes should NOT reset -> our method should fail / tie IDM.
turns -> "sentence" unit; bnd_top[i]=1 marks a top-level topic boundary at turn i (turn i-1
is segment-final). Each meeting = one doc_id.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-meetings", type=int, default=4)
    ap.add_argument("--topic-dir", type=Path, default=Path("data/ami/topic"))
    ap.add_argument("--out", type=Path, default=Path("outputs/eacl_phase0/dts_ami_master.jsonl"))
    args = ap.parse_args()
    files = sorted(args.topic_dir.glob("*.json"))[:args.n_meetings]
    out = []
    for f in files:
        d = json.loads(f.read_text())
        turns = [t["text"].strip() for t in d["turns"]]
        bnd = d["bnd_top"]  # per-turn 0/1
        doc_id = d["meeting"]
        for s in range(1, len(turns)):
            if not turns[s] or not turns[s - 1]:
                pass
            out.append({
                "doc_id": doc_id, "transition": s,
                "previous": turns[s - 1], "current": turns[s],
                "previous_ko_reference": "", "current_ko_reference": "",
                "boundary": int(bnd[s] == 1),  # boundary at turn s -> transition s
                "idm_score": None,
                "protagonist": None, "place": None, "time": None,
                "causality": None, "goal": None, "notes": "",
            })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out), encoding="utf-8")
    nb = sum(r["boundary"] for r in out)
    print(f"AMI: {len(files)} meetings, {len(out)} transitions, {nb} boundaries -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

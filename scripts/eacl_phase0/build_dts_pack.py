#!/usr/bin/env python3
"""Build an S5 master from a DTS session dataset (dialseg711/tiage/superseg) turns.

Theory-check (ledger 🔮 pre-declaration): topic-transition boundaries in dialogue should
NOT coincide with situation-model (place/time) resets, so our method should fail / tie IDM.
Turns become the "sentence" unit; [BOUNDARY] tokens mark topic boundaries (end-turn: the
turn before [BOUNDARY] is segment-final). Each dialogue = one doc_id (multi-doc master).
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

SPEAKER = re.compile(r"^(user|agent|[A-Za-z]{1,3}):\s*")


def parse_dialogue(text: str) -> tuple[list[str], set[int]]:
    """Return (turn texts, set of turn indices that are segment-final = boundary before next)."""
    parts = text.split("[NEWLINE]")
    turns: list[str] = []
    boundary_after: set[int] = set()
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if p == "[BOUNDARY]":
            if turns:
                boundary_after.add(len(turns) - 1)  # previous turn is segment-final
            continue
        turns.append(SPEAKER.sub("", p).strip())
    return turns, boundary_after


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tiage_test")
    ap.add_argument("--n-dialogues", type=int, default=20)
    ap.add_argument("--data-root", type=Path,
                    default=Path("benchmarks/Def-DTS/data/DTS_session_datasets"))
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/eacl_phase0/dts_tiage_master.jsonl"))
    args = ap.parse_args()

    rows_in = [json.loads(l) for l in (args.data_root / f"{args.dataset}.jsonl").read_text().splitlines() if l.strip()]
    out_rows = []
    n_docs = 0
    for r in rows_in[:args.n_dialogues]:
        turns, bafter = parse_dialogue(r["dialogue"])
        if len(turns) < 4:
            continue
        n_docs += 1
        doc_id = r["id"]
        for s in range(1, len(turns)):
            out_rows.append({
                "doc_id": doc_id, "transition": s,
                "previous": turns[s - 1], "current": turns[s],
                "previous_ko_reference": "", "current_ko_reference": "",
                "boundary": int((s - 1) in bafter),
                "idm_score": None,
                "protagonist": None, "place": None, "time": None,
                "causality": None, "goal": None, "notes": "",
            })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out_rows), encoding="utf-8")
    nb = sum(r["boundary"] for r in out_rows)
    print(f"{args.dataset}: {n_docs} dialogues, {len(out_rows)} transitions, {nb} boundaries -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Tier-2 held-out step B: sentence packs + M1 idm_score for each fetched story.

Uses the SAME sentence splitter as dev (`build_pieman_pack.split_sentences`). Text-only (no
word-onset alignment). Writes per-story master jsonl `outputs/eacl_phase0/tier2/{sid}_master.jsonl`
with previous/current/transition/doc_id, idm_score = raw-cosine mpnet drift (the fair M1-CosDrift
signal, project-HP-independent), and boundary=0 PLACEHOLDER (gold filled by build_tier2_gold.py).
5-axis fields = None (filled by llm_s5_states.py for M3). Sentence-split policy fixed here and
recorded in the pack meta (dev/test seal).

Usage: uv run python scripts/eacl_phase0/build_tier2_pack.py
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
from build_pieman_pack import split_sentences  # noqa: E402
from hi_ontop.embedding import make_encoder  # noqa: E402

TIER2_TXT = REPO / "data/eacl_phase0/tier2"
OUT = REPO / "outputs/eacl_phase0/tier2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="build packs for these story_ids only (protect applied masters)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    enc = make_encoder(model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    manifest = json.loads((TIER2_TXT / "manifest.json").read_text())
    meta_all = []
    for m in manifest:
        sid = m["story_id"]
        if args.only and sid not in args.only:
            continue
        text = (TIER2_TXT / f"{sid}.txt").read_text()
        sents = split_sentences(text)
        Z = np.asarray(enc.encode(sents), dtype=np.float64)
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
        rows = []
        for s in range(1, len(sents)):
            rows.append({
                "doc_id": sid, "transition": s,
                "previous": sents[s - 1], "current": sents[s],
                "previous_ko_reference": "", "current_ko_reference": "",
                "boundary": 0,  # PLACEHOLDER until gold (build_tier2_gold.py)
                "idm_encoder": "paraphrase-multilingual-mpnet-base-v2",
                "idm_score": float(1 - Z[s] @ Z[s - 1]),
                "protagonist": None, "place": None, "time": None, "causality": None, "goal": None,
                "notes": "",
            })
        (OUT / f"{sid}_master.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        meta = {"doc_id": sid, "author": m["author"], "gutenberg_id": m["gutenberg_id"],
                "n_sentences": len(sents), "n_transitions": len(rows), "words": m["words"],
                "sentence_split": "build_pieman_pack.split_sentences (regex, whitespace-normalized)"}
        meta_all.append(meta)
        print(f"{sid:22s} {len(sents):4d} sents / {len(rows):4d} transitions")
    if not args.only:  # don't clobber full meta when building a subset
        (OUT / "packs_meta.json").write_text(json.dumps(meta_all, ensure_ascii=False, indent=2))
    tot = sum(x["n_transitions"] for x in meta_all)
    print(f"\n{len(meta_all)} stories, {tot} transitions total -> {OUT}")


if __name__ == "__main__":
    main()

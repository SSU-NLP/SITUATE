#!/usr/bin/env python3
"""Build a Phase-0 master from the Pieman naturalistic free-recall dataset.

Source: phoebsc/Naturalistic-Free-Recall-Dataset (rep_results.csv + transcript).
Human boundaries are the 205-subject button-press event boundaries already mapped
to word onsets in ``rep_results.csv`` (column ``boundary``). We split the punctuated
transcript into sentences, align the word-level boundaries to sentence transitions,
and emit the same master schema consumed by ``llm_enum_states.py``.

Sentence policy (frozen 2026-07-12, EACL ledger): punctuation-based split
(build_narrative_pack.split_sentences), one "sentence" = one EISM state/reset unit.
A human boundary is assigned to the transition AFTER the sentence containing its
boundary word (word-onset alignment via difflib, not cumulative token count).
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
from pathlib import Path

SENTENCE_END = re.compile(r"(?<=[.!?])(?:[\"'”’)]*)\s+(?=[\"'“‘(]*[A-Z0-9])")
WORD = re.compile(r"[a-z]+")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in SENTENCE_END.split(text) if s.strip()]


def load_words(csv_path: Path) -> tuple[list[str], list[float], list[float], list[int]]:
    """Return (word tokens, onsets, offsets, boundary word indices)."""
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    words = [r["word"].strip().lower() for r in rows]
    onsets = [float(r["word_onsets"]) for r in rows]
    offsets = [float(r["word_offsets"]) for r in rows]
    bnd = [i for i, r in enumerate(rows) if r["boundary"] not in ("", "nan", "NaN")]
    return words, onsets, offsets, bnd


def align_words_to_sentences(words: list[str], sentences: list[str]) -> list[int]:
    """Map each csv word index -> sentence index via difflib on lowercased tokens.

    Handles contraction/segmentation drift (e.g. csv 'mcgowans' vs transcript
    'mcgowan' + 's', non-speech 'ns' markers) as insert/replace ops; unmatched csv
    words inherit the sentence id of the previous matched word.
    """
    trans_tokens: list[str] = []
    trans_sent: list[int] = []
    for sid, s in enumerate(sentences):
        for tok in WORD.findall(s.lower()):
            trans_tokens.append(tok)
            trans_sent.append(sid)

    word_sent = [-1] * len(words)
    sm = difflib.SequenceMatcher(None, words, trans_tokens, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                word_sent[i1 + k] = trans_sent[j1 + k]
    # fill unmatched csv words by forward/backward carry
    last = 0
    for i in range(len(word_sent)):
        if word_sent[i] == -1:
            word_sent[i] = last
        else:
            last = word_sent[i]
    return word_sent


def build(csv_path: Path, txt_path: Path, doc_id: str) -> tuple[list[dict], dict]:
    words, onsets, offsets, bnd_word_idx = load_words(csv_path)
    sentences = split_sentences(txt_path.read_text(encoding="utf-8"))
    word_sent = align_words_to_sentences(words, sentences)

    # per-sentence time span from aligned words
    sent_onset = [None] * len(sentences)
    sent_offset = [None] * len(sentences)
    for i, sid in enumerate(word_sent):
        if sent_onset[sid] is None:
            sent_onset[sid] = onsets[i]
        sent_offset[sid] = offsets[i]

    # human boundary -> sentence containing its boundary word -> transition after it
    boundary_after: dict[int, int] = {}  # sentence sid -> count (collisions)
    for i in bnd_word_idx:
        sid = word_sent[i]
        boundary_after[sid] = boundary_after.get(sid, 0) + 1
    collisions = {sid: n for sid, n in boundary_after.items() if n > 1}
    last_sid = len(sentences) - 1
    dropped_last = boundary_after.pop(last_sid, 0)  # boundary at final sentence has no transition after

    rows = []
    for s in range(1, len(sentences)):
        prev_sid = s - 1
        rows.append({
            "doc_id": doc_id,
            "transition": s,
            "previous": sentences[s - 1],
            "current": sentences[s],
            "previous_ko_reference": "",
            "current_ko_reference": "",
            "boundary": int(prev_sid in boundary_after),
            "prev_onset_s": sent_onset[s - 1],
            "prev_offset_s": sent_offset[s - 1],
            "cur_onset_s": sent_onset[s],
            "cur_offset_s": sent_offset[s],
            "idm_score": None,
            "protagonist": None, "place": None, "time": None,
            "causality": None, "goal": None, "notes": "",
        })
    meta = {
        "doc_id": doc_id,
        "n_words": len(words),
        "n_sentences": len(sentences),
        "n_transitions": len(rows),
        "n_boundary_words": len(bnd_word_idx),
        "n_boundary_transitions": sum(r["boundary"] for r in rows),
        "collisions_sentences_with_multiple_boundary_words": collisions,
        "boundary_words_dropped_at_final_sentence": dropped_last,
        "boundary_transitions": [r["transition"] for r in rows if r["boundary"]],
    }
    return rows, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path,
                    default=Path("data/eacl_phase0/pieman/pieman_rep_results.csv"))
    ap.add_argument("--txt", type=Path,
                    default=Path("data/eacl_phase0/pieman/pieman_transcript.txt"))
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/eacl_phase0/pieman_master.jsonl"))
    ap.add_argument("--doc-id", default="pieman")
    args = ap.parse_args()

    rows, meta = build(args.csv, args.txt, args.doc_id)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (args.out.parent / "pieman_pack_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

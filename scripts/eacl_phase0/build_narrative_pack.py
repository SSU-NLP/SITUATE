#!/usr/bin/env python3
"""Build an English-first concatenated-story Phase-0 pack from authorized JSONL.

Input row: story_id, title, text, source_url, license; optional translations[] aligned
to the sentence splitter output. Story order in the input is preserved.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SENTENCE_END = re.compile(r"(?<=[.!?])(?:[\"'”’)]*)\s+(?=[\"'“‘(]*[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in SENTENCE_END.split(text) if s.strip()]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def validate_story(s: dict, sentences: list[str]) -> None:
    required = ("story_id", "title", "text", "source_url", "license")
    missing = [k for k in required if not str(s.get(k, "")).strip()]
    if missing:
        raise ValueError(f"story {s.get('story_id', '?')}: missing provenance {missing}")
    if len(sentences) < 2:
        raise ValueError(f"story {s['story_id']}: fewer than 2 sentences")
    tr = s.get("translations")
    if tr is not None and len(tr) != len(sentences):
        raise ValueError(f"story {s['story_id']}: translations={len(tr)}, sentences={len(sentences)}")


def build(stories: list[dict]) -> list[dict]:
    sequence = []
    for story_no, s in enumerate(stories):
        sentences = split_sentences(s["text"])
        validate_story(s, sentences)
        translations = s.get("translations") or [""] * len(sentences)
        for sentence_no, (en, ko) in enumerate(zip(sentences, translations)):
            sequence.append({
                "story_no": story_no, "story_id": str(s["story_id"]),
                "sentence_no": sentence_no, "en": en, "ko_reference": str(ko),
                "title": s["title"], "source_url": s["source_url"], "license": s["license"],
            })
    rows = []
    for i in range(1, len(sequence)):
        a, b = sequence[i - 1], sequence[i]
        rows.append({
            "doc_id": "concat_" + "_".join(str(s["story_id"]) for s in stories),
            "transition": i,
            "previous": a["en"], "previous_ko_reference": a["ko_reference"],
            "current": b["en"], "current_ko_reference": b["ko_reference"],
            "previous_story_id": a["story_id"], "current_story_id": b["story_id"],
            "boundary": int(a["story_id"] != b["story_id"]),
            "idm_score": None, "protagonist": None, "place": None, "time": None,
            "causality": None, "goal": None, "notes": "",
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stories", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()
    stories = read_jsonl(args.stories)[:args.limit]
    if len(stories) < 2:
        raise SystemExit("At least two authorized stories are required.")
    rows = build(stories)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    n_boundaries = sum(r["boundary"] for r in rows)
    print(f"stories={len(stories)} transitions={len(rows)} free_boundaries={n_boundaries} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

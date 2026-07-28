#!/usr/bin/env python3
"""Deterministically export adequately long WritingPrompts stories for Phase 0."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_narrative_pack import split_sentences  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("output", type=Path)
    ap.add_argument("--split", default="train")
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--min-sentences", type=int, default=10)
    ap.add_argument("--max-sentences", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260711)
    args = ap.parse_args()
    from datasets import load_dataset
    ds = load_dataset("euclaise/writingprompts", split=args.split,
                      download_mode="reuse_dataset_if_exists")
    order = list(range(len(ds)))
    random.Random(args.seed).shuffle(order)
    chosen = []
    for idx in order:
        story = " ".join(str(ds[idx]["story"]).split())
        prompt = " ".join(str(ds[idx]["prompt"]).split())
        sentences = split_sentences(story)
        if not (args.min_sentences <= len(sentences) <= args.max_sentences):
            continue
        if len(story) < 800 or "[deleted]" in story.lower() or "[removed]" in story.lower():
            continue
        chosen.append({
            "story_id": f"writingprompts-{args.split}-{idx}",
            "title": prompt[:240], "prompt": prompt, "text": story,
            "source_url": "https://huggingface.co/datasets/euclaise/writingprompts",
            "license": "unspecified upstream user-content rights; internal research evaluation only",
            "dataset": "euclaise/writingprompts", "split": args.split,
            "row_index": idx, "sentence_count": len(sentences),
        })
        if len(chosen) == args.count:
            break
    if len(chosen) != args.count:
        raise SystemExit(f"only found {len(chosen)}/{args.count} eligible stories")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chosen), encoding="utf-8")
    print("selected rows:", [r["row_index"] for r in chosen])
    print("sentence counts:", [r["sentence_count"] for r in chosen])
    print(f"wrote {len(chosen)} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Tier-2 held-out step A: fetch public-domain short stories and slice single stories out.

Each story = ONE self-contained narrative-fiction short story with MULTIPLE in-story event
boundaries (no concatenation). Diverse authors/styles. Downloads Gutenberg plain text, strips PG
boilerplate, auto-detects ALL-CAPS blank-line-surrounded story headings, and slices each requested
story between its heading and the NEXT heading. Writes `data/eacl_phase0/tier2/{story_id}.txt` +
manifest (source/license/word count) for the ledger.

Usage: uv run python scripts/eacl_phase0/build_tier2_stories.py [--only story_id ...] [--list-headings GID]
"""
from __future__ import annotations
import argparse, json, re, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data/eacl_phase0/tier2"

# story_id -> (gutenberg_id, exact heading text, author, style-note)
# 12 stories / 5 authors (O.Henry reduced to 2 — no single-author dominance; canonical famous works).
MANIFEST = {
    # O. Henry — The Four Million (#2776): American ironic twist.
    "cop_and_anthem":       (2776, "THE COP AND THE ANTHEM", "O. Henry", "ironic episodic (Soapy's attempts)"),
    "the_furnished_room":   (2776, "THE FURNISHED ROOM", "O. Henry", "tragic search"),
    # Maupassant — Original Short Stories v4 (#3080): French realism. (avoid PART-subheaded stories)
    "the_diamond_necklace": (3080, "THE DIAMOND NECKLACE", "Maupassant", "10-year time jump"),
    "the_wolf":             (3080, "THE WOLF", "Maupassant", "hunt, framed tale"),
    "the_wreck":            (3080, "THE WRECK", "Maupassant", "shipwreck"),
    # Chekhov — Tales v3 (#13415): Russian realism.
    "the_lady_with_the_dog":(13415, "THE LADY WITH THE DOG", "Chekhov", "multi-city, time jumps"),
    "ionitch":              (13415, "IONITCH", "Chekhov", "years-spanning decline"),
    # (Poe gothic 3편 cask/masque/usher 제거 — gpt-4.1 content-moderation이 verbatim 재현 차단, gold 0개.
    #  D-M9 feasibility amendment 2026-07-15; 아래 non-gothic 3편으로 교체.)
    "the_green_door":       (2776, "THE GREEN DOOR", "O. Henry", "urban adventure"),
    "head_of_the_family":   (13415, "THE HEAD OF THE FAMILY", "Chekhov", "domestic comedy"),
    # Crane — The Open Boat and Other Stories (#45524): American realism.
    "bride_yellow_sky":     (45524, "THE BRIDE COMES TO YELLOW SKY", "Crane", "4 explicit scenes"),
    "a_man_and_some_others":(45524, "A MAN AND SOME OTHERS", "Crane", "frontier conflict"),
    "the_wise_men":         (45524, "THE WISE MEN", "Crane", "comic gambling tale"),
    # (the_open_boat excluded: 670 sentences, exceeds dev range (Tunnel 504) per tier2 spec §3)
}


def fetch(gid: int) -> str:
    url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def body_lines(text: str) -> list[str]:
    lines = text.splitlines()
    try:
        s = next(i for i, l in enumerate(lines) if "START OF THE PROJECT GUTENBERG" in l) + 1
    except StopIteration:
        s = 0
    try:
        e = next(i for i, l in enumerate(lines) if "END OF THE PROJECT GUTENBERG" in l)
    except StopIteration:
        e = len(lines)
    return lines[s:e]


def headings(lines: list[str]) -> list[tuple[int, str]]:
    out = []
    for i, l in enumerate(lines):
        t = l.strip()
        if t and t == t.upper() and re.match(r"^[A-Z][A-Z0-9 '\-\.,]+$", t) and len(t) > 4:
            prev_blank = (i == 0) or (not lines[i - 1].strip())
            next_blank = (i + 1 >= len(lines)) or (not lines[i + 1].strip())
            if prev_blank and next_blank:
                out.append((i, t))
    return out


def slice_story(lines: list[str], title: str) -> str:
    hs = headings(lines)
    idxs = [i for i, t in hs if t == title]
    if not idxs:
        raise ValueError(f"heading '{title}' not found. Available: {[t for _, t in hs][:40]}")
    start = idxs[-1] + 1  # body heading = last occurrence (earlier ones are TOC)
    later = [i for i, _ in hs if i > start]
    end = later[0] if later else len(lines)
    seg = "\n".join(lines[start:end]).strip()
    paras = [re.sub(r"\s*\n\s*", " ", p).strip() for p in re.split(r"\n\s*\n", seg)]
    return "\n\n".join(p for p in paras if p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--list-headings", type=int, help="print detected headings for a gutenberg id and exit")
    args = ap.parse_args()
    if args.list_headings:
        for i, t in headings(body_lines(fetch(args.list_headings))):
            print(i, t)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    cache: dict[int, list[str]] = {}
    manifest_out = []
    for sid in (args.only or list(MANIFEST)):
        gid, title, author, note = MANIFEST[sid]
        if gid not in cache:
            cache[gid] = body_lines(fetch(gid))
        story = slice_story(cache[gid], title)
        (OUT / f"{sid}.txt").write_text(story + "\n")
        nw = len(story.split())
        manifest_out.append(dict(story_id=sid, title=title, author=author, style=note,
                                 gutenberg_id=gid, words=nw,
                                 source=f"https://www.gutenberg.org/ebooks/{gid}",
                                 license="Public domain (Project Gutenberg)"))
        print(f"{sid:20s} {author:14s} {nw:5d} words")
    (OUT / "manifest.json").write_text(json.dumps(manifest_out, ensure_ascii=False, indent=2))
    print(f"\nmanifest -> {OUT/'manifest.json'} ({len(manifest_out)} stories)")


if __name__ == "__main__":
    main()

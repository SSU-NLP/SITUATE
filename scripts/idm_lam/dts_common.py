"""Shared DTS Data Hub loading + SBERT embedding cache for the IDM-LAM Go/No-Go.

DTS format: one JSON object per line, ``{"id":..., "dialogue": "<turn>[NEWLINE]<turn>...[NEWLINE][BOUNDARY][NEWLINE]<turn>..."}``.
``[BOUNDARY]`` markers sit *between* turns; the real turn immediately after a
marker opens a new segment. We parse each dialogue into ``(turns, boundaries)``
where ``boundaries`` is the set of turn indices that START a new segment
(index 0 is never a boundary — it opens the first segment implicitly).

Encoder = ``sentence-transformers/paraphrase-multilingual-mpnet-base-v2`` (768D,
L2-normalized) to match the HANDOFF_06 delta_eff baseline so that Gate-2's
"same embedding, different signal" comparison (baseline B') is fair.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DTS_DIR = ROOT / "benchmarks/Def-DTS/data/DTS_session_datasets"
CACHE_DIR = ROOT / "outputs/experiments/2026-07-09_idm_lam_gonogo/cache"
ENCODER_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
ENC_DIM = 768


def parse_dialogue(dialogue: str) -> tuple[list[str], set[int]]:
    """Return (turn_texts_without_speaker_prefix, boundary_turn_indices)."""
    turns: list[str] = []
    boundaries: set[int] = set()
    pending = False
    for item in dialogue.split("[NEWLINE]"):
        if item == "[BOUNDARY]":
            pending = True
            continue
        if not item:
            continue
        # strip leading "user: "/"agent: " speaker tag, keep content
        if item.startswith("user:") or item.startswith("agent:"):
            text = item.split(":", 1)[1].strip()
        else:
            text = item.strip()
        if pending and turns:
            boundaries.add(len(turns))
            pending = False
        turns.append(text)
    return turns, boundaries


def load_split(name: str) -> list[dict]:
    """Load a DTS split -> list of {id, turns:list[str], boundaries:set[int]}."""
    path = DTS_DIR / f"{name}.jsonl"
    out = []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            turns, bnd = parse_dialogue(r["dialogue"])
            if len(turns) < 2:
                continue
            out.append({"id": r["id"], "turns": turns, "boundaries": bnd})
    return out


_ENCODER = None


def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer

        _ENCODER = SentenceTransformer(ENCODER_NAME, device="cuda")
    return _ENCODER


def embed_split(name: str, force: bool = False) -> list[dict]:
    """Load split and attach ``emb`` (n_turns x 768, L2-normalized) per dialogue.

    Caches to ``cache/emb_<name>.npz`` keyed by a hash of the turn texts so a
    changed parse invalidates the cache.
    """
    data = load_split(name)
    all_turns = [t for d in data for t in d["turns"]]
    key = hashlib.md5(("␟".join(all_turns)).encode()).hexdigest()[:12]
    cache = CACHE_DIR / f"emb_{name}_{key}.npz"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if cache.exists() and not force:
        z = np.load(cache)
        embs = z["embs"]
    else:
        enc = _get_encoder()
        embs = enc.encode(
            all_turns,
            batch_size=256,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        np.savez_compressed(cache, embs=embs)
    # scatter back per dialogue
    i = 0
    for d in data:
        n = len(d["turns"])
        d["emb"] = embs[i : i + n]
        i += n
    return data


if __name__ == "__main__":
    import sys

    for split in sys.argv[1:] or [
        "superseg_train",
        "superseg_validation",
        "superseg_test",
        "tiage_test",
        "dialseg711_test",
    ]:
        data = embed_split(split)
        nt = sum(len(d["turns"]) for d in data)
        nb = sum(len(d["boundaries"]) for d in data)
        print(f"{split}: {len(data)} dialogues, {nt} turns, {nb} boundaries")

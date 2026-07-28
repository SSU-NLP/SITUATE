"""AMI held-out eval loader: embed eval meetings with OUR encoder + gold bnd_top.

AMI scorer convention (src/hi_ontop/ami_scoring.py): ±2 tolerant boundary F1, start-turn
alignment (shift 0), Pk/WD via nltk, Score=0.5*F1+0.25(1-Pk)+0.25(1-WD), macro over meetings.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_ami_eval():
    """Return list of {id, turns, emb(n x768 L2), bnd_top(list 0/1 start-turn)} for HELD-OUT meetings."""
    from sentence_transformers import SentenceTransformer
    from dts_common import ENCODER_NAME  # reuse same encoder

    split = json.loads((ROOT / "data/ami/split.json").read_text())
    enc = SentenceTransformer(ENCODER_NAME, device="cuda")
    out = []
    for mid in split["eval"]:
        d = json.loads((ROOT / f"data/ami/topic/{mid}.json").read_text())
        turns = [" ".join(t["text"].split()) for t in d["turns"]]
        emb = enc.encode(turns, batch_size=256, normalize_embeddings=True,
                         convert_to_numpy=True).astype(np.float32)
        out.append({"id": mid, "turns": turns, "emb": emb, "bnd_top": list(d["bnd_top"])})
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    data = load_ami_eval()
    nt = sum(len(d["turns"]) for d in data)
    nb = sum(sum(d["bnd_top"]) for d in data)
    print(f"AMI eval: {len(data)} meetings, {nt} turns, {nb} gold boundaries")

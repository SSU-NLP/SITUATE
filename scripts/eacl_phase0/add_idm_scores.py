#!/usr/bin/env python3
"""Fill Phase-0 ``idm_score`` with the official Hi-OnTop delta_eff stream."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from hi_ontop.embedding import make_encoder  # noqa: E402
from hi_ontop.hi_ontop import HiOnTop  # noqa: E402

DEFAULT_ENCODER = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("annotations", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--ctx-window", type=int, default=2)
    ap.add_argument("--ctx-decay", type=float, default=.7)
    ap.add_argument("--ctx-blend-a", type=float, default=.5)
    args = ap.parse_args()
    rows = [json.loads(x) for x in args.annotations.read_text(encoding="utf-8").splitlines() if x.strip()]
    by_doc: dict[str, list[dict]] = {}
    for r in rows:
        by_doc.setdefault(str(r["doc_id"]), []).append(r)
    enc = make_encoder(model=args.encoder)
    for doc_rows in by_doc.values():
        doc_rows.sort(key=lambda r: int(r["transition"]))
        texts = [doc_rows[0]["previous"], *[r["current"] for r in doc_rows]]
        embs = np.asarray(enc.encode(texts))
        probe = HiOnTop(dim=enc.dim, delta_star=2.0, ctx_window=args.ctx_window,
                        ctx_decay=args.ctx_decay, ctx_blend_a=args.ctx_blend_a)
        for v in embs:
            probe.assign(v)
        scores = [float(h["delta_eff"]) for h in probe.history()]
        for r, score in zip(doc_rows, scores[1:]):
            r["idm_score"] = score
            r["idm_encoder"] = args.encoder
    out = args.out or args.annotations
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"filled {len(rows)} IDM scores -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

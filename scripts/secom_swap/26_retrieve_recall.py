"""Retrieve top-k segments per question; measure gold-evidence recall@k by category.

For a density-matched segments file:
- embed each segment (its turn texts joined) with mpnet (multi-qa-mpnet-base-dot-v1),
- embed each question, cosine top-k segments,
- map retrieved segments -> constituent dia_ids,
- recall@k = |gold_evidence ∩ retrieved_dia_ids| / |gold_evidence|, averaged per
  LoCoMo category (1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial).

Also dumps the top-k retrieved segment texts per question (for the QA stage) at the
largest k requested.

Output: recall/<method>__tps{t}.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "outputs/experiments/2026-07-15_locomo_segmenter_swap"
DATA = BASE / "data/locomo_turns.jsonl"
SEGDIR = BASE / "segments"
OUTDIR = BASE / "recall"
CATNAME = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
DEV = ["conv-26", "conv-30", "conv-41", "conv-42", "conv-43"]
TEST = ["conv-44", "conv-47", "conv-48", "conv-49", "conv-50"]


def seg_text(seg):
    return "\n".join(f"{t['speaker']}: {t['text']}" for t in seg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", required=True, help="segments/<method>__tps{t}.json")
    ap.add_argument("--topks", default="1,2,3,5")
    args = ap.parse_args()
    topks = [int(x) for x in args.topks.split(",")]
    kmax = max(topks)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    convs = {json.loads(l)["conversation_id"]: json.loads(l) for l in DATA.open()}
    seg_rows = {r["conversation_id"]: r["segments"] for r in json.loads(Path(args.segments).read_text())}

    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(
        "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # recall accumulators: split -> k -> category -> list of per-question recall
    splits = {"all": set(convs), "dev": set(DEV), "test": set(TEST)}
    rec = {sp: {k: defaultdict(list) for k in topks} for sp in splits}
    qa_ctx = []  # for QA stage: {cid, qi, question, answer, category, contexts(topk_max)}
    for cid, conv in convs.items():
        which = [sp for sp, s in splits.items() if cid in s]
        segs = seg_rows[cid]
        seg_texts = [seg_text(s) for s in segs]
        seg_dia = [set(t["dia_id"] for t in s) for s in segs]
        seg_emb = np.asarray(enc.encode(seg_texts, normalize_embeddings=True,
                                        show_progress_bar=False, batch_size=128))
        qs = [q["question"] for q in conv["qa"]]
        if not qs:
            continue
        q_emb = np.asarray(enc.encode(qs, normalize_embeddings=True,
                                      show_progress_bar=False, batch_size=128))
        sims = q_emb @ seg_emb.T  # (Q, S)
        order = np.argsort(-sims, axis=1)  # descending
        for qi, q in enumerate(conv["qa"]):
            gold = set(q["evidence"])
            cat = q["category"]
            ranked = order[qi]
            for k in topks:
                topk_segs = ranked[:k]
                got = set().union(*[seg_dia[j] for j in topk_segs]) if len(topk_segs) else set()
                if gold:
                    r = len(gold & got) / len(gold)
                    for sp in which:
                        rec[sp][k][cat].append(r)
            top_ctx = [seg_texts[j] for j in ranked[:kmax]]
            qa_ctx.append({"conversation_id": cid, "qi": qi, "question": q["question"],
                           "answer": q["answer"], "category": cat, "contexts": top_ctx})

    # aggregate per split
    all_cats = sorted(CATNAME)
    summary = {"segments_file": Path(args.segments).name, "topks": topks, "splits": {}}
    for sp in splits:
        by_cat = {}
        overall = {}
        for k in topks:
            for cat in all_cats:
                vals = rec[sp][k][cat]
                by_cat.setdefault(str(cat), {"name": CATNAME[cat], "n": len(rec[sp][topks[0]][cat])})[
                    f"recall@{k}"] = float(np.mean(vals)) if vals else None
            allvals = [v for cat in all_cats for v in rec[sp][k][cat]]
            overall[f"recall@{k}"] = float(np.mean(allvals)) if allvals else None
        summary["splits"][sp] = {"by_category": by_cat, "overall": overall}

    name = Path(args.segments).stem
    (OUTDIR / f"{name}.json").write_text(json.dumps(summary, indent=2))
    (OUTDIR / f"{name}.qactx.json").write_text(json.dumps(qa_ctx, ensure_ascii=False))
    for sp in ["test", "dev", "all"]:
        print(f"== {name} [{sp}] ==")
        print(f"{'category':13} n    " + "  ".join(f"r@{k}" for k in topks))
        for cat in all_cats:
            row = summary["splits"][sp]["by_category"][str(cat)]
            cells = "  ".join(f"{row[f'recall@{k}']:.3f}" if row[f'recall@{k}'] is not None else "  -  " for k in topks)
            print(f"{CATNAME[cat]:13} {row['n']:<4} {cells}")
        ov = summary["splits"][sp]["overall"]
        print("overall           " + "  ".join(f"{ov[f'recall@{k}']:.3f}" for k in topks))
    print(f"saved -> recall/{name}.json")


if __name__ == "__main__":
    main()

"""Build a LARGE, DIVERSE, UNLABELED dialogue corpus and embed it (held-out eval untouched).

Sources (all UNLABELED, training only):
  * SODA (allenai/soda)                 — streamed, social commonsense dialogues (millions).
  * Synthetic-Persona-Chat (google)     — persona chit-chat.
  * AMI train meetings (data/ami/split.json 'train') — NOISY SPOKEN dialogue w/ fillers.

Eval sets (TIAGE/DialSeg711/SuperDialseg/AMI-eval) are NEVER touched here.

Output (to scratch, regenerable): emb_all.npy (fp16 U x 768, L2-normalized) + dlg_lens.npy
+ meta.json.  Training forms adjacent (z_t, z_{t+1}) pairs from dlg_lens offsets.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = Path(os.environ["CLAUDE_SCRATCH"]) if "CLAUDE_SCRATCH" in os.environ else Path(
    "/tmp/claude-0/-root/d13a28d4-7d56-4cb2-9443-281710e6c27a/scratchpad")
os.environ.setdefault("HF_DATASETS_CACHE", str(SCRATCH / "hf_datasets"))
CORPUS = SCRATCH / "idm_corpus"
ENCODER_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def clean(s: str) -> str:
    return " ".join(str(s).split()).strip()


def _turns_ok(turns):
    turns = [clean(t) for t in turns if clean(t)]
    return turns if len(turns) >= 2 else None


def soda_dialogues(n):
    from datasets import load_dataset
    seen = 0
    for sp in ["train", "validation", "test"]:
        ds = load_dataset("allenai/soda", split=sp, streaming=True)
        for r in ds:
            if seen >= n:
                return
            t = _turns_ok(r["dialogue"])
            if t:
                seen += 1
                yield t


def persona_synth_dialogues():
    from datasets import load_dataset
    ds = load_dataset("google/Synthetic-Persona-Chat", split="train")
    for r in ds:
        conv = r.get("Best Generated Conversation") or ""
        lines = [clean(l.split(":", 1)[1]) if ":" in l else clean(l)
                 for l in conv.split("\n") if clean(l)]
        t = _turns_ok(lines)
        if t:
            yield t


def bst_dialogues():
    from datasets import load_dataset
    ds = load_dataset("blended_skill_talk", split="train", streaming=True)
    for r in ds:
        turns = []
        for g, f in zip(r["guided_messages"], r["free_messages"]):
            turns += [g, f]
        t = _turns_ok(turns)
        if t:
            yield t


def emp_dialogues():
    from datasets import load_dataset
    ds = load_dataset("Estwld/empathetic_dialogues_llm", split="train", streaming=True)
    for r in ds:
        t = _turns_ok([c["content"] for c in r["conversations"]])
        if t:
            yield t


def personachat_dialogues():
    from datasets import load_dataset
    ds = load_dataset("AlekseyKorshuk/persona-chat", split="train", streaming=True)
    for r in ds:
        if not r["utterances"]:
            continue
        t = _turns_ok(list(r["utterances"][-1]["history"]))
        if t:
            yield t


def ultrachat_dialogues(cap):
    from datasets import load_dataset
    ds = load_dataset("stingning/ultrachat", split="train", streaming=True)
    seen = 0
    for r in ds:
        if seen >= cap:
            return
        t = _turns_ok(list(r["data"]))
        if t:
            seen += 1
            yield t


def ultrachat200k_dialogues():
    from datasets import load_dataset
    for sp in ["train_sft", "test_sft"]:
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=sp, streaming=True)
        for r in ds:
            t = _turns_ok([m["content"] for m in r["messages"]])
            if t:
                yield t


def hhrlhf_dialogues():
    from datasets import load_dataset
    import re
    ds = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
    for r in ds:
        parts = re.split(r"\n\n(?:Human|Assistant):\s*", r["chosen"])
        t = _turns_ok([p for p in parts if p.strip()])
        if t:
            yield t


def ami_train_dialogues():
    split = json.loads((ROOT / "data/ami/split.json").read_text())
    for mid in split["train"]:
        d = json.loads((ROOT / f"data/ami/topic/{mid}.json").read_text())
        t = _turns_ok([tt["text"] for tt in d["turns"]])
        if t:
            yield t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-soda", type=int, default=10_000_000)  # effectively all of SODA
    ap.add_argument("--n-ultrachat", type=int, default=1_500_000)  # effectively all
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--resume", action="store_true",
                    help="resume from last shard (skip already-embedded dialogues, no recompute)")
    ap.add_argument("--sources", nargs="*", default=None, help="restrict to these source names")
    args = ap.parse_args()
    CORPUS.mkdir(parents=True, exist_ok=True)

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(ENCODER_NAME, device="cuda")

    pending: list[np.ndarray] = []      # in-RAM emb chunks not yet written to a shard
    pending_n = 0
    dlg_lens: list[int] = []
    buf: list[str] = []
    buf_lens: list[int] = []
    counts: dict[str, int] = {}
    n_utts = 0
    shard_id = 0
    SHARD_UTTS = 2_000_000              # write a shard every ~2M utts (safety/resumability)

    # ---- RESUME: restore committed shard-boundary state; skip already-embedded dialogues ----
    skip_n = 0
    if args.resume and (CORPUS / "progress.json").exists():
        pj = json.loads((CORPUS / "progress.json").read_text())
        skip_n = pj["n_dialogues"]          # dialogues already committed to shards
        shard_id = pj["shards"]             # next shard id
        counts = dict(pj["counts"])         # per-source committed counts
        n_utts = pj["n_utts"]
        dlg_lens = list(np.load(CORPUS / "dlg_lens.npy").astype(int))
        assert len(dlg_lens) == skip_n, f"dlg_lens {len(dlg_lens)} != committed {skip_n}"
        print(f"[RESUME] skip {skip_n} dialogues ({n_utts} utts, {shard_id} shards already done); "
              f"only re-streaming (no re-embed) to fast-forward.", flush=True)
    g = 0  # global dialogue counter across all sources, in deterministic order

    def write_shard():
        nonlocal shard_id, pending, pending_n
        if not pending:
            return
        np.save(CORPUS / f"emb_shard_{shard_id:03d}.npy", np.concatenate(pending, 0))
        np.save(CORPUS / "dlg_lens.npy", np.array(dlg_lens, dtype=np.int32))
        (CORPUS / "progress.json").write_text(json.dumps(
            {"n_utts": n_utts, "n_dialogues": len(dlg_lens), "shards": shard_id + 1,
             "counts": counts}, indent=2))
        print(f"  [shard {shard_id:03d}] written; total {n_utts} utts, {len(dlg_lens)} dlgs", flush=True)
        shard_id += 1
        pending = []
        pending_n = 0

    def flush():
        nonlocal n_utts, pending_n
        if not buf:
            return
        e = enc.encode(buf, batch_size=args.batch, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
        pending.append(e)
        pending_n += len(e)
        dlg_lens.extend(buf_lens)
        n_utts += len(buf)
        buf.clear(); buf_lens.clear()
        if pending_n >= SHARD_UTTS:
            write_shard()

    def feed(genfn, name):
        nonlocal g
        counts.setdefault(name, 0)
        try:
            for turns in genfn():
                if g < skip_n:          # already committed in a prior run -> fast-forward, no embed
                    g += 1
                    continue
                g += 1
                buf.extend(turns)
                buf_lens.append(len(turns))
                counts[name] += 1
                if len(buf) >= 60000:
                    flush()
        except Exception as e:
            print(f"[{name}] error, partial kept: {e}", flush=True)
        print(f"  {name} dialogues: {counts[name]}", flush=True)

    sources = [
        ("soda", lambda: soda_dialogues(args.n_soda)),
        ("ultrachat", lambda: ultrachat_dialogues(args.n_ultrachat)),
        ("ultrachat_200k", ultrachat200k_dialogues),
        ("hh_rlhf", hhrlhf_dialogues),
        ("persona_synth", persona_synth_dialogues),
        ("blended_skill_talk", bst_dialogues),
        ("empathetic", emp_dialogues),
        ("personachat", personachat_dialogues),
        ("ami_train", ami_train_dialogues),
    ]
    if args.sources:
        keep = set(args.sources)
        sources = [(n, f) for n, f in sources if n in keep]
    for i, (name, fn) in enumerate(sources):
        print(f"[{i+1}/{len(sources)}] {name}...", flush=True)
        feed(fn, name)
    flush()
    write_shard()

    lens = np.array(dlg_lens, dtype=np.int32)
    n_pairs = int((lens - 1).sum())
    np.save(CORPUS / "dlg_lens.npy", lens)
    (CORPUS / "meta.json").write_text(json.dumps(
        {"n_utts": int(n_utts), "n_dialogues": int(len(lens)), "n_pairs": n_pairs,
         "n_shards": shard_id, "counts": counts, "encoder": ENCODER_NAME, "dim": 768}, indent=2))
    print(f"DONE: {n_utts} utts, {len(lens)} dialogues, {n_pairs} pairs, "
          f"{shard_id} shards -> {CORPUS}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Tier-2 held-out step C: LLM-silver event-boundary gold (B2 prompt + B3 procedure).

Per tier2-heldout-spec.md: send each story to GPT-4 with the VALIDATED B2/B3 zero-shot prompt
(verbatim), R times; the model copies the story and inserts a newline at each event boundary.
We map each run's inserted boundaries to sentence transitions by word-offset alignment (robust to
minor copy drift), then take a majority-vote consensus (boundary iff >= ceil(R/2) runs place it).

Cross-family (D-M6-b): gold = GPT-4 (OpenAI) vs M3 extractor = claude-fable-5 (Anthropic).
Circularity (D-M6-c): B2/B3 must NOT be baselines on this data.

Writes `outputs/eacl_phase0/tier2/{sid}_gold.json` (per-run + consensus boundary transitions) and,
with --apply, fills the `boundary` column of each `{sid}_master.jsonl`.

Usage:
  uv run python scripts/eacl_phase0/build_tier2_gold.py --reps 20 --temp 1.0 [--only sid] [--apply]
  uv run python scripts/eacl_phase0/build_tier2_gold.py --smoke  # 1 story, 1 rep, no write
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
from build_pieman_pack import split_sentences  # noqa: E402

OUT = REPO / "outputs/eacl_phase0/tier2"
TXT = REPO / "data/eacl_phase0/tier2"
# D-M9 feasibility amendment (codex 2026-07-15): frozen `gpt-4` (8k) truncates verbatim copies of
# stories >~2700 words. GPT-4.1 (1M ctx / 32k out) losslessly runs B3's procedure. Framed as the
# functional succession of B3's validated GPT-4 procedure, NOT "the exact model B3 validated"
# (B3 only reports model='gpt-4', no snapshot; GPT-4o is a *newer alternative* they cite, not used).
GOLD_MODEL = "openrouter/openai/gpt-4.1"

PROMPT_HEAD = ("An event is an ongoing coherent situation. The following story needs to be\n"
               "copied and segmented into large events. Copy the following story word-for-word\n"
               "and start a new line whenever one event ends, and another begins.\n"
               "This is the story:\n\n")
PROMPT_TAIL = ("\n\nThis is a word-for-word copy of the same story that is segmented into\n"
               "large event units:")

WORD = re.compile(r"\w+")


def client():
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")
    from openai import OpenAI
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"],
                  default_headers={"CF-Access-Client-Id": os.environ["CF_ACCESS_CLIENT_ID"],
                                   "CF-Access-Client-Secret": os.environ["CF_ACCESS_CLIENT_SECRET"]})


def sentence_word_bounds(sents: list[str]) -> list[int]:
    """cumulative word count at the END of each sentence -> word offset of each sentence boundary."""
    cum, tot = [], 0
    for s in sents:
        tot += len(WORD.findall(s))
        cum.append(tot)
    return cum  # cum[k] = words through sentence k (0-indexed)


def run_to_transitions(gpt_text: str, sents: list[str]) -> list[int]:
    """Map an LLM segmented copy -> gold boundary transitions (1-based, end-turn).

    Segment boundaries = blank/newline breaks. For each boundary, its word offset (cumulative words
    of preceding segments) is snapped to the nearest sentence boundary -> transition index.
    """
    segs = [s for s in re.split(r"\n\s*\n|\n", gpt_text.strip()) if s.strip()]
    cum_sent = sentence_word_bounds(sents)  # word offset per sentence end
    total = cum_sent[-1]
    # word offset at each inter-segment boundary
    off, acc = [], 0
    for seg in segs[:-1]:
        acc += len(WORD.findall(seg))
        off.append(acc)
    trans = set()
    for o in off:
        o = min(o, total)
        # nearest sentence index k such that boundary is AFTER sentence k (0-indexed) -> transition k+1
        k = min(range(len(cum_sent)), key=lambda i: abs(cum_sent[i] - o))
        t = k + 1  # transition between sentence k and k+1 (1-based)
        if 1 <= t <= len(sents) - 1:
            trans.add(t)
    return sorted(trans)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--temp", type=float, default=0.0)  # codex 2026-07-15: B3 used temp 0 (highest run-consistency; authors recommend 0)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--apply", action="store_true", help="write consensus into {sid}_master.jsonl boundary")
    ap.add_argument("--smoke", action="store_true", help="1 story, 1 rep, print, no write")
    ap.add_argument("--workers", type=int, default=8, help="concurrent API calls (protocol-neutral speedup)")
    ap.add_argument("--force", action="store_true", help="regenerate even if {sid}_gold.json exists")
    args = ap.parse_args()

    manifest = json.loads((TXT / "manifest.json").read_text())
    sids = [m["story_id"] for m in manifest]
    if args.only:
        sids = [s for s in sids if s in args.only]
    if args.smoke:
        sids, args.reps = sids[:1], 1
    c = client()
    _done = threading.Lock()
    _ctr = {"n": 0}

    meta = {"returned_models": set(), "finish_reasons": []}

    def one_valid_run(sid, story, sents, orig_words):
        """One valid GPT copy->transitions. Retries on transient API errors (5xx/timeout, backoff)
        AND on mechanical failure (truncation/incomplete copy). Thread-safe."""
        for attempt in range(10):
            try:
                resp = c.chat.completions.create(
                    model=GOLD_MODEL, temperature=args.temp, max_tokens=32000, timeout=300,
                    messages=[{"role": "user", "content": PROMPT_HEAD + story + PROMPT_TAIL}])
            except Exception as e:  # transient proxy/origin errors (504, timeouts) -> backoff + retry
                wait = min(120, 8 * (attempt + 1))
                print(f"[{sid}] API error {type(e).__name__} -> backoff {wait}s", flush=True)
                time.sleep(wait)
                continue
            out = resp.choices[0].message.content or ""
            fr = resp.choices[0].finish_reason
            if args.smoke:
                (OUT / "_smoke_raw.txt").write_text(out)
            # mechanical failure = truncated (finish_reason=length) or incomplete copy
            if fr != "length" and len(WORD.findall(out)) >= 0.8 * orig_words:
                meta["returned_models"].add(getattr(resp, "model", "?"))
                meta["finish_reasons"].append(fr)
                return run_to_transitions(out, sents)
            print(f"[{sid}] REJECT (finish={fr}) -> re-call", flush=True)
        raise RuntimeError(f"[{sid}] too many failures (API/mechanical)")

    for sid in sids:
        if not args.force and not args.smoke and (OUT / f"{sid}_gold.json").exists():
            print(f"[{sid}] skip (gold.json exists)", flush=True)
            continue
        text = (TXT / f"{sid}.txt").read_text()
        sents = split_sentences(text)
        story = re.sub(r"\s+", " ", text).strip()  # continuous block: GPT newline = boundary
        orig_words = len(WORD.findall(story))
        _ctr["n"] = 0
        meta["returned_models"].clear(); meta["finish_reasons"].clear()
        with ThreadPoolExecutor(max_workers=1 if args.smoke else args.workers) as ex:
            futs = [ex.submit(one_valid_run, sid, story, sents, orig_words) for _ in range(args.reps)]
            runs, n_fail = [], 0
            for f in as_completed(futs):
                try:
                    runs.append(f.result())
                except Exception as e:  # a run exhausted retries (origin overload) — don't kill the batch
                    n_fail += 1
                    continue
                with _done:
                    _ctr["n"] += 1
                    print(f"[{sid}] rep {_ctr['n']}/{args.reps}: {len(runs[-1])} boundaries", flush=True)
        if n_fail or len(runs) < args.reps:  # incomplete -> skip, no partial gold; re-resume later
            print(f"[{sid}] INCOMPLETE ({len(runs)}/{args.reps}, {n_fail} failed) -> skip, retry later", flush=True)
            continue
        # B3 top-n consensus (codex 2026-07-15, D-M9): n = half-up(mean per-run count); top-n by votes,
        # cutoff ties broken by fixed SHA-256 of the transition index. NO density adjustment.
        tally = {}
        for tr in runs:
            for t in tr:
                tally[t] = tally.get(t, 0) + 1
        n = int(math.floor(sum(len(tr) for tr in runs) / len(runs) + 0.5))
        ranked = sorted(tally.items(),
                        key=lambda kv: (-kv[1], hashlib.sha256(str(kv[0]).encode()).hexdigest()))
        consensus = sorted(t for t, _ in ranked[:n])
        rec = {"doc_id": sid, "model": GOLD_MODEL, "reps": args.reps, "temp": args.temp,
               "consensus_rule": "B3 top-n (n=half-up mean per-run count; SHA-256 tie-break)",
               "n_target": n, "n_sentences": len(sents),
               "consensus_boundaries": consensus, "n_consensus": len(consensus),
               "per_run_counts": [len(tr) for tr in runs], "tally": tally, "runs": runs,
               "qc_boundaries_per_1k_words": round(1000 * n / max(orig_words, 1), 2),
               "qc_unique_segmentations": len({tuple(tr) for tr in runs}),
               "returned_model": sorted(meta["returned_models"]),
               "finish_reasons": meta["finish_reasons"]}
        if args.smoke:
            print(json.dumps({k: rec[k] for k in ("doc_id", "n_sentences", "n_target",
                  "consensus_boundaries", "per_run_counts")}, ensure_ascii=False, indent=2))
            return
        (OUT / f"{sid}_gold.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2))
        print(f"[{sid}] consensus {len(consensus)} boundaries (n_target={n})", flush=True)
        if args.apply:
            mp = OUT / f"{sid}_master.jsonl"
            rows = [json.loads(l) for l in mp.read_text().splitlines() if l.strip()]
            cset = set(consensus)
            for row in rows:
                row["boundary"] = int(int(row["transition"]) in cset)
            mp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            print(f"[{sid}] applied gold -> {mp}", flush=True)


if __name__ == "__main__":
    main()

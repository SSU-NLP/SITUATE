"""Extract M3 5-axis states for ALL LoCoMo conversations (parallel, cached).

Reuses the frozen M3 Stage1/Stage2 (llm_s5_states). Parallelizes Stage2 batches
with a thread pool so all 10 conversations finish in minutes instead of hours.
States cache to ``m3_states/<cid>/<axis>.jsonl`` (resumable; skips done axes).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/eacl_phase0"))
from llm_s5_states import (  # noqa: E402
    AXES, STAGE1_PROMPT, STAGE1_SCHEMA, stage2_call, valid,
)


def stage1_big(c, model, story, max_tokens=6000):
    """Stage1 with a larger token budget (big LoCoMo conversations overflow the
    frozen 2000-token cap and truncate the JSON). Same prompt/schema."""
    import time as _t

    prompt = STAGE1_PROMPT.format(story=story)
    for a in range(3):
        try:
            r = c.chat.completions.create(
                model=model, temperature=0, max_tokens=max_tokens,
                response_format={"type": "json_schema", "json_schema":
                                 {"name": "story_entities", "strict": True, "schema": STAGE1_SCHEMA}},
                messages=[{"role": "system", "content":
                           "You enumerate the entities present in a text as normalized identifiers. "
                           "You never discuss structure, order, or boundaries."},
                          {"role": "user", "content": prompt}])
            return json.loads(r.choices[0].message.content), prompt
        except Exception:
            if a == 2:
                raise
            _t.sleep(2 * (a + 1))

DATA = REPO_ROOT / "outputs/experiments/2026-07-15_locomo_segmenter_swap/data/locomo_turns.jsonl"
CACHE = REPO_ROOT / "outputs/experiments/2026-07-15_locomo_segmenter_swap/m3_states"
SEED = 20260711


def make_client():
    load_dotenv(REPO_ROOT / ".env")
    from openai import OpenAI

    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        default_headers={
            "CF-Access-Client-Id": os.environ["CF_ACCESS_CLIENT_ID"],
            "CF-Access-Client-Secret": os.environ["CF_ACCESS_CLIENT_SECRET"],
        },
    )


def conv_texts(conv: dict) -> list[str]:
    return [f"{t['speaker']}: {t['text']}" for s in conv["sessions"] for t in s]


def extract_axis(c, model, lists, texts, axis, sp: Path, workers: int):
    done: dict[int, dict] = {}
    if sp.exists():
        for line in sp.open():
            x = json.loads(line)
            done[int(x["item_id"])] = x
    if len(done) >= len(texts):
        return
    ids = list(range(len(texts)))
    random.Random(SEED + AXES.index(axis)).shuffle(ids)
    todo = [i for i in ids if i not in done]
    batches = [todo[k : k + 20] for k in range(0, len(todo), 20)]

    def run_batch(batch):
        got, _ = stage2_call(c, model, axis, lists, [(i, texts[i]) for i in batch])
        out = {}
        for x in got:
            i = int(x["item_id"])
            if i not in set(batch):
                continue
            vals = valid(axis, x, lists)
            if vals is not None:
                out[i] = {"item_id": i, **vals}
        return out

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_batch, b) for b in batches]
        for f in as_completed(futs):
            done.update(f.result())
    # retry any missing individually
    missing = [i for i in ids if i not in done]
    if missing:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(run_batch, [i]) for i in missing]
            for f in as_completed(futs):
                done.update(f.result())
    with sp.open("w") as f:
        for i in sorted(done):
            f.write(json.dumps(done[i], ensure_ascii=False) + "\n")
    print(f"    {axis}: {len(done)}/{len(texts)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openrouter/openai/gpt-4o-mini")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    convs = [json.loads(l) for l in DATA.open()]
    c = make_client()
    for conv in convs:
        cid = conv["conversation_id"]
        texts = conv_texts(conv)
        ddir = CACHE / cid
        ddir.mkdir(parents=True, exist_ok=True)
        lp = ddir / "lists.json"
        if lp.exists():
            lists = json.loads(lp.read_text())
        else:
            lists, _ = stage1_big(c, args.model, "\n".join(texts))
            lp.write_text(json.dumps(lists, ensure_ascii=False, indent=2))
        # skip conv if all 5 axes already complete
        complete = all(
            (ddir / f"{ax}.jsonl").exists()
            and sum(1 for _ in (ddir / f"{ax}.jsonl").open()) >= len(texts)
            for ax in AXES
        )
        status = "cached" if complete else "extracting"
        print(f"[{cid}] {len(texts)} turns, cast={len(lists['cast'])} "
              f"loc={len(lists['locations'])} goals={len(lists['goals'])} [{status}]", flush=True)
        if complete:
            continue
        for axis in AXES:
            extract_axis(c, args.model, lists, texts, axis, ddir / f"{axis}.jsonl", args.workers)
    print("done — all conversations extracted", flush=True)


if __name__ == "__main__":
    main()

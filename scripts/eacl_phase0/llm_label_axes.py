#!/usr/bin/env python3
"""Generate boundary-blind, axis-independent LLM silver labels for Phase 0."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

SLOTS = ("protagonist", "place", "time", "causality", "goal")
DEFS = {
    "protagonist": "Did the active narrative protagonist/cast identity reset? Resolve who 'I', 'he', or 'she' is from context. Repeated pronouns or names do not imply continuity if this is a different narrative situation. A mere new mention is not a reset unless the central acting cast changes.",
    "place": "Did the active narrative stage/location reset discontinuously? A mentioned place is not enough; judge where the narrated action is currently situated.",
    "time": "Did narrative time jump discontinuously to a new temporal situation? Normal progression inside one event is continuity.",
    "causality": "Did the active causal chain reset, so the current event is not a consequence/continuation of the preceding event chain? A surprising consequence inside one chain is continuity.",
    "goal": "Did the active protagonist's operative goal/intention reset? New means a different active objective, not merely another action serving the same objective.",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def sequence(rows: list[dict]) -> list[str]:
    rows = sorted(rows, key=lambda r: int(r["transition"]))
    return [rows[0]["previous"], *[r["current"] for r in rows]]


def schema(n: int) -> dict:
    item = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "transition": {"type": "integer"},
            "value": {"type": "integer", "enum": [0, 1]},
            "explanation": {"type": "string"},
            "evidence": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["transition", "value", "explanation", "evidence", "confidence"],
    }
    return {
        "type": "object", "additionalProperties": False,
        # Anthropic structured outputs only supports minItems 0/1. Exact length and
        # transition identity are enforced after parsing instead.
        "properties": {"labels": {"type": "array", "items": item}},
        "required": ["labels"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("master", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--model", default="openrouter/anthropic/claude-fable-5")
    ap.add_argument("--chunk-size", type=int, default=40)
    ap.add_argument("--context-before", type=int, default=5)
    ap.add_argument("--context-after", type=int, default=3)
    ap.add_argument("--axis", choices=SLOTS, action="append")
    args = ap.parse_args()
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"],
                    default_headers={"CF-Access-Client-Id": os.environ["CF_ACCESS_CLIENT_ID"],
                                     "CF-Access-Client-Secret": os.environ["CF_ACCESS_CLIENT_SECRET"]})
    rows = read_jsonl(args.master)
    texts = sequence(rows)
    axes = args.axis or list(SLOTS)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for axis in axes:
        out_path = args.out_dir / f"{axis}.jsonl"
        done = {int(r["transition"]): r for r in read_jsonl(out_path)} if out_path.exists() else {}
        for lo in range(1, len(texts), args.chunk_size):
            targets = list(range(lo, min(len(texts), lo + args.chunk_size)))
            pending = [i for i in targets if i not in done]
            if not pending:
                continue
            c0, c1 = max(0, lo - args.context_before), min(len(texts), targets[-1] + args.context_after + 1)
            numbered = "\n".join(f"[{i}] {texts[i]}" for i in range(c0, c1))
            prompt = f"""AXIS: {axis}
DEFINITION: {DEFS[axis]}

Judge ONLY this one axis for each requested adjacent transition i-1 -> i.
Use the surrounding sentences to resolve narrative state and pronoun identity.
Do not infer or discuss story/document boundaries. You are not given boundary labels.
Do not judge any other situation-model axis. Treat every transition independently under the definition.
value=1 only for a genuine reset; when ambiguous use 0.
For value=1, explanation states concretely what changed. For value=0, explanation states why the state continues.
Evidence must quote or tightly point to words in the supplied English text.

REQUESTED TRANSITIONS: {pending}
CONTEXT:
{numbered}"""
            expected = len(pending)
            for attempt in range(3):
                try:
                    response = client.chat.completions.create(
                        model=args.model, temperature=0, max_tokens=12000,
                        response_format={"type": "json_schema", "json_schema": {
                            "name": f"{axis}_reset_labels", "strict": True,
                            "schema": schema(expected)}},
                        messages=[{"role": "system", "content":
                                   "You are a meticulous narrative situation-model annotator."},
                                  {"role": "user", "content": prompt}],
                    )
                    labels = json.loads(response.choices[0].message.content)["labels"]
                    got = {int(x["transition"]): x for x in labels}
                    if set(got) != set(pending):
                        raise ValueError(f"expected {pending}, got {sorted(got)}")
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    print(f"[{axis} {lo}] retry {attempt+1}: {e}", flush=True)
                    time.sleep(2 * (attempt + 1))
            for i in pending:
                x = got[i]
                done[i] = {"doc_id": rows[i-1]["doc_id"], "transition": i,
                           "axis": axis, "suggested_value": int(x["value"]),
                           "explanation": x["explanation"], "evidence": x["evidence"],
                           "confidence": max(0.0, min(1.0, float(x["confidence"]))),
                           "model": args.model}
            out_path.write_text("".join(json.dumps(done[i], ensure_ascii=False) + "\n"
                                         for i in sorted(done)), encoding="utf-8")
            print(f"{axis}: {len(done)}/{len(texts)-1}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

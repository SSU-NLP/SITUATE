#!/usr/bin/env python3
"""Add sentence-aligned Korean reference translations; English remains canonical."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from build_narrative_pack import split_sentences


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stories", type=Path)
    ap.add_argument("--model", default="openrouter/openai/gpt-4o-mini")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    load_dotenv(repo / ".env")
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"],
        default_headers={
            "CF-Access-Client-Id": os.environ["CF_ACCESS_CLIENT_ID"],
            "CF-Access-Client-Secret": os.environ["CF_ACCESS_CLIENT_SECRET"],
        })
    rows = [json.loads(x) for x in args.stories.read_text(encoding="utf-8").splitlines() if x.strip()]
    for n, row in enumerate(rows, 1):
        sentences = split_sentences(row["text"])
        if row.get("translations") and len(row["translations"]) == len(sentences):
            continue
        payload = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
        translations = []
        for attempt in range(3):
            response = client.chat.completions.create(
                model=args.model, temperature=0, max_tokens=4000,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": (
                        "Translate each numbered English narrative sentence into natural Korean. "
                        "Never merge or split sentences. Return only a JSON object named translations "
                        "whose keys are the exact input indices and whose values are translations, e.g. "
                        "{\"translations\":{\"0\":\"...\",\"1\":\"...\"}}. "
                        f"The required keys are exactly 0 through {len(sentences)-1}.")},
                    {"role": "user", "content": payload},
                ])
            parsed = json.loads(response.choices[0].message.content)
            obj = parsed.get("translations", {})
            if isinstance(obj, dict) and set(obj) == {str(i) for i in range(len(sentences))}:
                translations = [obj[str(i)] for i in range(len(sentences))]
                break
            print(f"  alignment retry {attempt + 1}/3 for {row['story_id']}", flush=True)
        if len(translations) != len(sentences):
            raise RuntimeError(
                f"story {row['story_id']}: got {len(translations)} translations for {len(sentences)} sentences")
        row["translations"] = [str(x).strip() for x in translations]
        row["translation_model"] = args.model
        # Incremental persistence makes reruns resume after the last completed story.
        args.stories.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"translated {n}/{len(rows)}: {row['story_id']} ({len(sentences)} sentences)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

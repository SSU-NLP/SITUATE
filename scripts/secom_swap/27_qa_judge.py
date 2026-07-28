"""QA answer generation + GPT4Score judge, per LoCoMo category (test split).

Consumes a recall/<name>.qactx.json (retrieved top-k segments per question).
For each TEST-split question:
  - generate an answer with `gen_model`, conditioned on the top-`qa_topk` retrieved
    segments (density-matched -> same #segments/token budget across methods),
  - judge the answer vs gold with `judge_model` (1-10), same rubric as 08_eval.
Reports mean GPT4Score (x10 -> /100) per category + overall, plus context tokens.

Output: qa/<name>.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "outputs/experiments/2026-07-15_locomo_segmenter_swap"
OUTDIR = BASE / "qa"
TEST = ["conv-44", "conv-47", "conv-48", "conv-49", "conv-50"]
CATNAME = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}

GEN_PROMPT = """You are answering a question about a long-term conversation between two people. Use ONLY the retrieved memory snippets below. If the answer is not in them, say "Not mentioned in the conversation." Answer concisely (a few words or one sentence).

[Retrieved memory]
{context}

[Question]
{question}

[Answer]"""

JUDGE_PROMPT = """You are an evaluator scoring a model's answer to a question about a long conversation. Compare the model's prediction to the ground-truth answer.

Scoring rubric (return a single integer 1-10):
- 10: Prediction conveys all key facts of the gold answer, no contradictions.
- 7-9: Mostly correct, minor missing detail or phrasing difference.
- 4-6: Partially correct; some key info missing or hedged.
- 1-3: Largely wrong, misses or contradicts the gold answer.

Reply with ONLY a single integer on the first line, no explanation.

[Question]
{question}

[Gold answer]
{gold}

[Model prediction]
{pred}

[Score (1-10)]"""


def make_client():
    load_dotenv(REPO_ROOT / ".env")
    import openai

    return openai.OpenAI(
        api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"],
        default_headers={"CF-Access-Client-Id": os.environ["CF_ACCESS_CLIENT_ID"],
                         "CF-Access-Client-Secret": os.environ["CF_ACCESS_CLIENT_SECRET"]},
    )


def call(client, model, prompt, max_tokens, retries=4):
    for a in range(retries):
        try:
            r = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens, temperature=0.0)
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            if a == retries - 1:
                return ""
            time.sleep(2 * (a + 1))
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qactx", required=True)
    ap.add_argument("--qa_topk", type=int, default=5)
    ap.add_argument("--gen_model", default="openrouter/openai/gpt-4o-mini")
    ap.add_argument("--judge_model", default="openrouter/openai/gpt-4o")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = json.loads(Path(args.qactx).read_text())
    rows = [r for r in rows if r["conversation_id"] in TEST]
    client = make_client()
    name = Path(args.qactx).name.replace(".qactx.json", "")

    # 1) generate answers (parallel)
    ctx_tokens = []

    def gen(i):
        r = rows[i]
        ctx = "\n\n".join(r["contexts"][: args.qa_topk])
        ctx_tokens.append(len(ctx.split()))
        prompt = GEN_PROMPT.format(context=ctx, question=r["question"])
        return i, call(client, args.gen_model, prompt, 120)

    preds = [""] * len(rows)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(gen, i) for i in range(len(rows))]
        for f in tqdm(as_completed(futs), total=len(futs), desc=f"gen {name}"):
            i, p = f.result(); preds[i] = p

    # 2) judge (parallel)
    def judge(i):
        r = rows[i]
        prompt = JUDGE_PROMPT.format(question=r["question"], gold=r["answer"], pred=preds[i])
        txt = call(client, args.judge_model, prompt, 8)
        m = re.search(r"\b([1-9]|10)\b", txt)
        return i, (int(m.group(1)) if m else 0)

    scores = [0] * len(rows)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(judge, i) for i in range(len(rows))]
        for f in tqdm(as_completed(futs), total=len(futs), desc=f"judge {name}"):
            i, s = f.result(); scores[i] = s

    # 3) aggregate by category
    by_cat = defaultdict(list)
    for r, s in zip(rows, scores):
        if s > 0:
            by_cat[r["category"]].append(s)
    out = {"qactx": Path(args.qactx).name, "qa_topk": args.qa_topk,
           "gen_model": args.gen_model, "judge_model": args.judge_model,
           "n_qa": len(rows), "avg_ctx_tokens": sum(ctx_tokens) / max(1, len(ctx_tokens)),
           "by_category": {}, "overall_gpt4score_x10": None}
    all_s = []
    for cat in sorted(CATNAME):
        v = by_cat[cat]
        out["by_category"][str(cat)] = {"name": CATNAME[cat], "n": len(v),
                                        "gpt4score_x10": (sum(v) / len(v) * 10) if v else None}
        all_s += v
    out["overall_gpt4score_x10"] = sum(all_s) / len(all_s) * 10 if all_s else None
    # save predictions for inspection
    out["_preds"] = [{"cid": r["conversation_id"], "cat": r["category"], "q": r["question"],
                      "gold": r["answer"], "pred": p, "score": s}
                     for r, p, s in zip(rows, preds, scores)]
    (OUTDIR / f"{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n== {name} (test) ==  avg_ctx_tokens={out['avg_ctx_tokens']:.0f}")
    for cat in sorted(CATNAME):
        row = out["by_category"][str(cat)]
        v = f"{row['gpt4score_x10']:.1f}" if row["gpt4score_x10"] is not None else "  -  "
        print(f"  {CATNAME[cat]:13} n={row['n']:<4} GPT4Score={v}")
    print(f"  OVERALL GPT4Score(x10) = {out['overall_gpt4score_x10']:.2f}")
    print(f"saved -> qa/{name}.json")


if __name__ == "__main__":
    main()

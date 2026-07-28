#!/usr/bin/env python3
"""오라클 전이유형 라벨링 (gonogo Tier0 §3.1). 에피소드=세션.

Phase A: 각 에피소드 1줄 화제 요약 (LLM).
Phase B: 각 에피소드 i(≥1)에 대해 과거 에피소드 요약들을 맥락으로 주고
         전이유형 ∈ {continue, elaborate, shift, return, introduce} 을 제로샷 분류.
         return 이면 되돌아가는 과거 에피소드 id(return_target) 도 출력.
  · ep_id=0 은 자동으로 introduce (과거 없음).

출력: outputs/exp_typed/episodes_typed.jsonl  (+summary, ttype, return_target)
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (LABEL_MODEL, REPO, chat_many, parse_json_lenient,
                    read_jsonl, write_jsonl)

IN = REPO / "outputs/exp_typed/episodes.jsonl"
OUT = REPO / "outputs/exp_typed/episodes_typed.jsonl"

TYPES = ["continue", "elaborate", "shift", "return", "introduce"]

SUMM_SYS = ("You summarize one session of a long-running conversation between two people "
            "into a single short topic phrase (max 12 words). Reply with ONLY the phrase.")

LABEL_SYS = f"""You label the topical transition type of the CURRENT conversation session relative to all PRIOR sessions between the same two people. Sessions are in chronological order.

Choose exactly one type:
- introduce: the current session's main topic is brand-new (not discussed in any prior session).
- continue: directly continues the topic of the IMMEDIATELY preceding session.
- elaborate: deepens / adds detail to the topic of the IMMEDIATELY preceding session.
- shift: moves to a different topic than the immediately preceding session, and it is NOT resuming any older topic.
- return: resumes a topic from an EARLIER session that is NOT the immediately preceding one (a come-back). In this case give the prior session id it returns to.

Reply ONLY as JSON: {{"type": one of {TYPES}, "return_target": <prior session id int, or null if not 'return'>}}"""


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=LABEL_MODEL, help="오라클 라벨링 LLM")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    model = args.model
    print(f"[오라클 모델] {model}", flush=True)

    eps = read_jsonl(IN)
    # Phase A: summaries
    print(f"[Phase A] summarizing {len(eps)} episodes ...", flush=True)
    summaries = chat_many(
        model, eps,
        lambda e: [{"role": "system", "content": SUMM_SYS},
                   {"role": "user", "content": e["text"][:2000]}],
        max_tokens=32, temperature=0.0, workers=8)
    for e, s in zip(eps, summaries):
        e["summary"] = (s or "").strip().strip('"')[:120]

    by_conv = defaultdict(list)
    for e in eps:
        by_conv[e["conv_id"]].append(e)
    for v in by_conv.values():
        v.sort(key=lambda e: e["ep_id"])

    # Phase B: labeling jobs (ep_id>=1)
    jobs = []
    for conv_id, group in by_conv.items():
        for e in group:
            if e["ep_id"] == 0:
                e["ttype"], e["return_target"] = "introduce", None
                continue
            prior = "\n".join(f"[session {p['ep_id']} | {p['date']}] {p['summary']}"
                              for p in group if p["ep_id"] < e["ep_id"])
            cur = f"[CURRENT session {e['ep_id']} | {e['date']}] {e['summary']}\n\nTurns:\n{e['text'][:1200]}"
            jobs.append((e, prior, cur))

    print(f"[Phase B] labeling {len(jobs)} episodes ...", flush=True)
    outs = chat_many(
        model, jobs,
        lambda j: [{"role": "system", "content": LABEL_SYS},
                   {"role": "user", "content": f"PRIOR sessions:\n{j[1]}\n\n{j[2]}"}],
        max_tokens=40, temperature=0.0, workers=8)

    for (e, _, _), raw in zip(jobs, outs):
        d = parse_json_lenient(raw) or {}
        tt = d.get("type")
        e["ttype"] = tt if tt in TYPES else "shift"   # 파싱 실패시 보수적 shift
        rt = d.get("return_target")
        try:
            rt = int(rt)
        except (TypeError, ValueError):
            rt = None
        # return_target 유효성: 자기 이전의 non-immediate 과거만 허용
        if e["ttype"] == "return" and (rt is None or rt >= e["ep_id"] or rt < 0):
            rt = None
        e["return_target"] = rt if e["ttype"] == "return" else None

    write_jsonl(args.out, eps)
    dist = Counter(e["ttype"] for e in eps)
    n_ret_valid = sum(1 for e in eps if e["ttype"] == "return" and e["return_target"] is not None)
    print(f"\nwrote {len(eps)} -> {args.out}")
    print("type dist:", dict(dist))
    print(f"return with valid target: {n_ret_valid}")


if __name__ == "__main__":
    main()

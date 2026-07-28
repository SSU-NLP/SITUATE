#!/usr/bin/env python3
"""B2/B3 (GPT-4 direct segmentation) TOPLINE on dev human gold — non-circular, unlike Tier2.

On dev (Pieman/Monkey/Tunnel, michelmann author-filtered HUMAN gold) B2/B3 are NOT gold generators,
so they can be compared fairly against our LLM-free methods. This measures the ORACLE/topline
("how well does direct LLM prompting segment?") that M3 approaches WITHOUT an LLM at inference (RQ2).

B2 prompt + B3 procedure (GPT-4.1, temp 0, R reps, top-n consensus) — same as Tier2 gold gen,
here applied to the dev stories and scored against their HUMAN boundaries. Reports B2/B3 (topline)
vs M3-S5 vs M1-CosDrift vs B1 (all vs the same human gold, deployable/official DTS Score).

Needs API budget (GPT-4.1 direct calls). ~3 docs x R reps.
Usage: uv run python scripts/eacl_phase0/dev_b2b3_topline.py --reps 20
"""
from __future__ import annotations
import argparse, hashlib, json, math, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from build_tier2_gold import (PROMPT_HEAD, PROMPT_TAIL, GOLD_MODEL, WORD,  # noqa: E402
                              client, run_to_transitions)
from build_pieman_pack import split_sentences  # noqa: E402

BASE = REPO / "outputs/eacl_phase0"
AX = ["protagonist_perspective", "space", "time", "causality", "intentionality"]
DEV = {  # name -> (michelmann master, s5 reset_matrix, b1 words)
    "pieman": ("pieman_michelmann_master.jsonl", "pieman_michelmann_s5/reset_matrix.jsonl", "b1_bayes/pieman_words.jsonl"),
    "tunnel": ("tunnel_michelmann_master.jsonl", "tunnel_michelmann_s5/reset_matrix.jsonl", "b1_bayes/tunnel_words.jsonl"),
    "monkey": ("monkey_michelmann_master.jsonl", "monkey_michelmann_s5/reset_matrix.jsonl", "b1_bayes/monkey_words.jsonl"),
}
# frozen-on-dev knobs (same as elsewhere)
C_M1, T_M3, C_B1 = 1.0, 3, 0.75


def read(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def endturn(n, idx):
    y = [0] * n
    for i in idx: y[i - 1] = 1
    return y
def adaptive(sig, c):
    s = np.asarray(sig, float); thr = s.mean() + c * s.std()
    return [t + 1 for t, v in enumerate(sig) if v >= thr]
def count_pred(sig, t): return [i + 1 for i, v in enumerate(sig) if v >= t]


def b2b3_boundaries(c, story, sents, reps, temp, workers):
    """GPT-4.1 direct segmentation, R reps, B3 top-n consensus -> boundary transitions."""
    orig = len(WORD.findall(story))
    def one():
        for _ in range(6):
            try:
                r = c.chat.completions.create(model=GOLD_MODEL, temperature=temp, max_tokens=32000,
                    timeout=300, messages=[{"role": "user", "content": PROMPT_HEAD + story + PROMPT_TAIL}])
            except Exception:
                continue
            out = r.choices[0].message.content or ""
            if r.choices[0].finish_reason != "length" and len(WORD.findall(out)) >= 0.8 * orig:
                return run_to_transitions(out, sents)
        raise RuntimeError("fail")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        runs = [f.result() for f in as_completed([ex.submit(one) for _ in range(reps)])]
    tally = {}
    for tr in runs:
        for t in tr: tally[t] = tally.get(t, 0) + 1
    n = int(math.floor(sum(len(tr) for tr in runs) / len(runs) + 0.5))
    ranked = sorted(tally.items(), key=lambda kv: (-kv[1], hashlib.sha256(str(kv[0]).encode()).hexdigest()))
    return sorted(t for t, _ in ranked[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    c = client()
    print(f"{'doc':8s} | {'B2/B3(topline)':>14s} {'M3-S5':>7s} {'M1-Cos':>7s} {'B1':>7s}  (vs HUMAN gold)")
    agg = {k: [] for k in ["B2B3", "M3", "M1", "B1"]}
    for name, (mp, rmp, b1p) in DEV.items():
        rows = sorted(read(BASE / mp), key=lambda r: int(r["transition"]))
        n = len(rows) + 1
        gold = endturn(n, [int(r["transition"]) for r in rows if int(r["boundary"])])
        sents = [rows[0]["previous"], *[r["current"] for r in rows]]
        story = re.sub(r"\s+", " ", " ".join(sents)).strip()
        # B2/B3 topline
        b2b3 = b2b3_boundaries(c, story, sents, args.reps, args.temp, args.workers)
        # our methods
        rm = {int(r["transition"]): r for r in read(BASE / rmp)}
        cnt = [sum(int(rm[int(r["transition"])][a]) for a in AX) for r in rows]
        idm = [float(r["idm_score"]) for r in rows]
        Wd = read(BASE / b1p); g0 = {}
        for gi, w in enumerate(Wd): g0.setdefault(w["sent"], gi)
        V = [w["transient"] for w in Wd]
        b1 = [float(max(V[g0.get(t, 0):g0.get(t + 1, len(V))]) if g0.get(t) is not None else 0) for t in range(1, n)]
        sc = {
            "B2B3": score_dialogues([gold], [endturn(n, b2b3)])["score"],
            "M3": score_dialogues([gold], [endturn(n, count_pred(cnt, T_M3))])["score"],
            "M1": score_dialogues([gold], [endturn(n, adaptive(idm, C_M1))])["score"],
            "B1": score_dialogues([gold], [endturn(n, adaptive(b1, C_B1))])["score"],
        }
        for k in agg: agg[k].append(sc[k])
        print(f"{name:8s} | {sc['B2B3']:14.3f} {sc['M3']:7.3f} {sc['M1']:7.3f} {sc['B1']:7.3f}")
    print(f"{'MEAN':8s} | " + " ".join(f"{np.mean(agg[k]):.3f}" for k in ["B2B3", "M3", "M1", "B1"]).replace(" ", "          ", 0))
    print("\nNote: B2/B3 = GPT-4.1 direct (offline, full-text) = TOPLINE; M1/M3/B1 = deployable LLM-free.")


if __name__ == "__main__":
    main()

"""M3 (5-axis LLM) granularity probe on LoCoMo — is it *coarser* (more turns/event)?

Adapts the frozen M3 spec (`scripts/eacl_phase0/llm_s5_states.py`) to LoCoMo
dialogue turns, reusing the exact Stage-1/Stage-2 prompts, closed enums, and
code-only reset rules. Target = each "Speaker: text" turn.

Per conversation:
  Stage 1 (whole conversation): cast / locations / goals closed lists.
  Stage 2 (per turn, per axis, shuffled, batched): pick from closed lists.
  Reset judgment (code): coordinated count_t = sum of 5 axis resets (0..5).
  Boundary <=> count_t >= t* (t*=3, frozen).  Segmentation per-session.

Reports the reset-count histogram + natural #boundaries + turns/segment, so we
can see (a) whether M3 under-segments casual dialogue and (b) whether its integer
signal has enough resolution to place boundaries at a matched density.

Usage: uv run python scripts/secom_swap/22_m3_granularity_probe.py --n_conv 2
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/eacl_phase0"))

from llm_s5_states import (  # noqa: E402
    AXES,
    carry_forward,
    reset_causality,
    reset_goal,
    reset_perspective,
    reset_place,
    reset_protagonist_charset,
    reset_time,
    stage1,
    stage2_call,
    valid,
)

DATA = REPO_ROOT / "outputs/experiments/2026-07-15_locomo_segmenter_swap/data/locomo_turns.jsonl"
OUT = REPO_ROOT / "outputs/experiments/2026-07-15_locomo_segmenter_swap/m3_granularity_probe.json"
CACHE = REPO_ROOT / "outputs/experiments/2026-07-15_locomo_segmenter_swap/m3_states"

TSTAR = 3
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


def extract_states(c, model, cid, texts):
    """Run Stage1 + Stage2 (all axes) over the full turn list; cache to disk."""
    ddir = CACHE / cid
    ddir.mkdir(parents=True, exist_ok=True)
    lp = ddir / "lists.json"
    if lp.exists():
        lists = json.loads(lp.read_text())
    else:
        lists, _ = stage1(c, model, "\n".join(texts))
        lp.write_text(json.dumps(lists, ensure_ascii=False, indent=2))
    print(f"[{cid}] stage1 cast={len(lists['cast'])} loc={len(lists['locations'])} "
          f"goals={len(lists['goals'])}", flush=True)

    states: dict[str, dict[int, dict]] = {}
    for axis in AXES:
        sp = ddir / f"{axis}.jsonl"
        done: dict[int, dict] = {}
        if sp.exists():
            for line in sp.open():
                x = json.loads(line)
                done[int(x["item_id"])] = x
        ids = list(range(len(texts)))
        random.Random(SEED + AXES.index(axis)).shuffle(ids)
        while len(done) < len(texts):
            todo = [i for i in ids if i not in done][:20]
            got, _ = stage2_call(c, model, axis, lists, [(i, texts[i]) for i in todo])
            acc = 0
            for x in got:
                i = int(x["item_id"])
                if i not in set(todo):
                    continue
                vals = valid(axis, x, lists)
                if vals is None:
                    continue
                done[i] = {"item_id": i, **vals}
                acc += 1
            if not acc:
                raise RuntimeError(f"[{cid}] {axis}: no valid outputs for {todo}")
            with sp.open("w") as f:
                for i in sorted(done):
                    f.write(json.dumps(done[i], ensure_ascii=False) + "\n")
            print(f"[{cid}] {axis}: {len(done)}/{len(texts)}", flush=True)
        states[axis] = done
    states["place"] = carry_forward(states["place"], "location", "unspecified")
    states["goal"] = carry_forward(states["goal"], "active_goal", "none")
    return states


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_conv", type=int, default=2)
    ap.add_argument("--model", default="openrouter/openai/gpt-4o-mini")
    args = ap.parse_args()

    convs = [json.loads(l) for l in DATA.open()][: args.n_conv]
    c = make_client()

    per_conv = []
    count_hist_all: Counter = Counter()
    for conv in convs:
        cid = conv["conversation_id"]
        sessions = conv["sessions"]
        # flat turn list with (session_index, within_session_index)
        flat = []
        for si, sess in enumerate(sessions):
            for t in sess:
                flat.append((si, f"{t['speaker']}: {t['text']}"))
        texts = [x[1] for x in flat]
        sess_of = [x[0] for x in flat]
        states = extract_states(c, args.model, cid, texts)

        # reset count for every WITHIN-session transition
        n_bnd = 0
        counts = []
        axis_fire = Counter()
        for i in range(1, len(texts)):
            if sess_of[i] != sess_of[i - 1]:
                continue  # session break = automatic boundary, not an M3 decision
            pc = reset_protagonist_charset(states["protagonist"][i - 1], states["protagonist"][i])
            pp = reset_perspective(states["protagonist"][i - 1], states["protagonist"][i])
            r = {
                "protagonist": int(pp),  # charset excluded per D-B7
                "place": reset_place(states["place"][i - 1], states["place"][i]),
                "time": reset_time(states["time"][i - 1], states["time"][i]),
                "causality": reset_causality(states["causality"][i - 1], states["causality"][i]),
                "goal": reset_goal(states["goal"][i - 1], states["goal"][i]),
            }
            cnt = sum(r.values())
            counts.append(cnt)
            count_hist_all[cnt] += 1
            for k, v in r.items():
                axis_fire[k] += v
            if cnt >= TSTAR:
                n_bnd += 1
        n_turns = len(texts)
        n_sess = len(sessions)
        n_segs = n_bnd + n_sess
        n_intra_trans = len(counts)
        pc_row = {
            "conversation_id": cid,
            "n_turns": n_turns,
            "n_sessions": n_sess,
            "n_intra_transitions": n_intra_trans,
            "m3_n_bnd_internal": n_bnd,
            "m3_n_segments": n_segs,
            "m3_turns_per_seg": n_turns / max(1, n_segs),
            "count_hist": dict(Counter(counts)),
            "axis_fire_rate": {k: axis_fire[k] / max(1, n_intra_trans) for k in
                               ["protagonist", "place", "time", "causality", "goal"]},
        }
        per_conv.append(pc_row)
        print(f"[{cid}] M3: {n_bnd} internal bnd / {n_intra_trans} intra-trans "
              f"-> {n_segs} segs, {pc_row['m3_turns_per_seg']:.1f} turns/seg", flush=True)
        print(f"       count_hist={pc_row['count_hist']} axis_fire={ {k: round(v,3) for k,v in pc_row['axis_fire_rate'].items()} }", flush=True)

    out = {
        "config": {"t_star": TSTAR, "model": args.model, "n_conv": len(per_conv)},
        "per_conv": per_conv,
        "count_hist_all": dict(sorted(count_hist_all.items())),
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()

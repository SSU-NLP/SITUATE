"""Session-prefix ONLINE M3 — removes the cross-session future-peek.

The only future leak in M3's reset signal is the Stage1 entity vocabulary that
`place` and `goal` are matched against (protagonist=perspective, time, causality
are FIXED enums -> already causal). This builds that vocabulary *causally*:

  when processing session s, the allowed locations/goals = union of entities seen
  in sessions 1..s only (no future sessions). place/goal Stage2 is re-run per
  session with that prefix vocabulary; protagonist/time/causality reuse the
  full-run cache (prefix-independent).

(Within a session the LLM still reads the whole session, matching SeCom's
session-batch construction; the removed leak is *future sessions*, which is the
dominant multi-session confound in LoCoMo.)

Outputs scores/<cid>.m3prefix.json : per-turn coordinated reset count (or null at
session breaks / first turns), so 25_segment can consume it as method `m3prefix`.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/eacl_phase0"))
from llm_s5_states import (  # noqa: E402
    carry_forward, reset_causality, reset_goal, reset_perspective,
    reset_place, reset_time, stage2_call, valid,
)
sys.path.insert(0, str(REPO_ROOT / "scripts/secom_swap"))
from importlib import import_module  # noqa: E402
stage1_big = import_module("23_m3_extract_all").stage1_big

BASE = REPO_ROOT / "outputs/experiments/2026-07-15_locomo_segmenter_swap"
DATA = BASE / "data/locomo_turns.jsonl"
SCORES = BASE / "scores"
FULL = BASE / "m3_states"
PREFIX = BASE / "m3_prefix_states"
MODEL = "openrouter/openai/gpt-4o-mini"
WORKERS = 16


def make_client():
    load_dotenv(REPO_ROOT / ".env")
    from openai import OpenAI

    return OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"],
                 default_headers={"CF-Access-Client-Id": os.environ["CF_ACCESS_CLIENT_ID"],
                                  "CF-Access-Client-Secret": os.environ["CF_ACCESS_CLIENT_SECRET"]})


def stage2_prefix(c, axis, lists, items):
    """Run Stage2 for one axis over `items` with a restricted `lists` vocab."""
    if not items:
        return {}
    batches = [items[k:k + 20] for k in range(0, len(items), 20)]
    out = {}

    def run(batch):
        got, _ = stage2_call(c, MODEL, axis, lists, batch)
        r = {}
        for x in got:
            i = int(x["item_id"])
            if i in {b[0] for b in batch}:
                v = valid(axis, x, lists)
                if v is not None:
                    r[i] = {"item_id": i, **v}
        return r

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(run, b) for b in batches]
        for f in as_completed(futs):
            out.update(f.result())
    return out


def load_full(cid, axis, n):
    d = {int(json.loads(l)["item_id"]): json.loads(l) for l in (FULL / cid / f"{axis}.jsonl").open()}
    return d


def main() -> None:
    PREFIX.mkdir(parents=True, exist_ok=True)
    c = make_client()
    convs = [json.loads(l) for l in DATA.open()]
    for conv in convs:
        cid = conv["conversation_id"]
        sessions = conv["sessions"]
        # global index bookkeeping
        flat = []
        sess_of = []
        for si, sess in enumerate(sessions):
            for t in sess:
                flat.append(f"{t['speaker']}: {t['text']}")
                sess_of.append(si)
        n = len(flat)
        pdir = PREFIX / cid
        pdir.mkdir(parents=True, exist_ok=True)

        # 1) per-session Stage1 (cache) -> cumulative prefix vocab
        sl_path = pdir / "session_lists.json"
        if sl_path.exists():
            sess_lists = json.loads(sl_path.read_text())
        else:
            sess_lists = []
            for si, sess in enumerate(sessions):
                text = "\n".join(f"{t['speaker']}: {t['text']}" for t in sess)
                lists, _ = stage1_big(c, MODEL, text, max_tokens=4000)
                sess_lists.append(lists)
            sl_path.write_text(json.dumps(sess_lists, ensure_ascii=False))
        # cumulative prefix locations/goals per session index
        cum_loc, cum_goal = [], []
        L, G = set(), set()
        for si in range(len(sessions)):
            L |= set(sess_lists[si].get("locations", []))
            G |= set(sess_lists[si].get("goals", []))
            cum_loc.append(sorted(L)); cum_goal.append(sorted(G))

        # 2) re-run place & goal Stage2 per session with prefix vocab (cache)
        prefix_states = {"place": {}, "goal": {}}
        for axis, key, cum in (("place", "locations", cum_loc), ("goal", "goals", cum_goal)):
            sp = pdir / f"{axis}.jsonl"
            if sp.exists():
                for line in sp.open():
                    x = json.loads(line); prefix_states[axis][int(x["item_id"])] = x
            if len(prefix_states[axis]) >= n:
                continue
            # process session by session (each session its own prefix vocab)
            for si, sess in enumerate(sessions):
                idxs = [gi for gi in range(n) if sess_of[gi] == si and gi not in prefix_states[axis]]
                if not idxs:
                    continue
                lists = {"cast": [], "locations": cum_loc[si], "goals": cum_goal[si]}
                items = [(gi, flat[gi]) for gi in idxs]
                got = stage2_prefix(c, axis, lists, items)
                prefix_states[axis].update(got)
            with sp.open("w") as f:
                for i in sorted(prefix_states[axis]):
                    f.write(json.dumps(prefix_states[axis][i], ensure_ascii=False) + "\n")
        print(f"[{cid}] prefix place={len(prefix_states['place'])}/{n} "
              f"goal={len(prefix_states['goal'])}/{n} "
              f"(final vocab loc={len(cum_loc[-1])} goal={len(cum_goal[-1])})", flush=True)

        # 3) protagonist/time/causality from full cache (prefix-independent)
        prot = load_full(cid, "protagonist", n)
        tim = load_full(cid, "time", n)
        cau = load_full(cid, "causality", n)
        place = carry_forward(prefix_states["place"], "location", "unspecified")
        goal = carry_forward(prefix_states["goal"], "active_goal", "none")

        # 4) coordinated reset count per within-session transition
        out = [None] * n
        for i in range(1, n):
            if sess_of[i] != sess_of[i - 1]:
                continue
            out[i] = int(
                reset_perspective(prot[i - 1], prot[i])
                + reset_place(place[i - 1], place[i])
                + reset_time(tim[i - 1], tim[i])
                + reset_causality(cau[i - 1], cau[i])
                + reset_goal(goal[i - 1], goal[i])
            )
        (SCORES / f"{cid}.m3prefix.json").write_text(json.dumps(out))
    print("done")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""LoCoMo → 에피소드. **에피소드 = 세션** (벤치마크 고유 단위, 분절기 교란 0).

gonogo Tier0 원칙: 분절은 기여물이 아니므로 최대한 단순·표준으로 재사용.
따라서 화제분절(δ_eff/EM-LLM) 안 씀 — 각 세션(날짜별 대화)을 1 에피소드로.
return 유형 = 후속 세션이 과거 세션의 화제로 복귀.

episode: {conv_id, ep_id(세션순서), session_id, date, dia_ids, n_turns, text}
출력: outputs/exp_typed/episodes.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, write_jsonl

SRC = REPO / "benchmarks/locomo/locomo10.json"
OUT = REPO / "outputs/exp_typed/episodes.jsonl"


def session_keys(conv):
    ks = [k for k in conv if k.startswith("session_") and not k.endswith("date_time")
          and isinstance(conv[k], list)]
    return sorted(ks, key=lambda k: int(k.split("_")[1]))


def turn_str(t):
    txt = (t.get("text") or "").strip()
    if not txt:
        cap = (t.get("blip_caption") or "").strip()
        txt = f"[shares an image] {cap}" if cap else "[non-text turn]"
    return f"{t.get('speaker','?')}: {txt}"


def main():
    data = json.loads(SRC.read_text())
    episodes = []
    for idx, sample in enumerate(data):
        conv = sample["conversation"]
        conv_id = str(sample.get("sample_id", idx))
        for ep_id, sk in enumerate(session_keys(conv)):
            turns = conv[sk]
            date = conv.get(f"{sk}_date_time", "")
            tstrs = [turn_str(t) for t in turns]
            episodes.append({
                "conv_id": conv_id, "ep_id": ep_id, "session_id": sk, "date": date,
                "dia_ids": [t.get("dia_id", "") for t in turns],
                "n_turns": len(turns),
                "text": "\n".join(tstrs),
            })
        print(f"conv {conv_id}: {ep_id + 1} episodes (sessions)", flush=True)
    write_jsonl(OUT, episodes)
    from collections import Counter
    per = Counter(e["conv_id"] for e in episodes)
    print(f"\nwrote {len(episodes)} episodes -> {OUT}")
    print("episodes/conv:", dict(per))
    import numpy as np
    print("turns/episode: mean=%.1f max=%d" % (
        np.mean([e["n_turns"] for e in episodes]), max(e["n_turns"] for e in episodes)))


if __name__ == "__main__":
    main()

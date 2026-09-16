#!/usr/bin/env python3
r"""**증거 턴 재현율** — LLM 없이 분절의 기여를 직접 재는 지표.

LoCoMo 의 각 질문에는 정답 근거가 되는 턴 목록(`evidence`, 예: `["D1:3"]`)이 붙어 있다.
검색이 돌려준 에피소드들이 그 턴들을 **실제로 담고 있었는지**를 세면,
QA 정확도(LLM 심판·생성 노이즈 포함)와 달리 **검색 단계까지의 품질만** 분리해서 볼 수 있다.

분절기를 바꾸면 바뀌는 것이 정확히 이것이다 — "정답 턴이 어느 에피소드에 묶여서,
질문 임베딩에 얼마나 잘 걸리는가". 그래서 이 지표가 우리 기여에 가장 가깝다.

    .venv/bin/python scripts/nemori_locomo/evidence_recall.py --name 2026-08-01_nemori_locomo_m5

산출: `<exp>/evidence_recall.json`
- `recall` = 질문별 (검색된 턴 ∩ evidence) / |evidence| 의 평균
- `full_hit` = evidence 를 **전부** 회수한 질문의 비율
- `turns_retrieved` = 질문당 회수된 원본 턴 수 (컨텍스트 비용 축)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nemori_env import REPO, NEMORI, bootstrap  # noqa: E402

DATA = REPO / "benchmarks/locomo/locomo10.json"


def turn_index(dataset: list[dict]) -> list[dict[str, str]]:
    """대화별 {메시지 content -> dia_id}. add.py 의 content 조립 규칙과 동일해야 한다."""
    out = []
    for item in dataset:
        conv = item.get("conversation", {})
        m: dict[str, str] = {}
        for key, chats in conv.items():
            if key in ("speaker_a", "speaker_b") or key.endswith("_date_time"):
                continue
            for chat in chats or []:
                parts = [chat.get("text", "")]
                if chat.get("blip_caption"):
                    parts.append(f"[Image: {chat['blip_caption']}]")
                if chat.get("query"):
                    parts.append(f"[Search: {chat['query']}]")
                m[" ".join(parts)] = chat.get("dia_id", "")
        out.append(m)
    return out


async def episode_sources(agent_id: str) -> dict[str, list[str]]:
    """episode_id -> 그 에피소드가 담고 있는 메시지 content 목록."""
    import asyncpg

    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        rows = await conn.fetch(
            "SELECT id::text AS id, source_messages FROM episodes WHERE agent_id=$1", agent_id)
    finally:
        await conn.close()
    out = {}
    for r in rows:
        src = r["source_messages"]
        if isinstance(src, str):
            src = json.loads(src)
        contents = []
        for m in src or []:
            c = m.get("content")
            if isinstance(c, list):  # 멀티모달
                c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
            contents.append(str(c or ""))
        out[r["id"]] = contents
    return out


def analyse(arm_dir: Path, dataset: list[dict]) -> dict:
    cfg = json.loads((arm_dir / "config.json").read_text(encoding="utf-8"))
    results = json.loads((arm_dir / "results.json").read_text(encoding="utf-8"))
    ep_src = asyncio.run(episode_sources(cfg["agent_id"]))
    idx = turn_index(dataset)

    per_cat: dict[str, list[float]] = {}
    recalls, full_hits, n_turns, missing_ep = [], [], [], 0
    for conv_key, items in results.items():
        cmap = idx[int(conv_key)]
        for it in items:
            evidence = [e for e in (it.get("evidence") or []) if isinstance(e, str)]
            if not evidence:
                continue
            got: set[str] = set()
            n_t = 0
            for mem in it.get("memories", []):
                if mem.get("type") != "episodic":
                    continue
                eid = mem.get("episode_id")
                contents = ep_src.get(eid)
                if contents is None:
                    missing_ep += 1
                    continue
                n_t += len(contents)
                for c in contents:
                    d = cmap.get(c)
                    if d:
                        got.add(d)
            hit = len(got & set(evidence)) / len(evidence)
            recalls.append(hit)
            full_hits.append(1.0 if got >= set(evidence) else 0.0)
            n_turns.append(n_t)
            per_cat.setdefault(str(it.get("category")), []).append(hit)

    if not recalls:
        return {"n_questions": 0}
    return {
        "n_questions": len(recalls),
        "recall": round(statistics.mean(recalls), 4),
        "full_hit": round(statistics.mean(full_hits), 4),
        "turns_retrieved_per_question": {
            "mean": round(statistics.mean(n_turns), 1),
            "median": statistics.median(n_turns),
            "p90": sorted(n_turns)[int(0.9 * (len(n_turns) - 1))],
        },
        "recall_by_category": {c: round(statistics.mean(v), 4)
                               for c, v in sorted(per_cat.items())},
        "episodes_not_found": missing_ep,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="2026-08-01_nemori_locomo_m5")
    args = ap.parse_args()
    bootstrap()

    dataset = json.loads(DATA.read_text(encoding="utf-8"))
    exp = REPO / "outputs/experiments" / args.name
    out = {}
    for d in sorted(exp.iterdir()):
        if d.is_dir() and (d / "results.json").exists():
            out[d.name] = analyse(d, dataset)
            print(d.name, json.dumps(out[d.name], ensure_ascii=False))
    (exp / "evidence_recall.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {exp / 'evidence_recall.json'}")


if __name__ == "__main__":
    main()

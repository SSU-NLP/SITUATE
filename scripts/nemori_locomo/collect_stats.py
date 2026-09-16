#!/usr/bin/env python3
r"""팔별 산출물에서 **비용·구조 지표**를 뽑는다 (정확도 지표는 metrics.json 이 따로 갖고 있다).

정확도만 보면 안 되는 이유: 분절기를 바꾸면 에피소드 **개수·길이**가 바뀌고, 그게 곧
(a) 하류 LLM 호출 수 (b) 검색 단위 크기 (c) QA 프롬프트에 실려 가는 **컨텍스트 토큰 수** 를 바꾼다.
프로젝트 규약대로 정확도 vs 컨텍스트 크기의 Pareto 로 읽어야 한다.

    .venv/bin/python scripts/nemori_locomo/collect_stats.py --name 2026-08-01_nemori_locomo_m5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nemori_env import REPO, bootstrap  # noqa: E402


def _tokens(text: str) -> int:
    """tiktoken 이 있으면 그걸로, 없으면 공백 토큰의 1.3배 근사."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return int(len(text.split()) * 1.3)


async def pg_stats(agent_id: str) -> dict:
    import asyncpg

    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        eps = await conn.fetch(
            "SELECT id, user_id, title, content, source_messages FROM episodes WHERE agent_id=$1",
            agent_id)
        n_sem = await conn.fetchval(
            "SELECT count(*) FROM semantic_memories WHERE agent_id=$1", agent_id)
    finally:
        await conn.close()

    sizes, ep_tokens, per_user = [], [], {}
    for e in eps:
        src = e["source_messages"]
        if isinstance(src, str):
            src = json.loads(src)
        n = len(src or [])
        sizes.append(n)
        ep_tokens.append(_tokens(f"{e['title']} {e['content']}"))
        per_user[e["user_id"]] = per_user.get(e["user_id"], 0) + 1
    if not sizes:
        return {"n_episodes": 0, "n_semantic": int(n_sem or 0)}
    return {
        "n_episodes": len(eps),
        "n_semantic": int(n_sem or 0),
        "episodes_per_conversation": dict(sorted(per_user.items())),
        "messages_covered": sum(sizes),
        "group_size": {
            "mean": round(statistics.mean(sizes), 2),
            "median": statistics.median(sizes),
            "p90": sorted(sizes)[int(0.9 * (len(sizes) - 1))],
            "max": max(sizes), "min": min(sizes),
        },
        "episode_tokens": {
            "mean": round(statistics.mean(ep_tokens), 1),
            "median": statistics.median(ep_tokens),
            "total": sum(ep_tokens),
        },
    }


def context_stats(results_path: Path) -> dict:
    """QA 프롬프트에 실제로 실린 검색 컨텍스트 크기."""
    if not results_path.exists():
        return {}
    data = json.loads(results_path.read_text(encoding="utf-8"))
    per_q_tokens, n_ep, n_sem = [], [], []
    for items in data.values():
        for it in items:
            mems = it.get("memories", [])
            text = "\n".join(str(m.get("content") or "") for m in mems)
            per_q_tokens.append(_tokens(text))
            n_ep.append(sum(1 for m in mems if m.get("type") == "episodic"))
            n_sem.append(sum(1 for m in mems if m.get("type") == "semantic"))
    if not per_q_tokens:
        return {}
    return {
        "n_questions": len(per_q_tokens),
        "context_tokens_per_question": {
            "mean": round(statistics.mean(per_q_tokens), 1),
            "median": statistics.median(per_q_tokens),
            "p90": sorted(per_q_tokens)[int(0.9 * (len(per_q_tokens) - 1))],
            "total": sum(per_q_tokens),
        },
        "retrieved_units_per_question": {
            "episodic_mean": round(statistics.mean(n_ep), 2),
            "semantic_mean": round(statistics.mean(n_sem), 2),
        },
    }


def score_stats(metrics_path: Path) -> dict:
    """evals.py 산출을 발표판 generate_scores.py 와 **같은 방식**으로 집계한다
    (카테고리별 평균 + 전체 평균). 카테고리 5는 evals.py 단계에서 이미 제외됐다."""
    if not metrics_path.exists():
        return {}
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    items = [x for v in data.values() for x in v]
    if not items:
        return {}
    by_cat: dict[str, list] = {}
    for it in items:
        by_cat.setdefault(str(it["category"]), []).append(it)

    def agg(rows):
        return {
            "bleu": round(statistics.mean(r["bleu_score"] for r in rows), 4),
            "f1": round(statistics.mean(r["f1_score"] for r in rows), 4),
            "llm": round(statistics.mean(r["llm_score"] for r in rows), 4),
            "count": len(rows),
        }

    return {
        "overall": agg(items),
        "by_category": {c: agg(rows) for c, rows in sorted(by_cat.items())},
    }


def llm_usage(run_log: Path) -> dict:
    """add.py 가 stdout 에 찍는 LLM 사용 요약을 run.log 에서 회수한다."""
    if not run_log.exists():
        return {}
    text = run_log.read_text(encoding="utf-8", errors="replace")
    out: dict = {}
    m = re.search(r"Total LLM calls:\s+(\d+)", text)
    if m:
        out["total_calls"] = int(m.group(1))
    m = re.search(r"Total tokens:\s+(\d+)", text)
    if m:
        out["total_tokens"] = int(m.group(1))
    m = re.search(r"Total episodes created:\s+(\d+)", text)
    if m:
        out["episodes_created"] = int(m.group(1))
    phases = re.findall(r"^\s{2}(\w+)\s+calls=\s*(\d+)\s+tokens=([\d,]+)", text, re.M)
    if phases:
        out["by_phase"] = {k: int(c) for k, c, _ in phases}
        out["by_phase_tokens"] = {k: int(t.replace(",", "")) for k, _, t in phases}
    return out


def collect_arm(arm_dir: Path) -> dict:
    cfg = json.loads((arm_dir / "config.json").read_text(encoding="utf-8"))
    seg = {}
    if (arm_dir / "segmenter_info.json").exists():
        seg = json.loads((arm_dir / "segmenter_info.json").read_text(encoding="utf-8"))
    out = {
        "arm": arm_dir.name,
        "agent_id": cfg["agent_id"],
        "segmenter": seg,
        "storage": asyncio.run(pg_stats(cfg["agent_id"])),
        "retrieval": context_stats(arm_dir / "results.json"),
        "scores": score_stats(arm_dir / "metrics.json"),
        "llm_usage_add": llm_usage(arm_dir / "run.log"),
    }
    trace = arm_dir / "m5_segmentation_trace.jsonl"
    if trace.exists():
        recs = [json.loads(l) for l in trace.read_text(encoding="utf-8").splitlines() if l.strip()]
        out["m5_trace"] = {
            "n_batches": len(recs),
            "total_messages": sum(r["n_messages"] for r in recs),
            "total_groups": sum(r["n_groups"] for r in recs),
            "total_seg_latency_ms": round(sum(r["seg_latency_ms"] for r in recs), 1),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="2026-08-01_nemori_locomo_m5")
    args = ap.parse_args()
    bootstrap()

    exp = REPO / "outputs/experiments" / args.name
    arms = sorted(d for d in exp.iterdir() if d.is_dir() and (d / "config.json").exists())
    stats = {d.name: collect_arm(d) for d in arms}
    out = exp / "stats.json"
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

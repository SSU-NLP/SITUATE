#!/usr/bin/env python3
r"""두 팔의 산출을 합쳐 REPORT.md 의 **표 부분**을 생성한다 (해석·판정은 사람이 쓴다).

    .venv/bin/python scripts/nemori_locomo/make_report.py --name 2026-08-01_nemori_locomo_m5

포함:
- 정확도 표 (전체 · 카테고리별) — 주지표 LLM judge
- **문항 짝지음 부트스트랩 95% CI** — run 1회·seed 없음이므로 Δ 의 불확실성을 이걸로 말한다
- 비용 표 (에피소드 수 · 그룹 크기 · 주입 LLM 호출/토큰 · 질문당 컨텍스트 토큰)
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nemori_env import REPO  # noqa: E402

CAT_NAMES = {"1": "Multi-Hop", "2": "Temporal", "3": "Open-Domain", "4": "Single-Hop"}


def load_metrics(arm_dir: Path) -> dict[tuple[str, str], dict]:
    """(대화키, 질문) -> 항목. 두 팔의 같은 질문을 짝지으려고 이렇게 키를 잡는다."""
    path = arm_dir / "metrics.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for conv_key, items in data.items():
        for it in items:
            out[(conv_key, it["question"])] = it
    return out


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 10000,
                     seed: int = 0) -> tuple[float, float, float]:
    """짝지은 차이(b − a)의 평균과 95% CI. 문항 단위 복원추출."""
    d = np.asarray(b, float) - np.asarray(a, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def acc_table(stats: dict) -> str:
    """baseline 을 기준으로 나머지 팔의 Δ 를 함께 낸다 (팔이 3개 이상이어도 동작)."""
    arms = list(stats)
    others = [a for a in arms if a != "baseline"]
    header = ["지표"] + arms + [f"Δ {a}" for a in others]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for key, label in (("llm", "**LLM judge (주지표)**"), ("f1", "token F1"), ("bleu", "BLEU-1")):
        def get(a):
            return stats[a].get("scores", {}).get("overall", {}).get(key)
        cells = [f"{get(a):.4f}" if get(a) is not None else "—" for a in arms]
        base = get("baseline") if "baseline" in stats else None
        deltas = [f"{get(a) - base:+.4f}" if (base is not None and get(a) is not None) else "—"
                  for a in others]
        lines.append(f"| {label} | " + " | ".join(cells + deltas) + " |")
    n = [str(stats[a].get("scores", {}).get("overall", {}).get("count", "—")) for a in arms]
    lines.append("| 문항 수 | " + " | ".join(n + ["" for _ in others]) + " |")
    return "\n".join(lines)


def cat_table(stats: dict) -> str:
    arms = list(stats)
    others = [a for a in arms if a != "baseline"]
    header = ["카테고리", "n"] + arms + [f"Δ {a}" for a in others]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    cats = sorted({c for a in arms for c in stats[a].get("scores", {}).get("by_category", {})})
    for c in cats:
        def get(a):
            return stats[a].get("scores", {}).get("by_category", {}).get(c, {}).get("llm")
        n = stats[arms[0]].get("scores", {}).get("by_category", {}).get(c, {}).get("count", "—")
        cells = [f"{get(a):.4f}" if get(a) is not None else "—" for a in arms]
        base = get("baseline") if "baseline" in stats else None
        deltas = [f"{get(a) - base:+.4f}" if (base is not None and get(a) is not None) else "—"
                  for a in others]
        lines.append(f"| {CAT_NAMES.get(c, c)} ({c}) | {n} | " + " | ".join(cells + deltas) + " |")
    return "\n".join(lines)


def cost_table(stats: dict) -> str:
    arms = list(stats)

    def g(a, *path, default="—"):
        cur = stats[a]
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur

    rows = [
        ("에피소드 수", lambda a: g(a, "storage", "n_episodes")),
        ("시맨틱 메모리 수", lambda a: g(a, "storage", "n_semantic")),
        ("그룹 크기 mean / median / max",
         lambda a: f'{g(a, "storage", "group_size", "mean")} / '
                   f'{g(a, "storage", "group_size", "median")} / '
                   f'{g(a, "storage", "group_size", "max")}'),
        ("주입 LLM 호출 수", lambda a: g(a, "llm_usage_add", "total_calls")),
        ("주입 LLM 토큰", lambda a: f'{g(a, "llm_usage_add", "total_tokens"):,}'
         if isinstance(g(a, "llm_usage_add", "total_tokens"), int) else "—"),
        ("분절 LLM 호출 수", lambda a: g(a, "llm_usage_add", "by_phase", "segmenter", default=0)),
        ("질문당 컨텍스트 토큰 (mean)",
         lambda a: g(a, "retrieval", "context_tokens_per_question", "mean")),
        ("질문당 컨텍스트 토큰 (p90)",
         lambda a: g(a, "retrieval", "context_tokens_per_question", "p90")),
    ]
    lines = ["| 항목 | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for label, fn in rows:
        lines.append(f"| {label} | " + " | ".join(str(fn(a)) for a in arms) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="2026-08-01_nemori_locomo_m5")
    args = ap.parse_args()

    exp = REPO / "outputs/experiments" / args.name
    stats = json.loads((exp / "stats.json").read_text(encoding="utf-8"))
    # baseline 을 먼저 오게 정렬 (Δ 방향 고정)
    # baseline 을 맨 앞에 (Δ 방향 고정)
    stats = ({"baseline": stats["baseline"]} if "baseline" in stats else {}) | \
            {k: v for k, v in sorted(stats.items()) if k != "baseline"}

    parts = ["## 정확도 (LoCoMo, 카테고리 5 제외 1,540문항)", "", acc_table(stats), "",
             "### 카테고리별 (LLM judge)", "", cat_table(stats), "",
             "## 비용·구조", "", cost_table(stats), ""]

    if "baseline" in stats:
        ma = load_metrics(exp / "baseline")
        rows = []
        for arm in [a for a in stats if a != "baseline"]:
            mb = load_metrics(exp / arm)
            common = sorted(set(ma) & set(mb))
            if not common:
                continue
            for key, label in (("llm_score", "LLM judge"), ("f1_score", "token F1"),
                               ("bleu_score", "BLEU-1")):
                d, lo, hi = paired_bootstrap([ma[k][key] for k in common],
                                             [mb[k][key] for k in common])
                rows.append(f"| {arm} | {label} | {len(common)} | {d:+.4f} | "
                            f"[{lo:+.4f}, {hi:+.4f}] | "
                            f"{'예 (구별 안 됨)' if lo <= 0 <= hi else '아니오'} |")
        if rows:
            parts += ["## 짝지음 부트스트랩 (baseline 대비 · 문항 복원추출 10,000회)", "",
                      "| 팔 | 지표 | 짝지은 문항 | Δ 평균 | 95% CI | 0 포함? |",
                      "|---|---|---|---|---|---|", *rows, ""]

    out = exp / "RESULTS_TABLES.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print("\n".join(parts))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""④ cross-model 비교 — 프롬프트 고정, 모델을 나란히 (어느 라벨러가 최고 Score?).

deepseek/GLM/Qwen32B/Qwen3-Next × {v2, v3}. deployable Score(5축 t*=3). API 0(reset_matrix 재사용).
v3 표는 전부 no-think(deepseek=v3-OFF)라 공정. 산출: outputs/experiments/2026-07-23_s5_redesign/CROSSMODEL.md
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402

FROZEN = ["protagonist", "space", "time", "causality", "intentionality"]
DOCS = ["pieman", "tunnel", "monkey"]
OUT = ROOT / "outputs/experiments/2026-07-23_s5_redesign"


def sc(suf: str, doc: str, t=3):
    f = ROOT / f"outputs/eacl_phase0/{doc}_michelmann_s5_{suf}/reset_matrix.jsonl"
    if not f.exists():
        return None
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if str(r["doc_id"]) == doc]
    if not rows:
        return None
    rows.sort(key=lambda r: int(r["transition"]))
    n = len(rows) + 1
    gold = [0] * n
    for r in rows:
        if int(r["boundary"]):
            gold[int(r["transition"]) - 1] = 1
    cnt = {int(r["transition"]): sum(int(r[a]) for a in FROZEN) for r in rows}
    bt = [i for i in cnt if cnt[i] >= t]
    pred = [0] * n
    for i in bt:
        pred[i - 1] = 1
    return score_dialogues([gold], [pred])["score"]


def table(title, rows):
    L = [f"## {title}", "", "| 모델 | pieman | tunnel | monkey | 평균 |", "|---|---|---|---|---|"]
    data = []
    for name, suf in rows:
        ss = [sc(suf, d) for d in DOCS]
        if any(s is None for s in ss):
            L.append(f"| {name} | _pending_ | | | |")
            continue
        data.append((name, ss, sum(ss) / 3))
    best = max((a for _, _, a in data), default=None)
    for name, ss, a in data:
        star = " ★" if a == best else ""
        L.append(f"| {name} | {ss[0]:.3f} | {ss[1]:.3f} | {ss[2]:.3f} | **{a:.3f}**{star} |")
    L.append("")
    return L


def build():
    L = ["# ④ cross-model Score 비교 (프롬프트 고정, dev-3, deployable t\\*=3)", "",
         "> 각 1 run(배치 비결정 ±0.01~0.05 노이즈). v3=전부 no-think(공정). deepseek v2=reasoning(교란).", ""]
    L += table("v2 프롬프트 (⚠️ deepseek=reasoning, 오픈=no-think)",
               [("deepseek-v2 (reason)", "eismv2"), ("GLM-v2", "glm_v2"),
                ("Qwen32B-v2", "qwen32b_v2"), ("Qwen3-Next-v2", "qwennext_v2"), ("deepseek-chat-v2", "dschat_v2")])
    L += table("v3 프롬프트 — 전부 no-think (공정 비교)",
               [("deepseek-v3-OFF", "eismv3_noreason"), ("GLM-v3", "glm_v3"),
                ("Qwen32B-v3", "qwen32b_v3"), ("Qwen3-Next-v3", "qwennext_v3"), ("deepseek-chat-v3", "dschat_v3")])
    L += ["> 참고 deepseek-v3-ON(reason) 평균 = 0.327 (v3에서 reasoning은 해로움).", ""]
    return "\n".join(L) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rep = build()
    (OUT / "CROSSMODEL.md").write_text(rep, encoding="utf-8")
    print(rep)


if __name__ == "__main__":
    main()

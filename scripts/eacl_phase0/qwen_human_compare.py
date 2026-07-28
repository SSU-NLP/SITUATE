#!/usr/bin/env python3
"""①B human 라벨 재사용 → Qwen v2 라벨 얹어 [사람 y/n · deepseek · Qwen] 3자 비교.

같은 repcorpus 100문장이라 ①B human_label(yes/no = 비중립/중립 블라인드 판정)이 그대로 유효.
재라벨 없이 Qwen(repcorpus_s5_q36_35b_v2) 라벨만 조회해 대조 → 축별 human-Qwen 일치율·✗(over/under) +
문장별로 [사람 / deepseek / Qwen] 나란히 표시(v3 재설계 근거). ⏱ 시간단축.

입력: ①B human_sample/{sheet.jsonl, filled.csv, answer_key.jsonl} + Qwen run-dir.
산출: outputs/experiments/2026-07-24_qwen_human_verify/COMPARE.md
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HS = ROOT / "outputs/experiments/2026-07-22_gate1b_repcorpus/human_sample"
QRUN = ROOT / "outputs/eacl_phase0/repcorpus_s5_q36_35b_v2"
OUT = ROOT / "outputs/experiments/2026-07-24_qwen_human_verify"

FIELD = {"space": "location", "intentionality": "active_goal", "time": "temporal_shift_marker",
         "protagonist": "perspective", "causality": "aspect"}
NEUTRAL = {"space": "unspecified", "intentionality": "none", "time": "absent",
           "protagonist": "none", "causality": "ongoing"}
AXES = ["space", "intentionality", "time", "protagonist", "causality"]


def rj(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def qwen_label(doc, item_id, axis):
    f = QRUN / doc / f"{axis}_states.jsonl"
    if not f.exists():
        return None
    for r in rj(f):
        if int(r["item_id"]) == int(item_id):
            return r.get(FIELD[axis])
    return None


def build():
    sheet = {r["row_id"]: r for r in rj(HS / "sheet.jsonl")}
    akey = {r["row_id"]: r for r in rj(HS / "answer_key.jsonl")}
    human = {}
    with (HS / "filled.csv").open() as f:
        for row in csv.DictReader(f):
            human[row["row_id"]] = row["human_label"].strip().lower()

    L = ["# Qwen v2 인간검증 (①B human 라벨 재사용, 재라벨 0)", "",
         "> 같은 repcorpus 100문장. 사람 y/n(비중립/중립 블라인드) 그대로. Qwen3.6-35B-A3B v2 라벨 얹어 대조.",
         "> **밑에 [사람 · deepseek · Qwen] 나란히** — 셋이 갈리는 곳이 v3 재설계 후보.",
         "> 사람 'no'=중립(unspecified/none/absent 등이 정답) · 'yes'=비중립(listed 매치). "
         "model 'neutral'=중립토큰 라벨.", ""]

    stats = {ax: {"n": 0, "agree": 0, "over": 0, "under": 0,  # human vs Qwen
                  "ds_agree": 0} for ax in AXES}
    rows_by_axis = {ax: [] for ax in AXES}
    for rid, sh in sheet.items():
        ax = sh["axis"]; doc = sh["doc_id"]; it = sh["item_id"]
        h = human.get(rid, "")          # yes/no
        ds = akey[rid]["assigned_label"]
        q = qwen_label(doc, it, ax)
        h_neutral = (h == "no")
        ds_neutral = (ds == NEUTRAL[ax])
        q_neutral = (q == NEUTRAL[ax]) if q is not None else None
        st = stats[ax]; st["n"] += 1
        if q_neutral is not None:
            if h_neutral == q_neutral:
                st["agree"] += 1
            elif h_neutral and not q_neutral:
                st["over"] += 1     # 사람=중립, Qwen=비중립 → 과잉귀속
            else:
                st["under"] += 1    # 사람=비중립, Qwen=중립 → 회피
        if h_neutral == ds_neutral:
            st["ds_agree"] += 1
        flag = ""
        if q_neutral is not None and h_neutral != q_neutral:
            flag = " 🔴over" if (h_neutral and not q_neutral) else " 🔴under"
        rows_by_axis[ax].append((rid, sh["sentence"][:120], h, ds, q, flag))

    # 요약
    L += ["## 축별 요약 — human vs Qwen (deepseek 대비)", "",
          "| 축 | n | human-Qwen 일치 | Qwen over | Qwen under | (참고)human-deepseek 일치 |",
          "|---|---|---|---|---|---|"]
    for ax in AXES:
        s = stats[ax]
        L.append(f"| {ax} | {s['n']} | {s['agree']}/{s['n']} ({s['agree']/s['n']:.0%}) | "
                 f"{s['over']} | {s['under']} | {s['ds_agree']}/{s['n']} ({s['ds_agree']/s['n']:.0%}) |")

    # 문장별 3자 비교 (불일치 우선)
    L += ["", "## 문장별 [사람 · deepseek · Qwen] (🔴=human↔Qwen 불일치)", ""]
    for ax in AXES:
        L += [f"### {ax} (scored={FIELD[ax]}, 중립={NEUTRAL[ax]})", ""]
        # 불일치 먼저
        rr = sorted(rows_by_axis[ax], key=lambda x: (x[5] == "", x[0]))
        for rid, sent, h, ds, q, flag in rr:
            L.append(f"- `{rid}`{flag}  사람=**{h}** · ds=`{ds}` · qwen=`{q}`")
            L.append(f"  > {sent}")
        L.append("")
    return "\n".join(L) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rep = build()
    (OUT / "COMPARE.md").write_text(rep, encoding="utf-8")
    print(rep[:2000])
    print(f"\n[written] {OUT/'COMPARE.md'}")


if __name__ == "__main__":
    main()

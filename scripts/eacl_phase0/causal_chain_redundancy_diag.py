#!/usr/bin/env python3
"""인과 사슬 축이 기존 `goals`(intentionality)와 **중복인지** 판별하는 제한적 진단 (D-B23 후속).

🔴 지위: **본 실험이 아니라 진단이다.** codex 권고(자문 7회차) —
*"새 causal chain 을 본 실험으로 추가하기보다는, goals 와의 **일치율·교차표·증분 lift** 를 보는
제한적 진단까지만. goals 와 **명확히 비중복**이라는 증거가 나올 때만 정식 축으로 재검토."*

**배경 정정(중요).** 앞서 나는 "사슬 ID = 사건 ID 라 전역 분절표가 된다"고 전제해 codex 판정을
왜곡했고, 실측으로 반증됐다 — pieman `goals` 6개 vs gold 경계 18개, intentionality lift 1.14~2.34
(space 는 5.35/5.39). **사슬은 사건보다 거칠다.** codex 는 이에 따라 누출 사유를 **철회**했고,
남은 쟁점은 **중복성**으로 바뀌었다.

**조작적 구분 (codex 지정, 이 정의로만 뽑는다):**
    goals        : 인물의 **의도·목적**이 무엇인가
    causal chain : **의도 여부와 무관하게** 어떤 상태/행동이 무엇을 야기했는가
      → 사고, 환경 변화, 타인의 **비의도적** 반응까지 묶는다.
      → 하나의 goal 이 여러 사슬을 포함하거나, 한 사슬이 여러 goal 을 가로지를 수 있어야 한다.
    **산출이 goals 와 거의 1:1 이면 같은 축을 두 번 계산하는 것이며 추가 가치가 없다.**

위생: G1(변화·경계·이전상태 미질의) · G2(Stage-2 target 단독) · G3(닫힌 목록) · G4(판정은 코드).
동결 파일 `llm_s5_states.py` 의 Stage-1/5축은 **건드리지 않는다** — 자체 Stage-1 호출을 쓴다.
🔒 dev-3 = 연습장(무제한 열람, D-M13③). 본 진단은 **탐색적**이며 확증 근거가 아니다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
import llm_s5_states as s5  # noqa: E402

DEV3 = ("pieman", "tunnel", "monkey")
BASE = REPO / "outputs/eacl_phase0"
OUT = REPO / "outputs/runs/2026-07-28_causalchain_diag"

CHAIN_STAGE1 = """Below is a complete narrative. List the CAUSAL CHAINS it contains, as normalized
snake_case identifiers.

A causal chain is a linked set of states and happenings where one brings about another, REGARDLESS
of whether anyone intended it. Include chains driven by accidents, environmental changes, and other
people's unintended reactions — not only chains someone pursues on purpose.

Name each chain by what it is about (e.g. camera_left_unattended_leads_to_selfies,
rain_floods_the_road_and_traps_the_car, rumor_spreads_and_damages_reputation).

Do not describe events, order, or structure. Do not mention boundaries, segments, changes, or
scenes. Do not number them by when they happen.

NARRATIVE:
{story}"""

CHAIN_SCHEMA = {"type": "object", "additionalProperties": False,
                "properties": {"chains": {"type": "array", "items": {"type": "string"}}},
                "required": ["chains"]}


def stage2_chain_prompt(chains: list[str], items: list[tuple[int, str]]) -> str:
    lst = ", ".join(chains) or "(none)"
    payload = "\n\n".join(f"ITEM {i}\nTARGET: {s}" for i, s in items)
    return f"""Label the causal chain each TARGET sentence belongs to, independently.
Choose exactly one value per field from its fixed list.

Fields:
causal_chain: the causal chain from the list that THIS sentence is part of — the linked set of
happenings that this sentence's content belongs to, whether or not anyone intended it. Judge only
THIS sentence's own content. Use none when this sentence belongs to no chain on the list.
Choose only from: [{lst}, none].

Items are shuffled and unrelated. Never compare items to each other. Do not reason about documents,
stories, their order, or where an item sits.

{payload}"""


def run_doc(c, model: str, doc: str, batch: int, eb: dict | None) -> dict:
    d = OUT / doc
    d.mkdir(parents=True, exist_ok=True)
    master = s5.read(BASE / f"{doc}_michelmann_master.jsonl")
    sents = s5.sequence(master)

    # ── Stage 1 (자체 호출, 동결 Stage-1 미사용) ──────────────────────────
    lp = d / "chain_lists.json"
    if lp.exists():
        chains = json.loads(lp.read_text())["chains"]
    else:
        r = c.chat.completions.create(
            model=model, temperature=0, max_tokens=8000,
            response_format={"type": "json_schema", "json_schema":
                             {"name": "causal_chains", "strict": True, "schema": CHAIN_SCHEMA}},
            messages=[{"role": "system", "content":
                       "You enumerate causal chains present in a text as normalized identifiers. "
                       "You never discuss structure, order, or boundaries."},
                      {"role": "user", "content": CHAIN_STAGE1.format(story="\n".join(sents))}],
            **({"extra_body": eb} if eb else {}))
        chains = sorted(set(json.loads(r.choices[0].message.content)["chains"]))
        lp.write_text(json.dumps({"chains": chains}, ensure_ascii=False, indent=2))
    print(f"[{doc}] Stage-1 사슬 {len(chains)}개", flush=True)

    # ── Stage 2 (target-only, 닫힌 목록) ─────────────────────────────────
    sp = d / "causalchain_states.jsonl"
    done = {int(x["item_id"]): x for x in s5.read(sp)}
    schema = {"type": "object", "additionalProperties": False, "properties": {"items": {
        "type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {
            "item_id": {"type": "integer"},
            "causal_chain": {"type": "string", "enum": chains + ["none"]}},
            "required": ["item_id", "causal_chain"]}}}, "required": ["items"]}
    ids = list(range(len(sents)))
    import random
    random.Random(20260711 + 6).shuffle(ids)          # 실험 축 seed(동결 0~5 와 겹치지 않음)
    while len(done) < len(sents):
        todo = [i for i in ids if i not in done][:batch]
        items = [(i, sents[i]) for i in todo]
        r = c.chat.completions.create(
            model=model, temperature=0, max_tokens=10000,
            response_format={"type": "json_schema",
                             "json_schema": {"name": "chain_states", "strict": True,
                                             "schema": schema}},
            messages=[{"role": "system", "content":
                       "You assign fixed categorical labels to one sentence at a time from a closed "
                       "list. You never compare sentences and never output free text."},
                      {"role": "user", "content": stage2_chain_prompt(chains, items)}],
            **({"extra_body": eb} if eb else {}))
        for x in json.loads(r.choices[0].message.content)["items"]:
            if int(x["item_id"]) in todo and x["causal_chain"] in set(chains) | {"none"}:
                done[int(x["item_id"])] = {"item_id": int(x["item_id"]),
                                           "causal_chain": x["causal_chain"]}
        s5.write(sp, [done[i] for i in sorted(done)])
        print(f"[{doc}] chain {len(done)}/{len(sents)}", flush=True)
    return {"chains": chains, "states": done, "n": len(sents)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen36-35b")
    ap.add_argument("--base-url", default="http://localhost:8001/v1")
    ap.add_argument("--batch", type=int, default=20)
    args = ap.parse_args()
    c = OpenAI(api_key="dummy", base_url=args.base_url, timeout=600)
    eb = {"chat_template_kwargs": {"enable_thinking": False}}

    res = {d: run_doc(c, args.model, d, args.batch, eb) for d in DEV3}

    L = ["# 인과 사슬 축 ↔ `goals` 중복성 진단 (D-B23 후속)\n",
         "> 🔴 **본 실험이 아니라 진단**(codex 권고). dev-3 = 연습장, **탐색적** — 확증 근거 아님.",
         "> 판정: goals 와 **명확히 비중복**이면 정식 축 재검토, 거의 1:1 이면 **중복으로 종결**.\n",
         "\n## 1. 목록 크기 대조\n",
         "| 문서 | 문장 | gold 경계 | goals | **causal chains** |", "|---|---|---|---|---|"]
    rows = {}
    for d in DEV3:
        gl = json.load(open(BASE / f"{d}_michelmann_s5_q36_35b_v2" / d / "story_lists.json"))["lists"]
        master = s5.read(BASE / f"{d}_michelmann_master.jsonl")
        gi = sum(1 for r in master if int(r["boundary"]))
        L.append(f"| {d} | {len(master)+1} | {gi} | {len(gl['goals'])} | "
                 f"**{len(res[d]['chains'])}** |")
        rows[d] = gl
    # ── 2. 문장별 배정 일치율 / 교차표 ────────────────────────────────────
    L += ["\n## 2. 문장별 배정 — goals 와 얼마나 같이 움직이나\n",
          "| 문서 | 사슬 none 율 | goal none 율 | **동시 변화 일치율** | **조정 Rand 지수** |",
          "|---|---|---|---|---|"]
    from sklearn.metrics import adjusted_rand_score
    per_doc = {}
    for d in DEV3:
        it = s5.carry_forward({int(x["item_id"]): x for x in s5.read(
            BASE / f"{d}_michelmann_s5_q36_35b_v2" / d / "intentionality_states.jsonl")},
            "active_goal", "none")
        ch = res[d]["states"]
        n = res[d]["n"]
        g_lab = [it[i]["active_goal"] for i in range(n)]
        c_lab = [ch[i]["causal_chain"] for i in range(n)]
        g_res = np.array([int(g_lab[i] != g_lab[i - 1]) for i in range(1, n)])
        c_res = np.array([int(c_lab[i] != c_lab[i - 1]) for i in range(1, n)])
        agree = float((g_res == c_res).mean())
        ari = float(adjusted_rand_score(g_lab, c_lab))
        L.append(f"| {d} | {np.mean([x=='none' for x in c_lab]):.3f} | "
                 f"{np.mean([x=='none' for x in g_lab]):.3f} | {agree:.3f} | **{ari:.3f}** |")
        per_doc[d] = (g_res, c_res, g_lab, c_lab)
    # ── 3. 증분 lift (gold 사용, 탐색적) ─────────────────────────────────
    L += ["\n## 3. 증분 lift — 사슬이 goals 를 넘어 더 주는 게 있나 (gold 사용, 탐색적)\n",
          "| 문서 | goal-reset lift | **chain-reset lift** | chain∧¬goal lift |", "|---|---|---|---|"]
    for d in DEV3:
        master = s5.read(BASE / f"{d}_michelmann_master.jsonl")
        gold = np.array([int(r["boundary"]) for r in sorted(master, key=lambda r: int(r["transition"]))])
        g_res, c_res, _, _ = per_doc[d]
        m = min(len(gold), len(g_res), len(c_res))
        gold, g_res, c_res = gold[:m], g_res[:m], c_res[:m]
        lf = lambda f: (f[gold == 1].mean() / f[gold == 0].mean()) if f[gold == 0].mean() > 0 else float("nan")
        only = ((c_res == 1) & (g_res == 0)).astype(int)
        L.append(f"| {d} | {lf(g_res):.2f} | **{lf(c_res):.2f}** | "
                 f"{lf(only) if only.sum() else float('nan'):.2f} |")
    L += ["\n> ARI(조정 Rand 지수) = 두 분할이 얼마나 같은가. **1.0 = 완전 동일, 0 = 무작위 수준.**",
          "> ARI 가 높으면 **같은 축을 두 번 계산**한 것이다(codex 판정 기준).",
          "\n⚠️ dev-3 = 연습장. 이 수치는 **탐색적 진단**이며 성능 주장·확증 근거가 아니다(D-B16)."]
    OUT.mkdir(parents=True, exist_ok=True)
    p = REPO / "outputs/experiments/m4m5_phase1_anchor/CAUSALCHAIN_REDUNDANCY.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()

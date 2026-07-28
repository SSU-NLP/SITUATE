#!/usr/bin/env python3
"""A/B/C 조건 인출 + (선택)QA/judge + 카테고리별 판정. (gonogo Tier0 §4,§5)

에피소드=세션. 예산 K 고정.
- A baseline : 질문 유사도 top-K 에피소드.
- B typed    : top-K 중 return 에피소드가 있으면 그 return_target 을 주입,
               예산 유지 위해 가장 낮은 유사도의 non-injected 를 제거(=K 고정).
- C random   : B와 동일 주입 개수, 단 target 을 같은 대화의 임의 과거 에피소드로.

먼저 --dry (무과금): B/C가 A와 갈리는 질문 수 = 검정력 진단.
그다음 전체: EMem conversation_qa 프롬프트로 QA(gpt-4o-mini) → EMem judge(CORRECT/WRONG)
           → 카테고리별 정확도 A/B/C.
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CATEGORY_NAMES, JUDGE_MODEL, QA_MODEL, REPO, chat_many,
                    embed, parse_json_lenient, read_jsonl)
import json

EPS = REPO / "outputs/exp_typed/episodes_typed.jsonl"
LOCOMO = REPO / "benchmarks/locomo/locomo10.json"

QA_SYS = ("You are an intelligent memory assistant answering a question using retrieved "
          "conversation memories (sessions with dates). Reason step by step, then give a "
          'concise final answer. Reply ONLY as JSON: {"final_answer": "..."}.')
JUDGE_SYS = ("Label an answer to a question as CORRECT or WRONG vs the gold answer. Be generous: "
             "if the generated answer refers to the same fact/topic/date as gold (even if longer or "
             'differently phrased), it is CORRECT. Reply ONLY as JSON: {"label": "CORRECT"|"WRONG"}.')


def load_qa():
    data = json.loads(LOCOMO.read_text())
    qa = {}
    for idx, s in enumerate(data):
        cid = str(s.get("sample_id", idx))
        rows = []
        for q in s.get("qa", []):
            ans = q.get("answer")
            if ans is None:
                ans = q.get("adversarial_answer")
            rows.append({"question": str(q["question"]), "answer": "" if ans is None else str(ans),
                         "category": q.get("category")})
        qa[cid] = rows
    return qa


def retrieve(sims, eps, K, mode, rng, ep_by_id):
    """sims: 질문-에피소드 유사도(len=n_ep). 반환: (선택 ep_id 리스트, n_inject)."""
    order = list(np.argsort(-sims))
    topK = order[:K]
    if mode == "A":
        return [eps[i]["ep_id"] for i in topK], 0
    # B/C: top-K 중 return 에피소드의 target 주입
    chosen = [eps[i]["ep_id"] for i in topK]
    chosen_set = set(chosen)
    injects = []
    for i in topK:
        e = eps[i]
        if e["ttype"] == "return" and e.get("return_target") is not None:
            tgt = e["return_target"]
            if mode == "C":
                cands = [p for p in ep_by_id if p < e["ep_id"] and p not in chosen_set]
                tgt = rng.choice(cands) if cands else None
            if tgt is not None and tgt not in chosen_set and tgt not in injects:
                injects.append(tgt)
    if not injects:
        return chosen, 0
    # 예산 유지: 주입 수만큼 가장 낮은 유사도 non-injected 제거
    keep = [eps[i]["ep_id"] for i in topK]
    for _ in injects:
        # 유사도 최저(=order 뒤쪽) 제거
        keep.pop()  # topK는 이미 유사도 내림차순
    final = keep + injects
    return final, len(injects)


def build_context(ep_ids, ep_by_id, max_chars=1200):
    parts = []
    for eid in ep_ids:
        e = ep_by_id[eid]
        parts.append(f"[Session {e['ep_id']} | {e['date']}]\n{e['text'][:max_chars]}")
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry", action="store_true", help="인출 발산만(무과금)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    eps_all = read_jsonl(EPS)
    qa = load_qa()
    by_conv = defaultdict(list)
    for e in eps_all:
        by_conv[e["conv_id"]].append(e)
    for v in by_conv.values():
        v.sort(key=lambda e: e["ep_id"])

    rng = random.Random(args.seed)
    # 인출 계획: 각 (conv, question) 에 대해 A/B/C ep_ids
    plan = []  # dict per question
    for cid, group in by_conv.items():
        ep_by_id = {e["ep_id"]: e for e in group}
        ep_ids_sorted = sorted(ep_by_id)
        texts = [e["text"][:2000] for e in group]
        ev = embed(texts)
        qrows = qa.get(cid, [])
        if not qrows:
            continue
        qv = embed([r["question"] for r in qrows])
        for qi, r in enumerate(qrows):
            sims = ev @ qv[qi]
            a_ids, _ = retrieve(sims, group, args.K, "A", rng, ep_ids_sorted)
            b_ids, nb = retrieve(sims, group, args.K, "B", rng, ep_ids_sorted)
            c_ids, nc = retrieve(sims, group, args.K, "C", rng, ep_ids_sorted)
            plan.append({"cid": cid, "qi": qi, "question": r["question"],
                         "answer": r["answer"], "category": r["category"],
                         "A": a_ids, "B": b_ids, "C": c_ids, "nB": nb, "nC": nc,
                         "ep_by_id": None})  # ep_by_id 재참조는 아래서
    # 발산 진단
    diffB = sum(1 for p in plan if set(p["A"]) != set(p["B"]))
    diffC = sum(1 for p in plan if set(p["A"]) != set(p["C"]))
    by_cat_diffB = defaultdict(int)
    for p in plan:
        if set(p["A"]) != set(p["B"]):
            by_cat_diffB[p["category"]] += 1
    print(f"\n=== 인출 발산 진단 (K={args.K}, seed={args.seed}) ===")
    print(f"총 질문: {len(plan)}")
    print(f"B가 A와 다른 질문: {diffB}  ({100*diffB/max(1,len(plan)):.1f}%)")
    print(f"C가 A와 다른 질문: {diffC}")
    print("B≠A 카테고리별:", {CATEGORY_NAMES.get(k, k): v for k, v in sorted(by_cat_diffB.items(), key=lambda x: str(x[0]))})
    if args.dry:
        print("\n[dry] QA/judge 생략. 위 발산 수가 검정력.")
        return

    # ── 전체 QA + judge ──
    ep_maps = {cid: {e["ep_id"]: e for e in g} for cid, g in by_conv.items()}

    def qa_msgs(job):
        p, cond = job
        ctx = build_context(p[cond], ep_maps[p["cid"]])
        return [{"role": "system", "content": QA_SYS},
                {"role": "user", "content": f"Memories:\n{ctx}\n\nQuestion: {p['question']}"}]

    conds = ["A", "B", "C"]
    jobs = [(p, c) for c in conds for p in plan]
    print(f"\n[QA] {len(jobs)} calls ({QA_MODEL}) ...", flush=True)
    qa_out = chat_many(QA_MODEL, jobs, qa_msgs, max_tokens=400, workers=args.workers)
    preds = {}
    for (p, c), raw in zip(jobs, qa_out):
        d = parse_json_lenient(raw) or {}
        preds[(id(p), c)] = str(d.get("final_answer", raw))[:400]

    def judge_msgs(job):
        p, cond = job
        pred = preds[(id(p), cond)]
        return [{"role": "system", "content": JUDGE_SYS},
                {"role": "user", "content": f"Question: {p['question']}\nGold answer: {p['answer']}\nGenerated answer: {pred}"}]

    print(f"[Judge] {len(jobs)} calls ({JUDGE_MODEL}) ...", flush=True)
    j_out = chat_many(JUDGE_MODEL, jobs, judge_msgs, max_tokens=16, workers=args.workers)
    correct = {}
    for (p, c), raw in zip(jobs, j_out):
        d = parse_json_lenient(raw) or {}
        lab = str(d.get("label", "")).upper()
        correct[(id(p), c)] = 1 if "CORRECT" in lab else 0

    # 카테고리별 정확도
    def acc_table(cond):
        tot, cor = defaultdict(int), defaultdict(int)
        for p in plan:
            tot[p["category"]] += 1
            cor[p["category"]] += correct[(id(p), cond)]
            tot["ALL"] += 1
            cor["ALL"] += correct[(id(p), cond)]
        return {k: 100 * cor[k] / max(1, tot[k]) for k in tot}, dict(tot)

    accs = {c: acc_table(c) for c in conds}
    cats = sorted({p["category"] for p in plan}, key=lambda x: str(x))
    print("\n=== 카테고리별 정확도 (%) ===")
    hdr = f"{'category':<14}{'n':>5} {'A':>7} {'B':>7} {'C':>7}  {'B-A':>6} {'B-C':>6}"
    print(hdr)
    _, tot = accs["A"]
    for k in list(cats) + ["ALL"]:
        name = CATEGORY_NAMES.get(k, str(k))
        a, b, c = accs["A"][0][k], accs["B"][0][k], accs["C"][0][k]
        print(f"{name:<14}{tot[k]:>5} {a:>7.1f} {b:>7.1f} {c:>7.1f}  {b-a:>+6.1f} {b-c:>+6.1f}")

    out = {"K": args.K, "seed": args.seed, "n_q": len(plan),
           "diffB": diffB, "diffC": diffC,
           "acc": {c: accs[c][0] for c in conds}, "n_by_cat": tot}
    Path(REPO / "outputs/exp_typed/results_seed%d.json" % args.seed).write_text(
        json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nsaved -> outputs/exp_typed/results_seed{args.seed}.json")


if __name__ == "__main__":
    main()

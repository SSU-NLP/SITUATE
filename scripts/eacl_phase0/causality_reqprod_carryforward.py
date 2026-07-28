#!/usr/bin/env python3
"""requires/produces 매칭에 carry-forward 규칙 적용 재측정.
strict(인접 produces[t-1]∩requires[t]) vs carry-forward(누적 produces ∩ requires[t]) 비교.
같은 추출에 규칙만 바꿔 공정 비교. frozen Qwen v2, N회.
"""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path
from dotenv import load_dotenv
REPO = Path(__file__).resolve().parents[2]
load_dotenv(REPO / ".env")
from openai import OpenAI  # noqa: E402
c = OpenAI(api_key="dummy", base_url="http://localhost:8001/v1", timeout=180, max_retries=1)
EXTRA = {"chat_template_kwargs": {"enable_thinking": False}}

def ask(sys, user):
    r = c.chat.completions.create(model="qwen36-35b", temperature=0, max_tokens=3000,
        response_format={"type": "json_object"}, extra_body=EXTRA,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}])
    return json.loads(r.choices[0].message.content)

S1 = ("You extract the atomic world-state conditions that turn true or false over a short story. "
      "Return JSON {\"states\": [...]} — short snake_case predicates (bars_intact, has_saw...). 8-20 items.")
S2 = ("For EACH numbered sentence's main action, choose from the STATE LIST: requires = conditions that "
      "must ALREADY be true; produces = conditions the action makes true/false. Copy predicates EXACTLY "
      "from the list; [] if none. Return JSON {\"items\":[{\"idx\":int,\"requires\":[...],\"produces\":[...]}]}.")

def extract(sents):
    states = [str(s).strip().lower() for s in ask(S1, "\n".join(sents)).get("states", [])]
    sset = set(states)
    body = "STATE LIST: [" + ", ".join(states) + "]\n\nSENTENCES:\n" + "\n".join(f"[{i}] {s}" for i, s in enumerate(sents))
    obj = ask(S2, body)
    st = {}
    for it in obj.get("items", []):
        i = int(it["idx"])
        rq = set(str(p).strip().lower() for p in it.get("requires", []) if str(p).strip().lower() in sset)
        pr = set(str(p).strip().lower() for p in it.get("produces", []) if str(p).strip().lower() in sset)
        st[i] = (rq, pr)
    return states, st

def bd_strict(st, n):
    return {t: (0 if (st.get(t-1,(set(),set()))[1] & st.get(t,(set(),set()))[0]) else 1) for t in range(1, n)}

def bd_carry(st, n, empty_boundary=True):
    """누적 produces ∩ requires[t]. requires 비면 empty_boundary 로."""
    b = {}
    active = set(st.get(0, (set(), set()))[1])
    for t in range(1, n):
        rq = st.get(t, (set(), set()))[0]
        if not rq:
            b[t] = 1 if empty_boundary else 0
        else:
            b[t] = 0 if (rq & active) else 1
        active |= st.get(t, (set(), set()))[1]
    return b

def metrics(bvecs, cont_tr, bound_tr, all_tr):
    agr = statistics.mean(max([bv[t] for bv in bvecs].count(0), [bv[t] for bv in bvecs].count(1)) / len(bvecs) for t in all_tr)
    cont = statistics.mean(statistics.mean(1 - bv[t] for t in cont_tr) for bv in bvecs)
    bnd = statistics.mean(statistics.mean(bv[t] for t in bound_tr) for bv in bvecs)
    return round(agr, 3), round(cont, 3), round(bnd, 3)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    a = ap.parse_args()
    rows = [json.loads(l) for l in (REPO / "outputs/experiments/2026-07-25_causality_reqprod_proto/design_passage.jsonl").open()]
    sents = [r["text"] for r in rows]; n = len(sents)
    cont_tr = [r["idx"] for r in rows if r["expect"] == "cont"]
    bound_tr = [r["idx"] for r in rows if r["expect"] == "bound"]
    all_tr = list(range(1, n))

    rules = {"strict": lambda st: bd_strict(st, n),
             "carry(empty=경계)": lambda st: bd_carry(st, n, True),
             "carry(empty=연속)": lambda st: bd_carry(st, n, False)}
    bv = {k: [] for k in rules}
    for r in range(a.runs):
        states, st = extract(sents)
        for k, fn in rules.items():
            bv[k].append(fn(st))
        if r == 0:
            print("=== run1 규칙별 경계 (설계: cont=", cont_tr, "bound=", bound_tr, ") ===")
            for k in rules:
                print(f"  {k:<16} 경계@{[t for t in all_tr if bv[k][0][t]==1]}")
    print(f"\n=== N={a.runs} 지표 (일치>0.85 · 인과연쇄→연속≥0.8 · 장면전환→경계=1.0) ===")
    print(f"{'규칙':<18}{'경계일치':<10}{'인과연쇄→연속':<15}{'장면전환→경계'}")
    for k in rules:
        agr, cont, bnd = metrics(bv[k], cont_tr, bound_tr, all_tr)
        pas = (agr > 0.85 and cont >= 0.8 and bnd >= 0.999)
        print(f"  {k:<16}{agr:<10}{cont:<15}{bnd}   {'PASS' if pas else 'FAIL'}")

if __name__ == "__main__":
    main()

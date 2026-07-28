#!/usr/bin/env python3
"""causality requires/produces 프로토타입 (사전등록 2026-07-25_causality_reqprod_proto/PRE_REGISTRATION.md).

목적: "produces[t-1] ∩ requires[t] ≠ ∅ → 인과 연결" 매칭이 신뢰성 있게 되는가를 소표본으로 측정.
두 변형: V-closed(Stage1 states 닫힌목록에서 선택) vs V-free(자유서술). frozen Qwen v2, temp=0, N회 반복.
통과조건(사전선언): 빈집합<40% · N회 경계일치>0.85 · 인과연쇄→연속≥0.8 · 장면전환→경계 2/2.
"""
from __future__ import annotations
import argparse, json, re, collections, statistics
from pathlib import Path
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[2]
load_dotenv(REPO / ".env")
from openai import OpenAI  # noqa: E402

BASE = "http://localhost:8001/v1"; MODEL = "qwen36-35b"
EXTRA = {"chat_template_kwargs": {"enable_thinking": False}}
c = OpenAI(api_key="dummy", base_url=BASE, timeout=180, max_retries=1)

STOP = {"the","a","an","of","his","her","he","she","it","to","and","in","on","at","is","was",
        "be","been","that","this","with","for","into","him","them","up","down","out"}

def ask_json(sys, user):
    r = c.chat.completions.create(model=MODEL, temperature=0, max_tokens=3000,
        response_format={"type": "json_object"}, extra_body=EXTRA,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}])
    return json.loads(r.choices[0].message.content)

# ---------- Stage 1 (closed-list states) ----------
S1_SYS = ("You extract the atomic world-state conditions that turn true or false over a short story. "
          "Return JSON {\"states\": [...]} — each a short snake_case predicate about the world "
          "(e.g. bars_intact, has_saw, inside_cell, phone_ringing, guards_alerted). 8-20 items. "
          "Conditions only — not events, not sentences.")

def stage1_states(passage):
    return [str(s).strip().lower() for s in ask_json(S1_SYS, passage).get("states", []) if str(s).strip()]

# ---------- Stage 2 closed ----------
S2C_SYS = ("For EACH numbered sentence's main action, choose from the STATE LIST: "
           "requires = conditions that must ALREADY be true for the action to occur; "
           "produces = conditions the action makes true or false. Choose ONLY predicates from the list "
           "(copy exactly); use [] if none apply. Judge each sentence on its own. "
           "Return JSON {\"items\":[{\"idx\":int,\"requires\":[...],\"produces\":[...]}]}.")

def stage2_closed(sents, states):
    lst = ", ".join(states)
    body = "STATE LIST: [" + lst + "]\n\nSENTENCES:\n" + "\n".join(f"[{i}] {s}" for i, s in enumerate(sents))
    obj = ask_json(S2C_SYS, body)
    sset = set(states)
    out = {}
    for it in obj.get("items", []):
        i = int(it["idx"])
        req = [p for p in it.get("requires", []) if str(p).strip().lower() in sset]
        pro = [p for p in it.get("produces", []) if str(p).strip().lower() in sset]
        out[i] = (set(str(x).strip().lower() for x in req), set(str(x).strip().lower() for x in pro))
    return out

# ---------- Stage 2 free ----------
S2F_SYS = ("For EACH numbered sentence's main action: requires = short phrases for what must ALREADY be "
           "true; produces = short phrases for what the action makes true/false. Free wording, 0-3 each, "
           "[] if none. Return JSON {\"items\":[{\"idx\":int,\"requires\":[...],\"produces\":[...]}]}.")

def _toks(phrase):
    return frozenset(w for w in re.findall(r"[a-z]+", phrase.lower()) if w not in STOP and len(w) > 2)

def stage2_free(sents):
    body = "SENTENCES:\n" + "\n".join(f"[{i}] {s}" for i, s in enumerate(sents))
    obj = ask_json(S2F_SYS, body)
    out = {}
    for it in obj.get("items", []):
        i = int(it["idx"])
        req = [_toks(str(x)) for x in it.get("requires", []) if str(x).strip()]
        pro = [_toks(str(x)) for x in it.get("produces", []) if str(x).strip()]
        out[i] = (req, pro)
    return out

# ---------- matching -> boundary ----------
def boundaries_closed(states_by_sent, n):
    b = {}
    for t in range(1, n):
        pr = states_by_sent.get(t - 1, (set(), set()))[1]
        rq = states_by_sent.get(t, (set(), set()))[0]
        b[t] = 0 if (pr & rq) else 1  # link -> 0 (not boundary), else 1
    return b

def boundaries_free(states_by_sent, n, jac=0.5):
    b = {}
    for t in range(1, n):
        pr = states_by_sent.get(t - 1, ([], []))[1]
        rq = states_by_sent.get(t, ([], []))[0]
        link = any(a and c2 and len(a & c2) / len(a | c2) >= jac for a in pr for c2 in rq)
        b[t] = 0 if link else 1
    return b

def empty_ratio(states_by_sent, n):
    e = sum(1 for i in range(n) if not states_by_sent.get(i, (set(), set()))[0]
            or not states_by_sent.get(i, (set(), set()))[1])
    return e / n

def agreement(bvecs, transitions):
    """mean over transitions of (majority count / N)."""
    N = len(bvecs)
    vals = []
    for t in transitions:
        labs = [bv[t] for bv in bvecs]
        vals.append(max(labs.count(0), labs.count(1)) / N)
    return statistics.mean(vals)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passage", type=Path,
                    default=REPO / "outputs/experiments/2026-07-25_causality_reqprod_proto/design_passage.jsonl")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", type=Path,
                    default=REPO / "outputs/experiments/2026-07-25_causality_reqprod_proto/proto_result.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.passage.open()]
    sents = [r["text"] for r in rows]
    n = len(sents)
    expect = {r["idx"]: r["expect"] for r in rows}
    cont_tr = [i for i in range(1, n) if expect.get(i) == "cont"]
    bound_tr = [i for i in range(1, n) if expect.get(i) == "bound"]
    all_tr = list(range(1, n))

    report = {"n_sent": n, "runs": args.runs, "cont_tr": cont_tr, "bound_tr": bound_tr, "variants": {}}
    for variant in ("closed", "free"):
        bvecs, empties, states_sizes = [], [], []
        for r in range(args.runs):
            if variant == "closed":
                states = stage1_states("\n".join(sents))
                states_sizes.append(len(states))
                s2 = stage2_closed(sents, states)
                b = boundaries_closed(s2, n)
            else:
                s2 = stage2_free(sents)
                b = boundaries_free(s2, n)
            bvecs.append(b)
            empties.append(empty_ratio(s2, n))
            print(f"[{variant}] run{r+1}: boundaries@{[t for t in all_tr if b[t]==1]} empty={empties[-1]:.2f}", flush=True)
        # metrics
        emp = statistics.mean(empties)
        agr = agreement(bvecs, all_tr)
        cont_acc = statistics.mean(statistics.mean(1 - bv[t] for t in cont_tr) for bv in bvecs) if cont_tr else None
        bound_acc = statistics.mean(statistics.mean(bv[t] for t in bound_tr) for bv in bvecs) if bound_tr else None
        report["variants"][variant] = {
            "empty_ratio": round(emp, 3), "boundary_agreement": round(agr, 3),
            "causalchain_continuous_acc": round(cont_acc, 3) if cont_acc is not None else None,
            "scenebreak_boundary_acc": round(bound_acc, 3) if bound_acc is not None else None,
            "states_sizes": states_sizes,
            "per_run_boundaries": [[t for t in all_tr if bv[t] == 1] for bv in bvecs],
        }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    # verdict vs pre-declared criteria
    print("\n=== 통과조건 판정 (사전선언) ===")
    for v in ("closed", "free"):
        m = report["variants"][v]
        c1 = m["empty_ratio"] < 0.40
        c2 = m["boundary_agreement"] > 0.85
        c3 = (m["causalchain_continuous_acc"] or 0) >= 0.80
        c4 = (m["scenebreak_boundary_acc"] or 0) >= 0.999
        pas = all([c1, c2, c3, c4])
        print(f"[V-{v}] 빈집합<40%={m['empty_ratio']:.2f}{'✓'if c1 else'✗'} | "
              f"경계일치>0.85={m['boundary_agreement']:.2f}{'✓'if c2 else'✗'} | "
              f"인과연쇄→연속≥0.8={m['causalchain_continuous_acc']}{'✓'if c3 else'✗'} | "
              f"장면전환→경계=2/2={m['scenebreak_boundary_acc']}{'✓'if c4 else'✗'} | "
              f"=> {'PASS' if pas else 'FAIL'}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()

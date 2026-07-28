#!/usr/bin/env python3
"""§6 채택바 평가 — dominant-actor turnover 통합 protagonist 축 후보 (사전등록 2026-07-22).

protagonist 자리에 {perspective, present, charset, dominant_turnover} 각각 대입 → BASE4와 합산 →
count>=t*, t*=argmax Score(pieman)→tunnel/monkey 적용. human gold vs. 동일 하네스(eval_seg_score와 동일).

protagonist-family reset 는 새 domact 추출(protagonist_states, dominant_actor 포함)에서 재계산.
BASE4(space/time/causality/intentionality) reset 은 frozen eismv2 reset_matrix 에서 그대로.
"""
from __future__ import annotations
import importlib.util, json
from pathlib import Path
from hi_ontop.dts_scoring import score_dialogues

_spec = importlib.util.spec_from_file_location("s5", Path(__file__).with_name("llm_s5_states.py"))
_s5 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_s5)

DOCS = ["pieman", "tunnel", "monkey"]
BASE4 = ["space", "time", "causality", "intentionality"]


def reset_dominant_turnover(a, b):  # §6 candidate (removed from frozen code; kept here for the record)
    return int(a["dominant_actor"] != b["dominant_actor"]
               and "none" not in (a["dominant_actor"], b["dominant_actor"]))


PROT = {  # display -> (reset fn on protagonist states a,b)
    "perspective": _s5.reset_perspective,
    "present":     _s5.reset_protagonist_present,
    "charset":     _s5.reset_protagonist_charset,
    "dominant":    reset_dominant_turnover,
}


def read(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def end_turn(n, bt):
    y = [0] * n
    for i in bt:
        y[i - 1] = 1
    return y


def prot_resets(doc: str) -> dict[str, dict[int, int]]:
    """per-transition reset for each protagonist variant, from domact extraction."""
    ddir = Path(f"outputs/eacl_phase0/{doc}_michelmann_s5_eismv2_domact/{doc}")
    rows = {r["item_id"]: r for r in read(ddir / "protagonist_states.jsonl")}
    st = _s5.carry_forward(rows, "dominant_actor", "none")  # §6 slow var (matches production)
    n = len(rows)
    return {name: {i: fn(st[i - 1], st[i]) for i in range(1, n)} for name, fn in PROT.items()}


def base_and_gold(doc: str):
    """BASE4 counts + gold boundaries from frozen eismv2 reset_matrix."""
    M = {int(r["transition"]): r for r in
         read(f"outputs/eacl_phase0/{doc}_michelmann_s5_eismv2/reset_matrix.jsonl")}
    base = {t: sum(int(M[t][a]) for a in BASE4) for t in M}
    gold_bt = [t for t in M if int(M[t]["boundary"])]
    return base, gold_bt, len(M) + 1


def main():
    data = {}
    for d in DOCS:
        base, gold_bt, n = base_and_gold(d)
        pr = prot_resets(d)
        data[d] = {"base": base, "gold_bt": gold_bt, "n": n, "pr": pr}

    def score(doc, variant, t):
        D = data[doc]
        pr = D["pr"][variant]
        cnt = {tr: D["base"][tr] + pr.get(tr, 0) for tr in D["base"]}
        bt = [tr for tr in cnt if cnt[tr] >= t]
        gold = end_turn(D["n"], D["gold_bt"])
        return score_dialogues([gold], [end_turn(D["n"], bt)])["score"]

    print(f"{'variant':<12} {'t*':>3} | {'pieman':>7} {'tunnel':>7} {'monkey':>7} | {'test평균':>7} {'3편평균':>7}")
    print("-" * 70)
    res = {}
    for v in PROT:
        tstar = max(range(1, 6), key=lambda t: score("pieman", v, t))
        sc = {d: score(d, v, tstar) for d in DOCS}
        testavg = (sc["tunnel"] + sc["monkey"]) / 2
        allavg = sum(sc.values()) / 3
        res[v] = (tstar, sc, testavg, allavg)
        print(f"{v:<12} {tstar:>3} | {sc['pieman']:>7.3f} {sc['tunnel']:>7.3f} {sc['monkey']:>7.3f} "
              f"| {testavg:>7.3f} {allavg:>7.3f}")

    print("\n=== §6 채택 판정 (사전등록 바) ===")
    dom_avg = res["dominant"][3]
    pres_avg = res["present"][3]
    persp_avg = res["perspective"][3]
    bar = pres_avg - 0.01
    print(f"dominant 3편평균 {dom_avg:.3f}  |  바1: >= present-0.01 = {bar:.3f}  ->"
          f" {'통과' if dom_avg >= bar else '미달'}")
    print(f"                        |  바2: > perspective = {persp_avg:.3f}  ->"
          f" {'통과' if dom_avg > persp_avg else '미달'}")
    print(f"  (참조: present(new){pres_avg:.3f} vs frozen 0.447; charset(new){res['charset'][3]:.3f})")
    adopt = dom_avg >= bar and dom_avg > persp_avg
    print(f"\n>>> 판정: {'✅ 통합축 dominant_turnover 채택' if adopt else '🔴 미채택 → present 동결, dominant는 LoCoMo 이월'}")


if __name__ == "__main__":
    main()

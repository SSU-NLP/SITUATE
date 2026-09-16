#!/usr/bin/env python3
r"""인과 축 조작화 일괄 대결 — dev-3, 기존 라벨 재계산(API 0).

사전등록 `outputs/experiments/2026-07-31_causality_bakeoff/PREREG.md` (계산 전 작성, 미수정).
후보를 **5축 count ≥ t*=3 규칙에 꽂아** deployable Score 를 낸다 — 축은 혼자 쓰이지 않으므로
축 단독 lift 가 아니라 이것이 주 지표다. 나머지 4축·t\*·채점기 전부 동결.

    .venv/bin/python scripts/eacl_phase0/causality_bakeoff.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

P0 = REPO / "outputs/eacl_phase0"
OUT = REPO / "outputs/experiments/2026-07-31_causality_bakeoff"
DOCS = ("pieman", "tunnel", "monkey")
REPS = ("q36_35b_v31", "q36_35b_v31_r2", "q36_35b_v31_r3")
OTHER4 = ("protagonist_present", "space", "time", "intentionality")
T_STAR = 3   # 🔒 동결

# 🔒 3군 접기 — 점수 보기 전 고정 (PREREG §2)
GROUP = {"physical_action": 0, "movement_or_travel": 0, "conflict_or_violence": 0,
         "transaction_or_exchange": 0, "work_or_creation": 0,
         "state_description": 1, "thought_or_emotion": 1, "perception_or_observation": 1,
         "dialogue": 2, "other": 3}
RE_OPEN = re.compile(r"^\W*(so|therefore|thus|hence|consequently|accordingly|as a result"
                     r"|because|since|as|then|for this reason)\b", re.I)


def load(doc: str, rep: str) -> dict:
    d = P0 / f"{doc}_michelmann_s5_{rep}"
    rm = sorted((json.loads(l) for l in (d / "reset_matrix.jsonl").read_text().splitlines()
                 if l.strip()), key=lambda r: r["transition"])
    st = sorted((json.loads(l) for l in (d / doc / "causality_states.jsonl").read_text().splitlines()
                 if l.strip()), key=lambda r: int(re.search(r"(\d+)$", str(r["item_id"])).group(1)))
    master = sorted((json.loads(l) for l in (P0 / f"{doc}_michelmann_master.jsonl").read_text()
                     .splitlines() if l.strip()), key=lambda r: r["transition"])
    txt = {int(r["transition"]): r["current"] for r in master}
    tr = [int(r["transition"]) for r in rm]
    et = [s.get("event_type") for s in st]
    asp = [s.get("aspect") for s in st]

    def at(seq, i):
        return seq[i] if 0 <= i < len(seq) else None

    o = {"gold": np.asarray([r["boundary"] for r in rm], np.int8),
         "other4": np.stack([np.asarray([r[a] for r in rm], np.int8) for a in OTHER4], 1),
         "frozen_causality": np.asarray([r["causality"] for r in rm], np.int8),
         "cm": np.asarray([r["causal_marker"] for r in rm], np.int8)}
    o["et_change"] = np.asarray([int(at(et, t - 1) != at(et, t)) for t in tr], np.int8)
    o["grp_change"] = np.asarray(
        [int(GROUP.get(at(et, t - 1), 3) != GROUP.get(at(et, t), 3)) for t in tr], np.int8)
    o["inc_b"] = np.asarray([int(at(asp, t) == "inceptive") for t in tr], np.int8)
    o["comp_b"] = np.asarray([int(at(asp, t) == "completive") for t in tr], np.int8)
    o["nonong_b"] = np.asarray([int(at(asp, t) not in (None, "ongoing")) for t in tr], np.int8)
    o["open"] = np.asarray([1 if RE_OPEN.match(txt[t]) else 0 for t in tr], np.int8)
    return o


def candidates(d: dict) -> dict[str, np.ndarray | None]:
    z = np.zeros_like(d["et_change"])
    return {
        "C0 동결 (et≠ ∨ inceptive)": ((d["et_change"] | d["inc_b"]) > 0).astype(np.int8),
        "C1 et≠ 단독": d["et_change"],
        "C2 inceptive 단독": d["inc_b"],
        "C3 completive 단독": d["comp_b"],
        "C4 ≠ongoing 단독": d["nonong_b"],
        "C5 et≠ ∨ completive": ((d["et_change"] | d["comp_b"]) > 0).astype(np.int8),
        "C6 3군 접기": d["grp_change"],
        "C7 3군 ∨ completive": ((d["grp_change"] | d["comp_b"]) > 0).astype(np.int8),
        "C8 causal_marker(LLM)": d["cm"],
        "C9 문두 표지(정규식)": d["open"],
        "C10 인과 없음(4축)": z,
    }


def evaluate(name: str, per_doc: list[tuple[dict, np.ndarray]]) -> dict:
    ys, ps, npred, lifts, fires, aucs = [], [], 0, [], [], []
    per = {}
    for (d, cau), doc in zip(per_doc, DOCS):
        g = d["gold"]
        cnt = d["other4"].sum(1) + cau
        b = (cnt >= T_STAR).astype(np.int8)
        ys.append(np.concatenate([g, [0]]).tolist())
        ps.append(np.concatenate([b, [0]]).tolist())
        npred += int(b.sum())
        base = g.mean()
        fires.append(float(cau.mean()))
        lifts.append(float(g[cau == 1].mean() / base) if cau.sum() and base else float("nan"))
        aucs.append(float(roc_auc_score(g, cnt)) if 0 < g.sum() < len(g) else float("nan"))
        per[doc] = {"score": float(score_dialogues([ys[-1]], [ps[-1]])["score"]),
                    "n_pred": int(b.sum()), "n_gold": int(g.sum()),
                    "axis_fire": fires[-1], "axis_lift": lifts[-1]}
    return {"score": float(score_dialogues(ys, ps)["score"]), "n_pred": npred,
            "per_doc": per, "axis_fire": float(np.mean(fires)),
            "axis_lift": float(np.nanmean(lifts)), "auc": float(np.nanmean(aucs))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="*", default=list(REPS))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    runs: dict[str, list[dict]] = {}
    for rep in a.reps:
        D = [load(doc, rep) for doc in DOCS]
        cands = [candidates(d) for d in D]
        for name in cands[0]:
            runs.setdefault(name, []).append(
                evaluate(name, [(d, c[name]) for d, c in zip(D, cands)]))

    base_mu = float(np.mean([r["score"] for r in runs["C0 동결 (et≠ ∨ inceptive)"]]))
    print(f"기준선 C0 = {base_mu:.4f} · t*={T_STAR} 동결 · 반복 {len(a.reps)}\n")
    print(f"{'후보':26s} {'Score':>16s} {'ΔC0':>8s} {'경계':>5s} {'발화율':>7s} "
          f"{'lift':>5s} {'AUC':>6s}  문서별 Δ")
    res = {}
    for name, rs in runs.items():
        sc = np.array([r["score"] for r in rs])
        mu, sd = float(sc.mean()), float(sc.std())
        d = mu - base_mu
        pd_delta = []
        for doc in DOCS:
            v = np.mean([r["per_doc"][doc]["score"] for r in rs])
            b = np.mean([r["per_doc"][doc]["score"] for r in runs["C0 동결 (et≠ ∨ inceptive)"]])
            pd_delta.append(v - b)
        ok3 = all(x >= -1e-9 for x in pd_delta)
        res[name] = {"score_mean": mu, "score_sd": sd, "delta": d,
                     "per_doc_delta": pd_delta, "all3_no_drop": bool(ok3),
                     "n_pred": float(np.mean([r["n_pred"] for r in rs])),
                     "axis_fire": float(np.mean([r["axis_fire"] for r in rs])),
                     "axis_lift": float(np.nanmean([r["axis_lift"] for r in rs])),
                     "auc": float(np.mean([r["auc"] for r in rs])),
                     "per_doc": {doc: {k: float(np.mean([r["per_doc"][doc][k] for r in rs]))
                                       for k in ("score", "n_pred", "axis_fire", "axis_lift")}
                                 for doc in DOCS}}
        flag = ""
        if d >= 0.03 and ok3 and sd < abs(d) / 2:
            flag = "  ← ①②③ 충족"
        print(f"{name:26s} {mu:.4f}±{sd:.4f} {d:+8.4f} {res[name]['n_pred']:5.0f} "
              f"{res[name]['axis_fire']:7.4f} {res[name]['axis_lift']:5.2f} {res[name]['auc']:6.3f}"
              f"  {' '.join(f'{x:+.3f}' for x in pd_delta)}{flag}")

    # 사전등록 §4 채택 규칙
    c10 = res["C10 인과 없음(4축)"]["score_mean"]
    print(f"\n=== 채택 규칙 (①ΔScore≥+0.03 ②3편 무하락 ③SD<Δ/2 ④C10({c10:.4f}) 초과) ===")
    win = [n for n, v in res.items()
           if n != "C0 동결 (et≠ ∨ inceptive)" and v["delta"] >= 0.03 and v["all3_no_drop"]
           and v["score_sd"] < abs(v["delta"]) / 2 and v["score_mean"] > c10]
    print("  통과:", win if win else "**없음 — 동결 유지, 재추출하지 않는다**")
    (OUT / "bakeoff.json").write_text(json.dumps(
        {"base": base_mu, "c10": c10, "winners": win, "candidates": res},
        indent=2, ensure_ascii=False))
    print(f"\n-> {OUT/'bakeoff.json'}")


if __name__ == "__main__":
    main()

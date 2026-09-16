#!/usr/bin/env python3
r"""인과 축 **단독** 대결 — 역사적 시도 전부 포함, dev-3. API 0.

사전등록 `outputs/experiments/2026-07-31_causality_bakeoff/PREREG.md` §AMENDMENT.
각 조작화를 **그 축만으로 경계를 찍게** 하고 dev-3 deployable Score 를 낸다(사용자 지정 기준).
저장된 dev-3 라벨이 있는 역사적 시도(반사실 창·의도 사슬·인과 사슬)를 전부 포함한다.

    .venv/bin/python scripts/eacl_phase0/causality_axis_only_bakeoff.py
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
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
from causality_bakeoff import DOCS, GROUP, RE_OPEN, REPS  # noqa: E402
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from llm_s5_states import reset_chainrole  # noqa: E402

P0 = REPO / "outputs/eacl_phase0"
RUNS = REPO / "outputs/runs"
OUT = REPO / "outputs/experiments/2026-07-31_causality_bakeoff"
CF = REPO / "outputs/experiments/2026-07-25_causality_cf_window/cf_dev3.jsonl"


def base_labels(doc: str, rep: str) -> dict:
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

    o = {"tr": tr, "gold": np.asarray([r["boundary"] for r in rm], np.int8),
         "cm": np.asarray([r["causal_marker"] for r in rm], np.int8),
         "frozen": np.asarray([r["causality"] for r in rm], np.int8)}
    o["et_change"] = np.asarray([int(at(et, t - 1) != at(et, t)) for t in tr], np.int8)
    o["grp_change"] = np.asarray(
        [int(GROUP.get(at(et, t - 1), 3) != GROUP.get(at(et, t), 3)) for t in tr], np.int8)
    o["inc"] = np.asarray([int(at(asp, t) == "inceptive") for t in tr], np.int8)
    o["comp"] = np.asarray([int(at(asp, t) == "completive") for t in tr], np.int8)
    o["nonong"] = np.asarray([int(at(asp, t) not in (None, "ongoing")) for t in tr], np.int8)
    # D-B21 규칙: 직전 completive ∧ 현재 ∈ {inceptive, ongoing}
    o["asp_trans"] = np.asarray(
        [int(at(asp, t - 1) == "completive" and at(asp, t) in ("inceptive", "ongoing")) for t in tr],
        np.int8)
    o["open"] = np.asarray([1 if RE_OPEN.match(txt[t]) else 0 for t in tr], np.int8)
    return o


def hist_labels(doc: str, tr: list[int]) -> dict:
    """저장된 역사적 시도 라벨을 전이 인덱스에 정렬."""
    o: dict[str, np.ndarray] = {}
    cf = {int(r["transition"]): int(r["causality_cf"]) for r in
          (json.loads(l) for l in CF.read_text().splitlines() if l.strip())
          if r["doc_id"] == doc}
    o["cf_window"] = np.asarray([cf.get(t, 0) for t in tr], np.int8)

    def states(path: Path, key: str):
        rows = sorted((json.loads(l) for l in path.read_text().splitlines() if l.strip()),
                      key=lambda r: int(r["item_id"]))
        return [r.get(key) for r in rows]

    p = RUNS / "2026-07-28_chainrole_dev3" / doc / "chainrole_states.jsonl"
    if p.exists():
        rows = sorted((json.loads(l) for l in p.read_text().splitlines() if l.strip()),
                      key=lambda r: int(r["item_id"]))
        o["chainrole"] = np.asarray(
            [reset_chainrole(rows[t - 1], rows[t]) if 0 < t < len(rows) else 0 for t in tr], np.int8)
    q = RUNS / "2026-07-28_causalchain_diag" / doc / "causalchain_states.jsonl"
    if q.exists():
        ch = states(q, "causal_chain")
        o["chain_loose"] = np.asarray(
            [int(ch[t - 1] != ch[t]) if 0 < t < len(ch) else 0 for t in tr], np.int8)
        o["chain_strict"] = np.asarray(
            [int(ch[t - 1] != ch[t] and "none" not in (ch[t - 1], ch[t])) if 0 < t < len(ch) else 0
             for t in tr], np.int8)
    return o


CANDS = [
    ("F0 동결 (et≠ ∨ inceptive)", "frozen", "#1 동결 기준선"),
    ("F1 et≠ 단독", "et_change", "#1 프록시"),
    ("F2 inceptive 단독", "inc", "동결의 aspect 항"),
    ("F3 completive 단독", "comp", "문헌(🔴 오늘 열람)"),
    ("F4 ≠ongoing 단독", "nonong", "🔴 오늘 열람"),
    ("F5 3군 접기", "grp_change", "감사 C안"),
    ("F6 aspect 전이 (D-B21)", "asp_trans", "#6"),
    ("F7 causal_marker(LLM)", "cm", "#1b"),
    ("F8 문두 표지(정규식)", "open", "오늘 A-E"),
    ("F9 반사실 창 (D-B13③)", "cf_window", "#5"),
    ("F10 의도 사슬 (D-B22-A1)", "chainrole", "#7"),
    ("F11 인과 사슬 loose (D-B23)", "chain_loose", "#8"),
    ("F12 인과 사슬 strict (D-B23)", "chain_strict", "#8"),
    ("F99 무경계 하한", "_zero", "하한"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="*", default=list(REPS))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    per_rep: dict[str, list[dict]] = {}
    for rep in a.reps:
        L = []
        for doc in DOCS:
            b = base_labels(doc, rep)
            b.update(hist_labels(doc, b["tr"]))
            b["_zero"] = np.zeros_like(b["gold"])
            L.append(b)
        for name, key, _src in CANDS:
            if any(key not in b for b in L):
                continue
            ys, ps, fires, lifts, npred, per = [], [], [], [], 0, {}
            for b, doc in zip(L, DOCS):
                g, f = b["gold"], b[key]
                ys.append(np.concatenate([g, [0]]).tolist())
                ps.append(np.concatenate([f, [0]]).tolist())
                npred += int(f.sum())
                fires.append(float(f.mean()))
                lifts.append(float(g[f == 1].mean() / g.mean()) if f.sum() else float("nan"))
                per[doc] = {"score": float(score_dialogues([ys[-1]], [ps[-1]])["score"]),
                            "n_pred": int(f.sum()), "fire": fires[-1], "lift": lifts[-1]}
            per_rep.setdefault(name, []).append(
                {"score": float(score_dialogues(ys, ps)["score"]), "n_pred": npred,
                 "fire": float(np.mean(fires)), "lift": float(np.nanmean(lifts)), "per_doc": per})

    base = float(np.mean([r["score"] for r in per_rep["F0 동결 (et≠ ∨ inceptive)"]]))
    floor = float(np.mean([r["score"] for r in per_rep["F99 무경계 하한"]]))
    print(f"동결 축 단독 = {base:.4f} · 무경계 하한 = {floor:.4f} · gold 66 · 반복 {len(a.reps)}\n")
    print(f"{'후보':30s} {'Score':>16s} {'Δ동결':>8s} {'예측':>5s} {'발화율':>7s} {'lift':>5s}"
          f"  문서별 Score (pie/tun/mon)")
    res = {}
    for name, key, src in CANDS:
        rs = per_rep.get(name)
        if not rs:
            print(f"{name:30s} — dev-3 라벨 없음 ({src})")
            continue
        sc = np.array([r["score"] for r in rs])
        mu, sd = float(sc.mean()), float(sc.std())
        pdoc = {d: float(np.mean([r["per_doc"][d]["score"] for r in rs])) for d in DOCS}
        pdel = {d: pdoc[d] - float(np.mean([r["per_doc"][d]["score"]
                                            for r in per_rep["F0 동결 (et≠ ∨ inceptive)"]]))
                for d in DOCS}
        fire = float(np.mean([r["fire"] for r in rs]))
        ok = (mu - base >= 0.03 and all(v >= 0 for v in pdel.values()) and mu > floor
              and 0.022 <= fire <= 0.20)
        res[name] = {"source": src, "score": mu, "sd": sd, "delta": mu - base,
                     "n_pred": float(np.mean([r["n_pred"] for r in rs])), "fire": fire,
                     "lift": float(np.nanmean([r["lift"] for r in rs])),
                     "per_doc": pdoc, "per_doc_delta": pdel, "pass_all": bool(ok)}
        print(f"{name:30s} {mu:.4f}±{sd:.4f} {mu-base:+8.4f} "
              f"{res[name]['n_pred']:5.0f} {fire:7.4f} {res[name]['lift']:5.2f}"
              f"  {pdoc['pieman']:.3f}/{pdoc['tunnel']:.3f}/{pdoc['monkey']:.3f}"
              f"{'  ← 전 조건 충족' if ok else ''}")

    win = [n for n, v in res.items() if v["pass_all"] and n != "F0 동결 (et≠ ∨ inceptive)"]
    print(f"\n=== 채택 규칙 (①Δ≥+0.03 ②3편 무하락 ③하한 {floor:.4f} 초과 ④발화율 0.022~0.20) ===")
    print("  통과:", win if win else "**없음 — 동결 유지, 앵커 재추출 안 함**")
    (OUT / "axis_only_bakeoff.json").write_text(json.dumps(
        {"base": base, "floor": floor, "winners": win, "candidates": res},
        indent=2, ensure_ascii=False))
    print(f"\n-> {OUT/'axis_only_bakeoff.json'}")


if __name__ == "__main__":
    main()

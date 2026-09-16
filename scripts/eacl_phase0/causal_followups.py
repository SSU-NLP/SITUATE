#!/usr/bin/env python3
r"""인과 축 후속 4종 — 전부 기존 라벨 재계산, API 0.

(a) 문서 층화 RR + Mantel-Haenszel — pooled RR 2.939 가 문서 기저율(0.19/0.037/0.079) 교락인지.
(b) LLM `causal_marker` × 문두 정규식 혼동행렬 — RR 이 0.683 대 2.939 로 반대인데 겹치긴 하는가.
(C) `aspect == completive` 단독 — 문헌은 **완료상**이 경계 표지라는데 동결 코드는 `inceptive` 를 쓴다.
(A) **합의 등급 gold** — gold 를 합의도 상/하로 갈라 축별 lift. 우변(기준)을 바꾸는 첫 시도.
     참가자 자료: pieman 2런 · tunnel 1런(`human_annotator_baseline.py` 파서 재사용). monkey 없음.

읽기 전용. 동결 설정·체크포인트·시험지 미접근.

    .venv/bin/python scripts/eacl_phase0/causal_followups.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
P0 = REPO / "outputs/eacl_phase0"
DATA = REPO / "data/eacl_phase0"
OUT = REPO / "outputs/experiments/2026-07-31_causal_cohesion_veto"
DOCS = ("pieman", "tunnel", "monkey")
REP = "q36_35b_v31"
AXES = ("protagonist_present", "space", "time", "causality", "intentionality")
RE_OPEN = re.compile(r"^\W*(so|therefore|thus|hence|consequently|accordingly|as a result"
                     r"|because|since|as|then|for this reason)\b", re.I)


def rr_ci(k1, n1, k0, n0):
    if min(k1, k0) == 0 or min(n1, n0) == 0:
        return (float("nan"),) * 3
    p1, p0 = k1 / n1, k0 / n0
    rr = p1 / p0
    se = np.sqrt(1 / k1 - 1 / n1 + 1 / k0 - 1 / n0)
    return rr, float(rr * np.exp(-1.96 * se)), float(rr * np.exp(1.96 * se))


def load_doc(doc: str):
    d = P0 / f"{doc}_michelmann_s5_{REP}"
    rm = sorted((json.loads(l) for l in (d / "reset_matrix.jsonl").read_text().splitlines() if l.strip()),
                key=lambda r: r["transition"])
    st = [json.loads(l) for l in (d / doc / "causality_states.jsonl").read_text().splitlines() if l.strip()]
    st.sort(key=lambda r: int(re.search(r"(\d+)$", str(r["item_id"])).group(1)))
    master = sorted((json.loads(l) for l in (P0 / f"{doc}_michelmann_master.jsonl").read_text().splitlines()
                     if l.strip()), key=lambda r: r["transition"])
    txt = {int(r["transition"]): r["current"] for r in master}
    out = {"gold": np.asarray([r["boundary"] for r in rm], np.int8),
           "cm": np.asarray([r["causal_marker"] for r in rm], np.int8),
           "trans": [int(r["transition"]) for r in rm]}
    for a in AXES:
        out[a] = np.asarray([r[a] for r in rm], np.int8)
    out["open"] = np.asarray([1 if RE_OPEN.match(txt[t]) else 0 for t in out["trans"]], np.int8)
    # 전이 t = (문장 t-1 -> t). states 는 문장 인덱스 0..n-1.
    asp = [s.get("aspect") for s in st]
    out["asp_b"] = np.asarray([1 if (t < len(asp) and asp[t] == "completive") else 0
                               for t in out["trans"]], np.int8)
    out["asp_a"] = np.asarray([1 if (t - 1 < len(asp) and asp[t - 1] == "completive") else 0
                               for t in out["trans"]], np.int8)
    out["asp_inc_b"] = np.asarray([1 if (t < len(asp) and asp[t] == "inceptive") else 0
                                   for t in out["trans"]], np.int8)
    return out


def line(tag, gold, flag):
    n1, n0 = int(flag.sum()), int((1 - flag).sum())
    k1, k0 = int(gold[flag == 1].sum()), int(gold[flag == 0].sum())
    rr, lo, hi = rr_ci(k1, n1, k0, n0)
    base = gold.mean()
    lift = (k1 / n1 / base) if n1 and base else float("nan")
    print(f"    {tag:26s} 발화율 {flag.mean():.4f} · P(경계) {k1/max(n1,1):.4f} ({k1}/{n1}) · "
          f"lift {lift:5.2f} · RR {rr:5.3f} [{lo:.3f},{hi:.3f}]")
    return {"fire": float(flag.mean()), "n": n1, "k": k1, "p": k1 / max(n1, 1),
            "lift": float(lift), "rr": rr, "ci": [lo, hi]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-consensus", action="store_true")
    a = ap.parse_args()
    D = {d: load_doc(d) for d in DOCS}
    res: dict = {}

    # ── (a) 문서 층화 + Mantel-Haenszel ──────────────────────────────────────
    print("=== (a) 문서 층화 RR + Mantel-Haenszel ===")
    for key, label in (("open", "문두 표지"), ("cm", "LLM 표지")):
        print(f"  [{label}]")
        num = den = 0.0
        st = {}
        for d in DOCS:
            g, f = D[d]["gold"], D[d][key]
            n1, n0 = int(f.sum()), int((1 - f).sum())
            k1, k0 = int(g[f == 1].sum()), int(g[f == 0].sum())
            n = n1 + n0
            st[d] = line(f"{d} (기저 {g.mean():.3f})", g, f)
            if n1 and n0:
                num += k1 * n0 / n
                den += k0 * n1 / n
        mh = num / den if den else float("nan")
        print(f"    -> Mantel-Haenszel RR = {mh:.3f}")
        # pieman 제외 재계산
        gg = np.concatenate([D[d]["gold"] for d in ("tunnel", "monkey")])
        ff = np.concatenate([D[d][key] for d in ("tunnel", "monkey")])
        sub = line("pieman 제외 pooled", gg, ff)
        res[f"a_{key}"] = {"per_doc": st, "mh_rr": float(mh), "no_pieman": sub}

    # ── (b) 혼동행렬 ─────────────────────────────────────────────────────────
    print("\n=== (b) LLM causal_marker × 문두 정규식 혼동행렬 (pooled) ===")
    cm = np.concatenate([D[d]["cm"] for d in DOCS])
    op = np.concatenate([D[d]["open"] for d in DOCS])
    tab = [[int(((cm == i) & (op == j)).sum()) for j in (0, 1)] for i in (0, 1)]
    print(f"            문두=0   문두=1")
    print(f"  LLM=0   {tab[0][0]:7d} {tab[0][1]:7d}")
    print(f"  LLM=1   {tab[1][0]:7d} {tab[1][1]:7d}")
    inter, union = tab[1][1], tab[0][1] + tab[1][0] + tab[1][1]
    phi = float(np.corrcoef(cm, op)[0, 1])
    print(f"  겹침 {inter} · Jaccard {inter/max(union,1):.4f} · φ {phi:.4f}")
    res["b_confusion"] = {"table": tab, "jaccard": inter / max(union, 1), "phi": phi}

    # ── (C) aspect == completive 단독 ────────────────────────────────────────
    print("\n=== (C) aspect 항 부호 검정 (pooled) ===")
    g = np.concatenate([D[d]["gold"] for d in DOCS])
    res["C"] = {}
    for key, label in (("asp_b", "b.aspect=completive"), ("asp_a", "a.aspect=completive"),
                       ("asp_inc_b", "b.aspect=inceptive (동결 항)"),
                       ("causality", "동결 causality 축 전체")):
        f = np.concatenate([D[d][key] for d in DOCS])
        res["C"][key] = line(label, g, f)
    print("  문서별 b.aspect=completive")
    for d in DOCS:
        res["C"][f"asp_b_{d}"] = line(f"  {d}", D[d]["gold"], D[d]["asp_b"])

    # ── (A) 합의 등급 gold ───────────────────────────────────────────────────
    if not a.skip_consensus:
        print("\n=== (A) 합의 등급 gold (pieman·tunnel; monkey 참가자 자료 없음) ===")
        import scipy.io as sio
        from human_annotator_baseline import SUBJ_SRC, build_doc, presses_ms  # noqa: E402
        res["A"] = {}
        for doc in ("pieman", "tunnel"):
            sentences, onsets, ms_to_trans = build_doc(doc)
            n_tr = len(D[doc]["gold"])
            cnt = np.zeros(n_tr + 2, np.int32)
            n_sub = 0
            for _, fn, hz in SUBJ_SRC[doc]:
                m = sio.loadmat(str(DATA / f"{doc}_michelmann/sourcedata" / fn))
                arr = m[[k for k in m if not k.startswith("__")][0]]
                for row in arr:
                    tr = ms_to_trans(presses_ms(np.asarray(row), hz))
                    n_sub += 1
                    for t in set(tr):
                        if 1 <= t <= n_tr:
                            cnt[t] += 1
            agree = cnt[1:n_tr + 1] / max(n_sub, 1)
            gold = D[doc]["gold"]
            ga = agree[gold == 1]
            med = float(np.median(ga))
            hi = ((gold == 1) & (agree > med)).astype(np.int8)
            lo = ((gold == 1) & (agree <= med)).astype(np.int8)
            print(f"  [{doc}] 참가자 {n_sub} · gold {int(gold.sum())} · 합의도 "
                  f"min {ga.min():.3f} 중앙 {med:.3f} max {ga.max():.3f}")
            res["A"][doc] = {"n_subjects": n_sub, "median_agreement": med,
                             "agreement_of_gold": [float(x) for x in ga], "strata": {}}
            for name, sel in (("상위 합의", hi), ("하위 합의", lo)):
                keep = (sel == 1) | (gold == 0)          # 반대 층은 분석에서 제외
                gs = sel[keep]
                print(f"   -- {name} (n={int(sel.sum())}, 기저 {gs.mean():.4f})")
                for ax in AXES + ("open", "cm", "asp_b"):
                    res["A"][doc]["strata"].setdefault(name, {})[ax] = line(
                        ax, gs, D[doc][ax][keep])

    (OUT / "followups.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float))
    print(f"\n-> {OUT/'followups.json'}")


if __name__ == "__main__":
    main()

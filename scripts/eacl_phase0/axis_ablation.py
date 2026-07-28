#!/usr/bin/env python3
"""S5 축 ablation (deployable): 쓸데없는 축을 빼면 나아지나.

coordinated count = Σ_{k∈subset} reset_k, deployable 규칙 count ≥ t* (t* dev 동결).
축 subset 선택도 dev(Pieman)에서. full5 / leave-one-out(4축) / leave-two-out(3축).
S5 reset_matrix 만 사용(임베딩·LLM 불필요).
"""
from __future__ import annotations
import json
from itertools import combinations
from pathlib import Path

import sys
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402

AX = ["protagonist_perspective", "space", "time", "causality", "intentionality"]
SHORT = {"protagonist_perspective": "persp", "space": "space", "time": "time",
         "causality": "caus", "intentionality": "intent"}
RM = {
    "pieman": "outputs/eacl_phase0/pieman_michelmann_s5/reset_matrix.jsonl",
    "tunnel": "outputs/eacl_phase0/tunnel_michelmann_s5/reset_matrix.jsonl",
    "monkey": "outputs/eacl_phase0/monkey_michelmann_s5/reset_matrix.jsonl",
    "tiage": "outputs/eacl_phase0/dts_tiage_s5/reset_matrix.jsonl",
    "dialseg711": "outputs/eacl_phase0/dts_dialseg711_s5/reset_matrix.jsonl",
    "superseg": "outputs/eacl_phase0/dts_superseg_s5/reset_matrix.jsonl",
    "ami": "outputs/eacl_phase0/dts_ami_s5/reset_matrix.jsonl",
}


def read(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def docs_of(name):
    by = {}
    for r in read(RM[name]):
        by.setdefault(r["doc_id"], []).append(r)
    out = []
    for doc, rs in by.items():
        rs = sorted(rs, key=lambda r: int(r["transition"]))
        n = len(rs) + 1
        gold = [i + 1 for i, r in enumerate(rs) if int(r["boundary"])]
        if gold:
            out.append((n, gold, rs))
    return out


def endturn(n, idxs):
    y = [0] * n
    for i in idxs:
        y[i - 1] = 1
    return y


def score_subset(docs, subset, t):
    golds, preds = [], []
    for n, gold, rs in docs:
        cnt = [sum(int(r[a]) for a in subset) for r in rs]
        pred = [i + 1 for i, c in enumerate(cnt) if c >= t]
        golds.append(endturn(n, gold)); preds.append(endturn(n, pred))
    return score_dialogues(golds, preds)["score"]


def best_t(docs, subset):
    return max(range(1, len(subset) + 1), key=lambda t: score_subset(docs, subset, t))


def main():
    data = {name: docs_of(name) for name in RM}
    dev = data["pieman"]
    names = list(RM)

    def row(subset, label):
        t = best_t(dev, subset)  # t* frozen on dev
        scores = {nm: score_subset(data[nm], subset, t) for nm in names}
        return label, t, scores

    subsets = [(AX, "full-5")]
    for drop in AX:
        subsets.append(([a for a in AX if a != drop], f"−{SHORT[drop]}"))
    # leave-two-out: best by dev
    l2 = []
    for d1, d2 in combinations(AX, 2):
        sub = [a for a in AX if a not in (d1, d2)]
        l2.append((sub, f"−{SHORT[d1]},{SHORT[d2]}"))

    rows = [row(s, lab) for s, lab in subsets]
    # pick best leave-two-out by dev score
    l2rows = [row(s, lab) for s, lab in l2]
    l2rows.sort(key=lambda r: -r[2]["pieman"])
    rows += l2rows[:3]

    hdr = f"{'subset':16s} {'t*':>2s} " + " ".join(f"{nm[:9]:>9s}" for nm in names)
    print(hdr)
    for lab, t, sc in rows:
        print(f"{lab:16s} {t:>2d} " + " ".join(f"{sc[nm]:9.3f}" for nm in names))
    print("\n(dev=pieman에서 subset·t* 선택 → test 적용. 각 행 t*는 그 subset의 dev-argmax.)")


if __name__ == "__main__":
    main()

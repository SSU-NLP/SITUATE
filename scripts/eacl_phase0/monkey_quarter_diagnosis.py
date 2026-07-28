"""Monkey 4구간 진단 (ledger D-M5-d).

Monkey는 참가자를 4그룹으로 나눠 각 그룹이 이야기의 1/4씩만 주석했다
(30/35/31/42명). 따라서 "저널리즘이라 상황모델이 정지 → 우리 방법 실패"라는
scope 해석에 **"주석 프로토콜 교란"이라는 대안 설명**이 존재한다.

이 스크립트는 Monkey를 위치 기준 4구간(=주석 그룹 근사)으로 나눠 구간별로
ours(coordinated 5-axis count) vs IDM(raw cosine / tuned)의 신호질을
matched@gold Score + AUC + 축별 lift로 측정한다.

판정 규약 (실행 전 고정, ledger D-M5-d):
  - 구간별 성능이 **들쭉날쭉** → 주석 품질 교란 → scope 해석 재작성 필요.
  - **균일하게 나쁨**       → 장르 요인 → scope 해석 지지.

⚠️ matched@gold는 oracle 경계 개수를 사용하는 **ranking 진단**이다(deployable 아님).
   구간별 K가 작아 free-threshold는 노이즈가 크므로, 여기서는 "신호가 경계를
   순위화하는가"만 본다 (CLAUDE.md가 허용한 matched@gold의 유일 용도).

usage: uv run python scripts/eacl_phase0/monkey_quarter_diagnosis.py
"""
from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from hi_ontop.embedding import make_encoder  # noqa: E402

AX = ["protagonist_perspective", "place", "time", "causality", "goal"]
RM_PATH = "outputs/eacl_phase0/monkey_michelmann_s5/reset_matrix.jsonl"
MASTER_A = "outputs/eacl_phase0/monkey_michelmann_master.jsonl"       # gold A (author-filtered, 30)
MASTER_B = "outputs/eacl_phase0/monkey_michelmann_raw_master.jsonl"   # gold B (raw+5s, 40)
ENCODER = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
N_QUARTERS = 4


def load_rows() -> list[dict]:
    rm = [json.loads(l) for l in open(RM_PATH)]
    rm.sort(key=lambda r: r["transition"])
    goldA = {json.loads(l)["transition"]: int(json.loads(l)["boundary"])
             for l in open(MASTER_A)}
    goldB = {json.loads(l)["transition"]: int(json.loads(l)["boundary"])
             for l in open(MASTER_B)}
    master = {json.loads(l)["transition"]: json.loads(l) for l in open(MASTER_A)}
    for r in rm:
        t = r["transition"]
        r["goldA"] = goldA.get(t, 0)
        r["goldB"] = goldB.get(t, 0)
        r["count"] = sum(int(r[a]) for a in AX)
        r["previous"] = master[t]["previous"]
        r["current"] = master[t]["current"]
    return rm


def raw_cosine(rows: list[dict]) -> dict[int, float]:
    enc = make_encoder(model=ENCODER)
    seq = [rows[0]["previous"]] + [r["current"] for r in rows]
    E = np.asarray(enc.encode(seq))
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    return {rows[k]["transition"]: float(1 - np.dot(En[k + 1], En[k]))
            for k in range(len(rows))}


def matched_score(gold_bin: list[int], signal: list[float]) -> float:
    """matched@gold: 예측 개수 = gold 경계 수, top-K 위치를 경계로."""
    n = len(gold_bin) + 1
    K = sum(gold_bin)
    if K == 0:
        return float("nan")
    m = len(signal)
    # 위치(local transition 1..m). 동점은 뒤쪽(-i) 우선(monkey_score_check와 동일).
    order = sorted(range(m), key=lambda i: (signal[i], -i), reverse=True)[:K]

    def endturn(idxs: list[int]) -> list[int]:
        y = [0] * n
        for i in idxs:
            y[i] = 1  # local index i (0-based transition) → 경계 turn
        return y

    gold_idx = [i for i, b in enumerate(gold_bin) if b]
    return score_dialogues([endturn(gold_idx)], [endturn(order)])["score"]


def safe_auc(labels: list[int], signal: list[float]) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, signal))


def rate(labels: list[int], flags: list[int]) -> tuple[float, float]:
    """(경계율, 내부율) = 축 reset이 경계/내부에서 켜지는 비율."""
    b = [f for f, y in zip(flags, labels) if y]
    i = [f for f, y in zip(flags, labels) if not y]
    return (float(np.mean(b)) if b else float("nan"),
            float(np.mean(i)) if i else float("nan"))


def eval_slice(rows: list[dict], idm_raw: dict[int, float], gold_key: str) -> dict:
    labels = [r[gold_key] for r in rows]
    count = [float(r["count"]) for r in rows]
    idmr = [idm_raw[r["transition"]] for r in rows]
    idmt = [float(r["idm_score"]) for r in rows]
    out = {
        "n": len(rows),
        "K": int(sum(labels)),
        "score_ours": matched_score(labels, count),
        "score_idm_raw": matched_score(labels, idmr),
        "score_idm_tuned": matched_score(labels, idmt),
        "auc_ours": safe_auc(labels, count),
        "auc_idm_raw": safe_auc(labels, idmr),
        "auc_idm_tuned": safe_auc(labels, idmt),
        "axes": {},
    }
    for a in AX:
        flags = [int(r[a]) for r in rows]
        br, ir = rate(labels, flags)
        out["axes"][a] = {
            "auc": safe_auc(labels, [float(f) for f in flags]),
            "total": int(sum(flags)),
            "b_rate": br,
            "i_rate": ir,
        }
    return out


def fmt(x: float) -> str:
    return "  nan" if (x != x) else f"{x:.3f}"


def main() -> None:
    rows = load_rows()
    idm_raw = raw_cosine(rows)
    transitions = [r["transition"] for r in rows]
    lo, hi = min(transitions), max(transitions)
    span = (hi - lo + 1) / N_QUARTERS

    quarters: list[list[dict]] = [[] for _ in range(N_QUARTERS)]
    for r in rows:
        q = min(N_QUARTERS - 1, int((r["transition"] - lo) / span))
        quarters[q].append(r)

    for gold_key, gname in [("goldA", "gold A (author-filtered, K=30)"),
                            ("goldB", "gold B (raw+5s, K=40)")]:
        print(f"\n{'=' * 78}\nMONKEY 4-QUARTER DIAGNOSIS — {gname}\n{'=' * 78}")
        full = eval_slice(rows, idm_raw, gold_key)
        print("\n[전체 doc 기준선]")
        print(f"  n={full['n']} K={full['K']}  "
              f"matched@gold Score: ours {fmt(full['score_ours'])} | "
              f"IDM_raw {fmt(full['score_idm_raw'])} | IDM_tuned {fmt(full['score_idm_tuned'])}")
        print(f"  {'':22s}AUC: ours {fmt(full['auc_ours'])} | "
              f"IDM_raw {fmt(full['auc_idm_raw'])} | IDM_tuned {fmt(full['auc_idm_tuned'])}")

        print("\n[구간별]  (Score=matched@gold, ranking 진단)")
        hdr = (f"  {'quarter':8s} {'n':>4s} {'K':>3s} | "
               f"{'ours':>6s} {'IDMraw':>7s} {'IDMtun':>7s} (Score) | "
               f"{'ours':>6s} {'IDMraw':>7s} {'IDMtun':>7s} (AUC)")
        print(hdr)
        qres = []
        for qi, q in enumerate(quarters, 1):
            res = eval_slice(q, idm_raw, gold_key)
            qres.append(res)
            t0, t1 = q[0]["transition"], q[-1]["transition"]
            print(f"  Q{qi} {t0:3d}-{t1:<3d} {res['n']:>4d} {res['K']:>3d} | "
                  f"{fmt(res['score_ours']):>6s} {fmt(res['score_idm_raw']):>7s} "
                  f"{fmt(res['score_idm_tuned']):>7s}        | "
                  f"{fmt(res['auc_ours']):>6s} {fmt(res['auc_idm_raw']):>7s} "
                  f"{fmt(res['auc_idm_tuned']):>7s}")

        print("\n[구간별 축 AUC]  (place=scope 판별자)")
        print(f"  {'quarter':8s} " + " ".join(f"{a[:9]:>9s}" for a in AX))
        for qi, res in enumerate(qres, 1):
            print(f"  Q{qi:<7d} " +
                  " ".join(f"{fmt(res['axes'][a]['auc']):>9s}" for a in AX))

        print("\n[구간별 축 reset 총계 / (경계율,내부율)]")
        for qi, res in enumerate(qres, 1):
            parts = []
            for a in AX:
                ax = res["axes"][a]
                parts.append(f"{a[:4]}={ax['total']:>2d}({fmt(ax['b_rate'])},{fmt(ax['i_rate'])})")
            print(f"  Q{qi}: " + "  ".join(parts))

        # 균일성 진단: 구간별 ours AUC의 변동
        ours_aucs = [r["auc_ours"] for r in qres if r["auc_ours"] == r["auc_ours"]]
        if ours_aucs:
            print(f"\n[균일성] 구간별 ours AUC: {[round(a,3) for a in ours_aucs]}  "
                  f"min={min(ours_aucs):.3f} max={max(ours_aucs):.3f} "
                  f"range={max(ours_aucs)-min(ours_aucs):.3f}")


if __name__ == "__main__":
    main()

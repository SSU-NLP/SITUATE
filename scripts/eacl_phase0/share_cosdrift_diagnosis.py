"""SHARE M1-CosDrift 진단 (학습·LLM 무관, 층위1/concat-easy 계측).

명명: baseline = **M1-CosDrift**(코사인 드리프트, `method-registry.md` §0). "IDM"이라
부르지 않는다. 여기서는 **CosDrift-raw**(a=1, 순수 1−cos(s_{t-1},s_t)) 사용.

메인 지표 = **DTS Score**(`dts_scoring.score_dialogues`). AUC·margin은 **보조**(층위1
concat-easy 계측용 ranking 진단).

질문: 대사 스트림에서 씬(세션) 전환을 M1-CosDrift가 얼마나 잡나?
- Score/AUC 높음(≈dialseg711·WP concat) → 경계가 concat-easy(큰 화제점프) = 층위1 현실화.
- 낮음(픽션-B 영역) → 임베딩만으로 안 잡힘 = M3(S5) 돌릴 가치.

참조 AUC: WP(A-concat) 0.914 · dialseg711(concat) 0.841 · tiage(대화) 0.786
         · Tunnel(픽션B) 0.553 · Pieman 0.629.

usage: uv run python scripts/eacl_phase0/share_cosdrift_diagnosis.py --split valid
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from hi_ontop.embedding import make_encoder  # noqa: E402

ENCODER = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
C_FROZEN = 1.0  # M1-CosDrift deployable 규칙 μ+cσ, c=1.0 (dev=Pieman 동결, D-M3)


def endturn(n: int, boundary_transitions: list[int]) -> list[int]:
    y = [0] * n
    for t in boundary_transitions:
        y[t - 1] = 1
    return y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid")
    ap.add_argument("--master", type=Path, default=None)
    args = ap.parse_args()
    master = args.master or REPO / f"outputs/eacl_phase0/share_{args.split}_master.jsonl"

    rows = [json.loads(l) for l in open(master)]
    docs: dict[str, list[dict]] = {}
    for r in rows:
        docs.setdefault(r["doc_id"], []).append(r)
    for d in docs.values():
        d.sort(key=lambda r: r["transition"])

    # encode all unique turn texts once
    texts: list[str] = []
    idx: dict[str, int] = {}
    def add(t: str) -> int:
        if t not in idx:
            idx[t] = len(texts); texts.append(t)
        return idx[t]
    for d in docs.values():
        add(d[0]["previous"])
        for r in d:
            add(r["current"])
    enc = make_encoder(model=ENCODER)
    E = np.asarray(enc.encode(texts))
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    all_lbl: list[int] = []
    all_score: list[float] = []
    per_doc_auc: list[float] = []
    dep_scores: list[float] = []       # deployable Score per doc (main)
    matched_scores: list[float] = []   # matched@gold Score per doc (aux ranking)
    b_dist: list[float] = []
    i_dist: list[float] = []
    for d in docs.values():
        lbl = [r["boundary"] for r in d]
        tx = [r["transition"] for r in d]
        sc = [1 - float(np.dot(E[idx[r["current"]]], E[idx[r["previous"]]])) for r in d]
        n = len(d) + 1
        gold_t = [t for t, y in zip(tx, lbl) if y]
        K = len(gold_t)
        for r, s in zip(d, sc):
            (b_dist if r["boundary"] else i_dist).append(s)
        all_lbl += lbl; all_score += sc
        if len(set(lbl)) == 2:
            per_doc_auc.append(float(roc_auc_score(lbl, sc)))
        # deployable: μ+cσ, c=1.0 (경계 개수 자가결정)
        arr = np.array(sc)
        thr = arr.mean() + C_FROZEN * arr.std()
        pred_dep = [t for t, s in zip(tx, sc) if s >= thr]
        gold_seq = endturn(n, gold_t)
        dep_scores.append(score_dialogues([gold_seq], [endturn(n, pred_dep)])["score"])
        # matched@gold: top-K (aux, oracle 개수)
        if K:
            top = [tx[i] for i in sorted(range(len(sc)), key=lambda i: (sc[i], -tx[i]),
                                         reverse=True)[:K]]
            matched_scores.append(score_dialogues([gold_seq], [endturn(n, top)])["score"])

    pda = np.array(per_doc_auc)
    print(f"\n=== SHARE {args.split} — M1-CosDrift 진단 (baseline; 메인=Score) ===")
    print(f"docs {len(docs)} | transitions {len(all_lbl)} | boundaries {sum(all_lbl)} "
          f"({100*sum(all_lbl)/len(all_lbl):.1f}%)")

    print(f"\n[메인] DTS Score (per-doc 평균)")
    print(f"  deployable (μ+cσ, c=1.0, 개수 자가결정): {np.mean(dep_scores):.3f} "
          f"± {np.std(dep_scores):.3f}")
    print(f"  matched@gold (보조·oracle 개수 ranking): {np.mean(matched_scores):.3f} "
          f"± {np.std(matched_scores):.3f}")

    print(f"\n[보조] 순위질·concat-easy 계측")
    print(f"  pooled AUC {float(roc_auc_score(all_lbl, all_score)):.3f} | "
          f"per-doc AUC mean {pda.mean():.3f} ± {pda.std():.3f} (median {np.median(pda):.3f})")
    print(f"  경계 vs 내부 코사인거리 margin {np.mean(b_dist)-np.mean(i_dist):+.3f} "
          f"(경계 {np.mean(b_dist):.3f} / 내부 {np.mean(i_dist):.3f}; 참조 mouthguard 0.04)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

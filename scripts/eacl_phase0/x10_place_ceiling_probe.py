#!/usr/bin/env python3
"""X1-0-lite — 정보 상한 게이트: 대사 임베딩이 place/time 신호를 나르는가? (plan Phase X, X1-0)

질문. SHARE 슬러그라인 place/time gold는 각본가 저작 = 최고 품질 라벨이다. 그러나 D-M7은
대사-only 모달리티가 그 축을 복원 못 함을 확정했다(M3 recall place 4% / time 0%, 장소 명시 6.1%).
그렇다면 그 gold는 **학습 가능한 타깃인가, 원리적으로 예측 불가능한 타깃인가?**

방법. frozen mpnet Δ특징 [z_{t-1}, z_t, z_t-z_{t-1}, z_t*z_{t-1}] (3072D)에서 라벨을 직접
로지스틱 회귀로 예측 → **doc 단위 leave-one-doc-out CV** → out-of-fold AUPRC. 모델 학습 없이
"입력에 신호가 있나"만 측정한다(상한: 슬롯 모델은 이보다 잘할 수 없다 — 같은 입력의 저차원 요약).

⭐ 양성 대조군. 동일 프로브를 **fiction place/time**(M3 silver)에도 돌린다. fiction에서 작동하고
SHARE에서 안 되면 원인은 프로브가 아니라 **모달리티**임이 분리된다. 이건 M3-독립적 확증이라
D-M7(추출기 기반)보다 강한 주장이다.

GATE. SHARE AUPRC >> base rate -> 대사가 신호를 나름 -> 3조건 ablation 진행 가치.
      SHARE AUPRC ~= base rate -> 44개 양성은 학습 불가 -> ablation 결과 사전 확정.

usage: uv run python scripts/eacl_phase0/x10_place_ceiling_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_share_turn_scene import read_jsonl, turns_of  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

BASE = REPO / "outputs/eacl_phase0"
SLOT = REPO / "outputs/experiments/m5_phase2_slotalign"
ALIGN = BASE / "share_turn_scene_align.jsonl"
OUT = REPO / "outputs/experiments/m5_phase2_corpus_ablation"


def delta_feats(E: np.ndarray) -> np.ndarray:
    """[T, d] 문장 임베딩 -> [T-1, 4d] 전이 특징. probe_slot_alignment와 동일 관용구."""
    a, b = E[:-1], E[1:]
    return np.concatenate([a, b, b - a, b * a], axis=1)


def loo_auprc(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """doc 단위 leave-one-doc-out, out-of-fold 예측을 모아 전체 AUPRC."""
    oof = np.zeros(len(y), dtype=float)
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if y[tr].sum() < 3 or len(np.unique(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    if y.sum() < 1:
        return float("nan")
    return float(average_precision_score(y, oof))


def share_data(enc):
    """SHARE 874 유효 전이: mpnet Δ특징 + 슬러그라인 gold place/time."""
    align = read_jsonl(ALIGN)
    by_doc: dict[str, dict[int, dict]] = {}
    for r in align:
        by_doc.setdefault(r["doc_id"], {})[r["transition"]] = r
    master = read_jsonl(BASE / "share_valid_probe30_master.jsonl")
    m_by_doc: dict[str, list[dict]] = {}
    for r in master:
        m_by_doc.setdefault(r["doc_id"], []).append(r)

    X, yp, yt, g = [], [], [], []
    for gi, did in enumerate(sorted(by_doc)):
        texts, _, rows = turns_of(m_by_doc[did])
        E = np.asarray(enc.encode(texts, normalize_embeddings=True, batch_size=128), dtype=np.float32)
        D = delta_feats(E)
        for r in rows:
            t = int(r["transition"])
            a = by_doc[did].get(t)
            if a is None or not a["place_mask"]:
                continue
            X.append(D[t - 1]); yp.append(a["gold_place"]); yt.append(a["gold_time"]); g.append(gi)
    return np.array(X), np.array(yp), np.array(yt), np.array(g)


def fiction_data():
    """양성 대조군: 기존 slot_dataset.npz(fiction 10 docs)의 place/time M3 silver."""
    d = np.load(SLOT / "slot_dataset.npz", allow_pickle=True)
    emb, doc_off = d["emb"], d["doc_offsets"]
    rst, r_off, doc_ids = d["resets"], d["reset_offsets"], d["doc_ids"]
    X, yp, yt, g = [], [], [], []
    for i in range(len(doc_off) - 1):
        E = emb[doc_off[i]:doc_off[i + 1]]
        R = rst[r_off[i]:r_off[i + 1]]
        D = delta_feats(E)
        X.append(D); yp.append(R[:, 1]); yt.append(R[:, 2])
        g.append(np.full(len(R), i))
    return (np.concatenate(X), np.concatenate(yp), np.concatenate(yt), np.concatenate(g))


def main() -> int:
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                              device="cuda")
    res = {}
    print("SHARE (대사-only, 슬러그라인 gold) 인코딩...", flush=True)
    Xs, yps, yts, gs = share_data(enc)
    print("fiction (서술, M3 silver) 로드...", flush=True)
    Xf, ypf, ytf, gf = fiction_data()

    rows = []
    for name, (X, y, g) in [
        ("SHARE place (슬러그라인 gold)", (Xs, yps, gs)),
        ("SHARE time  (슬러그라인 gold)", (Xs, yts, gs)),
        ("fiction place (M3 silver)", (Xf, ypf, gf)),
        ("fiction time  (M3 silver)", (Xf, ytf, gf)),
    ]:
        base = float(y.mean())
        ap = loo_auprc(X, y, g)
        lift = ap / base if base > 0 else float("nan")
        rows.append((name, len(y), int(y.sum()), base, ap, lift))
        print(f"{name:32s} n={len(y):5d} pos={int(y.sum()):4d} base={base:.3f} "
              f"AUPRC={ap:.3f}  lift={lift:.2f}x", flush=True)
        res[name] = {"n": len(y), "pos": int(y.sum()), "base": base, "auprc": ap, "lift": lift}

    print("\n=== X1-0 GATE ===")
    sp = res["SHARE place (슬러그라인 gold)"]
    st = res["SHARE time  (슬러그라인 gold)"]
    fp = res["fiction place (M3 silver)"]
    for label, r in [("SHARE place", sp), ("SHARE time ", st)]:
        verdict = "PASS (신호 있음)" if r["auprc"] > 0.15 and r["lift"] > 1.5 else "FAIL (신호 없음)"
        print(f"  {label}: AUPRC {r['auprc']:.3f} vs base {r['base']:.3f} ({r['lift']:.2f}x) -> {verdict}")
    print(f"  [대조] fiction place: AUPRC {fp['auprc']:.3f} vs base {fp['base']:.3f} "
          f"({fp['lift']:.2f}x)")
    print("\n해석: fiction에서 lift가 크고 SHARE에서 ~1x면, place/time 정보 부재는 "
          "추출기(M3) 탓이 아니라 **모달리티(서술 채널 부재)** 탓 — M3-독립적 확증.")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "x10_gate.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\n-> {OUT/'x10_gate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

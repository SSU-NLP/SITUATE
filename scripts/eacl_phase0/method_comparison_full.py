#!/usr/bin/env python3
"""M1 vs M4 vs M5 대조표 — **같은 인코더 · 같은 임계 규칙 · SuperDialseg 공식 test 전체**.

## 왜
D-B24 에서 두 가지 비교 불가 사유가 드러났다: ① DTS 부분집합(대화당 15전이 절단) ② M1 은
공식 실험(`multi-qa-mpnet`)과 본 파이프라인(`paraphrase-multilingual-mpnet`)의 인코더가 다름.
①은 공식 test 전체 확보로 해소했다. **②를 해소하려면 M1 도 본 파이프라인 임베딩으로 돌려야 한다.**

## 무엇을
동일 조건 3요소를 강제한다:
1. **같은 임베딩** — `paraphrase-multilingual-mpnet` frozen (M1~M7 공통, D-M11 공정비교 조건)
2. **같은 데이터** — SuperDialseg 공식 test 전체 + dev-3(서사)
3. **같은 임계 규칙** — 고정 percentile · otsu · offline/online μ+cσ 를 **전 방법에 동일 적용**

| 방법 | 신호 |
|---|---|
| **M1-CosDrift** | `δ_eff = a·(1−cos(s_{t-1},s_t)) + (1−a)·(1−cos(c_{t-1},s_t))`, m=2 ρ=0.7 a=0.5 |
| **M4-MonoAnchor** | 학습된 단일슬롯 모델의 `Σ_k P(reset_k)` |
| **M5-FactoredSlot** | 학습된 5슬롯 모델의 `Σ_k P(reset_k)` |

🔴 M4/M5 는 **서사 픽션으로만 학습**됐다(D-M8) — DTS 는 도메인 밖이다.
🔴 dev-3 = 연습장. c 를 여기서 고르면 확증 아님. 탐색적 대조표다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from hi_ontop.factored_slot import FactoredSlot  # noqa: E402
from hi_ontop.hi_ontop_v2 import adaptive_boundaries  # noqa: E402
from train_anchor_m4m5 import EXP, Corpus, axis_probs  # noqa: E402

SETS = ("dev3", "dts_full_tiage", "dts_full_dialseg711", "dts_full_superseg")
HDR = ["dev-3(서사)", "tiage(100)", "dialseg711(711)", "superseg(1322)"]
ANCHOR_RATE = 0.0649
# M1 canonical HP (registry §2)
M, RHO, A, DELTA_STAR = 2, 0.7, 0.5, 0.5594


def cosdrift(emb: np.ndarray) -> np.ndarray:
    """δ_eff 시퀀스 [n_sent-1]. emb 는 L2 정규화된 문장 임베딩 [n_sent, d]."""
    n = emb.shape[0]
    out = np.zeros(n - 1)
    for t in range(1, n):
        d_prev = 1.0 - float(emb[t - 1] @ emb[t])
        lo = max(0, t - M)
        w = np.array([RHO ** i for i in range(t - lo)])[::-1]      # 최근 가중↑
        ctx = (w[:, None] * emb[lo:t]).sum(0)
        nrm = np.linalg.norm(ctx)
        d_ctx = 1.0 - float(ctx @ emb[t] / nrm) if nrm > 0 else d_prev
        out[t - 1] = A * d_prev + (1 - A) * d_ctx
    return out


def rules(s: np.ndarray) -> dict[str, np.ndarray]:
    """전 방법 공통 임계 규칙. 전부 gold 개수 미사용."""
    seq = np.concatenate([[0.0], s]).tolist()
    r = {"고정 6.49%": (s >= np.quantile(s, 1 - ANCHOR_RATE)).astype(np.int8),
         "otsu": np.asarray(adaptive_boundaries(seq, mode="otsu"), np.int8)}
    for c in (0.5, 1.0, 1.5):
        r[f"online μ+{c}σ"] = np.asarray(adaptive_boundaries(seq, c=c, mode="online_nf"), np.int8)
    return {k: v[:len(s)] for k, v in r.items()}


def score_all(corp: Corpus, sig: list[np.ndarray]) -> dict[str, tuple[float, int]]:
    ys, acc, npred = [], {}, {}
    for i, s in enumerate(sig):
        g = corp.gold[int(corp.to[i]):int(corp.to[i + 1])].astype(np.int8)
        ys.append(np.concatenate([g, [0]]).tolist())
        for k, b in rules(s).items():
            acc.setdefault(k, []).append(np.concatenate([b, [0]]).tolist())
            npred[k] = npred.get(k, 0) + int(b.sum())
    return {k: (score_dialogues(ys, v)["score"], npred[k]) for k, v in acc.items()}


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    corp = {s: Corpus(s) for s in SETS}
    res: dict = {}

    # ── M1-CosDrift (같은 임베딩) ─────────────────────────────────────────
    for s in SETS:
        c = corp[s]
        sig = [cosdrift(np.asarray(c.emb[int(c.so[i]):int(c.so[i + 1])], np.float32))
               for i in range(c.n_docs)]
        res[("M1-CosDrift", s)] = score_all(c, sig)
        # canonical δ* 도 별도로
        ys = [np.concatenate([c.gold[int(c.to[i]):int(c.to[i+1])], [0]]).tolist()
              for i in range(c.n_docs)]
        ps = [np.concatenate([(x >= DELTA_STAR).astype(np.int8), [0]]).tolist() for x in sig]
        res[("M1-CosDrift", s)]["δ*=0.5594 (동결)"] = (
            score_dialogues(ys, ps)["score"], int(sum(sum(p) for p in ps)))
        print(f"  M1 {s} 완료", flush=True)

    # ── M4 / M5 ──────────────────────────────────────────────────────────
    for tag in ("M4", "M5"):
        ck = torch.load(EXP / f"{tag}_per_slot_slot_seed0.pt", map_location=dev, weights_only=False)
        m = FactoredSlot(slot_dim=ck["args"]["slot_dim"], mono=ck["mono"],
                         pos_weight=ck["pos_weight"], dynamics=ck.get("dynamics"),
                         pred_space=ck["args"].get("pred_space", "obs")).to(dev)
        m.load_state_dict(ck["state"])
        for s in SETS:
            c = corp[s]
            probs = axis_probs(m, c, list(range(c.n_docs)), dev, bs=8, with_anchor=False)
            res[(tag, s)] = score_all(c, [p.sum(1) for p in probs])
            print(f"  {tag} {s} 완료", flush=True)

    rule_names = ["δ*=0.5594 (동결)", "고정 6.49%", "otsu",
                  "online μ+0.5σ", "online μ+1.0σ", "online μ+1.5σ"]
    L = ["# M1 vs M4 vs M5 대조표 — 같은 인코더 · 같은 임계 규칙 · 공식 test 전체\n",
         "> **동일 조건 3요소:** ① 같은 임베딩(`paraphrase-multilingual-mpnet` frozen)"
         " ② SuperDialseg **공식 test 전체** ③ 같은 임계 규칙을 전 방법에 동일 적용.",
         "> 🔴 M4/M5 는 **서사 픽션으로만 학습**(D-M8) — DTS 는 도메인 밖이다.",
         "> 🔴 dev-3 = 연습장. **탐색적 대조표**이며 확증 근거가 아니다.",
         "> 괄호 = 예측 경계 수.\n"]
    for s, h in zip(SETS, HDR):
        c = corp[s]
        ys = [np.concatenate([c.gold[int(c.to[i]):int(c.to[i+1])], [0]]).tolist()
              for i in range(c.n_docs)]
        none = score_dialogues(ys, [[0] * len(y) for y in ys])["score"]
        L += [f"\n## {h}  ·  gold 경계 {int(c.gold.sum())} / 전이 {len(c.gold)}"
              f" (밀도 {c.gold.mean():.4f})  ·  하한(무예측) **{none:.4f}**\n",
              "| 임계 규칙 | M1-CosDrift | M4-MonoAnchor | M5-FactoredSlot |",
              "|---|---|---|---|"]
        for rn in rule_names:
            cells = []
            for meth in ("M1-CosDrift", "M4", "M5"):
                v = res.get((meth, s), {}).get(rn)
                cells.append(f"{v[0]:.4f} ({v[1]})" if v else "—")
            L.append(f"| {rn} | " + " | ".join(cells) + " |")
    L += ["\n## 참조 — 프로젝트 공식 DTS 수치 (`2026-06-13_dts_official_rescore`)\n",
          "| | tiage | dialseg711 | superseg |", "|---|---|---|---|",
          "| δ_eff best percentile | **0.476**(p70) | **0.602**(p80) | **0.405**(p60) |",
          "\n⚠️ 공식 수치는 **`multi-qa-mpnet`** 인코더 + label-free percentile 튜닝이다."
          " 본 표는 `paraphrase-multilingual-mpnet` 이라 **절대값이 직접 비교되지 않는다** —"
          " 본 표의 용도는 **같은 조건에서의 방법 간 상대 비교**다.",
          "\n⚠️ M4/M5 seed 1개(slot-native). CLAUDE.md 3-run 요건 미충족."]
    p = REPO / "outputs/experiments/m4m5_phase1_anchor/METHOD_COMPARISON_FULL.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()

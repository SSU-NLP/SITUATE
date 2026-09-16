#!/usr/bin/env python3
"""ICML 참관기(한국어 89문장) 분절 — M5 + 적응 임계 vs HANDOFF_06 기준선.

## 왜
`handoff/HANDOFF_06_icml-diary-baseline_0709.md` 가 *"앞으로 만들 새 화제분절 방법론의 **before**
비교대상"* 으로 두 방법의 경계를 문장 단위로 고정 기록해 두었다. **지금이 그 after 다.**

기준선(HANDOFF_06 §3):
- **A. Hi-OnTop δ_eff** (LLM 없음, c=1.0): 경계 13개
  `[1, 2, 7, 18, 20, 21, 27, 29, 34, 59, 61, 69, 79]`
- **B-1. LLM two-stage full-past**: 경계 4개 `[25, 34, 61, 69]`
- **B-2. LLM two-stage ctx_k=6**: 경계 9개 `[18, 25, 34, 57, 59, 61, 69, 71, 79]`

## 무엇을
동일 입력(89문장) · 동일 인코더(`paraphrase-multilingual-mpnet`, HANDOFF_06 §2 와 같음) 로
**M5-FactoredSlot + 적응 임계**(D-B25 채택)를 돌리고 **문장 단위 전량 기록**한다(요약 금지).

🔴 **gold 가 없다.** 이 글에는 사람 주석 경계가 없으므로 **Score 를 낼 수 없다.**
이 산출물의 용도는 **정성 대조**뿐이다 — 어느 문장을 경계로 삼았는지, 기준선과 어디서 갈리는지.
🔴 **도메인 밖 2중:** M5 는 **영어 서사 픽션**으로만 학습됐고(D-M8), 이 글은 **한국어 산문**이다.
인코더는 다국어라 임베딩은 되지만 축 라벨은 영어 데이터에서 배운 것이다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
from hi_ontop.factored_slot import AXES, FactoredSlot  # noqa: E402
from hi_ontop.hi_ontop_v2 import adaptive_boundaries  # noqa: E402
from train_anchor_m4m5 import EXP  # noqa: E402

BASE_A = [1, 2, 7, 18, 20, 21, 27, 29, 34, 59, 61, 69, 79]
BASE_BFULL = [25, 34, 61, 69]
BASE_BK6 = [18, 25, 34, 57, 59, 61, 69, 71, 79]
ENCODER = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sents = [s.strip() for s in (REPO / "icml_sentences.txt").read_text(encoding="utf-8").splitlines()
             if s.strip()]
    n = len(sents)
    print(f"[data] {n} 문장", flush=True)

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(ENCODER, device=dev)
    E = torch.tensor(np.asarray(enc.encode(sents, normalize_embeddings=True, batch_size=64),
                                np.float32), device=dev).unsqueeze(0)

    ck = torch.load(EXP / "M5_per_slot_slot_seed0.pt", map_location=dev, weights_only=False)
    m = FactoredSlot(slot_dim=ck["args"]["slot_dim"], mono=ck["mono"], pos_weight=ck["pos_weight"],
                     dynamics=ck.get("dynamics"),
                     pred_space=ck["args"].get("pred_space", "obs")).to(dev)
    m.load_state_dict(ck["state"])
    m.eval()
    with torch.no_grad():
        o = m.forward_batched(E, torch.zeros(1, n - 1, 5, dtype=torch.long, device=dev),
                              torch.zeros(1, n, 3, dtype=torch.long, device=dev),
                              torch.ones(1, n, device=dev), torch.ones(1, n - 1, device=dev))
        P = torch.stack([torch.sigmoid(o["change_logits"][a]) for a in AXES], -1)[0].cpu().numpy()
    s = P.sum(1)                                            # [n-1] 연속 점수
    seq = np.concatenate([[0.0], s]).tolist()

    variants = {f"M5 online μ+{c}σ": np.asarray(
        adaptive_boundaries(seq, c=c, mode="online_nf"), np.int8)[:n - 1] for c in (0.5, 1.0, 1.5)}
    variants["M5 otsu"] = np.asarray(adaptive_boundaries(seq, mode="otsu"), np.int8)[:n - 1]

    # 전이 i (0-based, sents[i] -> sents[i+1]) 가 1이면 **다음 문장**이 새 세그먼트를 연다
    bnd = {k: sorted(int(i) + 1 for i in np.flatnonzero(v)) for k, v in variants.items()}

    L = ["# ICML 참관기 분절 — M5 + 적응 임계 (HANDOFF_06 after)\n",
         "> 🔴 **gold 없음 → Score 계산 불가.** 이 문서는 **정성 대조**용이다.",
         "> 🔴 **도메인 밖 2중**: M5 는 영어 서사 픽션 학습(D-M8), 입력은 한국어 산문.",
         "> 동일 입력 89문장 · 동일 인코더(`paraphrase-multilingual-mpnet`, HANDOFF_06 §2 와 같음).",
         "> 경계 인덱스 = **새 세그먼트를 여는 문장 번호**(HANDOFF_06 규약과 동일).\n",
         "\n## 경계 대조\n", "| 방법 | 경계 수 | 경계 인덱스 |", "|---|---|---|",
         f"| A. Hi-OnTop δ_eff (기준선) | {len(BASE_A)} | {BASE_A} |",
         f"| B-1. LLM full-past (기준선) | {len(BASE_BFULL)} | {BASE_BFULL} |",
         f"| B-2. LLM ctx_k=6 (기준선) | {len(BASE_BK6)} | {BASE_BK6} |"]
    for k, b in bnd.items():
        L.append(f"| **{k}** | **{len(b)}** | {b} |")

    L += ["\n## 기준선과의 일치\n", "| 방법 | vs A(13) | vs B-1(4) | vs B-2(9) |", "|---|---|---|---|"]
    for k, b in bnd.items():
        row = []
        for ref in (BASE_A, BASE_BFULL, BASE_BK6):
            inter = len(set(b) & set(ref))
            row.append(f"{inter}/{len(ref)} (Jaccard {inter/max(1,len(set(b)|set(ref))):.2f})")
        L.append(f"| {k} | " + " | ".join(row) + " |")

    L += ["\n## 문장 단위 전량 기록 (요약 아님)\n",
          "표기: `▶A`=기준선 δ_eff · `▶B1`/`▶B2`=기준선 LLM · **`▶M5`=신규(online μ+1.0σ)** ·"
          " `s`=Σ_k P(reset_k) · 축별 확률 = prota/space/time/causa/inten\n"]
    main_b = set(bnd["M5 online μ+1.0σ"])
    for i, sent in enumerate(sents):
        mk = []
        if i in BASE_A:
            mk.append("▶A")
        if i in BASE_BFULL:
            mk.append("▶B1")
        if i in BASE_BK6:
            mk.append("▶B2")
        if i in main_b:
            mk.append("**▶M5**")
        tag = " ".join(mk) if mk else "—"
        sv = f"{s[i-1]:.2f}" if i > 0 else "----"
        ax = (" ".join(f"{P[i-1, k]:.2f}" for k in range(5)) if i > 0 else "")
        L.append(f"- `[{i:2d}]` {tag} · s={sv} · [{ax}]  |  {sent}")

    L += ["\n⚠️ M5 seed 1개(slot-native). 적응 임계 c 는 dev-3 에서 고른 값이다(D-B25) —"
          " 이 입력에 맞춰 고른 것이 아니다.",
          "⚠️ 이 글에는 사람 주석 경계가 없어 **어느 방법이 맞는지 판정할 수 없다.**"]
    p = REPO / "outputs/experiments/m4m5_phase1_anchor/ICML_DIARY_SEGMENT.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:40]))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()

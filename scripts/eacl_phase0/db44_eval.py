#!/usr/bin/env python3
r"""D-B44 셀 평가 — 손실 가중치 2×2 (ALPHA × pos_weight).

## 무엇을
학습된 셀들을 **동결 규칙 그대로**(`online_nf` c=1.0 · 신호 **S5 5축 전부**) 두 선택셋에서 평가한다.
codex 판정: *"1차 W 게이트에서는 경계 임계를 다시 맞추지 않는다 — 동일한 기존 deployable 규칙을
써야 loss 변경의 직접 영향이 보인다."*

| 선택셋 | 근거 |
|---|---|
| dev-3 (pieman·tunnel·monkey) | 서사 작동점 선택셋 (D-B41 §2) |
| superseg/validation 앞 60대화 | 대화 선택셋 (D-B41 §2, 공식 test 와 disjoint) |

🔴 DTS 공식 test·봉인 시험지 미사용.
🔴 dev-3 절대 Score 인용 금지 — **Δ·방향만**. "보고하는 셋에서 골랐다" 공시 대상.
🔴 축 제거 없음: 운영 신호는 **S5**. `S4`(causality 제외)는 **판독용으로만** 찍고 채택 후보로 안 올린다.

## W0 무결성 대조 (별도)
새 하네스의 W0int seed0 이 기존 W0 seed0 를 재현하는지. Score 만으로는 dev-3 3편 bootstrap 이
퇴화(std 0.0644)해 판정력이 없으므로, codex 지정대로 **전이 단위** 지표를 함께 본다:
logits 상관 · 평균절대차 · **경계 위치 일치율**. 이쪽은 1,005 전이 전체를 쓰므로 표본 문제가 없다.

## 실행
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/eacl_phase0/db44_eval.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
from db29_final_eval import load  # noqa: E402
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from hi_ontop.factored_slot import AXES  # noqa: E402
from hi_ontop.hi_ontop_v2 import adaptive_boundaries  # noqa: E402
from tolerance_bands import load_dialogue_val  # noqa: E402
from train_anchor_m4m5 import Corpus, axis_probs  # noqa: E402

OUT = REPO / "outputs/experiments/2026-07-30_m5_retrain_design"
N_DLG = 60
C_MAIN = 1.0
CAUSALITY = AXES.index("causality")

# 셀 -> (네임스페이스, seeds).  🔒 조건이 ns 에 드러난다(D-B44 덮어쓰기 사고 이후 규약)
CELLS = {
    "W0  (ALPHA 현행 · pw clip) 🔒동결": ("_per_slot_slot", (0, 1, 2)),
    "W0int (하네스 무결성 재학습)": ("_per_slot_slot_w0int", (0,)),
    "W3  (ALPHA 현행 · pw one)": ("_per_slot_slot_pwone", (0, 1, 2)),
    "W1  (ALPHA 0.63 · pw one) ★주안": ("_per_slot_slot_alphauniform_pwone", (0, 1, 2)),
    "W4  (ALPHA 0.63 · pw clip)": ("_per_slot_slot_alphauniform", (0, 1, 2)),
    # M6 = FactoredShared (5슬롯 + dynamics 공유). W3 와 동일 손실조건(ALPHA 현행 · pw one)
    "M6-W3 (dynamics 공유 · pw one)": ("_shared_slot_pwone", (0, 1, 2)),
    "M6-W0 (dynamics 공유 · pw clip)": ("_shared_slot", (0, 1, 2)),
}
MODEL_OF = {"M6-W3 (dynamics 공유 · pw one)": "M6", "M6-W0 (dynamics 공유 · pw clip)": "M6"}


def rule(s: np.ndarray) -> np.ndarray:
    seq = np.concatenate([[0.0], s]).tolist()
    return np.asarray(adaptive_boundaries(seq, c=C_MAIN, mode="online_nf"), np.int8)[:len(s)]


def score_of(P_list, G_list, drop_causality: bool = False):
    ys, ps, n = [], [], 0
    for P, g in zip(P_list, G_list):
        s = np.delete(P, CAUSALITY, axis=1).sum(1) if drop_causality else P.sum(1)
        b = rule(s)[:len(g)]
        ys.append(np.concatenate([g, [0]]).tolist())
        ps.append(np.concatenate([b, [0]]).tolist())
        n += int(b.sum())
    return score_dialogues(ys, ps)["score"], n


def main() -> None:
    ap = argparse.ArgumentParser(description="D-B44 셀 평가 (동결 규칙 · S5)")
    ap.add_argument("--cells", default="", help="쉼표 구분 접두어 필터 (예: W0,W3)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                              device=dev)
    dlg = load_dialogue_val(N_DLG)
    dlg_emb = [torch.tensor(np.asarray(enc.encode(u, normalize_embeddings=True, batch_size=64),
                                       np.float32), device=dev).unsqueeze(0) for u, _ in dlg]
    dlg_g = [y for _, y in dlg]
    dev3 = Corpus("dev3")
    dev3_g = [dev3.gold[int(dev3.to[i]):int(dev3.to[i + 1])].astype(np.int8)
              for i in range(dev3.n_docs)]

    @torch.no_grad()
    def probs_of(m):
        Pd = axis_probs(m, dev3, list(range(dev3.n_docs)), dev, bs=64, with_anchor=False)
        Pl = []
        for E in dlg_emb:
            n = E.shape[1]
            o = m.forward_batched(E, torch.zeros(1, n - 1, 5, dtype=torch.long, device=dev),
                                  torch.zeros(1, n, 3, dtype=torch.long, device=dev),
                                  torch.ones(1, n, device=dev), torch.ones(1, n - 1, device=dev))
            Pl.append(torch.stack([torch.sigmoid(o["change_logits"][x]) for x in AXES],
                                  -1)[0].cpu().numpy())
        return Pd, Pl

    res: dict = {}
    keep = [x.strip() for x in a.cells.split(",") if x.strip()]
    dev3_s: dict[str, dict[int, np.ndarray]] = {}
    for cell, (ns, seeds) in CELLS.items():
        if keep and not any(cell.startswith(k) for k in keep):
            continue
        rows = []
        for sd in seeds:
            tag = MODEL_OF.get(cell, "M5")
            ck = REPO / f"outputs/experiments/m4m5_phase1_anchor/{tag}{ns}_seed{sd}.pt"
            if not ck.exists():
                continue
            m = load(tag, sd, dev, ns=ns)[0]
            Pd, Pl = probs_of(m)
            dev3_s.setdefault(cell, {})[sd] = np.concatenate([p.sum(1) for p in Pd])
            d_s, d_n = score_of(Pd, dev3_g)
            d4, _ = score_of(Pd, dev3_g, drop_causality=True)
            g_s, g_n = score_of(Pl, dlg_g)
            rows.append({"seed": sd, "dev3": d_s, "dev3_n": d_n, "dev3_S4": d4,
                         "dlg": g_s, "dlg_n": g_n,
                         "meanP": [float(np.concatenate(Pd)[:, k].mean()) for k in range(5)]})
            print(f"  [{cell}] seed{sd}  dev-3 {d_s:.4f}({d_n})  대화 {g_s:.4f}({g_n})", flush=True)
            del m
            if dev == "cuda":
                torch.cuda.empty_cache()
        if rows:
            res[cell] = rows
    (OUT / "db44_eval.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float),
                                        encoding="utf-8")

    def ms(v):
        return f"{np.mean(v):.4f}" + (f" ± {np.std(v, ddof=1):.4f}" if len(v) > 1 else "")

    L = ["# D-B44 — 손실 가중치 2×2 (ALPHA × pos_weight) 평가\n",
         f"> 규칙 = **동결 그대로** `online_nf c={C_MAIN}` · 신호 = **S5(5축 전부)** · 모델만 다름",
         "> 🔴 dev-3 절대 Score 인용 금지 — **Δ·방향만**. 서사 작동점을 **보고하는 셋에서 골랐다**(공시).",
         "> 🔴 DTS 공식 test·봉인 시험지 미사용. 축 제거 없음(S4 는 판독용).",
         "> 🔒 1차 게이트에서 임계를 다시 맞추지 않는다(codex) — 손실 변경의 직접 영향만 본다.\n",
         "\n## 1. 셀별 결과\n",
         "| 셀 | dev-3 Score (예측수) | 대화 Score (예측수) | *[판독] dev-3 S4* |",
         "|---|---|---|---|"]
    for cell, rows in res.items():
        L.append(f"| {cell} | {ms([r['dev3'] for r in rows])}"
                 f" ({np.mean([r['dev3_n'] for r in rows]):.0f}) |"
                 f" {ms([r['dlg'] for r in rows])} ({np.mean([r['dlg_n'] for r in rows]):.0f}) |"
                 f" *{ms([r['dev3_S4'] for r in rows])}* |")
    L.append("\n> dev-3 gold 66 · 대화 gold 159. *S4 = causality 제외 판독값 — **채택 후보 아님**.*")

    base = next((c for c in res if c.startswith("W0 ")), None)
    if base:
        L += ["\n## 2. 🚦 Δ vs W0 (seed 짝지음)\n",
              "| 셀 | dev-3 Δ (seed별) | 평균 | 대화 Δ | 판정 |", "|---|---|---|---|---|"]
        b3 = {r["seed"]: r["dev3"] for r in res[base]}
        bd = {r["seed"]: r["dlg"] for r in res[base]}
        for cell, rows in res.items():
            if cell == base:
                continue
            sd = [r["seed"] for r in rows if r["seed"] in b3]
            d3 = [next(r["dev3"] for r in rows if r["seed"] == s) - b3[s] for s in sd]
            dd = [next(r["dlg"] for r in rows if r["seed"] == s) - bd[s] for s in sd]
            v = ("✅ 전 seed +" if all(x > 0 for x in d3) else
                 "❌ 전 seed −" if all(x < 0 for x in d3) else "⚠️ 부호 혼재")
            L.append(f"| {cell} | " + " · ".join(f"{x:+.4f}" for x in d3)
                     + f" | **{np.mean(d3):+.4f}** | {np.mean(dd):+.4f} | {v} |")
        L.append("\n> 🔒 **대화 비열등 허용폭 = −0.0470**(TOLERANCE_BANDS.md, 실행 전 잠금)."
                 " 대화 Δ 가 이보다 더 나쁘면 비열등 실패.")

    # W0 무결성 — 전이 단위 지표 (dev-3 3편 bootstrap 퇴화 우회)
    if base and "W0int (하네스 무결성 재학습)" in dev3_s:
        o = dev3_s[base][0]
        n_ = dev3_s["W0int (하네스 무결성 재학습)"][0]
        bo, bn = rule(o), rule(n_)
        L += ["\n## 3. 🚦 W0 무결성 — 새 하네스가 기존 W0 seed0 을 재현하나\n",
              "> Score 만으로는 판정 불가(dev-3 3편 bootstrap std 0.0644 = 퇴화)."
              " codex 지정대로 **전이 단위**(1,005 전이) 지표로 본다.\n",
              "| 지표 | 값 |", "|---|---|",
              f"| 신호 s 상관 (Pearson) | **{np.corrcoef(o, n_)[0,1]:.4f}** |",
              f"| 신호 s 평균절대차 | {np.abs(o-n_).mean():.4f} (s 평균 {o.mean():.3f}) |",
              f"| 경계 위치 일치율 | **{(bo == bn).mean():.4f}** ({int((bo==bn).sum())}/{len(bo)}) |",
              f"| 예측 경계 수 | 기존 {int(bo.sum())} vs 신규 {int(bn.sum())} |",
              f"| dev-3 Score | 기존 {res[base][0]['dev3']:.4f} vs 신규 "
              f"{res['W0int (하네스 무결성 재학습)'][0]['dev3']:.4f} |"]
    L += ["\n## 4. 축별 평균 확률 (보정 상태 진단, dev-3)\n",
          "| 셀 | " + " | ".join(AXES) + " | 합 |", "|---|" + "---|" * 6]
    for cell, rows in res.items():
        mp = np.mean([r["meanP"] for r in rows], 0)
        L.append(f"| {cell} | " + " | ".join(f"{x:.3f}" for x in mp) + f" | {mp.sum():.3f} |")
    L.append("\n> 앵커 실제 리셋율 = protagonist 0.305 · space 0.105 · time 0.021 ·"
             " causality 0.695 · intentionality 0.114 (합 1.240).")
    (OUT / "DB44_EVAL.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n-> {OUT / 'DB44_EVAL.md'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""M4/M5 slot-alignment probe — Phase-1 앵커(10,598편) 스케일판.

Phase 2 의 `probe_slot_alignment.py` 를 그대로 이식하되 규모만 맞췄다. **지표 정의는 불변**이므로
Phase 2 수치(M5 excess +0.072 / M4 +0.114 / factoring gain +0.042)와 같은 축에서 읽을 수 있다.

왜 이 지표인가: own-axis AUPRC 는 M5≈M4 로 나온다(둘 다 축별 헤드를 갖기 때문). **진짜 판별자는
교차 누출(cross-talk)** 이다 — 슬롯 k 의 전이 특징에서 *다른* 축 j 를 얼마나 읽어낼 수 있는가.
  M5(축별 5슬롯): 대각 높고 비대각 낮아야 함 = 슬롯이 축에 특화됨.
  M4(공유 1슬롯): 슬롯 0 에서 모든 축이 읽힘 = 얽힘(entangled).
비대각에서 **라벨 상관 천장 B[k,j]** (축 k 의 *참* 라벨만으로 축 j 를 예측한 AUPRC) 를 빼야 한다 —
축들이 원래 상관돼 있으면 완벽히 분리된 슬롯도 그만큼은 읽히기 때문(codex 2026-07-15 정정).

**factoring gain = M4 excess − M5 excess.** > 0 이면 D1 라우팅이 monolithic 이 얽는 것을 분리한 것.

🔒 D-B16: 이 스크립트는 dev-3·시험지를 쓰지 않는다. 전부 phase1 앵커 코퍼스 안에서만 계산한다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
from hi_ontop.factored_slot import AXES, FactoredSlot  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from train_anchor_m4m5 import EXP, Corpus, batches, make_batch  # noqa: E402


@torch.no_grad()
def slot_feats(model, c: Corpus, idx: list[int], dev: str, bs: int):
    """슬롯별 전이 특징 X[slot] [N,4*slot_dim] + reset 라벨 Y [N,5]."""
    model.eval()
    # D-B36: 읽기 단위 = 슬롯(M5/M6) · 단일 잠재(M4) · **블록**(구조화 벡터). axis_state 가 셋을
    # 모두 처리하므로 항상 5 단위로 뽑는다 — M4 는 5 행이 동일해지고(기존 동작과 같음),
    # block_readout 은 블록마다 다른 행이 나온다.
    n_unit = 1 if (model.mono and not getattr(model, "block_readout", False)) else 5
    feats: dict[int, list] = {s: [] for s in range(n_unit)}
    labels = []
    for chunk in batches(idx, c, bs):
        z, rs, st, sm, tm, order, ft = make_batch(
            c, chunk, dev, True, want_feat=getattr(model, 'tagger', None) is not None)
        S = model.encode_seq_batched(z, ft)  # [B,T,n_slots,dim]
        for b, i in enumerate(order):
            n = int(c.so[i + 1] - c.so[i])
            for s in range(n_unit):
                sk = model.axis_state(S[b:b+1, :n], s)[0]
                d = torch.cat([sk[:-1], sk[1:], sk[1:] - sk[:-1], sk[1:] * sk[:-1]], -1)
                feats[s].append(d.float().cpu().numpy())
            labels.append(c.resets[int(c.to[i]):int(c.to[i + 1])])
    return ({s: np.concatenate(feats[s]) for s in feats}, np.concatenate(labels))


def _fit_ap(Xtr, ytr, Xva, yva) -> float:
    if ytr.sum() < 3 or yva.sum() < 1:
        return float("nan")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    clf.fit(Xtr, ytr)
    return float(average_precision_score(yva, clf.predict_proba(Xva)[:, 1]))


def label_corr_ceiling(Ytr, Yva) -> np.ndarray:
    """B[k,j] = 축 k 의 *참* 라벨만으로 축 j 를 예측한 AUPRC (분리된 슬롯도 이만큼은 읽힌다)."""
    B = np.full((5, 5), np.nan)
    for k in range(5):
        for j in range(5):
            B[k, j] = 1.0 if k == j else _fit_ap(
                Ytr[:, [k]].astype(float), Ytr[:, j], Yva[:, [k]].astype(float), Yva[:, j])
    return B


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="phase1")
    ap.add_argument("--probe-train-docs", type=int, default=200,
                    help="프로브 학습용 문서 수(전량은 2.7M×512 라 비현실적)")
    ap.add_argument("--probe-val-docs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ns", default="", help="체크포인트 네임스페이스(예: _per_slot)")
    ap.add_argument("--m4-ns", default=None,
                    help="M4 만 다른 네임스페이스에서 읽는다(기본 = --ns). D-B33 애블레이션에서 "
                         "M4 는 재학습 대상이 아니라 기준선 ns 의 것을 그대로 참조하기 때문.")
    ap.add_argument("--models", default="M5,M4")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    c = Corpus(args.split)

    # 학습 때와 **같은 분할 seed** 로 train/val 을 재현한 뒤 각각에서 서브샘플
    rng = np.random.default_rng(20260728)
    perm = rng.permutation(c.n_docs)
    n_val = max(1, int(c.n_docs * 0.05))
    va_all, tr_all = perm[:n_val].tolist(), perm[n_val:].tolist()
    sub = np.random.default_rng(args.seed)
    tr = sub.choice(tr_all, min(args.probe_train_docs, len(tr_all)), replace=False).tolist()
    va = sub.choice(va_all, min(args.probe_val_docs, len(va_all)), replace=False).tolist()
    print(f"[probe] fit {len(tr)} docs / eval {len(va)} docs (학습 분할과 동일 seed)", flush=True)

    mats, owns, ckpaths = {}, {}, {}
    Ytr = Yva = None
    MONO = {"M5": False, "M4": True, "M6": False}
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        mono = MONO[name]
        ns = args.m4_ns if (name == "M4" and args.m4_ns is not None) else args.ns
        ckp = EXP / f"{name}{ns}_seed{args.seed}.pt"
        ckpaths[name] = str(ckp.relative_to(EXP))
        ck = torch.load(ckp, map_location=dev, weights_only=False)
        m = FactoredSlot(slot_dim=ck["args"]["slot_dim"], mono=mono,
                         pos_weight=ck["pos_weight"], dynamics=ck.get("dynamics"),
                         pred_weight=ck["args"].get("pred_weight", 1.0),
                         bottleneck=ck["args"].get("bottleneck", 8),
                         dropout=ck["args"].get("dropout", 0.4),
                         d1=ck["args"].get("input_path", "axis_extractor") == "axis_extractor",
                         shared_dim=ck["args"].get("shared_proj_dim") or None,
                         tagger=(ck["args"].get("tagger") or None),
                         tagger_axes=ck["args"].get("tagger_axes", 5),
                         pred_space=ck["args"].get("pred_space", "obs")).to(dev)
        m.load_state_dict(ck["state"])
        Xtr, Ytr = slot_feats(m, c, tr, dev, args.batch)
        Xva, Yva = slot_feats(m, c, va, dev, args.batch)
        n_unit = 1 if (m.mono and not getattr(m, "block_readout", False)) else 5
        M = np.full((n_unit, 5), np.nan)
        for s in range(n_unit):
            for j in range(5):
                M[s, j] = _fit_ap(Xtr[s], Ytr[:, j], Xva[s], Yva[:, j])
        mats[name] = M
        print(f"[probe] {name} 완료", flush=True)

    B = label_corr_ceiling(Ytr, Yva)
    # factored 모델은 M5 일 수도 M6 일 수도 있다(D-B33: 같은 5슬롯, dynamics 만 다름).
    fac = next(n for n in mats if n != "M4")
    M5, M4 = mats[fac], mats["M4"]
    off = ~np.eye(5, dtype=bool)
    excess5 = np.where(off, M5 - B, np.nan)
    m5_excess = float(np.nanmean(excess5))
    m4_excess = float(np.nanmean([M4[0, j] - np.nanmax([B[k, j] for k in range(5) if k != j])
                                  for j in range(5)]))

    L = [f"# M4/M5 slot-alignment probe — Phase-1 앵커 (seed {args.seed})\n",
         "> 지표 정의 = Phase 2 `probe_slot_alignment.py` 와 **동일**(비교 가능). "
         "프로브는 앵커 코퍼스 안에서만 학습·평가 — dev-3·시험지 미사용(D-B16).\n",
         "> 평가된 체크포인트: " + " · ".join(f"{k} `{v}`" for k, v in ckpaths.items()) +
         "  (표의 'M5' 행은 --ns 가 가리키는 조건의 factored 모델이다 — 이름이 아니라 경로를 본다)\n",
         f"프로브 학습 {len(tr)}편 / 평가 {len(va)}편 · 앵커 발화율 "
         + " ".join(f"{a[:5]} {Yva[:, k].mean():.3f}" for k, a in enumerate(AXES)) + "\n",
         "\n## M5 — probe AUPRC [slot k → axis j] (대각=자기 축, 비대각=교차 누출)\n",
         "| slot\\axis | " + " | ".join(a[:5] for a in AXES) + " |", "|---|" + "---|" * 5]
    for s in range(5):
        L.append(f"| slot {AXES[s][:5]} | " + " | ".join(
            (f"**{M5[s,j]:.3f}**" if s == j else f"{M5[s,j]:.3f}") for j in range(5)) + " |")
    L += ["\n## M5 — EXCESS off-axis leakage (라벨 상관 천장 B 제거)\n",
          "| slot\\axis | " + " | ".join(a[:5] for a in AXES) + " |", "|---|" + "---|" * 5]
    for s in range(5):
        L.append(f"| slot {AXES[s][:5]} | " + " | ".join(
            (f"own {M5[s,j]:.3f}" if s == j else f"{excess5[s,j]:+.3f}") for j in range(5)) + " |")
    L += ["\n## M4 (monolithic) — slot-0 → 각 축 (얽힘)\n",
          "| | " + " | ".join(a[:5] for a in AXES) + " |", "|---|" + "---|" * 5,
          "| slot 0 | " + " | ".join(f"{M4[0,j]:.3f}" for j in range(5)) + " |",
          "\n## 판정\n",
          "- M5 own-axis (대각): " + " ".join(f"{AXES[k][:5]} {M5[k,k]:.3f}" for k in range(5)),
          f"- **M5 평균 EXCESS 누출 = {m5_excess:+.4f}** (≈0 = 슬롯이 축에 특화)",
          f"- **M4 평균 excess = {m4_excess:+.4f}** (높을수록 단일 잠재가 전 축을 얽음)",
          f"- **factoring gain = M4 − M5 = {m4_excess - m5_excess:+.4f}** "
          "(>0 = D1 라우팅이 monolithic 이 얽는 것을 분리)",
          f"\n> 참조: Phase 2(10편·오염 Tier2 7편 포함) = M5 +0.072 / M4 +0.114 / gain **+0.042**.",
          "\n⚠️ 앵커 k=1 · S_self 0.659(상한) · 탐색적 — 성능 주장 아님(D-B16)."]
    out = EXP / f"PROBE{args.ns}_seed{args.seed}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

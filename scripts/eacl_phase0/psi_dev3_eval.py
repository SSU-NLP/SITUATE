#!/usr/bin/env python3
"""D-B62 — ψ 재개봉 조건들의 dev-3 개봉 평가.

🔴 **이 스크립트는 판정을 바꾸지 않는다.** ψ 채택 여부는 앵커 val OOF(J_anchor) 게이트가 이미
   결정했고(전부 미달), dev-3 는 *외적 방향 확인*일 뿐이다. dev-3 를 보고 조건을 다시 고르면
   그게 D-M5 재발(selection-on-dev)이다. 사용자 요청으로 개봉하되 그 지위를 명시한다.

지위 caveat: dev-3 = 3편(pieman/tunnel/monkey), 연습장. 앵커 k=1 · S_self 0.659(상한).
규칙: `train_direct_boundary.score_signals` 그대로 호출(재구현 금지) — 고정분위수(앵커 동결
ANCHOR_R) + 적응임계 둘 다. 메인은 Score, AUC 는 보조.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts/eacl_phase0")]
from full_comparison_dev3 import lift_l1, purity_coverage  # noqa: E402  🔒 동결 정의
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from hi_ontop.factored_slot import FactoredSlot  # noqa: E402
from train_anchor_m4m5 import Corpus, axis_probs  # noqa: E402
import train_direct_boundary as m8  # noqa: E402

EXP = REPO / "outputs/experiments/m4m5_phase1_anchor"
OUT = REPO / "outputs/experiments/psi_reopen_2026-08-01"
DOCS = ["pieman", "tunnel", "monkey"]

# (표시명, 네임스페이스) — ns 는 train_anchor_m4m5.py 의 args.ns 규칙이 만든 것 그대로.
COND = [
    ("기준선  ψ없음 · delta",        "_phase1_i2_per_slot_slot_pwone"),
    ("A  ψcummean · delta",          "_phase1_i2_per_slot_slot_psicummean_pwone"),
    ("B  ψema · delta",              "_phase1_i2_per_slot_slot_psiema_pwone"),
    ("C  ψcummean · resid_signed",   "_phase1_i2_per_slot_slot_resid_signed_psicummean_pwone"),
    ("D  ψ없음 · resid_signed",      "_phase1_i2_per_slot_slot_resid_signed_pwone"),
]
COND_CHAINQ = [(n, ns.replace("_phase1_i2_", "_phase1_chainq_i2_")) for n, ns in COND]
# D-B62 §8 책 단위 분할 배치 — ns 에 `_gsbook` 이 붙는다(psi 뒤, pos_weight 앞).
BOOK = [(n, ns.replace("_pwone", "_gsbook_pwone")) for n, ns in COND]
BOOK_CHAINQ = [(n, ns.replace("_pwone", "_gsbook_pwone")) for n, ns in COND_CHAINQ]


def load(ns: str, seed: int, dev: str):
    """🔴 db29_final_eval.load 는 head_feat/psi 를 안 넘긴다 — ψ·잔차 조건에는 못 쓴다."""
    ck = torch.load(EXP / f"M5{ns}_seed{seed}.pt", map_location=dev, weights_only=False)
    a = ck["args"]
    m = FactoredSlot(slot_dim=a["slot_dim"], mono=ck["mono"], pos_weight=ck["pos_weight"],
                     dynamics=ck.get("dynamics"), pred_space=a.get("pred_space", "obs"),
                     head_feat=a.get("head_feat", "delta"),
                     psi=(None if a.get("psi", "none") == "none" else a["psi"]),
                     psi_beta=a.get("psi_beta", 0.9), psi_dropout=a.get("psi_dropout", 0.3),
                     bottleneck=a.get("bottleneck", 8),
                     d1=a.get("input_path", "axis_extractor") == "axis_extractor",
                     shared_dim=a.get("shared_proj_dim") or None,
                     dropout=a.get("dropout", 0.4)).to(dev)
    m.load_state_dict(ck["state"])
    return m.eval(), ck


def row_metrics(P, G, n_gold: int) -> dict:
    """Score · BOR · purity · coverage · F(p,c) · L1 — 전부 동결 구현 호출(재구현 금지)."""
    ys = [np.concatenate([np.asarray(g, np.int8), [0]]).tolist() for g in G]
    ps = [np.concatenate([np.asarray(p[:len(g)], np.int8), [0]]).tolist() for p, g in zip(P, G)]
    s = score_dialogues(ys, ps)
    pur, cov = purity_coverage(P, G)
    n_pred = int(sum(int(np.asarray(p[:len(g)]).sum()) for p, g in zip(P, G)))
    return {"score": float(s["score"]), "pk": float(s["pk"]), "wd": float(s["wd"]),
            "f1": float(s["f1"]), "n_pred": n_pred, "bor": n_pred / max(n_gold, 1),
            "purity": float(pur), "coverage": float(cov),
            "f_pc": float(2 * pur * cov / (pur + cov)) if (pur + cov) else 0.0,
            "L1": float(lift_l1(P, G))}


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    dev3 = Corpus("dev3")
    G = [dev3.gold[int(dev3.to[i]):int(dev3.to[i + 1])].astype(np.int8)
         for i in range(dev3.n_docs)]
    print(f"dev-3 gold 경계: {[int(g.sum()) for g in G]} (총 {sum(int(g.sum()) for g in G)})\n")

    n_gold = int(sum(int(g.sum()) for g in G))
    z = [np.zeros(len(g), np.int8) for g in G]
    o = [np.ones(len(g), np.int8) for g in G]
    print(f"하한: 무경계 {row_metrics(z, G, n_gold)['score']:.4f} · "
          f"전부경계 {row_metrics(o, G, n_gold)['score']:.4f}\n")

    out: dict[str, dict] = {}
    sets = (("phase1_i2 · 책분할", BOOK), ("phase1_chainq_i2 · 책분할", BOOK_CHAINQ)) \
        if len(sys.argv) > 2 and sys.argv[2] == "book" \
        else (("phase1_i2", COND), ("phase1_chainq_i2", COND_CHAINQ))
    for split_name, conds in sets:
        sig: dict[str, list[np.ndarray]] = {}
        for name, ns in conds:
            if not (EXP / f"M5{ns}_seed{seed}.pt").exists():
                print(f"[{split_name}] 체크포인트 없음 — {ns}")
                continue
            m, _ = load(ns, seed, dev)
            with torch.no_grad():
                P = axis_probs(m, dev3, list(range(dev3.n_docs)), dev, bs=4, with_anchor=False)
            sig[name] = [x.sum(1) for x in P]
            del m
            torch.cuda.empty_cache()
        for rule, fn in (("고정분위수(앵커동결)", m8.cuts_pct), ("적응임계", m8.cuts_adaptive)):
            print(f"=== {split_name} · {rule} (seed {seed}) ===")
            print(f"{'조건':32s} {'Score':>7s} {'BOR':>6s} {'purity':>7s} {'cover':>6s} "
                  f"{'F(p,c)':>7s} {'L1':>5s}")
            for name in sig:
                Pb = [fn(s)[:len(g)] for s, g in zip(sig[name], G)]
                r = row_metrics(Pb, G, n_gold)
                out[f"{split_name}|{rule}|{name}"] = r
                print(f"{name:32s} {r['score']:7.4f} {r['bor']:6.2f} {r['purity']:7.3f} "
                      f"{r['coverage']:6.3f} {r['f_pc']:7.3f} {r['L1']:5.2f}")
            print()

    OUT.mkdir(parents=True, exist_ok=True)
    tag = "_book" if (len(sys.argv) > 2 and sys.argv[2] == "book") else ""
    (OUT / f"DEV3_PSI{tag}_seed{seed}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> {OUT / f'DEV3_PSI{tag}_seed{seed}.json'}")


if __name__ == "__main__":
    main()

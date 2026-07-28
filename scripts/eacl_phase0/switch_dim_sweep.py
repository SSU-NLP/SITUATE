#!/usr/bin/env python3
"""차원 불일치 값싼 판별 — 슬롯 타깃을 저차원 사영해 우도비를 다시 계산한다 (재학습 없음).

왜 (D-B18 후속② / ③ 진단)
--------------------------
설계 §4.2 의 분포족들(개체 집합 위 point process, 저차원 place/time 범주)은 **컴팩트한 타깃**을
전제하는데 우리 슬롯은 **128D 연속 잠재**다. 128D 로그밀도 합은 크기가 O(100~1000) 이라
sticky-CRP 사전(폭 1.7)이 묻히고(λ 게이트 INERT) 잡음도 함께 커진다(S/N 0.43).

VQ 도입 + 재학습 전에 **"차원만 줄여도 되는가"** 를 재학습 0 으로 판별한다:
슬롯별 PCA 로 d 차원 사영 → 같은 우도비/스위칭을 다시 계산 → S/N·AUC·Score·λ 민감도를 본다.

**판정 기준 (사전 선언, 2026-07-28 — 측정 전 확정):**
  통과 = 어떤 d 에서 **우도비 AUC 가 리셋확률 합 AUC(0.7006) 이상**  → VQ 재학습 착수 정당.
  회색 = AUC 가 0.65~0.70 이고 **λ 게이트가 살아남**(사전이 결정을 바꿈) → 부분 진전, 재검토.
  실패 = 전 d 에서 AUC < 0.65 → **스위칭 층을 접고** plan 시나리오 B(리셋확률 기반)로 확정.

🔒 D-B16: PCA·분포 적합은 **phase1 val 에서만**. dev-3 gold 는 채점에만 쓴다.
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
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from hi_ontop.factored_slot import FactoredSlot  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from switch_scorer import MAX_K, slot_states_and_pred  # noqa: E402
from train_anchor_m4m5 import EXP, T_STAR, Corpus  # noqa: E402
from hi_ontop.scrp import sticky_crp_unnormed  # noqa: E402

REF_AUC = 0.7006          # 리셋확률 합의 dev-3 AUC (같은 모델, D-B17)
PASS, GRAY = 0.7006, 0.65  # 사전선언 밴드


def fit_space(model, c: Corpus, va: list[int], dev: str, bs: int, dim: int, seed: int):
    """phase1 val 에서 슬롯별 PCA + 사영공간 분포(대각 가우시안)를 적합."""
    packs = slot_states_and_pred(model, c, va, dev, bs, True)
    n_slots = model.n_slots
    S_all = [[] for _ in range(n_slots)]
    H_all = [[] for _ in range(n_slots)]
    on_idx = []
    for i, (S, Sh) in zip(va, packs):
        R = c.resets[int(c.to[i]):int(c.to[i + 1])]
        on_idx.append(R.sum(1) >= T_STAR)
        for k in range(n_slots):
            S_all[k].append(S[1:, k])
            H_all[k].append(Sh[:, k])
    cal = {"pca": [], "mu_c": [], "v_c": [], "mu_o": [], "v_o": [],
           "ld_c": [], "ld_o": []}
    bnd = np.concatenate(on_idx)
    for k in range(n_slots):
        X = np.concatenate(S_all[k])
        H = np.concatenate(H_all[k])
        p = PCA(n_components=min(dim, X.shape[1]), random_state=seed).fit(X)
        Xp, Hp = p.transform(X), p.transform(H)
        r = Xp - Hp
        o = Xp[bnd] if bnd.any() else Xp
        eps = 1e-8
        cal["pca"].append(p)
        cal["mu_c"].append(r.mean(0)); cal["v_c"].append(r.var(0) + eps)
        cal["mu_o"].append(o.mean(0)); cal["v_o"].append(o.var(0) + eps)
        cal["ld_c"].append(float(np.log(r.var(0) + eps).sum()))
        cal["ld_o"].append(float(np.log(o.var(0) + eps).sum()))
    return cal


def loglik_ratio(S: np.ndarray, Sh: np.ndarray, cal: dict) -> np.ndarray:
    """전이별 Σ_k [log p(obs|new) − log p(obs|cont)] (사영공간, 정규화 상수 포함)."""
    n_slots = S.shape[1]
    ll = np.zeros(Sh.shape[0])
    for k in range(n_slots):
        p = cal["pca"][k]
        Xp, Hp = p.transform(S[1:, k]), p.transform(Sh[:, k])
        dn = Xp - cal["mu_o"][k]
        dc = (Xp - Hp) - cal["mu_c"][k]
        ll += (-0.5 * (dn * dn / cal["v_o"][k]).sum(1) - 0.5 * cal["ld_o"][k]) \
            - (-0.5 * (dc * dc / cal["v_c"][k]).sum(1) - 0.5 * cal["ld_c"][k])
    return ll


def segment(ll: np.ndarray, alpha: float, lmda: float) -> np.ndarray:
    counts = np.zeros(MAX_K)
    cur = 0
    counts[cur] = 1.0
    b = np.zeros(len(ll), dtype=np.int8)
    for t in range(len(ll)):
        pr = sticky_crp_unnormed(counts, cur, alpha, lmda, 1.0)
        nv = int(np.count_nonzero(counts))
        pn = pr[nv] if nv < MAX_K else 1e-12
        if np.log(max(pn, 1e-12)) - np.log(max(pr[cur], 1e-12)) + ll[t] > 0 and nv < MAX_K - 1:
            b[t] = 1; cur = nv; counts[cur] = 1.0
        else:
            counts[cur] += 1.0
    return b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="M5")
    ap.add_argument("--ns", default="_per_slot_slot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cal-docs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dims", default="4,8,16,32,64,128")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--lmdas", default="0,1,10,100")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    c, dev3 = Corpus("phase1"), Corpus("dev3")
    ck = torch.load(EXP / f"{args.model}{args.ns}_seed{args.seed}.pt",
                    map_location=dev, weights_only=False)
    m = FactoredSlot(slot_dim=ck["args"]["slot_dim"], mono=ck["mono"],
                     pos_weight=ck["pos_weight"], dynamics=ck["dynamics"],
                     pred_space=ck["args"].get("pred_space", "obs")).to(dev)
    m.load_state_dict(ck["state"])
    rng = np.random.default_rng(20260728)
    perm = rng.permutation(c.n_docs)
    va = np.random.default_rng(args.seed).choice(
        perm[:max(1, int(c.n_docs * 0.05))].tolist(), args.cal_docs, replace=False).tolist()
    d3 = slot_states_and_pred(m, dev3, list(range(dev3.n_docs)), dev, 4, False)
    golds = [dev3.gold[int(dev3.to[i]):int(dev3.to[i + 1])].astype(np.int8)
             for i in range(dev3.n_docs)]

    rows = []
    for dim in [int(x) for x in args.dims.split(",")]:
        cal = fit_space(m, c, va, dev, args.batch, dim, args.seed)
        lls = [loglik_ratio(S, Sh, cal) for S, Sh in d3]
        ll, g = np.concatenate(lls), np.concatenate(golds).astype(bool)
        auc = float(roc_auc_score(g, ll))
        sn = float((ll[g].mean() - ll[~g].mean()) / ll.std())
        best = None
        npred = {}
        for lm in [float(x) for x in args.lmdas.split(",")]:
            ys, ps = [], []
            for gg, l in zip(golds, lls):
                b = segment(l, args.alpha, lm)
                ys.append(np.concatenate([gg, [0]]).tolist())
                ps.append(np.concatenate([b, [0]]).tolist())
            sc = score_dialogues(ys, ps)["score"]
            npred[lm] = int(sum(sum(p) for p in ps))
            if best is None or sc > best[1]:
                best = (lm, sc)
        alive = len(set(npred.values())) > 1
        rows.append(dict(dim=dim, auc=auc, sn=sn, score=best[1], lmda=best[0],
                         npred=npred[best[0]], lam_alive=alive, sigma=float(ll.std())))
        print(f"  d={dim:>4d}  AUC {auc:.4f}  S/N {sn:+.3f}  σ {ll.std():8.1f}  "
              f"Score {best[1]:.4f}(λ={best[0]:g}, 경계 {npred[best[0]]})  "
              f"λ살아있음 {'✅' if alive else '🔴'}", flush=True)

    top = max(rows, key=lambda r: r["auc"])
    verdict = ("통과 — VQ 재학습 착수 정당" if top["auc"] >= PASS else
               "회색 — 부분 진전, 재검토" if top["auc"] >= GRAY and top["lam_alive"] else
               "실패 — 스위칭 층을 접고 시나리오 B(리셋확률)로 확정")
    L = [f"# 차원 불일치 값싼 판별 — {args.model}{args.ns} seed {args.seed}\n",
         "> 재학습 0. 슬롯별 PCA 사영 후 우도비 재계산. PCA·분포 적합은 phase1 val 에서만.",
         f"> **사전선언 밴드**: AUC ≥ {PASS}(=리셋확률 합) 통과 · {GRAY}~{PASS}+λ살아있음 회색 · <{GRAY} 실패\n",
         "\n| 사영 차원 d | 우도비 AUC | S/N | σ | Score(최적 λ) | 예측 경계 | λ 살아있음 |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['dim']} | **{r['auc']:.4f}** | {r['sn']:+.3f} | {r['sigma']:.1f} | "
                 f"{r['score']:.4f} (λ={r['lmda']:g}) | {r['npred']} | "
                 f"{'✅' if r['lam_alive'] else '🔴'} |")
    L += [f"| *참조: 리셋확률 합* | *{REF_AUC:.4f}* | — | — | *0.3987* | *67* | — |",
          f"\n**최고 = d={top['dim']} · AUC {top['auc']:.4f}** → **판정: {verdict}**",
          "\n⚠️ 앵커 k=1 · S_self 0.659(상한) · 탐색적 — 성능 주장 아님(D-B16). dev-3 = 연습장."]
    out = EXP / f"DIMSWEEP_{args.model}{args.ns}_seed{args.seed}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    (EXP / f"dimsweep_{args.model}{args.ns}_seed{args.seed}.json").write_text(
        json.dumps(rows, indent=2, default=float))
    print("\n".join(L))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

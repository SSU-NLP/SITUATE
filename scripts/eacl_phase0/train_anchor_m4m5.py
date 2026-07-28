#!/usr/bin/env python3
r"""M4-MonoAnchor vs M5-FactoredSlot — Phase-1 앵커(10,598편) 학습 + dev-3 평가.

🔒 지위 = **1차 탐색 패스** (ledger D-B16). 여기서 나오는 숫자는 확인적 성능이 아니다.
   - M5−M4 를 "가설 지지"라 부르지 않는다. **부호와 방향만** 보고한다.
   - dev-3 절대 Score 를 성능으로 인용하지 않는다(dev-3 = 연습장, pieman 선택압 9회 누적).
   - 봉인 시험지(Born a Crime 3편)는 **열지 않는다** — 이 스크립트는 접근조차 하지 않는다.
   - 앵커 k=1 · S_self 0.659(상한, 2026-07-28_anchor_k_stability) caveat 를 모든 표에 단다.

핵심 대조 = **M5 − M4 = 표현 분리(factored)의 인과적 기여.** 둘은 헤드·손실·데이터·seed 가 전부
같고 잠재 표현만 다르다(M5 = 축별 5슬롯 + D1 라우팅 / M4 = 공유 1슬롯). Phase 2 de-risk 는
10편·3,090전이(그중 7편이 오염된 Tier2)에서 gain +0.044±0.005 였고, 본 실행은 그것을
**10,598편·2,777,561전이 청정 앵커**로 대체한다.

경계 도출 규칙 — **둘 다 산출하고 둘 다 보고한다.** 어느 쪽도 dev-3 에서 캘리브레이션하지 않는다
(작동점은 전부 **phase1 val 분할**에서 정해 동결 -> deployable 규약 준수, oracle 누출 없음).

  세 규칙 전부 **앵커 코퍼스에서 얻은 경계율 0.0649 하나만** 쓰고 dev-3 gold 는 절대 쓰지 않는다.

  [주]   **per-doc percentile** — 연속 점수 `s = Σ_k P(reset_k)` 의 **문서 내** 상위 6.49% 를 경계로.
         문서마다 개수를 스스로 정하되 그 밀도는 앵커에서 동결. CLAUDE.md 가 명시 허용:
         *"개수를 문서 길이에 적응(percentile/경계밀도 사전/적응임계)시키는 것은 방법의 일부이며
         deployable 로 평가받아야 한다."* gold 개수 미사용 = oracle 누출 아님.
  [보조a] **global τ** — phase1 val 에서 예측 경계율 = 앵커 경계율이 되는 전역 분위수 τ. `s ≥ τ`.
  [보조b] **M3 충실판** — 축별 확률을 그 축 앵커 발화율 분위수로 이진화 -> `count ≥ t\*=3`.

  🔴 규칙 집합을 이렇게 고정한 경위 (파이프라인 스모크 후 · **실제 결과 보기 전**, 전부 보고함):
    스모크(30편·2에폭·slot_dim 32)에서 전역 임계 두 규칙이 M5 경계 **0개**를 냈다. 원인을 두 번
    잘못 짚었다가 실측으로 바로잡았으므로 그 경위를 남긴다:
      ✗ 가설1 "앵커 6.49% 는 5축 상관에서 나오는데 모델은 상관이 약하다" — **반증.**
        독립 가정 Poisson-binomial 이 0.0567, 실측 0.0649 로 차이 +0.008. **5축은 사실상 독립**이다.
      ✗ 가설2 "학습(PG 문어) → dev-3(구어 전사) 도메인 분포 이동" — **반증.**
        M5 의 val 평균 s=2.488(τ=2.491) vs dev-3 문서별 평균 2.486/2.489/2.488. 이동 없음.
      ✓ 실제 원인: **스모크 모델의 출력이 거의 상수**다(s 의 평균 2.488, p95 2.490 = 폭 0.005).
        학습이 안 된 모델이 모든 전이에 같은 값을 뱉으니 전역 임계가 평균에 딱 붙고, 소수점
        셋째 자리로 전부-아니면-전무가 갈린다. **스모크의 병리이지 본 학습의 결함이 아니다** —
        본 실행에서는 [보조a]도 정상 동작할 수 있다.
    그래도 [주]를 primary 로 두는 이유는 둘:
      ① 점수 스케일이 퇴화해도 **개수가 안정적**이다(분위수는 스케일 불변).
      ② M4·M5 에 **같은 밀도를 강제**해 작동점 차이를 제거하므로 M5−M4 가 표현 차이만 반영한다.
        (서로 다른 개수를 뱉으면 Pk/WD 벌점이 달라져 비교가 불공정해진다.)
    ⚠️ **t\*=3 은 동결 유지** — [보조b]가 그 규칙 그대로다. [주]/[보조a]는 t\* 를 바꾼 게 아니라
    *연속 점수*의 작동점을 앵커 경계율에 맞춘 것이다. 세 결과를 모두 실어 아무것도 숨기지 않는다.

Score = `hi_ontop.dts_scoring.score_dialogues` 직접 호출(재구현 금지). 메인 = Score, AUC 는 보조.
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
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from hi_ontop.factored_slot import AXES, STATE_AXES, FactoredSlot  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

EXP = REPO / "outputs/experiments/m4m5_phase1_anchor"
DATA = EXP / "data"
T_STAR = 3  # 동결
CAVEAT = "앵커 k=1 · S_self 0.659(상한) · 탐색적 — 성능 주장 아님"


# ── 데이터 ───────────────────────────────────────────────────────────────────
class Corpus:
    """memmap 임베딩 + 문서 인덱스. 문서 단위 접근."""

    def __init__(self, split: str):
        self.emb = np.load(DATA / f"{split}_emb.npy", mmap_mode="r")
        ix = np.load(DATA / f"{split}_index.npz", allow_pickle=True)
        self.doc_ids = [str(x) for x in ix["doc_ids"]]
        self.so = ix["sent_offsets"]
        self.to = ix["trans_offsets"]
        self.resets = ix["resets"] if "resets" in ix else None
        self.states = ix["states"] if "states" in ix else None
        self.gold = ix["gold"] if "gold" in ix else None
        self.n_docs = len(self.doc_ids)

    def lengths(self) -> np.ndarray:
        return (self.so[1:] - self.so[:-1]).astype(np.int64)


def make_batch(c: Corpus, idx: list[int], dev: str, with_anchor: bool):
    """길이 내림차순 문서 묶음 -> 패딩 배치. 패딩은 항상 뒤쪽."""
    idx = sorted(idx, key=lambda i: -(c.so[i + 1] - c.so[i]))
    T = int(max(c.so[i + 1] - c.so[i] for i in idx))
    B = len(idx)
    z = torch.zeros(B, T, 768, device=dev)
    sm = torch.zeros(B, T, device=dev)
    tm = torch.zeros(B, T - 1, device=dev)
    rs = torch.zeros(B, T - 1, 5, dtype=torch.long, device=dev)
    st = torch.zeros(B, T, len(STATE_AXES), dtype=torch.long, device=dev)
    for b, i in enumerate(idx):
        s0, s1 = int(c.so[i]), int(c.so[i + 1])
        n = s1 - s0
        z[b, :n] = torch.from_numpy(np.asarray(c.emb[s0:s1], dtype=np.float32)).to(dev)
        sm[b, :n] = 1.0
        tm[b, :n - 1] = 1.0
        if with_anchor:
            t0, t1 = int(c.to[i]), int(c.to[i + 1])
            rs[b, :n - 1] = torch.from_numpy(c.resets[t0:t1].astype(np.int64)).to(dev)
            st[b, :n] = torch.from_numpy(c.states[s0:s1].astype(np.int64)).to(dev)
    return z, rs, st, sm, tm, idx


def batches(order: list[int], c: Corpus, bs: int) -> list[list[int]]:
    """길이순 정렬 후 연속 묶음 = 패딩 최소화(bucketing)."""
    srt = sorted(order, key=lambda i: -(c.so[i + 1] - c.so[i]))
    return [srt[i:i + bs] for i in range(0, len(srt), bs)]


# ── 평가 ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def axis_probs(model, c: Corpus, idx: list[int], dev: str, bs: int, with_anchor: bool,
               want_err: bool = False):
    """문서별 축 확률 [n_trans,5] 리스트. want_err 면 (확률, 예측오차) 튜플."""
    model.eval()
    out: dict[int, np.ndarray] = {}
    err: dict[int, np.ndarray] = {}
    for chunk in batches(idx, c, bs):
        z, rs, st, sm, tm, order = make_batch(c, chunk, dev, with_anchor)
        o = model.forward_batched(z, rs, st, sm, tm)
        P = torch.stack([torch.sigmoid(o["change_logits"][a]) for a in AXES], -1)  # [B,T-1,5]
        E = o.get("pred_err")
        for b, i in enumerate(order):
            n = int(c.so[i + 1] - c.so[i])
            out[i] = P[b, :n - 1].float().cpu().numpy()
            err[i] = E[b, :n - 1].float().cpu().numpy() if E is not None else None
    P_list = [out[i] for i in idx]
    return (P_list, [err[i] for i in idx]) if want_err else P_list


def fit_axis_thresholds(probs: list[np.ndarray], target_rate: np.ndarray) -> np.ndarray:
    """[보조] 축별로 앵커 발화율과 같은 비율이 되는 분위수 임계 (rate-matching, gold 미사용)."""
    P = np.concatenate(probs)
    return np.asarray([np.quantile(P[:, k], 1.0 - target_rate[k]) for k in range(5)])


def fit_rate_threshold(probs: list[np.ndarray], target_bnd_rate: float) -> float:
    """[주] 연속 점수 s=ΣP 의 작동점 — val 예측 경계율 = 앵커 경계율이 되는 분위수."""
    s = np.concatenate(probs).sum(1)
    return float(np.quantile(s, 1.0 - target_bnd_rate))


def dev3_scores(model, dev3: Corpus, thr_axis: np.ndarray, tau: float, rate: float,
                dev: str) -> dict:
    """dev-3 deployable Score(메인) + AUC(보조). gold 는 채점에만 쓴다.

    두 작동점을 모두 산출: `primary` = boundary-rate matching(s ≥ τ), `m3_faithful` = 축별
    이진화 후 count ≥ t*=3. AUC 는 연속 점수 s 기준(작동점 무관, 순위 진단).
    """
    idx = list(range(dev3.n_docs))
    probs, errs = axis_probs(model, dev3, idx, dev, bs=4, with_anchor=False, want_err=True)
    has_err = errs[0] is not None
    rules: dict[str, list] = {"primary": [], "global_tau": [], "m3_faithful": []}
    if has_err:
        rules["pred_err"] = []   # SEM식: 예측오차가 큰 전이를 경계로 (①기대 -> ②오차의 직접 검정)
    ys = []
    per_doc: dict[str, dict] = {}
    auc_y, auc_s = [], []
    for i, pr in zip(idx, probs):
        t0, t1 = int(dev3.to[i]), int(dev3.to[i + 1])
        g = dev3.gold[t0:t1].astype(np.int8)
        s = pr.sum(1)
        # [주] 문서 내 상위 rate 분위수 (개수는 문서가 스스로 결정, 밀도만 앵커에서 동결)
        b_pri = (s >= np.quantile(s, 1.0 - rate)).astype(np.int8)
        b_tau = (s >= tau).astype(np.int8)
        b_m3 = ((pr >= thr_axis[None, :]).sum(1) >= T_STAR).astype(np.int8)
        # 끝-turn 규약: 문장 수 = 전이 수 + 1, 마지막 문장 라벨 0
        ys.append(np.concatenate([g, [0]]).tolist())
        pairs = [("primary", b_pri), ("global_tau", b_tau), ("m3_faithful", b_m3)]
        if has_err:
            e = errs[idx.index(i)]
            pairs.append(("pred_err", (e >= np.quantile(e, 1.0 - rate)).astype(np.int8)))
        for rule, b in pairs:
            rules[rule].append(np.concatenate([b, [0]]).tolist())
        auc_y.append(g)
        auc_s.append(s)
        per_doc[dev3.doc_ids[i]] = {
            "n_gold": int(g.sum()), "n_trans": len(g),
            "s_mean": float(s.mean()), "s_p95": float(np.quantile(s, 0.95)),
            "auc": float(roc_auc_score(g, s)) if 0 < g.sum() < len(g) else float("nan"),
            **{rule: {"score": score_dialogues([ys[-1]], [rules[rule][-1]])["score"],
                      "n_pred": int(rules[rule][-1][:-1].count(1) if isinstance(
                          rules[rule][-1], list) else 0)}
               for rule in rules},
        }
    out: dict = {"auc_pooled": float(roc_auc_score(np.concatenate(auc_y), np.concatenate(auc_s))),
                 "per_doc": per_doc}
    for rule, preds in rules.items():
        out[rule] = {
            "pooled_score": score_dialogues(ys, preds)["score"],
            "mean_doc_score": float(np.mean([per_doc[d][rule]["score"] for d in per_doc])),
            "n_pred_total": int(sum(sum(p) for p in preds)),
        }
    return out


@torch.no_grad()
def val_auprc(model, c: Corpus, idx: list[int], dev: str, bs: int) -> dict:
    probs = axis_probs(model, c, idx, dev, bs, with_anchor=True)
    P = np.concatenate(probs)
    Y = np.concatenate([c.resets[int(c.to[i]):int(c.to[i + 1])] for i in idx])
    return {a: (float(average_precision_score(Y[:, k], P[:, k]))
                if 0 < Y[:, k].sum() < len(Y) else float("nan")) for k, a in enumerate(AXES)}


# ── 학습 ─────────────────────────────────────────────────────────────────────
def train_one(name: str, mono: bool, dyn: str | None, tr: list[int], va: list[int], c: Corpus,
              dev3: Corpus, args, pw: list[float], dev: str) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = FactoredSlot(slot_dim=args.slot_dim, mono=mono, pos_weight=pw,
                         dynamics=dyn, pred_weight=args.pred_weight).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    for ep in range(args.epochs):
        model.train()
        order = list(tr)
        rng.shuffle(order)
        tot, nb = 0.0, 0
        for chunk in batches(order, c, args.batch):
            z, rs, st, sm, tm, _ = make_batch(c, chunk, dev, True)
            out = model.forward_batched(z, rs, st, sm, tm)
            opt.zero_grad()
            out["loss"].backward()
            opt.step()
            tot += float(out["loss"].detach())
            nb += 1
        if (ep + 1) % args.log_every == 0 or ep == args.epochs - 1:
            ap = val_auprc(model, c, va, dev, args.batch)
            print(f"[{name} seed{args.seed}] ep{ep+1:3d} loss {tot/max(nb,1):.4f} | val AUPRC " +
                  " ".join(f"{a[:5]} {ap[a]:.3f}" for a in AXES), flush=True)
    # 작동점: phase1 val 에서만 정한다 (dev-3 미사용 -> oracle 누출 없음)
    vr = c.resets[np.concatenate([np.arange(int(c.to[i]), int(c.to[i + 1])) for i in va])]
    base = vr.mean(0)                               # 축별 앵커 발화율
    anchor_bnd_rate = float((vr.sum(1) >= T_STAR).mean())   # 앵커 경계율 (≈0.065)
    vp = axis_probs(model, c, va, dev, args.batch, True)
    thr = fit_axis_thresholds(vp, base)             # [보조] M3 충실판
    tau = fit_rate_threshold(vp, anchor_bnd_rate)   # [주]  boundary-rate matching
    res = {"val_auprc": val_auprc(model, c, va, dev, args.batch),
           "axis_thresholds": thr.tolist(), "tau": tau,
           "val_score_mean": float(np.concatenate(vp).sum(1).mean()),
           "anchor_base_rate": base.tolist(), "anchor_boundary_rate": anchor_bnd_rate,
           "dev3": dev3_scores(model, dev3, thr, tau, anchor_bnd_rate, dev)}
    torch.save({"state": model.state_dict(), "mono": mono, "dynamics": dyn, "args": vars(args),
                "pos_weight": pw, "axis_thresholds": thr.tolist(), "tau": tau},
               EXP / f"{name}{args.ns}_seed{args.seed}.pt")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--slot-dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--limit-docs", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--split", default="phase1")
    ap.add_argument("--dynamics", default="per_slot", choices=["none", "per_slot", "shared"],
                    help="registry §3 dynamics 열. M5=per_slot · M6=shared · none=Phase2 anchor-only")
    ap.add_argument("--pred-weight", type=float, default=1.0)
    ap.add_argument("--models", default="M5,M4", help="쉼표 구분. M6 = factored + shared dynamics")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    EXP.mkdir(parents=True, exist_ok=True)

    c = Corpus(args.split)
    dev3 = Corpus("dev3")
    n = c.n_docs if not args.limit_docs else min(args.limit_docs, c.n_docs)
    rng = np.random.default_rng(20260728)  # 분할 seed 고정 = 모델·seed 간 동일 분할
    perm = rng.permutation(n)
    n_val = max(1, int(n * args.val_frac))
    va, tr = perm[:n_val].tolist(), perm[n_val:].tolist()
    print(f"[data] {args.split}: train {len(tr):,} docs / val {len(va):,} docs "
          f"({c.so[n]:,} sents) · dev3 {dev3.n_docs} docs", flush=True)

    R = c.resets[:int(c.to[n])]
    pos = np.clip(R.mean(0), 1e-3, 1 - 1e-3)
    pw = np.clip((1 - pos) / pos, 1, 30).tolist()
    print(f"[data] 앵커 발화율 {np.round(pos,4).tolist()} · pos_weight {[round(x,1) for x in pw]}",
          flush=True)

    dyn = None if args.dynamics == "none" else args.dynamics
    # 🔒 산출 네임스페이스: split·dynamics 가 다르면 파일이 절대 겹치지 않는다.
    #    (2026-07-28 사고: --split phase1_smoke 실행이 본 산출 results_seed0.json /
    #     M5_seed0.pt 를 덮어썼다. 파일명이 seed 로만 구분됐던 탓.)
    args.ns = ("" if args.split == "phase1" else f"_{args.split}") + \
              ("" if dyn is None else f"_{dyn}")
    # registry §3: M5 = factored + 축별 dynamics · M4 = monolithic + dynamics 1개 · M6 = factored + 공유
    SPEC = {"M5": (False, dyn), "M4": (True, dyn),
            "M6": (False, "shared" if dyn else None)}
    results = {}
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        mono, d = SPEC[name]
        results[name] = train_one(name, mono, d, tr, va, c, dev3, args, pw, dev)

    d5, d4 = results["M5"]["dev3"], results["M4"]["dev3"]  # M5/M4 는 항상 포함
    results["_meta"] = {"caveat": CAVEAT, "t_star": T_STAR, "seed": args.seed,
                        "delta_M5_minus_M4": {
                            r: d5[r]["pooled_score"] - d4[r]["pooled_score"]
                            for r in ("primary", "pred_err", "global_tau", "m3_faithful")
                            if r in d5 and r in d4},
                        "delta_auc": d5["auc_pooled"] - d4["auc_pooled"]}
    out = EXP / f"results{args.ns}_seed{args.seed}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"\n=== dev-3 (⚠️ {CAVEAT}) — seed {args.seed} ===")
    for rule, label in (("primary", "[주] per-doc percentile (문서 내 상위 앵커경계율)"),
                        ("pred_err", "[SEM식] 예측오차 상위 (①기대→②오차 직접 검정)"),
                        ("global_tau", "[보조a] global τ (phase1 val 분위수)"),
                        ("m3_faithful", "[보조b] M3 충실판 (축별 + count≥3)")):
        if rule not in d5:
            continue
        print(f"\n{label}")
        print(f"  {'':8s} {'pooled Score':>13s} {'mean-doc':>10s} {'예측경계':>9s}")
        for k in ("M5", "M4"):
            r = results[k]["dev3"][rule]
            print(f"  {k:8s} {r['pooled_score']:>13.4f} {r['mean_doc_score']:>10.4f} "
                  f"{r['n_pred_total']:>9d}")
        d = d5[rule]["pooled_score"] - d4[rule]["pooled_score"]
        print(f"  {'M5−M4':8s} {d:>13.4f}   <- 부호·방향만 (가설 지지 주장 아님)")
    print(f"\nAUC(보조, 연속점수 순위): M5 {d5['auc_pooled']:.4f}  M4 {d4['auc_pooled']:.4f}  "
          f"Δ {d5['auc_pooled'] - d4['auc_pooled']:+.4f}")
    print(f"\n{'doc':9s} {'gold':>5s} | {'M5 주 Score/pred':>19s} | {'M4 주 Score/pred':>19s}")
    for doc in d5["per_doc"]:
        a, b = d5["per_doc"][doc], d4["per_doc"][doc]
        print(f"{doc:9s} {a['n_gold']:>5d} | {a['primary']['score']:>13.4f}/"
              f"{a['primary']['n_pred']:<4d} | {b['primary']['score']:>13.4f}/"
              f"{b['primary']['n_pred']:<4d}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

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
from hi_ontop.factored_slot import ALPHA, AXES, STATE_AXES, FactoredSlot  # noqa: E402
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
        # D-B35: 축별 태거 피처 [N_sent, 40] (있을 때만). extract_axis_features.py 산출.
        fp = DATA / f"{split}_axisfeat.npy"
        self.feat = np.load(fp, mmap_mode="r") if fp.exists() else None
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


def make_batch(c: Corpus, idx: list[int], dev: str, with_anchor: bool, want_feat: bool = False):
    """길이 내림차순 문서 묶음 -> 패딩 배치. 패딩은 항상 뒤쪽.

    `want_feat` 면 축별 태거 피처 `ft [B,T,5,8]` 을 함께 싣는다(D-B35). 패딩 구간은 0 이며
    손실은 `sent_mask`/`trans_mask` 로 이미 배제되므로 상태 오염이 없다.
    """
    idx = sorted(idx, key=lambda i: -(c.so[i + 1] - c.so[i]))
    T = int(max(c.so[i + 1] - c.so[i] for i in idx))
    B = len(idx)
    z = torch.zeros(B, T, c.emb.shape[1], device=dev)
    ft = torch.zeros(B, T, 5, 8, device=dev) if want_feat else None
    sm = torch.zeros(B, T, device=dev)
    tm = torch.zeros(B, T - 1, device=dev)
    rs = torch.zeros(B, T - 1, 5, dtype=torch.long, device=dev)
    st = torch.zeros(B, T, len(STATE_AXES), dtype=torch.long, device=dev)
    for b, i in enumerate(idx):
        s0, s1 = int(c.so[i]), int(c.so[i + 1])
        n = s1 - s0
        z[b, :n] = torch.from_numpy(np.asarray(c.emb[s0:s1], dtype=np.float32)).to(dev)
        if ft is not None:
            if c.feat is None:
                raise SystemExit(f"🔴 {DATA} 에 축별 태거 피처가 없다 — "
                                 "extract_axis_features.py 를 먼저 돌린다")
            ft[b, :n] = torch.from_numpy(
                np.asarray(c.feat[s0:s1], dtype=np.float32).reshape(n, 5, 8)).to(dev)
        sm[b, :n] = 1.0
        tm[b, :n - 1] = 1.0
        if with_anchor:
            t0, t1 = int(c.to[i]), int(c.to[i + 1])
            rs[b, :n - 1] = torch.from_numpy(c.resets[t0:t1].astype(np.int64)).to(dev)
            st[b, :n] = torch.from_numpy(c.states[s0:s1].astype(np.int64)).to(dev)
    return z, rs, st, sm, tm, idx, ft      # ft 는 want_feat=False 면 None


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
        z, rs, st, sm, tm, order, ft = make_batch(
            c, chunk, dev, with_anchor, want_feat=model.tagger is not None)
        if getattr(model, "slot_reset", None) is not None:
            # 🔴 추론에는 앵커 경계가 없다 — 자기 예측 경계로 슬롯 메모리를 리셋하며 전진.
            P, E = model.change_probs_slot_rollout(z, feat=ft), None
        elif getattr(model, "psi", None) is not None:
            # 🔴 추론에는 앵커 경계가 없다 — 자기 예측 경계로 ψ 를 리셋하며 전진(rollout).
            #    앵커 경계를 추론에 쓰면 **누출**이다.
            P, E = model.change_probs_rollout(z, feat=ft), None
        else:
            o = model.forward_batched(z, rs, st, sm, tm, feat=ft)
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


def anchor_oof_score(model, c: Corpus, va: list[int], dev: str, bs: int,
                     groups: list[str] | None = None) -> dict:
    """🔒 **D-B29 단일 선택 지표** — 앵커 val 의사-gold 대비 5-fold OOF deployable Score.

    왜 이것 하나인가: dev-3 는 3편뿐이라 변형을 여러 개 비교하면 selection-on-dev 오염이
    확실하다(D-M5 재발). 그래서 **구조·HP·임계를 전부 앵커 val 에서** 고르고, dev-3 는 마지막에
    동결된 2개만 본다. 지표를 여러 개 두고 dev-3 와 맞는 걸 사후에 고르면 **그것도 누출**이므로
    지표는 이 하나로 고정한다.

    - 의사-gold = `count ≥ t*=3` (앵커 라벨. 사람 경계가 아니다 — 대리 지표임을 잊지 말 것).
    - 5-fold: 임계 규칙을 **나머지 4-fold 에서 고르고** held-out fold 에 적용 → 임계 선택조차
      out-of-fold 다.
    - 규칙 후보는 기본 분절기와 같은 족(고정 분위수 · otsu · offline/online μ+cσ). 축별 임계 5개를
      따로 탐색하지 않는다(D-B29: 합산 후 단일 규칙만).
    """
    from hi_ontop.hi_ontop_v2 import adaptive_boundaries
    probs = axis_probs(model, c, va, dev, bs, with_anchor=False)
    gold = [(c.resets[int(c.to[i]):int(c.to[i + 1])].sum(1) >= T_STAR).astype(np.int8) for i in va]
    rate = float(np.concatenate(gold).mean())

    def rules(s: np.ndarray) -> dict[str, np.ndarray]:
        seq = np.concatenate([[0.0], s]).tolist()
        r = {"고정분위수": (s >= np.quantile(s, 1 - rate)).astype(np.int8),
             "otsu": np.asarray(adaptive_boundaries(seq, mode="otsu"), np.int8)[:len(s)]}
        for cv in (0.5, 1.0, 1.5, 2.0):
            for md, tg in (("offline", "offline"), ("online_nf", "online")):
                r[f"{tg} μ+{cv}σ"] = np.asarray(
                    adaptive_boundaries(seq, c=cv, mode=md), np.int8)[:len(s)]
        return r

    S = [p.sum(1) for p in probs]
    R = [rules(s) for s in S]
    names = list(R[0])
    n_fold = min(5, len(va))                          # 소규모 스모크에서 빈 fold 방지
    if groups is not None:
        # D-B62: 책 단위 분할일 때는 fold 도 책 단위. 책 수가 5 미만이면 fold 수를 줄여
        # 빈 fold 를 막는다(빈 fold -> score_dialogues 가 빈 리스트에서 assert).
        n_fold = min(n_fold, len(set(groups)))
    if n_fold < 2:
        return {"J_anchor": float("nan"), "per_fold": [], "picked_rule": [],
                "pseudo_gold_rate": rate, "note": f"val {len(va)}편 — OOF 불가"}
    if groups is None:
        fold = np.arange(len(va)) % n_fold            # 고정 분할 (문서 순서 고정 -> 재현 가능)
    else:
        # D-B62: 책 단위 분할일 때는 **fold 도 책 단위**로 나눈다. 안 그러면 같은 책의 청크가
        # 서로 다른 fold 에 흩어져 임계 규칙 선택 단계에서 다시 누수된다.
        keys = list(dict.fromkeys(groups))
        gmap = {k: i % n_fold for i, k in enumerate(keys)}  # 책 단위 라운드로빈
        fold = np.asarray([gmap[g] for g in groups])
    oof, picked = [], []
    for f in range(n_fold):
        tr_i = np.flatnonzero(fold != f)
        te_i = np.flatnonzero(fold == f)
        best, best_v = names[0], -1.0
        for nm in names:                              # 임계 선택 = in-fold(나머지 4-fold)에서만
            y = [np.concatenate([gold[i], [0]]).tolist() for i in tr_i]
            p = [np.concatenate([R[i][nm], [0]]).tolist() for i in tr_i]
            v = score_dialogues(y, p)["score"]
            if v > best_v:
                best, best_v = nm, v
        y = [np.concatenate([gold[i], [0]]).tolist() for i in te_i]
        p = [np.concatenate([R[i][best], [0]]).tolist() for i in te_i]
        oof.append(score_dialogues(y, p)["score"])
        picked.append(best)
    return {"J_anchor": float(np.mean(oof)), "per_fold": [float(x) for x in oof],
            "picked_rule": picked, "pseudo_gold_rate": rate}


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
    # 퇴화 하한 — 경계를 하나도 예측하지 않으면 Pk=WD 라 Score 가 0.5(1-Pk) 로 붕괴한다.
    # 어떤 규칙의 Score 도 이 값을 못 넘으면 그 데이터에서 그 규칙은 의미가 없다.
    zero = [[0] * len(y) for y in ys]
    out["no_boundary_floor"] = score_dialogues(ys, zero)["score"]
    n_gold_all = int(sum(sum(y[:-1]) for y in ys))
    n_trans_all = int(sum(len(y) - 1 for y in ys))
    out["gold_density"] = n_gold_all / max(n_trans_all, 1)
    for rule, preds in rules.items():
        n_pred = int(sum(sum(p[:-1]) for p in preds))
        # 🔒 예측 위치에서의 정밀도 / 전체 경계 기저밀도. 전 문서 전이를 이어 붙여 계산한다.
        hit = int(sum(int(a) * int(b) for y, p in zip(ys, preds) for a, b in zip(y[:-1], p[:-1])))
        prec = hit / n_pred if n_pred else float("nan")
        out[rule] = {
            "pooled_score": score_dialogues(ys, preds)["score"],
            "mean_doc_score": float(np.mean([per_doc[d][rule]["score"] for d in per_doc])),
            "n_pred_total": n_pred,
            # BOR: 예측 경계 수 / 정답 경계 수. >1 과분절, <1 과소분절 (척도 불변)
            "bor": (n_pred / n_gold_all) if n_gold_all else float("nan"),
            # Lift L1 = P(y=1|ŷ=1) / P(y=1). 1=무작위 동등, <1=역신호
            "lift": (prec / out["gold_density"]) if (n_pred and out["gold_density"] > 0)
                    else float("nan"),
            "precision_at_pred": prec,
        }
    return out


@torch.no_grad()
def val_auprc(model, c: Corpus, idx: list[int], dev: str, bs: int) -> dict:
    probs = axis_probs(model, c, idx, dev, bs, with_anchor=True)
    P = np.concatenate(probs)
    Y = np.concatenate([c.resets[int(c.to[i]):int(c.to[i + 1])] for i in idx])
    return {a: (float(average_precision_score(Y[:, k], P[:, k]))
                if 0 < Y[:, k].sum() < len(Y) else float("nan")) for k, a in enumerate(AXES)}


@torch.no_grad()
def val_loss(model, c: Corpus, idx: list[int], dev: str, bs: int) -> float:
    """val 손실 — **학습과 같은 목적함수**. 조기종료·모델선택은 이것으로 한다.

    앵커 val 의 의사-gold(count>=t*) 분절 Score(J_anchor)는 사람 경계가 아니라 라벨러 산출이므로
    "모델이 만든 라벨로 모델을 채점"하는 성격이다. 분절 품질 판정은 평가셋(dev-3)에서 한다.
    """
    model.eval()
    tot, nb = 0.0, 0
    for chunk in batches(list(idx), c, bs):
        z, rs, st, sm, tm, _, ft = make_batch(c, chunk, dev, True,
                                              want_feat=model.tagger is not None)
        pb = (rs.sum(-1) >= T_STAR).float() if model.psi is not None else None
        sb = (rs.sum(-1) >= T_STAR).float() if model.slot_reset is not None else None
        out = model.forward_batched(z, rs, st, sm, tm, psi_bnd=pb, feat=ft, slot_bnd=sb)
        tot += float(out["loss"]); nb += 1
    model.train()
    return tot / max(nb, 1)


# ── 학습 ─────────────────────────────────────────────────────────────────────
def train_one(name: str, mono: bool, dyn: str | None, tr: list[int], va: list[int], c: Corpus,
              dev3: Corpus, args, pw: list[float], dev: str) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = FactoredSlot(z_dim=c.emb.shape[1], slot_dim=args.slot_dim, mono=mono, pos_weight=pw,
                         alpha=getattr(args, "_alpha", ALPHA),
                         d1=(args.input_path == "axis_extractor"),
                         shared_dim=(args.shared_proj_dim or None),
                         tagger=(None if args.tagger == "none" else args.tagger),
                         tagger_axes=args.tagger_axes,
                         block_readout=args.block_readout,
                         dynamics=dyn, pred_weight=args.pred_weight,
                         pred_space=args.pred_space, head_feat=args.head_feat,
                         slot_reset=(None if args.slot_reset == "none" else args.slot_reset),
                         slot_reset_dropout=args.slot_reset_dropout,
                         psi=(None if args.psi == "none" else args.psi),
                         psi_beta=args.psi_beta, psi_dropout=args.psi_dropout,
                         bottleneck=args.bottleneck, dropout=args.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    hist: list[dict] = []
    best_vl, best_ep, bad, best_state = float("inf"), 0, 0, None
    for ep in range(args.epochs):
        model.train()
        order = list(tr)
        rng.shuffle(order)
        tot, nb = 0.0, 0
        for chunk in batches(order, c, args.batch):
            z, rs, st, sm, tm, _, ft = make_batch(
                c, chunk, dev, True, want_feat=model.tagger is not None)
            # ψₑ 학습 리셋 = 앵커 경계(count ≥ t*). 모델 안에서 teacher reset dropout 이 걸린다.
            pb = (rs.sum(-1) >= T_STAR).float() if model.psi is not None else None
            sb = (rs.sum(-1) >= T_STAR).float() if model.slot_reset is not None else None
            out = model.forward_batched(z, rs, st, sm, tm, psi_bnd=pb, feat=ft, slot_bnd=sb)
            opt.zero_grad()
            out["loss"].backward()
            opt.step()
            tot += float(out["loss"].detach())
            nb += 1
        vl = val_loss(model, c, va, dev, args.batch)
        hist.append({"epoch": ep + 1, "train_loss": tot / max(nb, 1), "val_loss": vl})
        if (ep + 1) % args.log_every == 0 or ep == args.epochs - 1:
            ap = val_auprc(model, c, va, dev, args.batch)
            print(f"[{name} seed{args.seed}] ep{ep+1:3d} train {tot/max(nb,1):.4f} "
                  f"val {vl:.4f} | val AUPRC " +
                  " ".join(f"{a[:5]} {ap[a]:.3f}" for a in AXES), flush=True)
        # 중간 스냅숏 — 최종본과 **같은 형식**으로 남긴다(작동점까지 그 시점 val 에서 다시 계산).
        if args.ckpt_every and (ep + 1) % args.ckpt_every == 0:
            _vr = c.resets[np.concatenate([np.arange(int(c.to[i]), int(c.to[i + 1])) for i in va])]
            _rate = float((_vr.sum(1) >= T_STAR).mean())
            _vp = axis_probs(model, c, va, dev, args.batch, True)
            torch.save({"state": model.state_dict(), "mono": mono, "dynamics": dyn,
                        "args": vars(args), "pos_weight": pw,
                        "axis_thresholds": fit_axis_thresholds(_vp, _vr.mean(0)).tolist(),
                        "tau": fit_rate_threshold(_vp, _rate),
                        "epoch": ep + 1, "val_loss": vl, "anchor_boundary_rate": _rate},
                       EXP / f"{name}{args.ns}_ep{ep+1:03d}_seed{args.seed}.pt")
            print(f"[{name} seed{args.seed}] 💾 스냅숏 ep{ep+1} (val {vl:.4f})", flush=True)

        # 조기종료 — val **손실** 기준. 분절 Score(의사-gold)로 고르면 라벨러를 재학습하는 셈이다.
        if vl < best_vl - args.min_delta:
            best_vl, best_ep, bad = vl, ep + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if args.patience > 0 and bad >= args.patience:
                print(f"[{name} seed{args.seed}] ⏹ 조기종료 ep{ep+1} "
                      f"(best ep{best_ep} val {best_vl:.4f}, patience {args.patience})", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)   # 최적 epoch 가중치로 복원 후 평가·저장
    # 작동점: phase1 val 에서만 정한다 (dev-3 미사용 -> oracle 누출 없음)
    vr = c.resets[np.concatenate([np.arange(int(c.to[i]), int(c.to[i + 1])) for i in va])]
    base = vr.mean(0)                               # 축별 앵커 발화율
    anchor_bnd_rate = float((vr.sum(1) >= T_STAR).mean())   # 앵커 경계율 (≈0.065)
    vp = axis_probs(model, c, va, dev, args.batch, True)
    thr = fit_axis_thresholds(vp, base)             # [보조] M3 충실판
    tau = fit_rate_threshold(vp, anchor_bnd_rate)   # [주]  boundary-rate matching
    res = {"val_loss_best": best_vl, "val_loss_best_epoch": best_ep, "loss_history": hist,
           "val_auprc": val_auprc(model, c, va, dev, args.batch),
           "axis_thresholds": thr.tolist(), "tau": tau,
           "val_score_mean": float(np.concatenate(vp).sum(1).mean()),
           "anchor_base_rate": base.tolist(), "anchor_boundary_rate": anchor_bnd_rate,
           # 🔒 D-B29 단일 선택 지표. 탐색 단계에서는 **이것만** 본다.
           "anchor_oof": anchor_oof_score(model, c, va, dev, args.batch,
                                          groups=getattr(args, "_va_groups", None))}
    print(f"[{name} seed{args.seed}] 🔒 J_anchor = {res['anchor_oof']['J_anchor']:.4f}"
          f"  (folds {[round(x,3) for x in res['anchor_oof']['per_fold']]})", flush=True)
    # 🔴 D-B29: 탐색 중에는 dev-3 를 열지 않는다. 동결된 최종 2개만 --dev3 로 본다.
    if args.dev3:
        res["dev3"] = dev3_scores(model, dev3, thr, tau, anchor_bnd_rate, dev)
    torch.save({"state": model.state_dict(), "mono": mono, "dynamics": dyn, "args": vars(args),
                "pos_weight": pw, "axis_thresholds": thr.tolist(), "tau": tau},
               EXP / f"{name}{args.ns}_seed{args.seed}.pt")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    # 🔒 기본값 = 2026-08-04 확정 조건. 인자 없이 실행하면 그대로 재현된다.
    #   앵커  = phase1__personkey_i2 — 인물키 상한 20 + ScANT 격리 재빌드 코퍼스에
    #           **의도 축 I2**(carry-forward 미적용 + 채택만) 적용. 11,836편 · 3,013,224전이 ·
    #           경계율 0.0660. 다운스트림(EM-LLM `m5_default`)이 I2 본을 쓰므로 조건을 맞춘다.
    #   분할  = 작가 단위(작가·책 누수 0%%. none 은 val 의 96.1%% 가 train 에 같은 작가를 둔다)
    #   학습  = epochs 100 + val 손실 조기종료 patience 3
    #           (8 에서는 조기종료가 한 번도 안 걸렸고 40 도 seed 하나가 상한에 눌렸다)
    #   스냅숏 = 10 epoch 마다 완결 체크포인트(epoch-성능 곡선용)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=3,
                    help="조기종료 — val 손실이 이 횟수만큼 개선 없으면 중단(0=끄기). "
                         "최적 epoch 가중치로 복원한다.")
    ap.add_argument("--min-delta", type=float, default=1e-4, help="개선으로 인정할 최소 val 손실 감소")
    ap.add_argument("--ckpt-every", type=int, default=10,
                    help="N epoch 마다 **완결 체크포인트**를 남긴다(0=끄기). 파일명에 _ep{N} 이 붙고 "
                         "최종본과 같은 형식(작동점 thr·tau 포함)이라 그대로 평가·다운스트림에 꽂힌다. "
                         "epoch-성능 곡선을 그리려면 조기종료를 끄고(--patience 0) 쓴다.")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--slot-dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--limit-docs", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--split", default="phase1__personkey_i2")
    ap.add_argument("--dynamics", default="per_slot", choices=["none", "per_slot", "shared"],
                    help="registry §3 dynamics 열. M5=per_slot · M6=shared · none=Phase2 anchor-only")
    ap.add_argument("--pred-weight", type=float, default=1.0)
    # ── D-B29 배치 ────────────────────────────────────────────────────────────
    ap.add_argument("--bottleneck", type=int, default=8, help="H묶음: D1 extractor 폭")
    ap.add_argument("--dropout", type=float, default=0.4, help="H묶음")
    ap.add_argument("--head-feat", default="delta",
                    choices=["delta", "resid_signed", "resid_sq", "pred_centric"],
                    help="P묶음: §4.6 Axis1 predict-then-compare 를 head 입력 특징으로")
    ap.add_argument("--slot-reset", default="none", choices=["none", "hard"],
                    help="D-B63: hard = 경계 다음 문장부터 슬롯 메모리를 key 로 리셋(event switching). "
                         "학습=앵커 경계+teacher dropout · 추론=자기 예측 경계 rollout.")
    ap.add_argument("--slot-reset-dropout", type=float, default=0.3)
    ap.add_argument("--group-split", default="author", choices=["none", "book", "author"],
                    help="D-B62: book = 같은 책(pgXXX)의 청크를 train/val 한쪽에만 둔다. "
                         "author = 같은 **작가**(인물 키)를 한쪽에만 둔다 — 2026-08-03 실측으로 "
                         "none 은 val 의 96.1%%, book 도 55.7%% 가 train 에 같은 작가를 둔다. "
                         "기본 none = 청크 무작위(현행, 기존 산출 재현용).")
    ap.add_argument("--author-map",
                    default=str(REPO / "data/eacl_phase1/pg_shorts_personkey_cap20_scant.jsonl"),
                    help="--group-split author 용 doc_id→저자 매핑 소스(코퍼스 jsonl).")
    ap.add_argument("--psi", default="none", choices=["none", "cummean", "ema"],
                    help="Ψ묶음: §4.3 에피소드 스키마 조건부 dynamics")
    ap.add_argument("--psi-beta", type=float, default=0.9)
    ap.add_argument("--psi-dropout", type=float, default=0.3, help="teacher reset dropout")
    ap.add_argument("--dev3", action="store_true",
                    help="🔴 dev-3 평가. D-B29 탐색 단계에서는 켜지 않는다(selection-on-dev 오염).")
    ap.add_argument("--pred-space", default="obs", choices=["obs", "slot", "both"],
                    help="§4.6 Axis2. slot = slot-native(D3 [v1]) · obs = 768D 의미공간(누출 위험)")
    ap.add_argument("--models", default="M5,M4", help="쉼표 구분. M6 = factored + shared dynamics")
    # ── D-B44 손실 가중치 2×2 (ALPHA × pos_weight) ────────────────────────────
    # 🔒 codex 판정(2026-07-30): 확률을 왜곡하는 것은 pos_weight 뿐이고 ALPHA 는 축별 경사 배분이다.
    #    둘을 분리해 보려면 ALPHA 균등화는 **합을 보존**해야 한다 — 안 그러면 총 손실 스케일(=실효 lr)과
    #    `gamma·l_collapse` 대비 비중이 같이 변해 원인 귀속이 안 된다. 그래서 1.0 이 아니라 0.63.
    ap.add_argument("--ns-suffix", default="",
                    help="산출 네임스페이스 뒤에 붙일 조건 이름. 조건이 ns 에 안 드러나는 재현 런은 "
                         "이걸로 이름을 분리한다(D-B44 덮어쓰기 사고 방지).")
    ap.add_argument("--overwrite-ckpt", action="store_true",
                    help="🔴 기존 체크포인트를 의도적으로 덮어쓴다. 기본은 차단.")
    ap.add_argument("--alpha-mode", default="current", choices=["current", "uniform"],
                    help="current = (1,.5,1,.4,.25) 현행 · uniform = 합보존 균등 ΣALPHA/5 = 0.63")
    ap.add_argument("--pos-weight-mode", default="clip", choices=["clip", "one", "exact"],
                    help="clip = (1-p)/p 를 [1,30] 으로 자름(현행) · one = 전축 1(plain BCE) · "
                         "exact = clip 없는 참 odds (🔴 time 붕괴 시에만 발동하는 rescue 조건)")
    # ── D-B33 입력경로 애블레이션 ──────────────────────────────────────────────
    ap.add_argument("--input-path", default="axis_extractor",
                    choices=["axis_extractor", "shared_proj"],
                    help="axis_extractor = 축별 E_k 라우팅(현행) · shared_proj = 공유 투영 한 "
                         "벡터를 5슬롯 전부에 브로드캐스트(스키마 구조가 슬롯 상태·dynamics 에만)")
    ap.add_argument("--shared-proj-dim", type=int, default=0,
                    help="shared_proj 폭. 0 = slot_dim. 40 = 5×bottleneck(=D1 총 입력폭 매칭)")
    # ── D-B35 §4.6 진짜 D1: 태거가 정의한 축 공간 ─────────────────────────────
    ap.add_argument("--tagger", default="none", choices=["none", "replace", "hybrid"],
                    help="replace = 축 입력을 태거 피처로 **대체**(순수 D1) · "
                         "hybrid = [E_k(z), 태거] concat")
    ap.add_argument("--block-readout", action="store_true",
                    help="D-B36: **하나의 상태 벡터**를 5블록으로 나눠 head k 가 자기 블록만 "
                         "읽는다(SEM 식 구조화 스키마). --models M4 + slot_dim 5의 배수와 함께 쓴다")
    ap.add_argument("--tagger-axes", type=int, default=5, choices=[3, 5],
                    help="3 = protagonist/space/time 만 대체(설계문서 §10: causal/goal 은 "
                         "깨끗한 태거가 없다) · 5 = 전 축")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    EXP.mkdir(parents=True, exist_ok=True)

    c = Corpus(args.split)
    args.z_dim = int(c.emb.shape[1])
    dev3 = Corpus("dev3")
    n = c.n_docs if not args.limit_docs else min(args.limit_docs, c.n_docs)
    rng = np.random.default_rng(20260728)  # 분할 seed 고정 = 모델·seed 간 동일 분할
    n_val = max(1, int(n * args.val_frac))
    if args.group_split == "author":
        # 작가 단위. 인물 키(정규화 이름 + 생몰연도) = 코퍼스 상한·격리와 같은 정의를 쓴다.
        # 같은 작가의 서로 다른 책이 train/val 로 갈리면 문체 누수가 남는다(book 분할의 한계).
        import sys as _sys
        _sys.path.insert(0, str(REPO / "scripts/eacl_phase1"))
        from author_review_list import parse_person  # noqa: E402
        amap: dict[str, tuple] = {}
        for _l in open(args.author_map, encoding="utf-8"):
            _r = json.loads(_l)
            _n, _y, _ = parse_person(_r["author"].split(";")[0])
            amap[_r["doc_id"]] = (_n.split("(")[0].strip().rstrip(",").lower(), _y)
        groups: dict[object, list[int]] = {}
        for i in range(n):
            groups.setdefault(amap.get(c.doc_ids[i], ("__unknown__", c.doc_ids[i])), []).append(i)
        keys = sorted(groups, key=lambda k: (str(k[0]), str(k[1])))
        va = []
        for b in rng.permutation(len(keys)):
            if len(va) >= n_val:
                break
            va.extend(groups[keys[b]])
        vs = set(va)
        tr = [i for i in range(n) if i not in vs]
    elif args.group_split == "book":
        # D-B62: 청크 무작위 분할은 val 의 80.2% 가 "같은 책의 다른 청크"를 train 에 두고 있었다
        # (책 단위 누수). 책(pgXXX)을 통째로 한쪽에만 넣는다.
        books: dict[str, list[int]] = {}
        for i in range(n):
            books.setdefault(c.doc_ids[i].rsplit("_", 1)[0], []).append(i)
        keys = sorted(books)
        va = []
        for b in rng.permutation(len(keys)):
            if len(va) >= n_val:
                break
            va.extend(books[keys[b]])
        vs = set(va)
        tr = [i for i in range(n) if i not in vs]
    else:
        perm = rng.permutation(n)
        va, tr = perm[:n_val].tolist(), perm[n_val:].tolist()
    args._va_groups = ([c.doc_ids[i].rsplit("_", 1)[0] for i in va]
                       if args.group_split in ("book", "author") else None)
    print(f"[data] {args.split}: train {len(tr):,} docs / val {len(va):,} docs "
          f"({c.so[n]:,} sents) · dev3 {dev3.n_docs} docs", flush=True)

    R = c.resets[:int(c.to[n])]
    pos = np.clip(R.mean(0), 1e-3, 1 - 1e-3)
    if args.pos_weight_mode == "clip":
        pw = np.clip((1 - pos) / pos, 1, 30).tolist()      # 현행(W0/W4)
    elif args.pos_weight_mode == "one":
        pw = [1.0] * 5                                     # plain BCE ⇒ 출력이 곧 참 사후확률
    else:
        pw = ((1 - pos) / pos).tolist()                    # exact odds (clip 없음)
    # 🔒 ALPHA 균등화는 **합 보존**(ΣALPHA 불변) — 총 손실 스케일과 gamma 대비 비중을 건드리지 않는다.
    alpha = tuple(ALPHA) if args.alpha_mode == "current" else (sum(ALPHA) / 5,) * 5
    print(f"[data] 앵커 발화율 {np.round(pos,4).tolist()} · pos_weight {[round(x,2) for x in pw]}"
          f" ({args.pos_weight_mode}) · ALPHA {[round(x,3) for x in alpha]}"
          f" (Σ={sum(alpha):.2f}, {args.alpha_mode})", flush=True)
    args._alpha = alpha

    dyn = None if args.dynamics == "none" else args.dynamics
    # 🔒 산출 네임스페이스: split·dynamics 가 다르면 파일이 절대 겹치지 않는다.
    #    (2026-07-28 사고: --split phase1_smoke 실행이 본 산출 results_seed0.json /
    #     M5_seed0.pt 를 덮어썼다. 파일명이 seed 로만 구분됐던 탓.)
    # 🔴 산출 파일명 네임스페이스 — **모든 조건이 이름에 들어가야 한다.**
    #    스모크가 본 산출을 덮어쓴 사고(REPORT §1)와 같은 계열의 재발 방지.
    args.ns = ("" if args.split == "phase1" else f"_{args.split}") + \
              ("" if dyn is None else f"_{dyn}") + \
              ("" if args.pred_space == "obs" else f"_{args.pred_space}") + \
              ("" if args.head_feat == "delta" else f"_{args.head_feat}") + \
              ("" if args.psi == "none" else f"_psi{args.psi}") + \
              ("" if args.slot_reset == "none" else f"_sreset{args.slot_reset}") + \
              ("" if args.group_split == "none" else f"_gs{args.group_split}") + \
              ("" if args.slot_dim == 128 else f"_sd{args.slot_dim}") + \
              ("" if args.bottleneck == 8 else f"_bn{args.bottleneck}") + \
              ("" if args.dropout == 0.4 else f"_do{args.dropout}") + \
              ("" if args.lr == 3e-4 else f"_lr{args.lr:g}") + \
              ("" if args.epochs == 8 else f"_ep{args.epochs}") + \
              ("" if args.input_path == "axis_extractor"
               else f"_sharedinput{args.shared_proj_dim or args.slot_dim}") + \
              ("" if args.tagger == "none" else f"_tag{args.tagger}{args.tagger_axes}") + \
              ("_blocks" if args.block_readout else "") + \
              ("" if args.alpha_mode == "current" else f"_alpha{args.alpha_mode}") + \
              ("" if args.pos_weight_mode == "clip" else f"_pw{args.pos_weight_mode}") + \
              (f"_{args.ns_suffix}" if args.ns_suffix else "")
    # 🔴 모델 태그 — `--models M6` 를 기본 ns 로 돌리면 M5/M4 의 결과 json 을 통째로 덮어쓴다.
    #    (체크포인트는 이름이 달라 안전하지만 results{ns}_seed{n}.json 은 겹친다.)
    mtag = "" if args.models.replace(" ", "") == "M5,M4" else \
        "_" + "-".join(m.strip() for m in args.models.split(",") if m.strip())
    # registry §3: M5 = factored + 축별 dynamics · M4 = monolithic + dynamics 1개 · M6 = factored + 공유
    SPEC = {"M5": (False, dyn), "M4": (True, dyn),
            "M6": (False, "shared" if dyn else None)}
    results = {}
    # 🔴 D-B44 사고 재발 방지: 체크포인트 경로는 `{model}{ns}_seed{n}.pt` 라 **mtag 가 안 붙는다**.
    #    조건이 ns 에 드러나지 않는 실행(= 현행값 재현 런)은 동결 산출을 그대로 덮어쓴다.
    #    2026-07-30 에 W0 무결성 런이 `M5_per_slot_slot_seed0.pt` 를 덮어써 복구한 사고가 있었다.
    #    ⇒ 이미 있는 체크포인트를 건드리려 하면 **멈춘다.** 재현 런은 `--ns-suffix` 로 이름을 분리한다.
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        ck_path = EXP / f"{name}{args.ns}_seed{args.seed}.pt"
        if ck_path.exists() and not args.overwrite_ckpt:
            raise SystemExit(
                f"🔴 덮어쓰기 차단: {ck_path.name} 이 이미 있다.\n"
                f"   조건이 네임스페이스에 안 드러나는 실행이다 — `--ns-suffix <조건이름>` 을 붙여"
                f" 이름을 분리하거나, 의도적 재생성이면 `--overwrite-ckpt` 를 명시하라.")
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        mono, d = SPEC[name]
        results[name] = train_one(name, mono, d, tr, va, c, dev3, args, pw, dev)

    if not args.dev3:
        # D-B29 탐색 모드 — dev-3 를 열지 않았으므로 J_anchor 만 기록하고 끝낸다.
        results["_meta"] = {"caveat": CAVEAT, "t_star": T_STAR, "seed": args.seed,
                            "mode": "D-B29 search (dev-3 unopened)", "ns": args.ns,
                            "J_anchor": {k: v["anchor_oof"]["J_anchor"]
                                         for k, v in results.items() if "anchor_oof" in v}}
        p = EXP / f"results{args.ns}{mtag}_seed{args.seed}.json"
        p.write_text(json.dumps(results, indent=2, default=float))
        print(f"\n[D-B29] dev-3 미개봉. J_anchor = " +
              " · ".join(f"{k} {v:.4f}" for k, v in results["_meta"]["J_anchor"].items()))
        print(f"-> {p}")
        return

    # 🔴 M5/M4 가 **둘 다 있을 때만** Δ 를 낸다. 모델을 쪼개 돌리면(예: M4 만 VESSL)
    #    한쪽이 없어 KeyError 로 죽었다 — 학습·체크포인트는 멀쩡한데 결과 JSON 만 못 썼다(2026-08-04).
    results["_meta"] = {"caveat": CAVEAT, "t_star": T_STAR, "seed": args.seed}
    if "M5" in results and "M4" in results and "dev3" in results["M5"] and "dev3" in results["M4"]:
        d5, d4 = results["M5"]["dev3"], results["M4"]["dev3"]
        results["_meta"]["delta_M5_minus_M4"] = {
            r: d5[r]["pooled_score"] - d4[r]["pooled_score"]
            for r in ("primary", "pred_err", "global_tau", "m3_faithful")
            if r in d5 and r in d4}
        results["_meta"]["delta_auc"] = d5["auc_pooled"] - d4["auc_pooled"]
    else:
        results["_meta"]["note"] = ("단일 모델 실행 — Δ(M5−M4) 는 두 모델을 함께 돌린 실행에서만 낸다. "
                                   f"이 실행 모델 = {[k for k in results if k != '_meta']}")
    out = EXP / f"results{args.ns}{mtag}_seed{args.seed}.json"
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

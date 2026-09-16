#!/usr/bin/env python3
"""대화 도메인 신호조립·임계 작동점 선택 — 사전등록 PREREG.md 대로만 실행한다.

🔴 **앵커/의사-gold 는 쓰지 않는다.** 작동점 선택셋은 DTS dev 3세트(사람 gold)뿐이다
   (`phase0-decisions-and-runs.md:4736` · D-B41 "앵커 선택셋 무효").
🔴 dev 와 **내용이 겹치는** test 문서는 확인셋에서 제외한다(ID 체계가 달라 이름 대조로는 안 잡힌다).

    .venv/bin/python scripts/eacl_phase0/dialogue_operating_point.py          # dev 격자 + 확인
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts/eacl_phase0")]
from hi_ontop.ami_scoring import score_meetings  # noqa: E402
from hi_ontop.hi_ontop_v2 import adaptive_boundaries  # noqa: E402
from psi_dev3_eval import load, row_metrics  # noqa: E402
from train_anchor_m4m5 import Corpus, axis_probs  # noqa: E402
from train_direct_boundary import ANCHOR_R  # noqa: E402  🔒 동결 밀도 사전
from sklearn.metrics import roc_auc_score  # noqa: E402

OUT = REPO / "outputs/experiments/2026-08-01_dialogue_operating_point"
DATA = REPO / "outputs/experiments/m4m5_phase1_anchor/data"
DEV = ["dts_tiage", "dts_dialseg711", "dts_superseg"]          # 🔒 선택셋
TEST = ["dts_full_tiage", "dts_full_dialseg711", "dts_full_superseg"]   # 🔒 확인셋
# 🔒 PREREG §2 — 작동점 선택에 쓰는 유일한 체크포인트
NS_MAIN = "_phase1_i2_per_slot_slot_gsbook_pwone"
NS_ALT = "_phase1_chainq_i2_per_slot_slot_gsbook_pwone"        # 적용만, 재선택 없음


# ── 신호 조립 (문서 내에서만 계산) ───────────────────────────────────────────
def sig_raw(P: np.ndarray) -> np.ndarray:
    return P.sum(1)


def sig_zaxis(P: np.ndarray) -> np.ndarray:
    return ((P - P.mean(0)) / (P.std(0) + 1e-8)).sum(1)


def sig_rankaxis(P: np.ndarray) -> np.ndarray:
    return (np.argsort(np.argsort(P, 0), 0) / max(len(P) - 1, 1)).sum(1)


SIGNALS = {"raw": sig_raw, "zaxis": sig_zaxis, "rankaxis": sig_rankaxis}


# ── 임계 규칙 (동결 구현 호출) ───────────────────────────────────────────────
def cut_pct(s: np.ndarray) -> np.ndarray:
    return (s >= np.quantile(s, 1.0 - ANCHOR_R)).astype(np.int8)


def cut_adapt(s: np.ndarray, cv: float) -> np.ndarray:
    seq = np.concatenate([[0.0], s]).tolist()
    return np.asarray(adaptive_boundaries(seq, c=cv, mode="online_nf", warmup=5), np.int8)[:len(s)]


RULES = {"pct": cut_pct}
RULES.update({f"adapt_c{cv}": (lambda cv: (lambda s: cut_adapt(s, cv)))(cv)
              for cv in (0.5, 1.0, 1.5, 2.0)})
CURRENT = ("raw", "adapt_c1.0")          # 현행 조합
SIMPLE_ORDER = {"raw": 0, "zaxis": 1, "rankaxis": 2}
RULE_ORDER = {"adapt_c1.0": 0, "pct": 1, "adapt_c0.5": 2, "adapt_c1.5": 3, "adapt_c2.0": 4}


def doc_key(split: str) -> dict[bytes, str]:
    """문서 앞 3문장 임베딩 = 내용 지문. ID 체계가 달라도 같은 대화를 잡는다."""
    ix = np.load(DATA / f"{split}_index.npz", allow_pickle=True)
    e = np.load(DATA / f"{split}_emb.npy", mmap_mode="r")
    so = ix["sent_offsets"]
    return {np.asarray(e[int(so[i]):int(so[i]) + 3], dtype=np.float32).tobytes(): str(d)
            for i, d in enumerate(ix["doc_ids"])}


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    # 🔴 확인셋 오염 제거 — dev 와 내용이 겹치는 test 문서 인덱스를 뺀다
    keep: dict[str, list[int]] = {}
    for dv, ts in zip(DEV, TEST):
        dk = set(doc_key(dv))
        tk = doc_key(ts)
        ix = np.load(DATA / f"{ts}_index.npz", allow_pickle=True)
        drop = {v for k, v in tk.items() if k in dk}
        keep[ts] = [i for i, d in enumerate(ix["doc_ids"]) if str(d) not in drop]
        print(f"[확인셋] {ts}: {len(ix['doc_ids'])}편 중 dev 중복 {len(drop)}편 제외 "
              f"→ {len(keep[ts])}편", flush=True)
    print()

    def probs(ns: str, split: str, idx: list[int] | None = None):
        c = Corpus(split)
        ids = idx if idx is not None else list(range(c.n_docs))
        m, _ = load(ns, 0, dev)
        with torch.no_grad():
            P = axis_probs(m, c, ids, dev, bs=8, with_anchor=False)
        G = [c.gold[int(c.to[i]):int(c.to[i + 1])].astype(np.int8) for i in ids]
        del m
        torch.cuda.empty_cache()
        return P, G

    def grid(P, G, ami: bool = False) -> dict:
        n_gold = int(sum(int(g.sum()) for g in G))
        out = {}
        for sn, sf in SIGNALS.items():
            S = [sf(p) for p in P]
            auc = float(roc_auc_score(np.concatenate(G), np.concatenate(S)))
            for rn, rf in RULES.items():
                Pb = [rf(s)[:len(g)] for s, g in zip(S, G)]
                if ami:
                    g2 = [np.concatenate([[0], x]).tolist() for x in G]
                    p2 = [np.concatenate([[0], x]).tolist() for x in Pb]
                    r = {"score": float(score_meetings(g2, p2)["score"]),
                         "bor": sum(int(b.sum()) for b in Pb) / max(n_gold, 1)}
                else:
                    r = row_metrics(Pb, G, n_gold)
                r["auc"] = auc
                out[f"{sn}|{rn}"] = r
        return out

    res: dict[str, dict] = {}
    print("=== 1. dev 격자 (선택셋 — 여기서만 고른다) ===", flush=True)
    for ds in DEV:
        P, G = probs(NS_MAIN, ds)
        res[ds] = grid(P, G)
    combos = list(res[DEV[0]])
    mean_dev = {k: float(np.mean([res[d][k]["score"] for d in DEV])) for k in combos}
    cur = f"{CURRENT[0]}|{CURRENT[1]}"
    print(f"{'조합':22s} {'dev평균':>8s} {'Δ현행':>8s} " + " ".join(f"{d.split('_')[-1]:>11s}" for d in DEV))
    for k in sorted(combos, key=lambda x: -mean_dev[x]):
        wins = sum(res[d][k]["score"] > res[d][cur]["score"] for d in DEV)
        mark = " ← 현행" if k == cur else (f"  ({wins}/3 승)" if mean_dev[k] > mean_dev[cur] else "")
        print(f"{k:22s} {mean_dev[k]:8.4f} {mean_dev[k]-mean_dev[cur]:+8.4f} "
              + " ".join(f"{res[d][k]['score']:11.4f}" for d in DEV) + mark)

    # 🔒 PREREG §4 선택 규칙
    best = max(combos, key=lambda k: mean_dev[k])
    wins = sum(res[d][best]["score"] > res[d][cur]["score"] for d in DEV)
    tie = mean_dev[best] - mean_dev[cur] < 0.01
    if tie or wins < 2:
        why = ("1위와 현행 차이 <0.01 → 동률, 더 단순한 쪽" if tie
               else f"dev 3세트 중 {wins}세트만 개선 (2세트 미달)")
        picked, verdict = cur, f"현행 유지 ({why})"
    else:
        picked, verdict = best, f"채택 후보 (Δ {mean_dev[best]-mean_dev[cur]:+.4f}, {wins}/3 승)"
    print(f"\n🔒 선택: **{picked}** — {verdict}")

    print("\n=== 2. 확인 (test 1회, dev 중복 제외) ===", flush=True)
    conf: dict[str, dict] = {}
    for ds in TEST:
        P, G = probs(NS_MAIN, ds, keep[ds])
        conf[ds] = grid(P, G)
    print(f"{'조합':22s} {'test평균':>9s} " + " ".join(f"{d.split('_')[-1]:>11s}" for d in TEST))
    for k in (cur, picked) if picked != cur else (cur,):
        mt = float(np.mean([conf[d][k]["score"] for d in TEST]))
        print(f"{k:22s} {mt:9.4f} " + " ".join(f"{conf[d][k]['score']:11.4f}" for d in TEST))

    print("\n=== 3. 병기 (판정 미포함) — dev-3 서사 · AMI · chainq 라벨 ===", flush=True)
    side: dict[str, dict] = {}
    P, G = probs(NS_MAIN, "dev3")
    side["dev3"] = grid(P, G)
    P, G = probs(NS_MAIN, "dts_ami")
    side["dts_ami"] = grid(P, G, ami=True)
    for ds in DEV:
        P, G = probs(NS_ALT, ds)
        side[f"{ds}|chainq"] = grid(P, G)
    print(f"{'조합':22s} {'dev3':>8s} {'ami':>8s} {'chainq dev평균':>14s}")
    for k in sorted(combos, key=lambda x: -mean_dev[x])[:6]:
        cq = float(np.mean([side[f"{d}|chainq"][k]["score"] for d in DEV]))
        print(f"{k:22s} {side['dev3'][k]['score']:8.4f} {side['dts_ami'][k]['score']:8.4f} {cq:14.4f}")

    (OUT / "results.json").write_text(json.dumps(
        {"dev": res, "test": conf, "side": side, "mean_dev": mean_dev,
         "picked": picked, "verdict": verdict,
         "kept_docs": {k: len(v) for k, v in keep.items()}},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {OUT / 'results.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""BayesSurprise(B1) 를 서사 6편에 올려 예측 덤프를 만든다 — **온라인**.

## 방법
Kumar et al. 의 Bayesian surprise(연속 두 다음-단어 분포의 KL)를 GPT-2-XL 로 계산하고
(`scripts/eacl_phase0/b1_bayes_surprise.py`, 무수정), 전이 단위 readout 을 적용한다.

**readout = `newsent_max` transient** — 새 문장 안의 최대 transient 서프라이즈.
**임계 = μ + c·σ (running, c=0.75 고정)** — dev=Pieman 에서 argmax-Score 로 동결한 값
(ledger L1102: *"동결(Pieman): M3-S5 t*=3 · M1 c=1.0 · **B1 c=0.75**"*). **재튜닝 없음.**

## 온라인 등급 = zero-lookahead
전이 t 의 판정에 **t 이하의 단어만** 쓴다:
- transient 정규화는 과거 이동평균(drift=0.10)만 본다
- 임계 μ·σ 는 **그 전이까지 관측된 점수**로만 갱신한다
⇒ 미래 문맥 0. `-online` 접미사 없이도 B1 은 원래 인과적이다.

🔴 **onset-max 가 아니다.** codex 초기 권고였던 onset-max 는 구어 서사에서 **역상관**
(AUC 0.35)이었다 — 사건 도입부("And so…")가 예측 가능해 서프라이즈가 낮다.
`newsent_max` 가 dev 에서 최선·최견고였고 그것을 동결했다.

    .venv/bin/python EACL/methods/bayes_surprise/run_narrative.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))

B1 = REPO / "outputs/eacl_phase0/b1_bayes"
PRED = REPO / "outputs/experiments/2026-07-29_baseline_compare/preds"
OUT = REPO / "outputs/experiments/2026-08-04_bayessurprise_narrative"
C_FROZEN = 0.75          # dev=Pieman 동결 (ledger L1102). 재튜닝 금지
WARMUP = 3


def read_scores(doc: str):
    """전이별 **newsent_max transient** — 새 문장 전체의 최대 transient 서프라이즈.

    🔴 `{doc}_b1.jsonl` 의 `b1_n1/n3/n5` 를 **쓰면 안 된다.** 그건 새 문장의 앞 N 단어만 보는
    **onset-max** 계열이고, ledger 가 구어 서사에서 **역상관**(AUC 0.35)이라 기각한 readout 이다
    (*"사건 도입부('And so…')가 예측 가능해 서프라이즈가 낮다"*). 동결된 readout 은
    `newsent_max` 이며 b1.jsonl 에 저장돼 있지 않으므로 **단어 덤프에서 직접 만든다.**

    단어 덤프 `{doc}_words.jsonl` = {doc_id, sent, pos, transient, raw, ...}.
    전이 t(문장 t 앞의 경계)의 점수 = **문장 t 안 모든 단어의 transient 최댓값**.
    """
    f = B1 / f"{doc}_words.jsonl"
    if not f.exists():
        return None
    best: dict[int, float] = {}
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        w = json.loads(line)
        s_i = int(w["sent"])
        v = float(w["transient"])
        if s_i not in best or v > best[s_i]:
            best[s_i] = v
    if not best:
        return None
    return best, "newsent_max(transient)"


def online_threshold(scores: list[float], c: float, warmup: int) -> list[int]:
    """running μ+c·σ — 전이 t 의 임계는 **t 이전 점수만**으로 만든다 (zero-lookahead)."""
    out, seen = [], []
    for s in scores:
        if len(seen) < warmup:
            out.append(0)
        else:
            a = np.asarray(seen)
            out.append(int(s > a.mean() + c * a.std()))
        seen.append(s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=float, default=C_FROZEN, help="🔒 동결값. 재튜닝 금지")
    ap.add_argument("--datasets", default="dev3,test3")
    args = ap.parse_args()

    from baseline_compare import load_dataset            # noqa: PLC0415
    from hi_ontop.dts_scoring import score_dialogues     # noqa: PLC0415

    print(f"B1 · readout=newsent_max transient · c={args.c} (Pieman 동결) · zero-lookahead")
    rows = {}
    for ds in args.datasets.split(","):
        data = load_dataset(ds)
        preds, missing, used_key = {}, [], None
        for doc, _sents, gold in data:
            got = read_scores(doc)
            if got is None:
                missing.append(doc)
                continue
            sc_map, used_key = got
            # 전이 t(0-based) = 문장 t+1 앞의 경계 → 새 문장은 인덱스 t+1
            series = [sc_map.get(t + 1, 0.0) for t in range(len(gold))]
            preds[doc] = online_threshold(series, args.c, WARMUP)
        if missing:
            print(f"  🔴 {ds}: b1 점수 없음 {missing} — 건너뜀")
            continue

        docs = [d for d, _s, _g in data]
        G = [g for _d, _s, g in data]
        ys = [np.concatenate([np.asarray(g, np.int8), [0]]).tolist() for g in G]
        ps = [np.concatenate([np.asarray(preds[d][:len(g)], np.int8), [0]]).tolist()
              for d, g in zip(docs, G)]
        s = score_dialogues(ys, ps)
        PRED.mkdir(parents=True, exist_ok=True)
        (PRED / f"BayesSurprise_{ds}.json").write_text(json.dumps(preds))
        rows[ds] = {"pk": s["pk"], "wd": s["wd"], "f1": s["f1"], "score": s["score"],
                    "n_pred": sum(sum(v) for v in preds.values()),
                    "n_gold": int(sum(int(g.sum()) for g in G)), "readout_key": used_key}
        print(f"  BayesSurprise {ds:<6} Pk {s['pk']:.4f} WD {s['wd']:.4f} F1 {s['f1']:.4f} "
              f"Score {s['score']:.4f} · 예측 {rows[ds]['n_pred']}/gold {rows[ds]['n_gold']}"
              f" · readout={used_key}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "b1_narrative.json").write_text(json.dumps({"config": vars(args), "rows": rows},
                                                      ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

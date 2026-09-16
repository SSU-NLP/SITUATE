#!/usr/bin/env python3
"""B1-BayesSurprise 를 **폐쇄 API 모델**로 실행 (verbatim 복사 + top-k logprobs).

왜 가능한가. B1 의 원 정의는 전체 어휘 KL 이고 API 는 top-20 만 준다. 그 절단이 신호를 보존하는지
`b1_topk_fidelity.py` 로 먼저 실측했다 (Qwen3.6-35B, pieman): 단어 KL Spearman **0.9866**,
문장 신호 **0.9736**, Score 0.4050 vs 전체 0.4062. ⇒ top-20 근사는 정당하다.

어떻게. 모델에 문서를 **그대로 복사**시키고(temp 0), 생성 토큰마다 top-20 logprobs 를 받는다.
복사가 원문과 일치하면 생성 위치 = 원문 위치이므로, 위치 j 의 top_logprobs 가
"토큰 j-1 까지 읽은 뒤의 다음 토큰 분포" 가 된다. 나머지(transient·readout·임계·채점)는
`b1_bayes_surprise` / `b1_deployable` 의 동결 함수를 그대로 쓴다.

🔴 GPT-5 는 provider 가 logprobs 를 거부한다 — 실행 불가. 이 스크립트로도 안 된다.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import b1_bayes_surprise as B  # noqa: E402
import b1_deployable as D  # noqa: E402
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402

PROMPT = ("Copy the following text word-for-word. Output only the text itself, "
          "with no preamble, no commentary, and no changes.\n\nThis is the text:\n{text}")


def get_copy_logprobs(client, model, text, max_tokens):
    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": PROMPT.format(text=text)}],
        temperature=0, max_tokens=max_tokens, logprobs=True, top_logprobs=20)
    ch = r.choices[0]
    lp = ch.logprobs.content
    return ([{"tok": t.token, "top": {a.token: a.logprob for a in t.top_logprobs}} for t in lp],
            ch.finish_reason)


def align(toks, spans, full_text):
    """생성 토큰 -> 원문 단어 인덱스. 공백을 제거한 문자 포인터로 대조한다.

    Returns (word_last_tok, n_mismatch, n_matched_chars, n_total_chars).
    """
    stripped, char2word = [], []
    for wi, (a, b, _si, _wp) in enumerate(spans):
        for ch in full_text[a:b]:
            if not ch.isspace():
                stripped.append(ch)
                char2word.append(wi)
    ptr, mism = 0, 0
    word_last_tok = [-1] * len(spans)
    for ti, t in enumerate(toks):
        for ch in t["tok"]:
            if ch.isspace():
                continue
            if ptr >= len(stripped):
                break
            if ch.lower() != stripped[ptr].lower():
                mism += 1
            word_last_tok[char2word[ptr]] = ti
            ptr += 1
    return word_last_tok, mism, ptr, len(stripped)


def dist_at(entry):
    """top-20 logprob dict -> {token: prob} (재정규화 없이 원값)."""
    return {k: math.exp(v) for k, v in entry["top"].items()}


def kl_union(p: dict, q: dict) -> float:
    """API 사용자가 계산 가능한 값: 두 top-k 합집합 위에서 각각 재정규화한 KL."""
    S = set(p) | set(q)
    pv = np.array([p.get(t, 0.0) for t in S]) + 1e-12
    qv = np.array([q.get(t, 0.0) for t in S]) + 1e-12
    pv, qv = pv / pv.sum(), qv / qv.sum()
    return float(np.sum(pv * (np.log(pv) - np.log(qv))))


def doc_signal(toks, rows, drift):
    sentences = B.reconstruct_sentences(rows)
    full, spans = B.build_words(sentences)
    wlt, mism, matched, total = align(toks, spans, full)
    cov = matched / total if total else 0.0
    acc = 1.0 - (mism / matched) if matched else 0.0
    missing = [w for w, t in enumerate(wlt) if t < 0]

    raw = [0.0]
    for i in range(1, len(spans)):
        gi, gj = wlt[i], wlt[i - 1]
        # "토큰 g 까지 읽은 뒤의 분포" = 위치 g+1 의 top_logprobs
        if gi < 0 or gj < 0 or gi + 1 >= len(toks) or gj + 1 >= len(toks):
            raw.append(0.0)
            continue
        raw.append(max(kl_union(dist_at(toks[gi + 1]), dist_at(toks[gj + 1])), 0.0))
    trans = B.transient_from_raw(raw, drift)
    words = [{"sent": si, "pos": wp, "transient": t}
             for (_a, _b, si, wp), t in zip(spans, trans)]
    return words, {"coverage": cov, "char_accuracy": acc, "n_missing_words": len(missing),
                   "n_words": len(spans), "n_tokens": len(toks)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openrouter/openai/gpt-4.1")
    ap.add_argument("--label", default=None)
    ap.add_argument("--docs", nargs="*", default=["pieman", "tunnel", "monkey"])
    ap.add_argument("--drift", type=float, default=0.10)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--gate-coverage", type=float, default=0.80,
                    help="이 미만이면 그 문서는 실행 실패로 기록 (B3 verbatim 게이트와 동일 기준)")
    ap.add_argument("--cache-dir", type=Path,
                    default=REPO / "outputs/eacl_phase0/b1_bayes/api_logprobs")
    args = ap.parse_args()
    label = args.label or args.model.split("/")[-1]
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    from secom_llm_eval import make_client
    client = make_client()

    docs, meta, failed = {}, {}, []
    for d in args.docs:
        rows = B.read(REPO / f"outputs/eacl_phase0/{d}_michelmann_master.jsonl")
        cache = args.cache_dir / f"{d}_{label}.json"
        if cache.exists():
            blob = json.loads(cache.read_text())
            toks, fin = blob["toks"], blob["finish_reason"]
            print(f"[{d}] 캐시 재사용 ({len(toks)} 토큰)")
        else:
            sentences = B.reconstruct_sentences(rows)
            text = " ".join(sentences)
            print(f"[{d}] 복사 요청 중 ({len(text.split())} 단어) ...")
            toks, fin = get_copy_logprobs(client, args.model, text, args.max_tokens)
            cache.write_text(json.dumps({"toks": toks, "finish_reason": fin}, ensure_ascii=False))
        words, m = doc_signal(toks, rows, args.drift)
        m["finish_reason"] = fin
        meta[d] = m
        print(f"[{d}] coverage {m['coverage']:.1%} · char_acc {m['char_accuracy']:.1%} · "
              f"미매칭 단어 {m['n_missing_words']} · finish={fin}")
        if m["coverage"] < args.gate_coverage:
            print(f"  🔴 verbatim 게이트 미달 ({m['coverage']:.1%} < {args.gate_coverage:.0%}) — 실행 실패로 기록")
            failed.append(d)
            continue
        gold = [int(r["transition"]) for r in sorted(rows, key=lambda r: int(r["transition"]))
                if int(r["boundary"])]
        docs[d] = {"n": len(rows) + 1, "gold": gold,
                   "b1": D.b1_newsent_max(words, len(rows)),
                   "y": [int(r["boundary"]) for r in sorted(rows, key=lambda r: int(r["transition"]))]}

    if "pieman" not in docs:
        raise SystemExit("🔴 pieman 이 게이트를 통과하지 못해 c 동결이 불가능하다. 중단.")
    c = max(D.C_GRID, key=lambda cc: D.score_docs([docs["pieman"]], "b1", cc, "adapt"))

    out = {"label": label, "model": args.model, "c_frozen_on_pieman": c,
           "topk_approximation": "top-20 union-renormalized KL "
                                 "(fidelity: word-KL rho 0.9866 vs full vocab, Qwen3.6-35B/pieman)",
           "failed_docs": failed, "meta": meta, "per_doc": {}}
    golds, preds = [], []
    for d, x in docs.items():
        pred = D.adaptive_pred(x["b1"], c)
        out["per_doc"][d] = {"score": D.score_docs([x], "b1", c, "adapt"),
                             "auc": D.auc(x["y"], x["b1"]),
                             "n_pred": len(pred), "n_gold": len(x["gold"]),
                             "bor": len(pred) / len(x["gold"])}
        golds.append(D.endturn(x["n"], x["gold"]))
        preds.append(D.endturn(x["n"], pred))
    out["pooled"] = score_dialogues(golds, preds)

    print(f"\n[{label}] c(dev=pieman argmax) = {c}")
    print(f"{'doc':8s} {'Score':>8s} {'AUC':>7s} {'pred':>6s} {'gold':>5s} {'BOR':>6s}")
    for d, v in out["per_doc"].items():
        print(f"{d:8s} {v['score']:8.4f} {v['auc']:7.4f} {v['n_pred']:6d} "
              f"{v['n_gold']:5d} {v['bor']:6.2f}")
    print(f"{'pooled':8s} {out['pooled']['score']:8.4f}  (문서 {len(docs)}편"
          + (f", 실패 {failed}" if failed else "") + ")")

    dest = REPO / "outputs/experiments/2026-07-31_b1_model_scale" / f"b1_api_{label}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n→ {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

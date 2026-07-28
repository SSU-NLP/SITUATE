#!/usr/bin/env python3
"""B1-BayesSurprise (Kumar et al. 2023, Cognitive Science) — faithful reimplementation.

Kumar et al. compute, at each word, GPT-2's next-word predictive distribution and measure
**Bayesian surprise** = KL divergence between consecutive next-word distributions (NOT surprisal).
Transient-normalized (Reynolds 2007). Their aggregation was a LASSO trailing-window regression to
human button-press; OUR pipeline instead needs one continuous score per transition for the
deployable μ+cσ DTS evaluation. Per codex decision (2026-07-14) the faithful adaptation is:

  - model         : GPT-2-XL (1.5B), HuggingFace, moving 1024-token context window (Kumar §2.4)
  - context       : story-continuous; reset only at story start, NOT at sentence boundaries
  - word surprise : S_i = KL( P(·|w<=i) || P(·|w<=i-1) ), distributions taken at the LAST BPE token
                    of each whitespace word (word endpoint-to-endpoint; no subword sum/mean)
  - transient     : mu_1 = mean_w S_w ; mu_t = mu_{t-1} + (S_t-mu_{t-1})*drift ;
                    Transient_t = S_t / mu_{t-1}          (drift = 0.10 frozen; 0.03/0.05 robustness)
  - transition agg: b1_nN[t] = max Transient over the first N words of transition t's `current`
                    sentence (onset-max). N in {1,3,5} emitted; (N,c) frozen on Pieman downstream.
                    => B1 uses an N-word onset lookahead (bounded, consistent with RQ2).

Output: one jsonl per doc with {doc_id, transition, b1_n1, b1_n3, b1_n5} aligned to the master's
`transition` index (1..n-1), so it plugs into the same deployable machinery as IDM/LAM/S5.

Usage:
  uv run python scripts/eacl_phase0/b1_bayes_surprise.py \
      outputs/eacl_phase0/pieman_michelmann_master.jsonl \
      --out outputs/eacl_phase0/b1_bayes/pieman_b1.jsonl [--drift 0.10 --model gpt2-xl]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F

ONSET_NS = (1, 3, 5)


def read(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def reconstruct_sentences(rows):
    """master rows (one doc) -> ordered sentence list. seq = [row0.previous, *rows.current]
    (identical to deployable_3way.doc_signals). transition t (1..n-1) => current sentence index t."""
    rows = sorted(rows, key=lambda r: int(r["transition"]))
    return [rows[0]["previous"], *[r["current"] for r in rows]]


def build_words(sentences):
    """Return full_text, and word_spans list of (char_start, char_end, sent_idx, pos_in_sent)."""
    full, spans = "", []
    for si, s in enumerate(sentences):
        for wp, w in enumerate(s.split()):
            if full:
                full += " "
            start = len(full)
            full += w
            spans.append((start, len(full), si, wp))
    return full, spans


@torch.no_grad()
def word_endpoint_dists(model, tok, full_text, word_spans, ctx=1024, stride=512, device="cuda"):
    """Compute, for each whitespace word, the next-token distribution at its LAST BPE token,
    using a moving `ctx`-token context window (finalize each token once, in order, HF-stride style).
    Returns list of dist tensors (len = n_words), each on `device` (V,)."""
    enc = tok(full_text, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]
    offs = enc["offset_mapping"]
    N = len(ids)
    # map each token to a word index via char-offset containment (token start within a word span)
    tok2word = [-1] * N
    wi = 0
    nW = len(word_spans)
    for ti, (a, b) in enumerate(offs):
        if a == b:  # empty (shouldn't happen w/ gpt2 offsets, but guard)
            tok2word[ti] = tok2word[ti - 1] if ti else 0
            continue
        while wi < nW and a >= word_spans[wi][1]:
            wi += 1
        tok2word[ti] = min(wi, nW - 1)
    # last token index per word
    word_last_tok = [-1] * nW
    for ti, w in enumerate(tok2word):
        word_last_tok[w] = ti
    ids_t = torch.tensor(ids, dtype=torch.long, device=device)

    dists = [None] * nW
    prev_end = 0
    for begin in range(0, N, stride):
        end = min(begin + ctx, N)
        window = ids_t[begin:end].unsqueeze(0)
        logits = model(window).logits[0]  # (end-begin, V) — logits[k] predicts token begin+k+1
        # finalize global positions [prev_end, end): distribution AFTER reading token g = softmax(logits[g-begin])
        for g in range(prev_end, end):
            w = tok2word[g]
            if word_last_tok[w] == g:  # g is this word's endpoint
                dists[w] = F.softmax(logits[g - begin].float(), dim=-1)
        prev_end = end
        if end == N:
            break
    # guard: every word must have a dist
    missing = [i for i, d in enumerate(dists) if d is None]
    if missing:
        raise RuntimeError(f"{len(missing)} words without endpoint dist (first={missing[:5]})")
    # first BPE token id per word (for surprisal of the NEXT word)
    word_first_tok = [-1] * nW
    for ti in range(N - 1, -1, -1):
        word_first_tok[tok2word[ti]] = ti
    first_tok_id = [ids[word_first_tok[w]] for w in range(nW)]
    return dists, first_tok_id


def transient_from_raw(raw, drift):
    """Reynolds transient: mu_1 = global mean; mu_t = mu_{t-1}+(S_t-mu_{t-1})*drift; T_t=S_t/mu_{t-1}."""
    n = len(raw)
    if n == 0:
        return []
    mu = sum(raw) / n  # mu_1 init (Kumar)
    out = []
    for s in raw:
        out.append(s / mu if mu > 1e-12 else 0.0)
        mu = mu + (s - mu) * drift
    return out


def doc_b1(model, tok, rows, drift, ctx, stride, device, words_dump=None):
    sentences = reconstruct_sentences(rows)
    full, spans = build_words(sentences)
    dists, first_tok_id = word_endpoint_dists(model, tok, full, spans, ctx, stride, device)
    # raw word-level Bayesian surprise S_i = KL(dist_i || dist_{i-1}); first word = 0
    # surprisal_i = -log P(word_i | context up to i-1) = -log dist_{i-1}[first_tok_id(word i)]
    raw = [0.0]
    surp = [0.0]
    for i in range(1, len(dists)):
        p, q = dists[i], dists[i - 1]
        kl = torch.sum(p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).item()
        raw.append(max(kl, 0.0))
        surp.append(float(-torch.log(dists[i - 1][first_tok_id[i]] + 1e-12).item()))
    trans = transient_from_raw(raw, drift)
    surp_tr = transient_from_raw(surp, drift)
    if words_dump is not None:  # per-word diagnostic dump (dev analysis only)
        doc = rows[0]["doc_id"]
        for (a, b, si, wp), tr, rw, sp, sptr in zip(spans, trans, raw, surp, surp_tr):
            words_dump.append({"doc_id": doc, "sent": si, "pos": wp, "transient": tr, "raw": rw,
                               "surprisal": sp, "surprisal_tr": sptr})
    # words grouped by sentence, in order
    by_sent: dict[int, list[float]] = {}
    for (a, b, si, wp), t in zip(spans, trans):
        by_sent.setdefault(si, []).append((wp, t))
    n_sent = len(sentences)
    out = []
    for t in range(1, n_sent):  # transition t => current sentence index t
        words = [v for _, v in sorted(by_sent.get(t, []))]
        row = {"doc_id": rows[0]["doc_id"], "transition": t}
        for N in ONSET_NS:
            head = words[:N]
            row[f"b1_n{N}"] = float(max(head)) if head else 0.0
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("master", type=Path, help="*_master.jsonl (has previous/current/doc_id/transition)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="gpt2-xl")
    ap.add_argument("--drift", type=float, default=0.10)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp32", action="store_true", help="use float32 (Bayesian-surprise KL is precision-sensitive)")
    ap.add_argument("--dump-words", type=Path, default=None,
                    help="also write per-word {doc_id,sent,pos,transient,raw} for dev aggregation analysis")
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"[b1] loading {args.model} on {args.device} ...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    dtype = torch.float32 if (args.fp32 or args.device != "cuda") else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(args.device).eval()

    rows = read(args.master)
    by = {}
    for r in rows:
        by.setdefault(r["doc_id"], []).append(r)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_docs = len(by)
    all_out = []
    words_dump = [] if args.dump_words else None
    for di, (doc, rs) in enumerate(by.items(), 1):
        print(f"[b1] doc {di}/{n_docs} {doc} ({len(rs)} transitions) ...", file=sys.stderr)
        all_out.extend(doc_b1(model, tok, rs, args.drift, args.ctx, args.stride, args.device, words_dump))
    args.out.write_text("\n".join(json.dumps(r) for r in all_out) + "\n")
    print(f"[b1] wrote {len(all_out)} rows -> {args.out}", file=sys.stderr)
    if words_dump is not None:
        args.dump_words.parent.mkdir(parents=True, exist_ok=True)
        args.dump_words.write_text("\n".join(json.dumps(r) for r in words_dump) + "\n")
        print(f"[b1] wrote {len(words_dump)} word rows -> {args.dump_words}", file=sys.stderr)


if __name__ == "__main__":
    main()

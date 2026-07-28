"""Precompute per-transition boundary signals for all LoCoMo conversations.

Saves, per conversation, everything the segmenters and retriever need:
- turn metadata (dia_id, speaker, text, session_idx) — preserves the mapping to
  gold-evidence dia_ids for recall@k,
- mpnet turn embeddings (for CosDrift + segment/query retrieval),
- CosDrift delta_eff (M1) and proxy-LM surprise (gpt2-xl) per within-session
  transition (NaN at session breaks = automatic boundaries).

Output: scores/<cid>.npz  (+ scores/<cid>.turns.json)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "outputs/experiments/2026-07-15_locomo_segmenter_swap"
DATA = BASE / "data/locomo_turns.jsonl"
OUTDIR = BASE / "scores"

M, RHO, A = 2, 0.7, 0.5
SURP_WINDOW = 2


def cosdrift(embs: np.ndarray) -> np.ndarray:
    T = len(embs)
    out = np.full(T, np.nan)
    for t in range(1, T):
        s_t, s_prev = embs[t], embs[t - 1]
        d_prev = 1.0 - float(s_prev @ s_t)
        w = [RHO ** i for i in range(min(M, t))]
        ctx = np.zeros_like(s_t)
        for i, wi in enumerate(w):
            ctx += wi * embs[t - 1 - i]
        n = np.linalg.norm(ctx)
        if n > 0:
            ctx = ctx / n
        d_ctx = 1.0 - float(ctx @ s_t)
        out[t] = A * d_prev + (1 - A) * d_ctx
    return out


def build_surpriser():
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = GPT2TokenizerFast.from_pretrained("gpt2-xl")
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl").to(dev).eval()

    @torch.no_grad()
    def surprise(prev_turns, turn):
        ctx = "\n".join(prev_turns[-SURP_WINDOW:])
        prefix = (ctx + "\n") if ctx else ""
        pref_ids = tok(prefix, return_tensors="pt").input_ids.to(dev)
        full_ids = tok(prefix + turn, return_tensors="pt").input_ids.to(dev)
        if full_ids.shape[1] > 1024:
            full_ids = full_ids[:, -1024:]
        n_pref = pref_ids.shape[1]
        if full_ids.shape[1] <= n_pref:
            return float("nan")
        logits = model(full_ids).logits
        logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
        tok_logp = logp.gather(2, full_ids[:, 1:].unsqueeze(-1)).squeeze(-1)[0]
        idx = torch.arange(tok_logp.shape[0], device=dev) + 1 >= n_pref
        sel = tok_logp[idx]
        return float(-sel.mean()) if sel.numel() else float("nan")

    return surprise


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    convs = [json.loads(l) for l in DATA.open()]
    surprise = build_surpriser()
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(
        "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    for conv in convs:
        cid = conv["conversation_id"]
        turns, sess_idx, texts = [], [], []
        for si, sess in enumerate(conv["sessions"]):
            for t in sess:
                turns.append({"dia_id": t["dia_id"], "speaker": t["speaker"],
                              "text": t["text"], "session_idx": si})
                sess_idx.append(si)
                texts.append(f"{t['speaker']}: {t['text']}")
        embs = np.asarray(
            enc.encode(texts, normalize_embeddings=True, show_progress_bar=False,
                       batch_size=128), dtype=np.float32)
        T = len(texts)
        cos = np.full(T, np.nan)
        surp = np.full(T, np.nan)
        sess = np.asarray(sess_idx)
        # per-session cosdrift (reset context each session)
        for s in sorted(set(sess_idx)):
            idxs = np.where(sess == s)[0]
            sub = embs[idxs].astype(np.float64)
            cs = cosdrift(sub)
            for j, gi in enumerate(idxs):
                cos[gi] = cs[j]
        # surprise per turn conditioned on prior turns within session
        for gi in range(T):
            s = sess_idx[gi]
            start = next(k for k in range(T) if sess_idx[k] == s)
            if gi == start:
                continue  # first turn of session = no transition
            surp[gi] = surprise([texts[k] for k in range(start, gi)], texts[gi])
        np.savez(OUTDIR / f"{cid}.npz", emb=embs, cos=cos, surp=surp, sess=sess)
        (OUTDIR / f"{cid}.turns.json").write_text(json.dumps(turns, ensure_ascii=False))
        nb_cos = int(np.isfinite(cos).sum())
        print(f"[{cid}] T={T} sessions={len(set(sess_idx))} "
              f"cos_finite={nb_cos} surp_finite={int(np.isfinite(surp).sum())}", flush=True)
    print("done")


if __name__ == "__main__":
    main()

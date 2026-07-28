"""LLM-free granularity probe: natural boundary density + turns/segment.

Answers the confound question directly for the two LLM-free signals:
- **M1-CosDrift** (delta_eff, canonical HP m=2, rho=0.7, a=0.5) — method-registry §1.
- **proxy-LM surprise** (gpt2-xl mean NLL of turn t given a window of prior turns).

Segmentation follows SeCom's per-session convention: candidate boundaries live
*inside* each session; session breaks are automatic segment boundaries.

For each signal we report, per conversation and pooled:
- natural boundary count under the method's natural rule (score >= mu + c*sigma),
- resulting #segments and mean turns/segment,
- the raw score distribution (so density-matching later is calibrated).

Output: outputs/experiments/2026-07-15_locomo_segmenter_swap/granularity_probe.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = (
    REPO_ROOT
    / "outputs/experiments/2026-07-15_locomo_segmenter_swap/data/locomo_turns.jsonl"
)
OUT = (
    REPO_ROOT
    / "outputs/experiments/2026-07-15_locomo_segmenter_swap/granularity_probe.json"
)

M, RHO, A = 2, 0.7, 0.5  # CosDrift canonical HP
C_STD = 1.0  # natural rule: boundary if score >= mu + C_STD*sigma
SURP_WINDOW = 2  # prior turns fed as context to the proxy LM


def load_convs() -> list[dict]:
    return [json.loads(l) for l in DATA.open()]


def cosdrift_scores(embs: np.ndarray) -> np.ndarray:
    """delta_eff per transition within one session. embs: (T, d) L2-normalized.
    Returns array length T (index 0 = NaN, no transition before first turn)."""
    T = len(embs)
    out = np.full(T, np.nan)
    for t in range(1, T):
        s_t = embs[t]
        s_prev = embs[t - 1]
        d_prev = 1.0 - float(s_prev @ s_t)
        # context vector over prev min(M, t) turns, recency-weighted
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
    def surprise(prev_turns: list[str], turn: str) -> float:
        """Mean NLL (nats/token) of `turn` conditioned on prior turns."""
        ctx = "\n".join(prev_turns[-SURP_WINDOW:])
        prefix = (ctx + "\n") if ctx else ""
        pref_ids = tok(prefix, return_tensors="pt").input_ids.to(dev)
        full_ids = tok(prefix + turn, return_tensors="pt").input_ids.to(dev)
        # truncate context to keep within 1024
        if full_ids.shape[1] > 1024:
            full_ids = full_ids[:, -1024:]
            pref_len = max(0, pref_ids.shape[1] - (full_ids.shape[1] - 0))
        n_pref = pref_ids.shape[1]
        if full_ids.shape[1] <= n_pref:
            return float("nan")
        logits = model(full_ids).logits
        # predict token i from position i-1
        shift_logits = logits[:, :-1, :]
        shift_labels = full_ids[:, 1:]
        logp = torch.log_softmax(shift_logits, dim=-1)
        tok_logp = logp.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)[0]
        # target token positions belonging to `turn` are those with label index >= n_pref
        # label at position j corresponds to full_ids[:, j+1]; belongs to turn if j+1 >= n_pref
        idx = torch.arange(tok_logp.shape[0], device=dev) + 1 >= n_pref
        sel = tok_logp[idx]
        if sel.numel() == 0:
            return float("nan")
        return float(-sel.mean())

    return surprise


def natural(scores: np.ndarray, c: float = C_STD):
    """Return boundary mask + stats using score >= mu + c*sigma (finite scores)."""
    fin = scores[np.isfinite(scores)]
    if len(fin) == 0:
        return np.zeros(len(scores), dtype=bool), 0.0, 0.0
    mu, sd = float(fin.mean()), float(fin.std())
    thr = mu + c * sd
    mask = np.isfinite(scores) & (scores >= thr)
    return mask, mu, sd


def main() -> None:
    convs = load_convs()
    surprise = build_surpriser()

    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(
        "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    report = {"per_conv": [], "config": {"M": M, "RHO": RHO, "A": A,
              "C_STD": C_STD, "SURP_WINDOW": SURP_WINDOW}}
    agg = {
        "n_turns": 0, "n_sessions": 0, "n_transitions": 0,
        "cos": {"n_bnd": 0}, "surp": {"n_bnd": 0},
    }
    all_cos, all_surp = [], []

    for conv in convs:
        cid = conv["conversation_id"]
        sessions = conv["sessions"]
        n_turns = sum(len(s) for s in sessions)
        n_sess = len(sessions)
        # per-session boundaries
        cos_bnd = surp_bnd = n_trans = 0
        for sess in sessions:
            texts = [f"{t['speaker']}: {t['text']}" for t in sess]
            if len(texts) < 2:
                continue
            embs = enc.encode(texts, normalize_embeddings=True,
                              show_progress_bar=False)
            embs = np.asarray(embs, dtype=np.float64)
            cs = cosdrift_scores(embs)
            # surprise per turn (transition score attached to turn t>=1)
            ss = np.full(len(texts), np.nan)
            for t in range(1, len(texts)):
                ss[t] = surprise([texts[j] for j in range(t)], texts[t])
            n_trans += len(texts) - 1
            m_cos, _, _ = natural(cs)
            m_surp, _, _ = natural(ss)
            cos_bnd += int(m_cos.sum())
            surp_bnd += int(m_surp.sum())
            all_cos.extend(cs[np.isfinite(cs)].tolist())
            all_surp.extend(ss[np.isfinite(ss)].tolist())
        # #segments = session-internal boundaries + n_sessions (each session starts a seg)
        cos_segs = cos_bnd + n_sess
        surp_segs = surp_bnd + n_sess
        pc = {
            "conversation_id": cid,
            "n_turns": n_turns,
            "n_sessions": n_sess,
            "n_transitions": n_trans,
            "cos": {"n_bnd_internal": cos_bnd, "n_segments": cos_segs,
                    "turns_per_seg": n_turns / max(1, cos_segs)},
            "surp": {"n_bnd_internal": surp_bnd, "n_segments": surp_segs,
                     "turns_per_seg": n_turns / max(1, surp_segs)},
        }
        report["per_conv"].append(pc)
        agg["n_turns"] += n_turns
        agg["n_sessions"] += n_sess
        agg["n_transitions"] += n_trans
        agg["cos"]["n_bnd"] += cos_bnd
        agg["surp"]["n_bnd"] += surp_bnd
        print(f"conv {cid}: {n_turns}t/{n_sess}sess | "
              f"CosDrift {cos_bnd} bnd -> {pc['cos']['turns_per_seg']:.1f} t/seg | "
              f"surprise {surp_bnd} bnd -> {pc['surp']['turns_per_seg']:.1f} t/seg",
              flush=True)

    cos_segs_tot = agg["cos"]["n_bnd"] + agg["n_sessions"]
    surp_segs_tot = agg["surp"]["n_bnd"] + agg["n_sessions"]
    agg_out = {
        "n_turns": agg["n_turns"],
        "n_sessions": agg["n_sessions"],
        "n_transitions": agg["n_transitions"],
        "cos": {
            "n_bnd_internal": agg["cos"]["n_bnd"],
            "n_segments": cos_segs_tot,
            "turns_per_seg": agg["n_turns"] / max(1, cos_segs_tot),
            "bnd_rate": agg["cos"]["n_bnd"] / max(1, agg["n_transitions"]),
            "score_mean": float(np.mean(all_cos)), "score_std": float(np.std(all_cos)),
        },
        "surp": {
            "n_bnd_internal": agg["surp"]["n_bnd"],
            "n_segments": surp_segs_tot,
            "turns_per_seg": agg["n_turns"] / max(1, surp_segs_tot),
            "bnd_rate": agg["surp"]["n_bnd"] / max(1, agg["n_transitions"]),
            "score_mean": float(np.mean(all_surp)), "score_std": float(np.std(all_surp)),
        },
    }
    report["aggregate"] = agg_out
    OUT.write_text(json.dumps(report, indent=2))
    print("\n=== AGGREGATE ===")
    print(json.dumps(agg_out, indent=2))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()

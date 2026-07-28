"""Run our no-parameter segmenter on the ICML diary (89 KO sentences, sentence-as-turn).
Multilingual encoder handles Korean. Compare boundaries to HANDOFF_06 method A."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dts_common import ENCODER_NAME  # noqa: E402
from lam import LAM  # noqa: E402
from adaptive_robust import otsu, _nms_from_thresh  # noqa: E402
from capstone_segment import ms_mean, raw_cos  # noqa: E402

OUT = HERE.parents[1] / "outputs/experiments/2026-07-09_idm_lam_gonogo/results"
ROOT = HERE.parents[1]
DEV = "cuda"
HANDOFF_A = [1, 2, 7, 18, 20, 21, 27, 29, 34, 59, 61, 69, 79]  # HANDOFF_06 우리방법(δ_eff, c=1.0)


def local_maxima(s):
    return [t for t in range(len(s))
            if (t == 0 or s[t] >= s[t - 1]) and (t + 1 >= len(s) or s[t] >= s[t + 1])]


def otsu_pk(s):
    pk = local_maxima(s)
    if len(pk) < 3:
        return set(pk)
    vals = np.array([s[t] for t in pk]); th = otsu(vals)
    return {t for t in pk if s[t] > th}


def lex_ko(turns, w=2):
    toks = [Counter(turns[i].split()) for i in range(len(turns))]
    n = len(turns); out = []
    for t in range(n - 1):
        L = Counter(); R = Counter()
        for i in range(max(0, t - w + 1), t + 1):
            L.update(toks[i])
        for i in range(t + 1, min(n, t + 1 + w)):
            R.update(toks[i])
        ks = set(L) | set(R)
        a = np.array([L[k] for k in ks], float); b = np.array([R[k] for k in ks], float)
        out.append(1 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)) if ks else 0.0)
    return np.array(out)


def fdm_best(model, emb):
    z = torch.tensor(emb, device=DEV); zt, z1 = z[:-1], z[1:]
    with torch.no_grad():
        cb = model.vq._codebook.embed.reshape(-1, model.code_dim)
        errs = [((model.fdm(torch.cat([zt, cb[k:k+1].expand(len(zt), -1)], -1)) - z1) ** 2).sum(-1)
                for k in range(cb.shape[0])]
        return torch.stack(errs, 0).min(0).values.cpu().numpy()


def main():
    sents = [l.strip() for l in (ROOT / "icml_sentences.txt").read_text().splitlines() if l.strip()]
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(ENCODER_NAME, device="cuda")
    emb = enc.encode(sents, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    ck = torch.load(OUT / "lam_corpus_f1.0.pt", weights_only=False)
    a = ck["args"]
    model = LAM(code_dim=a["code_dim"], codebook_size=a["codebook_size"],
               commitment_weight=a["commitment"]).to(DEV)
    model.load_state_dict(ck["state"]); model.eval()

    sigs = {"cosine": raw_cos(emb), "ms_mean": ms_mean(emb),
            "FDM_best": fdm_best(model, emb), "C_lex": lex_ko(sents)}
    # boundary at transition t -> new segment starts at sentence t+1
    results = {}
    for name, s in sigs.items():
        results[name] = sorted(t + 1 for t in otsu_pk(s))

    print("=== ICML diary (89 sentences) — no-parameter otsu_pk boundaries (new-segment sentence idx) ===")
    print(f"HANDOFF_06 A (δ_eff c=1.0): {HANDOFF_A}\n")
    for name, b in results.items():
        print(f"{name:10s} ({len(b)} bnd): {b}")
    print("\n=== ms_mean+otsu_pk (our deployable no-param) boundary sentences ===")
    for b in results["ms_mean"]:
        print(f"  [{b:2d}] {sents[b][:75]}")


if __name__ == "__main__":
    main()

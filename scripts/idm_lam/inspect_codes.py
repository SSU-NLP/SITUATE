"""What did the 16 codes actually cluster into? Per-code samples + characteristic words.

Uses the final corpus-trained model (corpus_f1.0) on held-out sets. For each code:
  - n, boundary-rate (does it fire at topic shifts?)
  - top content words of the NEXT utterance z_{t+1} (reveals topic/content clustering)
  - sample (z_t -> z_{t+1}) transitions to read the transition CHARACTER.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dts_common import embed_split  # noqa: E402
from lam import LAM  # noqa: E402
from gate2_segmentation import code_seq  # noqa: E402

OUT = HERE.parents[1] / "outputs/experiments/2026-07-09_idm_lam_gonogo/results"
DEV = "cuda"
WORD = re.compile(r"[a-zA-Z']+")
STOP = set("the a an to of and i you it is are was were be do does did have has had "
           "in on at for with that this my your me we they he she what how can could "
           "would will yes no not so if but or as about get got would'nt n't 'm 're 's "
           "there here their them then than out up down".split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="corpus_f1.0")
    ap.add_argument("--datasets", nargs="+", default=["superseg_test", "tiage_test"])
    args = ap.parse_args()
    ck = torch.load(OUT / f"lam_{args.tag}.pt", weights_only=False)
    a = ck["args"]; K = a["codebook_size"]
    model = LAM(code_dim=a["code_dim"], codebook_size=K, commitment_weight=a["commitment"]).to(DEV)
    model.load_state_dict(ck["state"]); model.eval()

    # gather transitions across datasets
    codes, bnd, tt, t1, nextwords = [], [], [], [], []
    for ds in args.datasets:
        for d in embed_split(ds):
            turns, b = d["turns"], d["boundaries"]
            c = code_seq(model, d["emb"], DEV)
            for t in range(len(turns) - 1):
                codes.append(int(c[t])); bnd.append(1 if (t + 1) in b else 0)
                tt.append(turns[t]); t1.append(turns[t + 1])
    codes = np.array(codes); bnd = np.array(bnd)
    base = bnd.mean()

    # global document frequency for tf-idf-ish weighting
    df = Counter()
    for s in t1:
        for w in set(x.lower() for x in WORD.findall(s) if x.lower() not in STOP and len(x) > 2):
            df[w] += 1
    N = len(t1)

    lines = [f"# How the {K} codes typed  (model={args.tag}, {N} transitions, "
             f"base boundary-rate={base:.2f})\n"]
    order = sorted(range(K), key=lambda k: -(bnd[codes == k].mean() if (codes == k).any() else 0))
    rng = np.random.default_rng(0)
    for k in order:
        idx = np.where(codes == k)[0]
        if len(idx) == 0:
            continue
        br = bnd[idx].mean()
        # characteristic words: sum tf * log(N/df) over this code's next-utterances
        score = Counter()
        for j in idx:
            for w in set(x.lower() for x in WORD.findall(t1[j]) if x.lower() not in STOP and len(x) > 2):
                score[w] += np.log(N / (df[w] + 1))
        top = ", ".join(w for w, _ in score.most_common(12))
        lines.append(f"\n## code {k}  n={len(idx)}  boundary-rate={br:.2f} (base {base:.2f})")
        lines.append(f"   characteristic next-utterance words: {top}")
        for j in rng.choice(idx, size=min(6, len(idx)), replace=False):
            mark = " <<BND" if bnd[j] else ""
            lines.append(f"     [{tt[j][:60]}]  ->  [{t1[j][:60]}]{mark}")
    txt = "\n".join(lines)
    (OUT / f"code_typing_{args.tag}.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()

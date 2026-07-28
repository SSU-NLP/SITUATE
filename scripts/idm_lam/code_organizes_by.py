"""What does the code organize by? NMI(code ; X) for X = content-topic vs form/register features
of the NEXT utterance (is_question, length bucket, speaker). Highest = what the code splits on.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dts_common import embed_split, DTS_DIR, parse_dialogue  # noqa: E402
from lam import LAM  # noqa: E402
from gate2_segmentation import code_seq  # noqa: E402
from sklearn.metrics import adjusted_mutual_info_score as AMI  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402

OUT = HERE.parents[1] / "outputs/experiments/2026-07-09_idm_lam_gonogo/results"
DEV = "cuda"


def speakers_for(name):
    """Re-parse raw to recover per-turn speaker (user/agent), aligned with parse_dialogue turns."""
    out = []
    with open(DTS_DIR / f"{name}.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            turns, _ = parse_dialogue(r["dialogue"])
            spk = []
            for item in r["dialogue"].split("[NEWLINE]"):
                if item == "[BOUNDARY]" or not item:
                    continue
                spk.append(0 if item.startswith("user:") else 1)
            if len(turns) >= 2:
                out.append(spk[:len(turns)])
    return out


def main():
    ck = torch.load(OUT / "lam_corpus_f1.0.pt", weights_only=False)
    a = ck["args"]; K = a["codebook_size"]
    model = LAM(code_dim=a["code_dim"], codebook_size=K, commitment_weight=a["commitment"]).to(DEV)
    model.load_state_dict(ck["state"]); model.eval()

    for ds in ["superseg_test", "tiage_test"]:
        data = embed_split(ds)
        spk_all = speakers_for(ds)
        codes, isq, lenb, spk, Z1 = [], [], [], [], []
        for d, sp in zip(data, spk_all):
            turns = d["turns"]; c = code_seq(model, d["emb"], DEV)
            for t in range(len(turns) - 1):
                nxt = turns[t + 1]
                codes.append(int(c[t]))
                isq.append(1 if "?" in nxt else 0)
                L = len(nxt.split()); lenb.append(0 if L <= 5 else 1 if L <= 15 else 2)
                spk.append(sp[t + 1] if t + 1 < len(sp) else 0)
                Z1.append(d["emb"][t + 1])
        codes = np.array(codes); Z1 = np.stack(Z1)
        # content at 2 granularities to show cardinality is handled by AMI
        topic30 = KMeans(30, n_init=3, random_state=0).fit_predict(Z1)
        topic6 = KMeans(6, n_init=3, random_state=0).fit_predict(Z1)
        feats = {
            "내용(topic-30 KMeans)": topic30,
            "내용(topic-6 KMeans) ": topic6,
            "말투: 질문여부(?)     ": np.array(isq),
            "말투: 길이(짧/중/긺)  ": np.array(lenb),
            "말투: 화자(user/agent)": np.array(spk),
        }
        rng = np.random.default_rng(0)
        shuf = rng.permutation(codes)   # chance baseline: code label shuffled
        print(f"=== {ds} ({len(codes)} transitions) — AMI(우연보정) ===")
        print(f"  {'축':22s}{'AMI(code;X)':>12s}{'AMI(shuffled;X)':>16s}")
        for name, x in feats.items():
            print(f"  {name:22s}{AMI(codes, x):>12.3f}{AMI(shuf, x):>16.3f}")


if __name__ == "__main__":
    main()

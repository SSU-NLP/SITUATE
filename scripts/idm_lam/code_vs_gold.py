"""Does the code partition actually recover the GOLD topic segmentation?
Per dialogue: turn t's code = c_{t-1} (incoming transition). gold_seg_id[t] = #boundaries <= t.
Compare partition(turn->code) vs partition(turn->gold_seg) via NMI/homogeneity, averaged over dialogues.
Baselines: KMeans-content proxy (what 'splits by content' was measured against), shuffled code, random.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dts_common import embed_split  # noqa: E402
from lam import LAM  # noqa: E402
from gate2_segmentation import code_seq  # noqa: E402
from sklearn.metrics import normalized_mutual_info_score as NMI, homogeneity_score  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402

OUT = HERE.parents[1] / "outputs/experiments/2026-07-09_idm_lam_gonogo/results"
DEV = "cuda"


def main():
    ck = torch.load(OUT / "lam_corpus_f1.0.pt", weights_only=False)
    a = ck["args"]; K = a["codebook_size"]
    model = LAM(code_dim=a["code_dim"], codebook_size=K, commitment_weight=a["commitment"]).to(DEV)
    model.load_state_dict(ck["state"]); model.eval()

    for ds in ["tiage_test", "superseg_test", "dialseg711_test"]:
        data = embed_split(ds)
        nmi_code, nmi_km, nmi_shuf, hom_code = [], [], [], []
        rng = np.random.default_rng(0)
        for d in data:
            turns, bnd = d["turns"], d["boundaries"]
            n = len(turns)
            if n < 6:
                continue
            c = code_seq(model, d["emb"], DEV)          # transitions 0..n-2
            code_of_turn = c[:n - 1]                     # turn t (1..n-1) code = c[t-1]
            seg_id = np.cumsum([1 if t in bnd else 0 for t in range(1, n)])  # per turn 1..n-1
            # content proxy: kmeans on the turn embeddings (same #clusters as gold segments)
            kseg = max(2, len(set(seg_id.tolist())))
            km = KMeans(kseg, n_init=3, random_state=0).fit_predict(d["emb"][1:n])
            nmi_code.append(NMI(seg_id, code_of_turn))
            nmi_km.append(NMI(seg_id, km))
            nmi_shuf.append(NMI(seg_id, rng.permutation(code_of_turn)))
            hom_code.append(homogeneity_score(seg_id, code_of_turn))
        print(f"{ds:16s} NMI(code;gold_seg)={np.mean(nmi_code):.3f}  "
              f"NMI(kmeans_content;gold_seg)={np.mean(nmi_km):.3f}  "
              f"NMI(shuffled_code;gold_seg)={np.mean(nmi_shuf):.3f}  "
              f"homogeneity(code)={np.mean(hom_code):.3f}")


if __name__ == "__main__":
    main()

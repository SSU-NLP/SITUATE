"""작품별 지표 비교 — dev-3 3편 + 시험지 3편, epoch 8 vs 40.

작동점 = 적응임계(문서 내 상위 rate 분위수), rate 는 앵커에서 동결(0.0657). gold 밀도 미사용.
지표 4종: Score / BOR / Lift / AUC.

⚠️ dev-3 과 시험지는 gold 밀도가 다르다(0.0657 vs 0.094~0.116). 퇴화 하한도 다르므로
   **작품 간 절대 Score 를 직접 비교하지 않는다.** 같은 작품 안에서 8 vs 40 만 비교한다.
   6편 평균은 요약일 뿐이며, 밀도가 다른 데이터를 섞은 값임을 명시한다.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
import train_anchor_m4m5 as T  # noqa: E402


def load_model(path: str, dev: str):
    b = torch.load(path, map_location=dev, weights_only=False)
    ta = argparse.Namespace(**b["args"])
    m = T.FactoredSlot(slot_dim=ta.slot_dim, mono=b["mono"], pos_weight=b["pos_weight"],
                       d1=(ta.input_path == "axis_extractor"),
                       shared_dim=(ta.shared_proj_dim or None),
                       tagger=(None if ta.tagger == "none" else ta.tagger),
                       tagger_axes=ta.tagger_axes, block_readout=ta.block_readout,
                       dynamics=b["dynamics"], pred_weight=ta.pred_weight,
                       pred_space=ta.pred_space, head_feat=ta.head_feat,
                       slot_reset=(None if ta.slot_reset == "none" else ta.slot_reset),
                       slot_reset_dropout=ta.slot_reset_dropout,
                       psi=(None if ta.psi == "none" else ta.psi), psi_beta=ta.psi_beta,
                       psi_dropout=ta.psi_dropout, bottleneck=ta.bottleneck,
                       dropout=ta.dropout).to(dev)
    m.load_state_dict(b["state"])
    return m


def per_doc(model, corpus, dev: str, rate: float) -> dict[str, dict]:
    probs = T.axis_probs(model, corpus, list(range(corpus.n_docs)), dev, 4, with_anchor=False)
    out = {}
    for i, pr in enumerate(probs):
        g = corpus.gold[int(corpus.to[i]):int(corpus.to[i + 1])].astype(np.int8)
        s = pr.sum(1)
        b = (s >= np.quantile(s, 1.0 - rate)).astype(np.int8)
        y = np.concatenate([g, [0]]).tolist()
        p = np.concatenate([b, [0]]).tolist()
        n_gold, n_pred = int(g.sum()), int(b.sum())
        hit = int((g & b).sum())
        dens = n_gold / len(g)
        prec = hit / n_pred if n_pred else float("nan")
        out[corpus.doc_ids[i]] = {
            "score": score_dialogues([y], [p])["score"],
            "floor": score_dialogues([y], [[0] * len(y)])["score"],
            "bor": n_pred / n_gold if n_gold else float("nan"),
            "lift": prec / dens if (n_pred and dens > 0) else float("nan"),
            "auc": float(roc_auc_score(g, s)) if 0 < n_gold < len(g) else float("nan"),
            "n_gold": n_gold, "n_trans": len(g), "gold_rate": dens,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e8-glob", required=True)
    ap.add_argument("--e40-glob", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rate", type=float, default=0.0657)
    a = ap.parse_args()

    corpora = {"dev-3": T.Corpus("dev3"), "test3": T.Corpus("test3")}
    res: dict[str, dict[str, list[dict]]] = {}
    for tag, gl in (("e8", a.e8_glob), ("e40", a.e40_glob)):
        acc: dict[str, list[dict]] = {}
        for path in sorted(glob.glob(gl)):
            m = load_model(path, a.device)
            for cname, c in corpora.items():
                for doc, v in per_doc(m, c, a.device, a.rate).items():
                    acc.setdefault(f"{cname}/{doc}", []).append(v)
        res[tag] = acc
        print(f"[{tag}] 체크포인트 {len(glob.glob(gl))}개 · 작품 {len(acc)}편")

    def agg(rows: list[dict], k: str) -> tuple[float, float]:
        v = [r[k] for r in rows if r[k] == r[k]]
        return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)

    docs = list(res["e40"].keys())
    print(f"\n{'작품':22s} {'gold밀도':>8s} {'하한':>7s} │ "
          + " ".join(f"{m:>17s}" for m in ("Score", "BOR", "Lift", "AUC")))
    lines = []
    for d in docs:
        r8, r40 = res["e8"][d], res["e40"][d]
        f = agg(r40, "floor")[0]
        cells = []
        for k in ("score", "bor", "lift", "auc"):
            m8, _ = agg(r8, k)
            m40, _ = agg(r40, k)
            mark = "▲" if m40 > m8 else ("▼" if m40 < m8 else "=")
            cells.append(f"{m8:6.3f}→{m40:6.3f}{mark}")
        print(f"{d:22s} {agg(r40,'gold_rate')[0]:8.4f} {f:7.4f} │ " + " ".join(cells))
        lines.append({"doc": d, "floor": f,
                      **{f"{k}_{t}": agg(res[t][d], k)[0]
                         for k in ("score", "bor", "lift", "auc") for t in ("e8", "e40")}})

    print(f"\n{'6편 평균':22s} {'':8s} {'':7s} │ ", end="")
    cells = []
    for k in ("score", "bor", "lift", "auc"):
        m8 = st.mean([st.mean([r[k] for r in res["e8"][d] if r[k] == r[k]]) for d in docs])
        m40 = st.mean([st.mean([r[k] for r in res["e40"][d] if r[k] == r[k]]) for d in docs])
        mark = "▲" if m40 > m8 else "▼"
        cells.append(f"{m8:6.3f}→{m40:6.3f}{mark}")
    print(" ".join(cells))
    print("\n⚠️ dev-3 과 시험지는 gold 밀도·퇴화 하한이 다르다 — 작품 간 절대 Score 비교 금지.")

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "epoch_compare_alldocs.json").write_text(
        json.dumps({"per_doc": lines, "raw": {t: {d: res[t][d] for d in docs} for t in res}},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {a.out/'epoch_compare_alldocs.json'}")


if __name__ == "__main__":
    main()

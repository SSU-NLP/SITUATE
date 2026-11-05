"""6편(dev-3 3 + 시험지 3) × 4모델(M5·M4·M8-pair·M8-ctx) 작품별 지표.

작동점 = 적응임계(문서 내 상위 rate 분위수), rate 는 앵커 동결값. gold 밀도 미사용.
지표 = Score / BOR / Lift / AUC + 퇴화 하한.

⚠️ dev-3 과 시험지는 gold 밀도·퇴화 하한이 다르다 — **작품 간 절대 Score 비교 금지.**
   같은 작품 안에서 모델끼리만 비교한다.
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
from hi_ontop.hi_ontop_v2 import adaptive_boundaries  # noqa: E402
import train_anchor_m4m5 as T  # noqa: E402
import train_direct_boundary as B  # noqa: E402

# 작동점(임계 규칙). 기본은 `quantile` = 문서 내 상위 rate 분위수.
# `online_nf` = 적응임계 s_t >= 누적평균(s_<t) + c·누적표준편차(s_<t) — 미래 미사용.
# 🔴 둘은 성질이 다르다: 분위수는 예측 **개수를 고정**하므로 모델 간 BOR 이 같아지고 Score
#    차이가 위치 차이만 반영한다. 적응임계는 개수를 고정하지 않아 모델마다 BOR 이 달라진다.
#    따라서 두 규칙의 Score 를 나란히 비교하면 안 된다.
THRESH_MODE = "quantile"
THRESH_C = 1.0


def binarize(s: np.ndarray, rate: float) -> np.ndarray:
    if THRESH_MODE == "quantile":
        return (s >= np.quantile(s, 1.0 - rate)).astype(np.int8)
    # adaptive_boundaries 규약: 0번 sentinel 을 앞에 붙인 시퀀스를 받는다
    return np.asarray(adaptive_boundaries(np.concatenate([[0.0], s]).tolist(),
                                          c=THRESH_C, mode=THRESH_MODE), np.int8)[:len(s)]


def metrics(g: np.ndarray, s: np.ndarray, rate: float) -> dict:
    """🔒 작동점 = **채택된 적응임계** `online_nf`, c=1.0 (D-B39).

    `s_t >= 지금까지의 mu + 1.0*sigma` — **미래를 쓰지 않는다**(온라인 적법).
    이전 구현은 문서 전체 분포의 분위수(top rate%)를 썼는데, 그건 문서 전체가 있어야
    임계가 정해지므로 **오프라인**이다. 온라인 분절 주장과 어긋난다(2026-08-04 정정).
    `rate` 는 더 이상 임계에 쓰이지 않으며 기록용으로만 남긴다.
    """
    from hi_ontop.hi_ontop_v2 import adaptive_boundaries
    n = len(s)
    b = np.asarray(adaptive_boundaries(np.concatenate([[0.0], s]).tolist(),
                                       c=1.0, mode="online_nf"), np.int8)[:n]
    y = np.concatenate([g, [0]]).tolist()
    p = np.concatenate([b, [0]]).tolist()
    ng, npd = int(g.sum()), int(b.sum())
    hit = int((g & b).sum())
    dens = ng / len(g)
    prec = hit / npd if npd else float("nan")
    return {"score": score_dialogues([y], [p])["score"],
            "floor": score_dialogues([y], [[0] * len(y)])["score"],
            "bor": npd / ng if ng else float("nan"),
            "lift": prec / dens if (npd and dens > 0) else float("nan"),
            "auc": float(roc_auc_score(g, s)) if 0 < ng < len(g) else float("nan"),
            "gold_rate": dens, "n_pred": npd}


def m5_scores(path: str, corpora: dict, dev: str, rate: float) -> dict[str, dict]:
    b = torch.load(path, map_location=dev, weights_only=False)
    ta = argparse.Namespace(**b["args"])
    m = T.FactoredSlot(z_dim=getattr(ta, "z_dim", 768), slot_dim=ta.slot_dim, mono=b["mono"], pos_weight=b["pos_weight"],
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
    out = {}
    for cname, c in corpora.items():
        probs = T.axis_probs(m, c, list(range(c.n_docs)), dev, 4, with_anchor=False)
        for i, pr in enumerate(probs):
            g = c.gold[int(c.to[i]):int(c.to[i + 1])].astype(np.int8)
            out[f"{cname}/{c.doc_ids[i]}"] = metrics(g, pr.sum(1), rate)
    return out


@torch.no_grad()
def m8_scores(path: str, corpora: dict, dev: str, rate: float) -> dict[str, dict]:
    b = torch.load(path, map_location=dev, weights_only=False)
    variant = b["variant"]
    in_dim = 768 * (5 if variant == "ctx" else 4)
    m = B.DirectBoundaryHead(in_dim, b.get("hidden", 256), b.get("dropout", 0.4)).to(dev)
    m.load_state_dict(b["state"])
    m.eval()
    out = {}
    for cname, c in corpora.items():
        emb = torch.from_numpy(np.asarray(c.emb)).to(dev)
        ctx = B.causal_context(emb, c.so, dev) if variant == "ctx" else None
        for i in range(c.n_docs):
            s0, s1 = int(c.so[i]), int(c.so[i + 1])
            j = torch.arange(s0, s1 - 1, device=dev)
            x = B.feats(emb[j].float(), emb[j + 1].float(),
                        ctx[j].float() if ctx is not None else None)
            p = torch.sigmoid(m(x)).cpu().numpy()
            g = c.gold[int(c.to[i]):int(c.to[i + 1])].astype(np.int8)
            out[f"{cname}/{c.doc_ids[i]}"] = metrics(g, p, rate)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rate", type=float, default=0.0660)   # I2 앵커 경계율 (quantile 모드 전용)
    ap.add_argument("--mode", default="quantile",
                    choices=["quantile", "online_nf", "offline", "otsu", "ewma"],
                    help="작동점. quantile=문서 내 상위 rate 분위수(개수 고정), "
                         "online_nf=적응임계 누적 mu+c*sigma(개수 미고정)")
    ap.add_argument("--c", type=float, default=1.0, help="적응임계 계수 (mode!=quantile 일 때)")
    a = ap.parse_args()

    global THRESH_MODE, THRESH_C
    THRESH_MODE, THRESH_C = a.mode, a.c
    op = f"문서 내 상위 {a.rate:.4f} 분위수" if a.mode == "quantile" else f"적응임계 {a.mode} c={a.c}"
    print(f"■ 작동점 = {op}")

    EXP = REPO / "outputs/experiments/m4m5_phase1_anchor"
    M8D = REPO / "outputs/experiments/2026-07-31_m8_direct_boundary__personkey_i2_e100"
    corpora = {"dev-3": T.Corpus("dev3"), "test3": T.Corpus("test3")}
    models = {
        "M5": (sorted(glob.glob(str(EXP / "M5_phase1__personkey_i2_per_slot_gsauthor_ep100_i2e100_seed*.pt"))), m5_scores),
        "M4": (sorted(glob.glob(str(EXP / "M4_phase1__personkey_i2_per_slot_gsauthor_ep100_i2e100_seed*.pt"))), m5_scores),
        "M8-pair": (sorted(glob.glob(str(M8D / "M8_pair_seed*.pt"))), m8_scores),
        "M8-ctx": (sorted(glob.glob(str(M8D / "M8_ctx_seed*.pt"))), m8_scores),
    }
    res: dict[str, dict[str, list[dict]]] = {}
    for name, (paths, fn) in models.items():
        acc: dict[str, list[dict]] = {}
        for p in paths:
            for doc, v in fn(p, corpora, a.device, a.rate).items():
                acc.setdefault(doc, []).append(v)
        res[name] = acc
        print(f"[{name}] 체크포인트 {len(paths)}개 · 작품 {len(acc)}편")

    docs = list(res["M5"].keys())
    names = list(models)

    def m(model: str, doc: str, k: str) -> float:
        v = [r[k] for r in res[model][doc] if r[k] == r[k]]
        return st.mean(v) if v else float("nan")

    for k, lab in (("score", "Score"), ("bor", "BOR"), ("lift", "Lift"), ("auc", "AUC")):
        print(f"\n■ {lab}")
        print(f"   {'작품':22s} {'하한':>7s} " + " ".join(f"{n:>9s}" for n in names))
        for d in docs:
            best = max(names, key=lambda n: (m(n, d, k) if m(n, d, k) == m(n, d, k) else -9))
            cells = " ".join(f"{m(n,d,k):9.3f}" + ("*" if n == best else " ") for n in names)
            print(f"   {d:22s} {m('M5',d,'floor'):7.4f} {cells}")
        avg = {n: st.mean([m(n, d, k) for d in docs if m(n, d, k) == m(n, d, k)]) for n in names}
        bn = max(avg, key=avg.get)
        print(f"   {'6편 평균':22s} {'':7s} " +
              " ".join(f"{avg[n]:9.3f}" + ("*" if n == bn else " ") for n in names))

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "all_models_alldocs.json").write_text(
        json.dumps({n: {d: res[n][d] for d in docs} for n in names}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"\n⚠️ dev-3 과 시험지는 gold 밀도·하한이 다르다 — 작품 간 절대 Score 비교 금지.")
    print(f"-> {a.out/'all_models_alldocs.json'}")


if __name__ == "__main__":
    main()

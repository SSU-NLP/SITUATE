#!/usr/bin/env python3
"""M5 Phase 2 GATE — cross-axis probe: does factoring (M5) separate slots that monolithic (M4) can't?

Own-axis AUPRC (from training) shows M5≈M4 (both have per-axis heads). The real discriminator is
CROSS-TALK: fit a FRESH linear probe from each slot's transition feature to each axis's reset.
  - M5 (5 typed slots): probe[slot k → axis j] should be HIGH on diagonal (k=j), LOW off-diagonal
    = slots are separated / axis-specialized.
  - M4 (1 shared slot):  slot-0 → every axis is decodable = entangled (no separation).
Probes fit on TRAIN docs' slot states, evaluated (AUPRC) on VAL docs' — held-out linear decodability.

Separation score per axis k: AUPRC(k→k) − max_{j≠k} AUPRC(k→j)  (codex GATE: >0 for all axes = aligned).
Output: outputs/experiments/m5_phase2_slotalign/REPORT.md
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
from hi_ontop.factored_slot import FactoredSlot, AXES  # noqa: E402
from train_factored_slot import load_docs, VAL_DOCS  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

OUT = REPO / "outputs/experiments/m5_phase2_slotalign"


def slot_transition_feats(model, docs):
    """Return per-slot transition features X[slot] (stacked over docs) and reset labels Y[:,5]."""
    model.eval()
    n_slots = model.n_slots
    feats = {s: [] for s in range(n_slots)}
    labels = []
    with torch.no_grad():
        for d in docs:
            S = model.encode_seq(d["z"])  # [T, n_slots, dim]
            for s in range(n_slots):
                sk = S[:, s, :]
                dd = torch.cat([sk[:-1], sk[1:], sk[1:] - sk[:-1], sk[1:] * sk[:-1]], -1)
                feats[s].append(dd.cpu().numpy())
            labels.append(d["resets"].cpu().numpy())
    X = {s: np.concatenate(feats[s]) for s in range(n_slots)}
    Y = np.concatenate(labels)  # [N, 5]
    return X, Y


def _fit_ap(Xtr, ytr, Xva, yva):
    if ytr.sum() < 3 or yva.sum() < 1:
        return np.nan
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    clf.fit(Xtr, ytr)
    return average_precision_score(yva, clf.predict_proba(Xva)[:, 1])


def probe_matrix(model, tr, va):
    Xtr, Ytr = slot_transition_feats(model, tr)
    Xva, Yva = slot_transition_feats(model, va)
    n_slots = model.n_slots
    M = np.full((n_slots, 5), np.nan)
    for s in range(n_slots):
        for j in range(5):
            M[s, j] = _fit_ap(Xtr[s], Ytr[:, j], Xva[s], Yva[:, j])
    return M


def label_corr_ceiling(tr, va):
    """B[k,j] = AUPRC predicting axis j from axis k's TRUE reset label alone (label-correlation limit).
    A slot that encodes ONLY axis k can still decode axis j up to B[k,j] via label correlation."""
    Ytr = np.concatenate([d["resets"].cpu().numpy() for d in tr])
    Yva = np.concatenate([d["resets"].cpu().numpy() for d in va])
    B = np.full((5, 5), np.nan)
    for k in range(5):
        for j in range(5):
            if k == j:
                B[k, j] = 1.0
                continue
            B[k, j] = _fit_ap(Ytr[:, [k]].astype(float), Ytr[:, j], Yva[:, [k]].astype(float), Yva[:, j])
    return B


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    docs = load_docs(dev)
    tr = [d for d in docs if d["name"] not in VAL_DOCS]
    va = [d for d in docs if d["name"] in VAL_DOCS]
    base = {k: float(np.concatenate([d["resets"].cpu().numpy() for d in va])[:, i].mean())
            for i, k in enumerate(AXES)}

    B = label_corr_ceiling(tr, va)  # label-correlation ceiling
    mats = {}
    for name, mono in [("M5", False), ("M4", True)]:
        ck = torch.load(OUT / f"{name}.pt", map_location=dev, weights_only=False)
        m = FactoredSlot(slot_dim=ck["args"]["slot_dim"], mono=mono, pos_weight=ck["pos_weight"]).to(dev)
        m.load_state_dict(ck["state"])
        mats[name] = probe_matrix(m, tr, va)

    lines = ["# M5 Phase 2 — slot-alignment probe (cross-axis, held-out val)\n",
             f"val docs = {sorted(VAL_DOCS)} · probes fit on {len(tr)} train docs, AUPRC on val.\n",
             "base rate (val): " + " ".join(f"{k[:4]} {base[k]:.2f}" for k in AXES) + "\n"]
    # M5 5x5 matrix
    lines.append("\n## M5 — probe AUPRC [slot k → axis j]  (diagonal=own, off=cross-talk)\n")
    lines.append("| slot\\axis | " + " | ".join(a[:5] for a in AXES) + " |")
    lines.append("|---|" + "---|" * 5)
    M5 = mats["M5"]
    for s in range(5):
        row = " | ".join(f"**{M5[s,j]:.2f}**" if s == j else f"{M5[s,j]:.2f}" for j in range(5))
        lines.append(f"| slot {AXES[s][:5]} | {row} |")
    # EXCESS leakage vs label-correlation ceiling (codex correction): off-diag slot_k->j minus B[k,j]
    lines.append("\n## M5 — EXCESS off-axis leakage  slot_k→j − B(k→j)  (label-corr ceiling removed)\n")
    lines.append("| slot\\axis | " + " | ".join(a[:5] for a in AXES) + " |")
    lines.append("|---|" + "---|" * 5)
    excess = np.full((5, 5), np.nan)
    for s in range(5):
        cells = []
        for j in range(5):
            if s == j:
                cells.append(f"own {M5[s,j]:.2f}")
            else:
                excess[s, j] = M5[s, j] - B[s, j]
                cells.append(f"{excess[s,j]:+.2f}")
        lines.append(f"| slot {AXES[s][:5]} | " + " | ".join(cells) + " |")
    mean_excess = np.nanmean(excess[~np.eye(5, dtype=bool)])
    # M4 single slot + its excess (slot-0 encodes all; ceiling = best single-axis correlation into j)
    lines.append("\n## M4 (monolithic) — slot-0 → each axis (entanglement)\n")
    lines.append("| | " + " | ".join(a[:5] for a in AXES) + " |")
    lines.append("|---|" + "---|" * 5)
    lines.append("| slot 0 | " + " | ".join(f"{mats['M4'][0,j]:.2f}" for j in range(5)) + " |")
    m4_excess = np.nanmean([mats["M4"][0, j] - np.nanmax([B[k, j] for k in range(5) if k != j])
                            for j in range(5)])
    # verdict
    own = np.diag(M5)
    lines.append("\n## Verdict")
    lines.append(f"- **M5 own-axis AUPRC** (diag): " + " ".join(f"{AXES[k][:4]} {own[k]:.2f}" for k in range(5)))
    lines.append(f"- **M5 mean EXCESS off-axis leakage: {mean_excess:+.3f}** (≈0 = slots specialized; "
                 "high = slot k over-encodes other axes beyond label correlation).")
    lines.append(f"- **M4 mean excess (slot-0 vs best-single-axis ceiling): {m4_excess:+.3f}** "
                 "(high = single latent entangles all axes).")
    lines.append(f"- **Factoring gain = M4 excess − M5 excess = {m4_excess - mean_excess:+.3f}** "
                 "(>0 = D1 routing separates what monolithic entangles).")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

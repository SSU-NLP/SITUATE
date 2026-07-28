#!/usr/bin/env python3
"""M5 Phase 2 — train FactoredSlot (M5) and the monolithic control (M4) on frozen embeddings.

Anchor-only training (no dynamics). Doc-level train/val split. Per-axis pos_weight for the rare
reset axes (place/time). Reports val own-axis AUPRC per epoch; saves both checkpoints. Loop template:
scripts/idm_lam/train_lam.py. Anchor loss = codex 2026-07-15 (see factored_slot.py).

Usage: uv run python scripts/eacl_phase0/train_factored_slot.py [--epochs 300 --slot-dim 128]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from hi_ontop.factored_slot import FactoredSlot, AXES, STATE_AXES  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

OUT = REPO / "outputs/experiments/m5_phase2_slotalign"
STATE_FIELDS = ["perspective", "temporal_shift_marker", "event_type"]  # global-enum aux
VAL_DOCS = {"tunnel_michelmann", "the_lady_with_the_dog"}  # held-out for probe (fixed)


def load_docs(dev, dataset="slot_dataset.npz"):
    """dataset 기본값 = 기존 Phase 2 파일 -> 하위호환(기존 호출부 수치 불변).

    통합 npz(slot_dataset_all.npz)에는 mask/corpus 배열이 추가돼 있다. 없으면 fiction 전체 유효로 간주.
    """
    d = np.load(OUT / dataset, allow_pickle=True)
    emb = d["emb"]; doc_off = d["doc_offsets"]; rst = d["resets"]; r_off = d["reset_offsets"]
    doc_ids = d["doc_ids"]
    msk = d["mask"] if "mask" in d else None
    corp = d["corpus"] if "corpus" in d else None
    states = {f: d[f"state_{f}"] for f in STATE_FIELDS}
    docs = []
    names = [doc_ids[doc_off[i]] for i in range(len(doc_off) - 1)]
    for i, name in enumerate(names):
        s0, s1 = doc_off[i], doc_off[i + 1]
        t0, t1 = r_off[i], r_off[i + 1]
        docs.append(dict(
            name=str(name),
            corpus=str(corp[doc_off[i]]) if corp is not None else "fiction",
            z=torch.tensor(emb[s0:s1], dtype=torch.float32, device=dev),
            resets=torch.tensor(rst[t0:t1], dtype=torch.long, device=dev),
            mask=(torch.tensor(msk[t0:t1], dtype=torch.float32, device=dev)
                  if msk is not None else None),
            states={f: torch.tensor(states[f][s0:s1], dtype=torch.long, device=dev) for f in STATE_FIELDS},
        ))
    return docs


def val_auprc(model, val):
    model.eval()
    logits = {k: [] for k in AXES}; labels = {k: [] for k in AXES}
    with torch.no_grad():
        for doc in val:
            out = model(doc["z"], doc["resets"], doc["states"])
            for ki, k in enumerate(AXES):
                logits[k].append(torch.sigmoid(out["change_logits"][k]).cpu().numpy())
                labels[k].append(doc["resets"][:, ki].cpu().numpy())
    ap = {}
    for k in AXES:
        y = np.concatenate(labels[k]); p = np.concatenate(logits[k])
        ap[k] = average_precision_score(y, p) if 0 < y.sum() < len(y) else float("nan")
    return ap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--slot-dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    docs = load_docs(dev)
    train = [d for d in docs if d["name"] not in VAL_DOCS]
    val = [d for d in docs if d["name"] in VAL_DOCS]
    # pos_weight per axis from TRAIN resets (neg/pos, capped) for imbalance
    allr = torch.cat([d["resets"] for d in train]).float()
    pos = allr.mean(0).clamp(1e-3, 1 - 1e-3)
    pw = ((1 - pos) / pos).clamp(1, 30).tolist()
    print(f"train {len(train)} docs / val {len(val)} docs · pos-rate {pos.round(decimals=2).tolist()} · pos_weight {[round(x,1) for x in pw]}")

    results = {}
    for name, mono in [("M5", False), ("M4", True)]:
        torch.manual_seed(args.seed)
        model = FactoredSlot(slot_dim=args.slot_dim, mono=mono, pos_weight=pw).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        hist = []
        for ep in range(args.epochs):
            model.train(); np.random.shuffle(train); tot = 0.0
            for doc in train:
                out = model(doc["z"], doc["resets"], doc["states"], mask=doc["mask"])
                opt.zero_grad(); out["loss"].backward(); opt.step()
                tot += float(out["loss"])
            if (ep + 1) % 50 == 0 or ep == args.epochs - 1:
                ap = val_auprc(model, val)
                hist.append({"epoch": ep + 1, "train_loss": tot / len(train), **ap})
                print(f"[{name}] ep{ep+1:3d} loss {tot/len(train):.3f} | val AUPRC " +
                      " ".join(f"{k[:4]} {ap[k]:.2f}" for k in AXES), flush=True)
        torch.save({"state": model.state_dict(), "args": vars(args), "mono": mono, "pos_weight": pw},
                   OUT / f"{name}.pt")
        results[name] = hist[-1]
    # base rate reference (predict base rate -> AUPRC = positive rate)
    base = {k: float(torch.cat([d["resets"] for d in val])[:, i].float().mean()) for i, k in enumerate(AXES)}
    print("\n=== val own-axis AUPRC (final) ===")
    print(f"{'axis':14s} {'M5':>6s} {'M4':>6s} {'base':>6s}")
    for k in AXES:
        print(f"{k:14s} {results['M5'][k]:6.2f} {results['M4'][k]:6.2f} {base[k]:6.2f}")
    (OUT / "train_history.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

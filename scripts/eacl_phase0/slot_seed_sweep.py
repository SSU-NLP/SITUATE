#!/usr/bin/env python3
"""M5 Phase 2 — seed-stability sweep of the D1 factoring gain + 학습 코퍼스 ablation.

For each seed: train M5(D1) and M4(mono), probe cross-axis with label-correlation ceiling, record
own-axis AUPRC (M5) and factoring gain (M4 excess − M5 excess). Aggregate mean±std over seeds.

**코퍼스 ablation (2026-07-16).** `--train-corpus {fiction,share,both}` 로 학습 문서만 갈아끼우고
평가는 항상 고정한다 — 모델이 유일한 처치가 되도록:
  - val = fiction 2문서 (VAL_DOCS) 고정, 조건 불문 동일.
  - probe fit + label-correlation ceiling B = **항상 fiction-train** 고정 (C4: 조건마다 B가 움직이면
    factoring gain 기준선이 이동해 조건 간 비교가 깨진다).
  - `--pos-weight-from fiction` (C7: 손실 스케일을 조건 간 동일하게).
  - `--chunk-len` + `--max-steps` (C3: 문서당 opt.step 1회라 에폭 고정 시 조건별 업데이트 수가 다름).

Usage:
  # 기존 동작 (fiction, 하위호환)
  uv run python scripts/eacl_phase0/slot_seed_sweep.py --seeds 5 --epochs 150 [--variant weak]
  # 코퍼스 ablation
  uv run python scripts/eacl_phase0/slot_seed_sweep.py --dataset slot_dataset_all.npz \
      --train-corpus both --chunk-len 128 --max-steps 3000 --pos-weight-from fiction --tag both
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
from hi_ontop.factored_slot import FactoredSlot, AXES  # noqa: E402
from train_factored_slot import load_docs, VAL_DOCS  # noqa: E402
from probe_slot_alignment import probe_matrix, label_corr_ceiling  # noqa: E402


def chunk_docs(docs, L):
    """문서를 L 전이 청크로 분할 (C3: 문서당 1 step이므로 길이 편차가 업데이트 크기 편차가 됨).

    문장 슬라이스 [s:e] 의 내부 전이 = resets[s:e-1] (resets[i] = 쌍(문장 i, 문장 i+1)).
    """
    if not L:
        return docs
    out = []
    for d in docs:
        n = d["z"].shape[0]
        for s in range(0, n - 1, L):
            e = min(s + L + 1, n)
            if e - s < 2:
                continue
            out.append(dict(name=d["name"], corpus=d["corpus"], z=d["z"][s:e],
                            resets=d["resets"][s:e - 1],
                            mask=d["mask"][s:e - 1] if d["mask"] is not None else None,
                            states={f: v[s:e] for f, v in d["states"].items()}))
    return out


def train_one(model, train, max_steps, lr, seed, axis_mask=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.RandomState(seed)
    steps = 0
    while steps < max_steps:
        model.train()
        for i in rng.permutation(len(train)):
            d = train[i]
            out = model(d["z"], d["resets"], d["states"], mask=d.get("mask"), axis_mask=axis_mask)
            opt.zero_grad(); out["loss"].backward(); opt.step()
            steps += 1
            if steps >= max_steps:
                break
    return model


def excess(P, B, mono=False):
    if mono:
        return float(np.nanmean([P[0, j] - np.nanmax([B[k, j] for k in range(5) if k != j]) for j in range(5)]))
    off = ~np.eye(5, dtype=bool)
    return float(np.nanmean((P - B)[off]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--slot-dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--variant", choices=["base", "weak"], default="base")
    ap.add_argument("--dataset", default="slot_dataset.npz")
    ap.add_argument("--train-corpus", choices=["fiction", "share", "both"], default="fiction")
    ap.add_argument("--chunk-len", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = epochs * n_train_docs")
    ap.add_argument("--pos-weight-from", choices=["self", "fiction"], default="self")
    ap.add_argument("--axis-mask", default="11111", help="축별 손실 on/off, 예 01111 = protagonist 제외")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dump", type=Path, default=None)
    ap.add_argument("--subsample-trans", type=int, default=0,
                    help="train 청크를 이 전이 수까지 서브샘플 (C2: 크기 교란 통제)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    axis_mask = tuple(c == "1" for c in args.axis_mask)
    assert len(axis_mask) == 5, "--axis-mask 는 5자리"

    docs = load_docs(dev, args.dataset)
    val = [d for d in docs if d["name"] in VAL_DOCS]
    assert len(val) == 2, f"val set drift: {[d['name'] for d in val]}"
    fiction_tr = [d for d in docs if d["name"] not in VAL_DOCS and d["corpus"] == "fiction"]
    sel = {"fiction": {"fiction"}, "share": {"share"}, "both": {"fiction", "share"}}[args.train_corpus]
    tr = [d for d in docs if d["name"] not in VAL_DOCS and d["corpus"] in sel]

    # pos_weight (C7): 조건 간 손실 스케일 고정을 위해 fiction 기준 옵션 제공
    pw_src = fiction_tr if args.pos_weight_from == "fiction" else tr
    allr = torch.cat([d["resets"] for d in pw_src]).float()
    pos = allr.mean(0).clamp(1e-3, 1 - 1e-3)
    cap = 60 if args.variant == "weak" else 30
    pw = ((1 - pos) / pos).clamp(1, cap).tolist()
    bottleneck = 16 if args.variant == "weak" else 8

    # 청크 분할 (C3) — 조건 간 업데이트 단위 균질화
    tr_chunks = chunk_docs(tr, args.chunk_len)
    if args.subsample_trans:                      # C2: 크기 교란 통제
        rng = np.random.RandomState(20260716)
        order = rng.permutation(len(tr_chunks))
        keep, tot = [], 0
        for i in order:
            if tot >= args.subsample_trans:
                break
            keep.append(tr_chunks[i]); tot += len(tr_chunks[i]["resets"])
        tr_chunks = keep
    max_steps = args.max_steps or args.epochs * len(tr_chunks)
    n_trans = sum(len(d["resets"]) for d in tr_chunks)

    # probe fit + ceiling 은 항상 fiction-train 고정 (C4): 모델만이 처치가 되도록
    B = label_corr_ceiling(fiction_tr, val)

    print(f"=== corpus={args.train_corpus} tag={args.tag or args.variant} ===")
    print(f"train {len(tr)} docs -> {len(tr_chunks)} chunks / {n_trans} trans (chunk_len={args.chunk_len}) | "
          f"val {len(val)} docs | max_steps={max_steps}")
    print(f"pos_weight from {args.pos_weight_from}: {[round(x,1) for x in pw]} | axis_mask {args.axis_mask}")

    diag, gains = {k: [] for k in AXES}, []
    for seed in range(args.seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        P = {}
        for name, mono in [("M5", False), ("M4", True)]:
            torch.manual_seed(seed)
            m = FactoredSlot(slot_dim=args.slot_dim, mono=mono, pos_weight=pw, bottleneck=bottleneck).to(dev)
            train_one(m, list(tr_chunks), max_steps, args.lr, seed, axis_mask)
            P[name] = probe_matrix(m, fiction_tr, val)   # 고정 probe 데이터
        for k in range(5):
            diag[AXES[k]].append(P["M5"][k, k])
        g = excess(P["M4"], B, mono=True) - excess(P["M5"], B, mono=False)
        gains.append(g)
        print(f"[seed {seed}] own-axis " + " ".join(f"{AXES[k][:4]} {P['M5'][k,k]:.2f}" for k in range(5)) +
              f" | factoring gain {g:+.3f}", flush=True)

    print(f"\n=== {args.train_corpus} · {args.variant} · {args.seeds} seeds "
          f"(bottleneck={bottleneck}, pw_cap={cap}) ===")
    print(f"{'axis':14s} {'own-axis mean±std':>20s}")
    for k in AXES:
        print(f"{k:14s} {np.mean(diag[k]):.2f} ± {np.std(diag[k]):.2f}")
    g = np.array(gains)
    print(f"\nfactoring gain: {g.mean():+.3f} ± {g.std():.3f}  (min {g.min():+.3f}, max {g.max():+.3f})")
    print(f"  -> {'ROBUST >0' if g.mean() - g.std() > 0 else 'within noise (mean-std <= 0)'}")

    if args.dump:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        rec = {"tag": args.tag or args.variant, "corpus": args.train_corpus, "variant": args.variant,
               "seeds": args.seeds, "n_docs": len(tr), "n_chunks": len(tr_chunks), "n_trans": n_trans,
               "chunk_len": args.chunk_len, "max_steps": max_steps, "axis_mask": args.axis_mask,
               "pos_weight_from": args.pos_weight_from, "dataset": args.dataset,
               "own_axis": {k: diag[k] for k in AXES}, "gains": gains}
        with args.dump.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"-> {args.dump}")


if __name__ == "__main__":
    main()

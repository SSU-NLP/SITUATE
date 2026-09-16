#!/usr/bin/env python3
r"""실험 ① 교사 애블레이션 — 학생(직행 경계 헤드) 고정, **교사 라벨만** 바꾼다.

사전등록 `outputs/experiments/2026-07-31_teacher_and_combiner/PREREG.md` (실행 전 작성, 미수정).
학생·학습설정·평가규칙은 `train_direct_boundary.py`(M8) 를 그대로 재사용한다 — 재구현 금지.

교사 6종: axis5_t3(현행) · axis5_t2 · axis5_t4 · cosdrift(스칼라) · space_only(단일축) ·
random(밀도만 일치 = 바닥 통제).

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/eacl_phase0/teacher_ablation.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
import train_direct_boundary as m8  # noqa: E402
from train_anchor_m4m5 import Corpus  # noqa: E402

OUT = REPO / "outputs/experiments/2026-07-31_teacher_and_combiner"
RATE = 0.0650          # 🔒 앵커 실측 경계율. 교사 밀도를 이 값으로 맞춘다.
RANDOM_SEED = 20260731  # 🔒 고정
AXIS_NAMES = ["protagonist", "space", "time", "causality", "intentionality"]  # 앵커 resets 열 순서
AXIS_SPACE = 1         # 레거시 `space_only` 용
TEACHERS = ["axis5_t3", "axis5_t2", "axis5_t4", "cosdrift", "space_only", "random"]


def per_doc_topr(score: np.ndarray, n: int, rate: float) -> np.ndarray:
    """문서 내 상위 rate 를 1 로. 밀도가 문서마다 일정해진다."""
    k = max(1, int(round(len(score) * rate)))
    lab = np.zeros(len(score), np.int8)
    lab[np.argpartition(-score, k - 1)[:k]] = 1
    return lab


def build_labels(teacher: str, c: Corpus, n_docs: int, emb: torch.Tensor,
                 dev: str) -> torch.Tensor:
    """전역 문장 인덱스에 교사 라벨을 싣는다(전이 t 의 라벨 = 왼쪽 문장 위치)."""
    y = torch.zeros(int(c.so[n_docs]), dtype=torch.int8, device=dev)
    rng = np.random.default_rng(RANDOM_SEED)
    for i in range(n_docs):
        s0, s1 = int(c.so[i]), int(c.so[i + 1])
        t0, t1 = int(c.to[i]), int(c.to[i + 1])
        R = c.resets[t0:t1]
        nt = t1 - t0
        if teacher.startswith("axis5_t"):
            lab = (R.sum(1) >= int(teacher[-1])).astype(np.int8)
        elif teacher == "space_only":
            lab = R[:, AXIS_SPACE].astype(np.int8)
        elif teacher.startswith("only_"):
            # 축 단독 교사 — 그 축의 리셋만 경계로. 밀도는 축마다 다르나 추론 임계가
            # 문서 내 분위수(r 동결)라 예측 밀도는 조건 간 동일하다.
            lab = R[:, AXIS_NAMES.index(teacher[len("only_"):])].astype(np.int8)
        elif teacher == "cosdrift":
            z = emb[s0:s1].float()
            d = (1.0 - (z[:-1] * z[1:]).sum(1)).cpu().numpy()   # 1 − cos, 인접쌍
            lab = per_doc_topr(d, nt, RATE)
        elif teacher == "random":
            lab = per_doc_topr(rng.random(nt), nt, RATE)
        else:
            raise SystemExit(f"🔴 알 수 없는 교사: {teacher}")
        y[s0:s0 + nt] = torch.from_numpy(lab.astype(np.int8)).to(dev)
    return y


def main() -> None:
    ap = argparse.ArgumentParser(description="교사 애블레이션 (학생 = M8 직행 헤드 고정)")
    ap.add_argument("--teachers", nargs="*", default=TEACHERS)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--limit-docs", type=int, default=0)
    ap.add_argument("--tag", default="",
                    help="🔒 산출 json 네임스페이스. 병렬 실행 시 반드시 지정(덮어쓰기 방지)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    m8.OUT = OUT                                   # 체크포인트 산출 경로 분리

    c, dev3 = Corpus("phase1"), Corpus("dev3")
    n = c.n_docs if not a.limit_docs else min(a.limit_docs, c.n_docs)
    tr_docs, va_docs = m8.doc_split(n)
    tr_ix, va_ix = m8.trans_left_index(c.so, tr_docs), m8.trans_left_index(c.so, va_docs)
    emb = torch.from_numpy(np.asarray(c.emb[:int(c.so[n])], dtype=np.float16)).to(dev)
    G = [dev3.gold[int(dev3.to[i]):int(dev3.to[i + 1])].astype(np.int8) for i in range(dev3.n_docs)]
    print(f"[data] 문서 {len(tr_docs):,}/{len(va_docs):,} · 전이 {len(tr_ix):,}/{len(va_ix):,} · "
          f"dev-3 gold {sum(int(g.sum()) for g in G)}", flush=True)

    results: dict = {"_meta": {"prereg": "PREREG.md 실험①", "rate": RATE,
                               "random_seed": RANDOM_SEED, "student": "M8 pair (786,945 params)",
                               "epochs": a.epochs, "lr": a.lr, "batch": a.batch}}
    for teacher in a.teachers:
        y = build_labels(teacher, c, n, emb, dev)
        pos = float(y[torch.from_numpy(tr_ix).to(dev)].float().mean())
        print(f"\n=== 교사 {teacher} · 라벨 양성률 {pos:.4f} ===", flush=True)
        runs = []
        for sd in a.seeds:
            args = argparse.Namespace(epochs=a.epochs, batch=a.batch, lr=a.lr,
                                      hidden=a.hidden, dropout=a.dropout)
            model, info = m8.train_one(f"{teacher}", sd, emb, None, y, tr_ix, va_ix, args, dev)
            r = m8.score_signals(m8.m8_dev3_signal(model, "pair", dev3, dev), G)
            r.update({"seed": sd, "n_params": info["n_params"],
                      "teacher_val_auprc": info["history"][-1]["val_auprc"]})
            runs.append(r)
            torch.save({"state": model.state_dict(), "teacher": teacher, "seed": sd},
                       OUT / f"student_{teacher}_seed{sd}.pt")
            print(f"  [{teacher} seed{sd}] dev-3 적응 {r['adaptive']['score']:.4f} "
                  f"({r['adaptive']['n_pred']}) · 분위수 {r['percentile']['score']:.4f} "
                  f"· AUC {r['auc_pooled']:.4f}", flush=True)
        results[teacher] = {"label_pos_rate": pos, "runs": runs, **m8.agg(runs)}
        jpath = OUT / f"teacher_ablation{('_' + a.tag) if a.tag else ''}.json"
        jpath.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    ref = json.loads((OUT / "teacher_ablation.json").read_text())
    base = results.get("axis5_t3") or ref["axis5_t3"]     # 현행 교사 = 기준(기존 런 재사용)
    print(f"\n=== 판정 (사전등록 §① · Δ = axis5_t3 − 대안, 실질성 한계 0.06) ===")
    for t in a.teachers:
        if t == "axis5_t3" or t not in results:
            continue
        for rule in ("adaptive", "percentile"):
            d = base[rule]["mean"] - results[t][rule]["mean"]
            print(f"  {t:12s} {rule:11s} {results[t][rule]['mean']:.4f} · Δ {d:+.4f}"
                  f"{'  ← 실질적' if d > 0.06 else ''}")
    print(f"\n-> {jpath}")


if __name__ == "__main__":
    main()

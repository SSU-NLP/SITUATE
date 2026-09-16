#!/usr/bin/env python3
r"""🔒 사전등록용 **허용폭(tolerance band) 산출** — 결과를 보기 전에 잠글 숫자.

## 왜
codex 판정(2026-07-30, `outputs/experiments/2026-07-30_m5_retrain_design/CODEX_ANSWER_2.md`)이
**현재 정보로 확정 불가**라고 남긴 두 숫자를 여기서 산출한다:

1. **W0 재현 허용범위** — 새 학습 하네스로 같은 seed 를 다시 돌렸을 때 "재현됐다"고 볼 폭.
   완전 결정론이 보장되지 않으므로 **기존 seed 간 변동폭**을 허용범위로 쓴다(codex 지시).
2. **대화 비열등 허용폭** — W1 이 대화 선택셋에서 이만큼까지 나빠지는 것은 감수한다는 폭.

## 무엇을
동결 규칙(`online_nf` c=1.0) · 신호 S5 · 기존 W0 체크포인트 3 seed 로,
**두 선택셋에서만** 잡음 크기를 잰다:

| 선택셋 | 근거 |
|---|---|
| **dev-3** (pieman·tunnel·monkey) | 서사 작동점 선택셋 (D-B41 §2) |
| **superseg/validation 앞 60대화** | 대화 선택셋 (D-B41 §2, 공식 test 와 disjoint) |

🔴 **DTS 공식 test·봉인 시험지는 건드리지 않는다** — 판독 전용.
🔴 이 스크립트는 **성능을 비교하지 않는다.** 잡음 크기만 잰다 ⇒ 조건 선택이 아니므로 오염이 아니다.

두 종류의 변동을 분리해서 낸다:
- **seed 변동**: 같은 설정을 seed 만 바꿔 학습했을 때의 pooled Score 퍼짐 (학습 확률성)
- **문서단위 bootstrap**: 문서를 복원추출해 재집계했을 때의 퍼짐 (표본 확률성)

⚠️ dev-3 는 **문서가 3편뿐**이라 bootstrap 이 거의 퇴화한다(가능한 재표본 조합 10가지).
그 사실을 숨기지 말고 같이 적는다 — ledger 가 이미 지목한 `bootstrap_by_doc` degenerate 문제다.

## 실행
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/eacl_phase0/tolerance_bands.py
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
from db29_final_eval import load  # noqa: E402
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from hi_ontop.hi_ontop_v2 import adaptive_boundaries  # noqa: E402
from train_anchor_m4m5 import Corpus, axis_probs  # noqa: E402

OUT = REPO / "outputs/experiments/2026-07-30_m5_retrain_design"
SDS = REPO / "benchmarks/superdialseg_data"
SEEDS = (0, 1, 2)
NS = "_per_slot_slot"
C_MAIN = 1.0                     # 🔒 동결 규칙 그대로. 여기서 c 를 고르지 않는다
N_DLG = 60                       # 대화 선택셋 크기 (D-B41 §2)
B = 10000                        # bootstrap 반복
BOOT_SEED = 20260730


def rule(s: np.ndarray) -> np.ndarray:
    seq = np.concatenate([[0.0], s]).tolist()
    return np.asarray(adaptive_boundaries(seq, c=C_MAIN, mode="online_nf"), np.int8)[:len(s)]


def load_dialogue_val(n_doc: int):
    """superseg/**validation** 앞 n_doc 대화. 공식 test 와 disjoint (D-B41 §2 확인 완료).

    라벨 규약 = 세그먼트의 **마지막** turn 이 1 (`baseline_opt_freeze.load_dialogue` 와 동일).
    """
    raw = json.loads((SDS / "superseg/segmentation_file_validation.json").read_text(encoding="utf-8"))
    arr = raw["dial_data"][list(raw["dial_data"])[0]]
    out = []
    for d in arr[:n_doc]:
        utts = [t["utterance"] for t in d["turns"]]
        y = np.asarray([int(t.get("segmentation_label", 0)) for t in d["turns"]], np.int8)
        if len(utts) >= 3:
            out.append((utts, y[:len(utts) - 1]))       # 전이 단위 = turn 수 - 1
    return out


_BD: dict = {}


def _binit(per_doc):
    _BD["d"] = per_doc


def _bone(idx):
    d = _BD["d"]
    return score_dialogues([d[i][0] for i in idx], [d[i][1] for i in idx])["score"]


def boot_ci(per_doc, rng, workers: int = 32, n_boot: int = B):
    """문서 복원추출 -> pooled Score 분포. 반환 (std, p2.5, p97.5).

    병목은 GPU 가 아니라 **nltk Pk/WindowDiff 채점(CPU)** 이다 — 재표본을 프로세스 병렬로 돌린다.
    (단일 코어로 돌리면 60대화 × 10,000회 = 60만 채점에 40분 이상 걸린다.)
    """
    n = len(per_doc)
    idxs = [rng.integers(0, n, n) for _ in range(n_boot)]
    if workers > 1 and n_boot > 100:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers, initializer=_binit, initargs=(per_doc,)) as ex:
            vals = list(ex.map(_bone, idxs, chunksize=max(1, n_boot // (workers * 8))))
    else:
        _binit(per_doc)
        vals = [_bone(i) for i in idxs]
    v = np.asarray(vals)
    return float(v.std(ddof=1)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description="사전등록 허용폭 산출 (잡음 크기만 측정)")
    ap.add_argument("--boot", type=int, default=B)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    # 대화 선택셋은 memmap 이 없으므로 여기서 직접 인코딩 (frozen mpnet, 전 방법 공통)
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                              device=dev)
    dlg = load_dialogue_val(N_DLG)
    print(f"[대화 선택셋] superseg/validation 앞 {len(dlg)}대화 · "
          f"{sum(len(y) for _, y in dlg):,} 전이 · gold {sum(int(y.sum()) for _, y in dlg)}", flush=True)
    dlg_emb = [torch.tensor(np.asarray(enc.encode(u, normalize_embeddings=True, batch_size=64),
                                       np.float32), device=dev).unsqueeze(0) for u, _ in dlg]
    dev3 = Corpus("dev3")
    print(f"[서사 선택셋] dev-3 {dev3.n_docs}편 · {len(dev3.gold):,} 전이 · "
          f"gold {int(dev3.gold.sum())}", flush=True)

    res: dict = {"c": C_MAIN, "rule": "online_nf", "signal": "S5", "boot": a.boot, "sets": {}}
    per_seed_docs: dict[str, dict[int, list]] = {"dev3": {}, "superseg_val60": {}}
    from hi_ontop.factored_slot import AXES
    for seed in SEEDS:
        m = load("M5", seed, dev, ns=NS)[0]
        with torch.no_grad():
            P = axis_probs(m, dev3, list(range(dev3.n_docs)), dev, bs=64, with_anchor=False)
            per_seed_docs["dev3"][seed] = [
                (np.concatenate([dev3.gold[int(dev3.to[i]):int(dev3.to[i + 1])], [0]]).tolist(),
                 np.concatenate([rule(p.sum(1)), [0]]).tolist()) for i, p in enumerate(P)]
            docs = []
            for E, (_, y) in zip(dlg_emb, dlg):
                n = E.shape[1]
                o = m.forward_batched(
                    E, torch.zeros(1, n - 1, 5, dtype=torch.long, device=dev),
                    torch.zeros(1, n, 3, dtype=torch.long, device=dev),
                    torch.ones(1, n, device=dev), torch.ones(1, n - 1, device=dev))
                s = torch.stack([torch.sigmoid(o["change_logits"][ax]) for ax in AXES],
                                -1)[0].cpu().numpy().sum(1)
                docs.append((np.concatenate([y, [0]]).tolist(),
                             np.concatenate([rule(s)[:len(y)], [0]]).tolist()))
            per_seed_docs["superseg_val60"][seed] = docs
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()
        print(f"  seed{seed} 완료", flush=True)

    rng = np.random.default_rng(BOOT_SEED)
    L = ["# 🔒 사전등록 허용폭 — 결과 보기 전에 잠그는 숫자\n",
         f"> M5 W0(현행 동결) 3 seed · 규칙 `online_nf c={C_MAIN}` · 신호 **S5(5축 전부)** ·"
         f" bootstrap {a.boot:,}회(seed {BOOT_SEED})",
         "> 🔴 이 표는 **성능 비교가 아니다. 잡음 크기만 잰다** ⇒ 조건 선택이 아니므로 오염이 아니다.",
         "> 🔴 DTS 공식 test·봉인 시험지 미사용. 선택셋 두 개만.\n",
         "\n## 1. 실측 변동\n",
         "| 선택셋 | seed별 Score | seed std | 최대-최소 | bootstrap std | 95% CI 폭 |",
         "|---|---|---|---|---|---|"]
    for name in ("dev3", "superseg_val60"):
        sc = []
        for seed in SEEDS:
            d = per_seed_docs[name][seed]
            sc.append(score_dialogues([x[0] for x in d], [x[1] for x in d])["score"])
        bs, lo, hi = boot_ci(per_seed_docs[name][SEEDS[0]], rng, n_boot=a.boot)
        res["sets"][name] = {"per_seed": sc, "seed_std": float(np.std(sc, ddof=1)),
                             "seed_range": float(max(sc) - min(sc)),
                             "boot_std": bs, "boot_ci": [lo, hi],
                             "n_docs": len(per_seed_docs[name][SEEDS[0]])}
        L.append(f"| {name} ({res['sets'][name]['n_docs']}편) | "
                 + " · ".join(f"{x:.4f}" for x in sc)
                 + f" | **{np.std(sc, ddof=1):.4f}** | {max(sc)-min(sc):.4f} | "
                   f"**{bs:.4f}** | {hi-lo:.4f} |")

    d3, dl = res["sets"]["dev3"], res["sets"]["superseg_val60"]
    # 허용폭 = max(seed std, bootstrap std) × 2  (2σ = 통상적 잡음 상한)
    rep = 2 * max(d3["seed_std"], d3["boot_std"])
    nin = 2 * max(dl["seed_std"], dl["boot_std"])
    res["tolerance"] = {"w0_reproduction_band_dev3": rep, "dialogue_noninferiority_band": nin,
                        "definition": "2 × max(seed_std, bootstrap_std)"}
    L += ["\n## 2. 🔒 잠그는 허용폭\n",
          "> 정의 = **2 × max(seed std, bootstrap std)**. 두 잡음원 중 큰 쪽의 2σ 를 쓴다"
          " — 어느 하나만 쓰면 다른 쪽 잡음을 실제 효과로 오독한다.\n",
          "| 항목 | 값 | 쓰는 곳 |", "|---|---|---|",
          f"| **W0 재현 허용범위** (dev-3) | **±{rep:.4f}** | 새 하네스로 W0 seed0 재학습 시,"
          f" dev-3 Score 가 이 안이면 기존 3 seed 재사용 가능. 벗어나면 W0 도 3 seed 전부 재학습 |",
          f"| **대화 비열등 허용폭** | **−{nin:.4f}** | W1 이 superseg/validation 60 에서 W0 대비"
          f" 이보다 더 나빠지면 비열등 실패 |",
          "\n🔴 **이 두 숫자는 지금 잠긴다.** 결과를 본 뒤 넓히거나 좁히지 않는다.",
          "\n## 3. 한계 (제거 금지)\n",
          f"- 🔴 **dev-3 는 문서가 {d3['n_docs']}편뿐이라 bootstrap 이 거의 퇴화한다**"
          " (복원추출 가능 조합 10가지). ledger 가 지목한 `bootstrap_by_doc` degenerate 문제 그대로다."
          " dev-3 의 bootstrap 폭은 **참고값**이며, seed std 를 우선한다.",
          "- seed 가 3개뿐이라 seed std 자체의 불확실성이 크다(자유도 2).",
          "- seed 변동은 **다른 seed** 간 폭이다. 같은 seed 재학습의 결정론 여부는 측정하지 않았다"
          " — 그래서 codex 지시대로 이 폭을 재현 허용범위의 **보수적 상한**으로 쓴다.",
          "- 이 폭은 W0(현행 손실)에서 잰 것이다. W1 의 잡음이 더 클 수 있다.",
          "- 대화 선택셋 인코딩은 여기서 즉석 계산했다(memmap 아님). 인코더·규약은 동일."]
    (OUT / "TOLERANCE_BANDS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "tolerance_bands.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print("\n".join(L))
    print(f"\n-> {OUT / 'TOLERANCE_BANDS.md'}")


if __name__ == "__main__":
    main()

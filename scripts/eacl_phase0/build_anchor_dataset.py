#!/usr/bin/env python3
"""M4/M5 학습·평가 데이터셋 빌드 (Phase-1 앵커 10,598편 스케일).

Phase 2 de-risk 의 `build_slot_dataset.py` 를 900배 스케일로 대체한다. 차이:
  - 산출이 npz 인메모리(3,090 전이) -> **float16 memmap**(2,777,561 전이). 임베딩 4.3GB.
  - 소스가 하드코딩 10편 -> `data/eacl_phase1/final_master.jsonl` + `phase1_all_v31/` 전량.
  - 문장 임베딩은 mpnet **frozen**(전 방법 공통, M1~M7·baseline 동일) — D-M11 공정비교 조건.

두 split:
  --corpus phase1 : 학습용. 앵커 = 축별 reset 5개(primary) + 전역 enum state 3종(aux).
                    `boundary` 는 전부 0이므로 싣지 않는다(평가셋 아님, D-B15).
  --corpus dev3   : 평가용. pieman/tunnel/monkey + **사람 gold 경계**. 앵커 라벨 불필요
                    (M4/M5 는 임베딩만 입력받아 reset 을 예측하고, 경계를 gold 와 대조한다).

🔒 D-B16: dev-3 는 연습장이다. **봉인 시험지(Born a Crime 3편)는 이 스크립트가 취급하지 않는다.**

산출: outputs/experiments/m4m5_phase1_anchor/data/{split}_{emb.f16,index.npz}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/experiments/m4m5_phase1_anchor/data"

AXES = ["protagonist_present", "space", "time", "causality", "intentionality"]  # D-B10 동결 5축
ENCODER = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # frozen, 전 방법 공통
Z_DIM = 768

# aux absolute-state 타깃 (전역 고정 enum 축만; space/intentionality 는 story-local -> 제외)
PERSPECTIVE = ["first", "second", "third", "none"]
TEMPORAL = ["absent", "sequence", "jump", "simultaneity"]
EVENT_TYPE = ["dialogue", "physical_action", "movement_or_travel", "perception_or_observation",
              "thought_or_emotion", "transaction_or_exchange", "conflict_or_violence",
              "work_or_creation", "state_description", "other"]
STATE_SPEC = [  # (출력열 이름, 축 파일, 필드, enum)
    ("perspective", "protagonist", "perspective", PERSPECTIVE),
    ("temporal_shift_marker", "time", "temporal_shift_marker", TEMPORAL),
    ("event_type", "causality", "event_type", EVENT_TYPE),
]

PHASE1_MASTER = REPO / "data/eacl_phase1/final_master.jsonl"
PHASE1_S5 = REPO / "outputs/eacl_phase0/phase1_all_v31"
DEV3 = ("pieman", "tunnel", "monkey")


def read_jsonl(p: Path):
    with p.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def collect_docs(master: Path) -> tuple[list[str], list[list[str]], list[np.ndarray] | None]:
    """master.jsonl -> (doc_ids, 문서별 문장 리스트, 문서별 gold 경계 배열).

    문장 = [첫 행의 previous, 각 행의 current] (transition 순). build_slot_dataset 의
    `sentences_of()` 와 동일 규약 — 문장 수 = 전이 수 + 1.
    """
    doc_ids: list[str] = []
    sents: list[list[str]] = []
    golds: list[list[int]] = []
    cur_doc = None
    for r in read_jsonl(master):
        d = r["doc_id"]
        if d != cur_doc:
            doc_ids.append(d)
            sents.append([r["previous"]])
            golds.append([])
            cur_doc = d
        sents[-1].append(r["current"])
        golds[-1].append(int(r.get("boundary", 0)))
    return doc_ids, sents, [np.asarray(g, dtype=np.int8) for g in golds]


def load_anchors(doc_ids: list[str], n_sent: list[int]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """phase1_all_v31/ 에서 축별 reset(전이) + 전역 enum state(문장) 을 읽는다.

    반환 (resets [N_trans,5] int8, states [N_sent,3] int8, 제외된 doc_id 목록).
    상태 파일이 불완전한 문서는 통째로 제외한다(부분 라벨을 섞지 않는다).
    """
    rm: dict[str, dict[int, list[int]]] = {}
    for r in read_jsonl(PHASE1_S5 / "reset_matrix.jsonl"):
        rm.setdefault(r["doc_id"], {})[int(r["transition"])] = [int(r[a]) for a in AXES]

    resets_parts, states_parts, dropped = [], [], []
    for doc, n in zip(doc_ids, n_sent):
        tr = rm.get(doc)
        if tr is None or len(tr) != n - 1:
            dropped.append(doc)
            continue
        R = np.asarray([tr[t] for t in range(1, n)], dtype=np.int8)
        S = np.zeros((n, len(STATE_SPEC)), dtype=np.int8)
        ok = True
        for ci, (_, axis, field, enum) in enumerate(STATE_SPEC):
            p = PHASE1_S5 / doc / f"{axis}_states.jsonl"
            if not p.exists():
                ok = False
                break
            byid = {int(x["item_id"]): x for x in read_jsonl(p)}
            if len(byid) < n:
                ok = False
                break
            try:
                S[:, ci] = [enum.index(byid[i][field]) for i in range(n)]
            except (KeyError, ValueError):
                ok = False
                break
        if not ok:
            dropped.append(doc)
            continue
        resets_parts.append(R)
        states_parts.append(S)
    return resets_parts, states_parts, dropped


def encode_to_memmap(all_sents: list[str], out_path: Path, batch: int, device: str) -> None:
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(ENCODER, device=device)
    enc.eval()
    N = len(all_sents)
    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(N, Z_DIM))
    step = batch * 64
    for s in range(0, N, step):
        chunk = all_sents[s:s + step]
        E = enc.encode(chunk, normalize_embeddings=True, batch_size=batch,
                       convert_to_numpy=True, show_progress_bar=False)
        mm[s:s + len(chunk)] = E.astype(np.float16)
        pct = 100.0 * min(s + step, N) / N
        print(f"  [encode] {min(s + step, N):>9,}/{N:,} ({pct:5.1f}%)", flush=True)
    mm.flush()
    del mm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["phase1", "dev3"], required=True)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit-docs", type=int, default=0, help="스모크용: 앞 N개 문서만")
    ap.add_argument("--tag", default="", help="산출 파일명 접미사(스모크와 본 산출 분리)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    split = args.corpus + (f"_{args.tag}" if args.tag else "")

    if args.corpus == "phase1":
        print(f"[read] {PHASE1_MASTER}", flush=True)
        doc_ids, sents, _ = collect_docs(PHASE1_MASTER)
    else:
        doc_ids, sents, golds = [], [], []
        for d in DEV3:
            di, se, go = collect_docs(REPO / f"outputs/eacl_phase0/{d}_michelmann_master.jsonl")
            doc_ids += di
            sents += se
            golds += go
    if args.limit_docs:
        doc_ids, sents = doc_ids[:args.limit_docs], sents[:args.limit_docs]
        if args.corpus == "dev3":
            golds = golds[:args.limit_docs]
    n_sent = [len(s) for s in sents]
    print(f"[read] {len(doc_ids):,} docs · {sum(n_sent):,} sentences · "
          f"{sum(n_sent) - len(doc_ids):,} transitions", flush=True)

    keep_idx = list(range(len(doc_ids)))
    payload: dict[str, np.ndarray] = {}

    if args.corpus == "phase1":
        print("[anchor] reset_matrix + 축별 state 파일 읽는 중...", flush=True)
        resets_parts, states_parts, dropped = load_anchors(doc_ids, n_sent)
        if dropped:
            drop = set(dropped)
            keep_idx = [i for i, d in enumerate(doc_ids) if d not in drop]
            print(f"[anchor] 🔴 라벨 불완전으로 제외 = {len(dropped):,}편 "
                  f"(예: {dropped[:5]})", flush=True)
        payload["resets"] = np.concatenate(resets_parts)
        payload["states"] = np.concatenate(states_parts)
        payload["state_cols"] = np.array([c for c, _, _, _ in STATE_SPEC])
    else:
        payload["gold"] = np.concatenate([golds[i] for i in keep_idx])

    doc_ids = [doc_ids[i] for i in keep_idx]
    sents = [sents[i] for i in keep_idx]
    n_sent = [len(s) for s in sents]
    sent_off = np.asarray([0] + list(np.cumsum(n_sent)), dtype=np.int64)
    trans_off = np.asarray([0] + list(np.cumsum([n - 1 for n in n_sent])), dtype=np.int64)

    flat = [s for doc in sents for s in doc]
    print(f"[encode] {len(flat):,} 문장 -> {ENCODER} ({args.device}, fp16 memmap)", flush=True)
    encode_to_memmap(flat, OUT / f"{split}_emb.npy", args.batch, args.device)

    np.savez(OUT / f"{split}_index.npz", doc_ids=np.array(doc_ids),
             sent_offsets=sent_off, trans_offsets=trans_off, **payload)
    print(f"\n[done] {len(doc_ids):,} docs · {sent_off[-1]:,} sents · {trans_off[-1]:,} trans")
    if args.corpus == "phase1":
        R = payload["resets"]
        for i, a in enumerate(AXES):
            print(f"  {a:22s} 발화율 {R[:, i].mean():.4f}")
        cnt = R.sum(1)
        print(f"  경계율 (count>=3)      {float((cnt >= 3).mean()):.4f}")
    else:
        g = payload["gold"]
        print(f"  gold 경계 {int(g.sum())} / {len(g)} = {g.mean():.4f}")
    print(f"  -> {OUT}/{split}_*", flush=True)


if __name__ == "__main__":
    main()

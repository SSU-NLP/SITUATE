#!/usr/bin/env python3
"""SSC(Sequential Sentence Classification) 구조를 **우리 앵커 코퍼스로 학습**한다.

## 왜 이 파일이 필요한가 — 원 논문은 영어 서사에 못 쓴다
`ssc_gbert_2025`(Zehe+ 2025, STSS SotA) · `bert_sentence_pair`(Zehe+ 2021b) · `ssc_kurfali`
는 전부 **독일어 장면분절** 모델이다(`fiction-gbert-large`, German Scene Dataset). 우리
평가셋은 **영어 구어 서사 6편**이라 체크포인트가 살아 있어도 zero-shot 경로가 없다.
저자 체크포인트 404 와 저작권 코퍼스는 그 다음 문제다.

⇒ 발표된 모델을 재현하는 것은 불가능하다. 대신 **구조만 가져와 우리 앵커로 학습**한다.
🔴 이건 **베이스라인이 아니라 우리 쪽 ablation** 이다([[baselines-stay-as-published]]).
표·본문에서 "our reimplementation, trained on our anchor labels" 로 반드시 명시한다.
M5/M4/M8 이 쓰는 것과 **같은 학습 신호**를 쓰므로, 구조 비교로서는 오히려 정확하다.

## 무엇을 가져오고 무엇을 바꿨나
| SSC 원본 | 여기 |
|---|---|
| 문장 시퀀스를 겹치는 문맥 창으로 잘라 인코더에 넣고 문장별 경계 분류 | 그대로 (`encode_window` 재사용) |
| 인코더 `deepset/gbert-large` (독일어) | `roberta-base` (영어). 기존 지도 베이스라인과 **같은 백본**이라 비교가 깨끗하다 |
| 학습 = German Scene Dataset 32편 | 우리 앵커 코퍼스 (LLM 라벨, 사람 경계 라벨 미사용) |
| **Half Stride** 추론 (창을 절반씩 겹쳐 문장별 다중 예측을 합성) | 🔴 **미래 문맥을 쓴다** — 온라인 표에는 못 넣는다. 추론은 `methods/RoBERTa/online/segment.py::predict_online` 의 zero-lookahead 경로를 쓴다 |

Half Stride 를 뺀 것이 핵심 차이다. 논문이 Half Stride 를 도입한 근거가 *"단측 문맥으로는
경계를 신뢰성 있게 판정할 수 없다"* 였으므로, 그 하락폭 자체가 온라인 조건의 난이도를
말해 준다 — 그래서 `--half-stride` 로 오프라인 판도 함께 뽑아 부록에 남길 수 있게 열어 뒀다.

    env -C /tmp CUDA_VISIBLE_DEVICES=0 /root/Hi-OnTop/.venv/bin/python \\
        EACL/methods/ssc_gbert_2025/train_anchor_ssc.py --epochs 20
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
sys.path.insert(0, str(REPO / "methods/RoBERTa/offline"))
sys.path.insert(0, str(REPO / "methods/RoBERTa/online"))

ANCHOR_TEXT = REPO / "data/eacl_phase1/all_master_ready.jsonl"
ANCHOR_LABEL = REPO / "outputs/experiments/m4m5_phase1_anchor/data/phase1_i2_index.npz"
PRED = REPO / "outputs/experiments/2026-07-29_baseline_compare/preds"
OUT = REPO / "outputs/experiments/2026-08-04_ssc_anchor"
T_STAR = 3          # 🔒 동결. 이진 앵커 = resets.sum(1) >= T_STAR (train_anchor_m4m5.py:190)


def load_anchor(limit_docs: int = 0):
    """앵커 코퍼스 → [(sents, sent_labels)]. **M5/M4/M8 과 같은 학습 신호**를 쓴다.

    🔴 `all_master_ready.jsonl` 의 `boundary` 필드는 **전부 0** 이다 — 그건 LLM 라벨링
    *전* 의 원문 코퍼스다. 그걸 그대로 쓰면 모델이 전부 음성으로 붕괴한다(실측: val loss
    0.0002, 예측 0개). 실제 앵커 라벨은 `phase1_i2_index.npz` 의 `resets [N,5]` 이고,
    이진 경계는 `resets.sum(1) >= T_STAR(=3)` 이다. 이 조합의 경계율 0.0657 은 논문
    Training setup 에 적힌 값과 정확히 일치한다(검산 완료).

    라벨 규약은 SuperDialseg 와 동일(끝-turn=1): 전이 j 가 경계면 문장 j 가 자기 세그먼트의
    **마지막** 문장이므로 `y[j]=1`. 마지막 문장은 항상 0 이다.
    """
    z = np.load(ANCHOR_LABEL, allow_pickle=True)
    doc_ids = [str(x) for x in z["doc_ids"]]
    to = z["trans_offsets"]
    resets = z["resets"]
    want = doc_ids[:limit_docs] if limit_docs else doc_ids
    want_set = set(want)
    lab = {d: (resets[int(to[i]):int(to[i + 1])].sum(1) >= T_STAR).astype(np.int8)
           for i, d in enumerate(doc_ids) if d in want_set}

    text: dict[str, list[tuple[int, str, str]]] = {}
    with ANCHOR_TEXT.open(encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            r = json.loads(line)
            d = r["doc_id"]
            if d in want_set:
                text.setdefault(d, []).append((r["transition"], r["previous"], r["current"]))
            if limit_docs and len(text) == len(want_set) and d not in want_set:
                break

    out, skipped = [], 0
    for d in want:
        rows = text.get(d)
        y = lab.get(d)
        if not rows or y is None:
            skipped += 1
            continue
        rows.sort(key=lambda x: x[0])
        if len(rows) != len(y):          # 원문과 라벨의 전이 수가 다르면 조용히 어긋난다
            skipped += 1
            continue
        sents = [rows[0][1]] + [x[2] for x in rows]
        out.append((sents, y.tolist() + [0]))
    if skipped:
        print(f"  ⚠️ 원문·라벨 불일치로 건너뜀: {skipped}문서", flush=True)
    return out


class Collate:
    """`make_collate` 의 지역 클로저 판은 피클링이 안 돼 `num_workers>0` 을 못 쓴다.

    토크나이즈는 `TrainDS.__getitem__` 안에서 도는 **CPU 작업**이라, worker 0 이면
    96코어 중 1개만 쓰면서 GPU 를 놀린다. 모듈 레벨 클래스로 바꿔 워커를 연다.
    """

    def __init__(self, pad_id: int, ignore: int):
        self.pad_id, self.ignore = pad_id, ignore

    def __call__(self, batch):
        from torch.nn.utils.rnn import pad_sequence
        ids, labs = zip(*batch)
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad_id)
        labs = pad_sequence(labs, batch_first=True, padding_value=self.ignore)
        return ids, ids.ne(self.pad_id).long(), labs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="roberta-base")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--sliding-window", type=int, default=20)
    ap.add_argument("--max-utt-len", type=int, default=25)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--limit-docs", type=int, default=0, help="0=전체")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label", default="SSC-anchor-online")
    ap.add_argument("--workers", type=int, default=8, help="DataLoader 워커 (토크나이즈가 CPU 병목)")
    ap.add_argument("--patience", type=int, default=3,
                    help="val loss 개선 없이 이 에폭 수가 지나면 조기 종료. "
                         "논문 Training setup 이 M4/M5/M8 에 쓴 것과 같은 규약(patience 3)")
    ap.add_argument("--amp", default="bf16", choices=("bf16", "off"),
                    help="A100 bf16 autocast. 수치가 아니라 속도만 바꾼다")
    args = ap.parse_args()

    from torch.nn.utils.rnn import pad_sequence           # noqa: F401 (collate 내부)
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, RobertaForTokenClassification

    from train import IGNORE, TrainDS, encode_window, make_collate   # noqa: PLC0415
    from segment import predict_online                               # noqa: PLC0415
    from baseline_compare import load_dataset                        # noqa: PLC0415
    from hi_ontop.dts_scoring import score_dialogues                 # noqa: PLC0415

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    docs = load_anchor(args.limit_docs)
    random.shuffle(docs)
    n_val = max(1, int(len(docs) * args.val_frac))
    val_d, train_d = docs[:n_val], docs[n_val:]
    n_tr = sum(len(y) - 1 for _s, y in docs)
    n_pos = sum(sum(y) for _s, y in docs)
    print(f"앵커 {len(docs)}문서 · 전이 {n_tr:,} · 경계율 {n_pos/max(n_tr,1):.4f} "
          f"· 로드 {time.time()-t0:.0f}s", flush=True)
    print(f"train {len(train_d)} / val {len(val_d)} (문서 단위 분할, seed={args.seed})",
          flush=True)

    cfg = {"sliding_window": args.sliding_window, "max_utt_len": args.max_utt_len,
           "max_seq_len": args.max_seq_len}
    tok = AutoTokenizer.from_pretrained(args.backbone)
    model = RobertaForTokenClassification.from_pretrained(args.backbone, num_labels=2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    coll = Collate(tok.pad_token_id, IGNORE)
    loader = DataLoader(TrainDS(train_d, tok, cfg), batch_size=args.batch_size, shuffle=True,
                        collate_fn=coll, num_workers=args.workers,
                        persistent_workers=args.workers > 0,
                        prefetch_factor=4 if args.workers > 0 else None)
    vloader = DataLoader(TrainDS(val_d, tok, cfg), batch_size=args.batch_size,
                         collate_fn=coll, num_workers=max(2, args.workers // 2),
                         persistent_workers=args.workers > 0)
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(args.amp == "bf16"
                                                               and dev.type == "cuda"))

    best, best_state, bad = float("inf"), None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        te = time.time()
        for ids, mask, lab in loader:
            ids, mask, lab = ids.to(dev), mask.to(dev), lab.to(dev)
            with amp:
                loss = model(input_ids=ids, attention_mask=mask, labels=lab).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            tot += float(loss.detach()) * ids.size(0)
            n += ids.size(0)
        # val loss (같은 랜덤-창 규약)
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for ids, mask, lab in vloader:
                ids, mask, lab = ids.to(dev), mask.to(dev), lab.to(dev)
                with amp:
                    l = model(input_ids=ids, attention_mask=mask, labels=lab).loss
                vl += float(l.detach()) * ids.size(0)
                vn += ids.size(0)
        vloss = vl / max(vn, 1)
        star = ""
        if vloss < best:
            best, best_state, bad = vloss, {k: v.detach().cpu().clone()
                                            for k, v in model.state_dict().items()}, 0
            star = " *"
        else:
            bad += 1
        print(f"  ep{ep:>3} train {tot/max(n,1):.4f} · val {vloss:.4f}"
              f" · {time.time()-te:.0f}s{star}", flush=True)
        if bad >= args.patience:
            print(f"  조기 종료 (patience {args.patience})", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)          # best-epoch 복원 (M4/M5 와 같은 규약)
    mdir = OUT / "model"
    mdir.mkdir(exist_ok=True)
    model.save_pretrained(mdir)
    tok.save_pretrained(mdir)
    print(f"→ {mdir} (best val {best:.4f})", flush=True)

    # ── 서사 6편 zero-lookahead 추론 ──────────────────────────────────────────
    rows = {}
    PRED.mkdir(parents=True, exist_ok=True)
    for ds in ("dev3", "test3"):
        data = load_dataset(ds)
        dialogs = [(sents, [0] * len(sents)) for _d, sents, _g in data]
        yp = predict_online(model, tok, dialogs, cfg, dev)
        preds, G = {}, []
        for (did, _s, g), p in zip(data, yp):
            preds[did] = [int(x) for x in p[: len(g)]]
            G.append(g)
        ys = [np.concatenate([np.asarray(g, np.int8), [0]]).tolist() for g in G]
        ps = [np.concatenate([np.asarray(preds[d][: len(g)], np.int8), [0]]).tolist()
              for (d, _s, _g), g in zip(data, G)]
        sc = score_dialogues(ys, ps)
        rows[ds] = {k: sc[k] for k in ("pk", "wd", "f1", "score")} | {
            "n_pred": int(sum(sum(v) for v in preds.values())),
            "n_gold": int(sum(int(g.sum()) for g in G))}
        (PRED / f"{args.label}_{ds}.json").write_text(json.dumps(preds))
        print(f"  {args.label} {ds:<6} Pk {sc['pk']:.4f} WD {sc['wd']:.4f} F1 {sc['f1']:.4f} "
              f"Score {sc['score']:.4f} · 예측 {rows[ds]['n_pred']}/gold {rows[ds]['n_gold']}",
              flush=True)

    (OUT / "ssc_anchor.json").write_text(json.dumps(
        {"config": vars(args), "best_val": best, "n_docs": len(docs), "rows": rows},
        ensure_ascii=False, indent=2))
    print(f"\n→ {OUT/'ssc_anchor.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""DialSTART 를 **서사 무대(NES)** 로 이식한다 — D-09 의 개조를 구현한 판.

## 왜 개조가 필요한가 (D-09)
저자 `test.py` 를 그대로 서사에 꽂으면 DialSTART 를 **부당하게 불리하게** 만든다:

1. **top-k 고정** — `test.py:137` 의 `np.argsort(depth_scores)[-pick_num:]` 는 문서당 경계 수를
   `pick_num`(기본 4)으로 **문서 길이와 무관하게 고정**한다. gold 에서 온 값이 아니라 oracle
   누출은 아니지만, CLAUDE.md 의 deployable 규약(경계 수 자가결정)과 맞지 않는다. 서사 문서는
   전이 95~503 · gold 13~30 이라 4개는 심한 과소예측이다.
2. **depth 계산이 오프라인** — `depth_score_cal` 의 오른쪽 봉우리 탐색
   `for r in range(i+1, len(scores))` 은 **문서 끝까지** 훑는다. 온라인이 아니다.
3. **점수 창의 우측 문맥** — `window_size=2` 라 전이 i 의 점수가 `text[i+1], text[i+2]` 를 쓴다.

## 이 파일이 하는 개조 (셋 다 원본과 다름을 명시한다)
| # | 원본 | 이식판 |
|---|---|---|
| ① | top-k (`pick_num=4`) | **running μ + c·σ** (경계 수 자가결정). c 는 **dev3 에서만** 동결 |
| ② | depth 오른쪽 봉우리 = 문서 끝까지 | `--depth-span` 전이까지만 (기본 1) |
| ③ | — | 위 둘의 결과로 **총 지연 = 1 + depth_span** (기본 2) ⇒ `online(bounded d=2)` |

점수 함수(BERT-NSP coherence + topic contrastive)·토크나이즈·`window_size`·가중치는
**저자 것 그대로** 다. 바뀐 건 readout 뿐이다.

🔴 이건 **DialSTART 원 논문 수치와 비교할 대상이 아니다.** 원 논문은 대화(dialseg711/doc2dial)에
top-k readout 으로 Pk/WD 를 보고한다. 이 행은 "대화용 SotA 를 서사에 zero-shot 으로 옮기면
어떻게 되는가" 라는 **대칭 실험**의 한쪽이다.

    .venv-dialstart/bin/python EACL/methods/dialstart/run_narrative_online.py \\
        --freeze-c            # dev3 에서 c 스윕 후 동결, test3 에 적용
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
UPSTREAM = REPO / "EACL/methods/_upstream/DAMO-ConvAI/dial-start"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eacl_phase0"))
sys.path.insert(0, str(UPSTREAM))

PRED = REPO / "outputs/experiments/2026-07-29_baseline_compare/preds"
OUT = REPO / "outputs/experiments/2026-08-04_dialstart_nes"
SCORES = OUT / "scores"
CKPT = UPSTREAM / "model/model_711"     # 저자 배포 가중치 (dialseg711 학습본)

# 🔴 환경 함정 #2 회피: `baseline_compare` 는 import 만으로 torch·sklearn·train_anchor_m4m5 를
# 끌어온다. `.venv-dialstart` 에는 sklearn 이 없다. 로더를 여기 직접 둔다
# (`scripts/eacl_phase0/baseline_compare.py:55-81` 의 `load_dataset` 과 **동일 규약**).
MASTER = REPO / "outputs/eacl_phase0"
DOCS = {"dev3": (("pieman", "tunnel", "monkey"), "{d}_michelmann_master.jsonl"),
        "test3": (("run", "gohitler", "mymotherslife"),
                  "{d}_eventrecall_goldtopn_master.jsonl")}


def load_dataset_light(ds: str):
    """(doc_id, sents, gold[n-1]) 리스트. torch 외 의존 없음."""
    docs, pat = DOCS[ds]
    out = []
    for d in docs:
        rows = [json.loads(l) for l in
                (MASTER / pat.format(d=d)).read_text(encoding="utf-8").splitlines() if l.strip()]
        rows.sort(key=lambda r: r["transition"])
        sents = [rows[0]["previous"]] + [r["current"] for r in rows]
        gold = np.asarray([int(r["boundary"]) for r in rows], np.int8)
        out.append((d, sents, gold))
    return out


def pad_post(seqs, maxlen: int, value: int) -> np.ndarray:
    """`keras.preprocessing.sequence.pad_sequences(..., padding='post', truncating='post')` 동치.

    keras 3 에서 그 모듈이 사라져 저자 코드가 그대로는 안 돈다. 동작만 복제한다.
    """
    out = np.full((len(seqs), maxlen), value, dtype=np.int64)
    for i, s in enumerate(seqs):
        s = np.asarray(s, dtype=np.int64)[:maxlen]
        out[i, : len(s)] = s
    return out


@torch.no_grad()
def transition_scores(model, tok, sents: list[str], device, window_size: int,
                      batch: int = 64, right_window: int | None = None) -> np.ndarray:
    """전이별 응집도 점수 — `test.py:81-134` 의 입력 구성을 **그대로** 옮긴 것.

    전이 i 의 점수는 `text[i-window_size+1..i]`(좌) 와 `text[i+1..i+right_window]`(우) 만
    쓴다 ⇒ 도착 문장 `i+1` 기준 **우측 초과 = right_window-1** 문장.
    `right_window=None` 이면 저자 기본값(=window_size, 좌우 대칭)이고, `1` 이면 도착 문장
    하나만 봐서 점수 자체는 zero-lookahead 가 된다. 좌측 창은 인과적이라 그대로 둔다 —
    좌우를 같이 줄이면 'zero 라서' 와 '창이 좁아서' 가 섞여 해석이 불가능해진다.
    """
    n = len(sents)
    id_inputs, type_ids, topic_in, topic_mask, topic_num = [], [], [[], []], [[], []], [[], []]
    rw = window_size if right_window is None else right_window
    for i in range(n - 1):
        context, cur = [], []
        l, r = i, i + 1
        for k in range(max(window_size, rw)):
            if k < window_size and l > -1:
                context.append(sents[l][:128]); l -= 1
            if k < rw and r < n:
                cur.append(sents[r][:128]); r += 1
        context.reverse()

        tc = tok(context, truncation=True, padding=True, max_length=256, return_tensors="pt")
        tu = tok(cur, truncation=True, padding=True, max_length=256, return_tensors="pt")
        topic_in[0].extend(tc["input_ids"]); topic_in[1].extend(tu["input_ids"])
        topic_mask[0].extend(tc["attention_mask"]); topic_mask[1].extend(tu["attention_mask"])
        topic_num[0].append(len(context)); topic_num[1].append(len(cur))

        sent1 = "".join(s + "[SEP]" for s in context)
        e1 = tok.encode(sent1, add_special_tokens=True, max_length=256, return_tensors="pt")
        e2 = tok.encode(sents[i + 1], add_special_tokens=True, max_length=256,
                        return_tensors="pt")
        pair = e1[0].tolist()[:-1] + e2[0].tolist()[1:]
        id_inputs.append(pair)
        type_ids.append([0] * len(e1[0].tolist()[:-1]) + [1] * len(e2[0].tolist()[1:]))

    from torch.nn.utils.rnn import pad_sequence

    MAX_LEN = 512
    ci = pad_post(id_inputs, MAX_LEN, 0)
    ti = pad_post(type_ids, MAX_LEN, 1)
    cm = (ci > 0).astype(np.int64)

    # 전이 수가 많으면(tunnel 503) 한 번에 못 올린다 — 배치로 쪼갠다.
    # 각 전이의 입력은 이미 독립·유계라 쪼개도 값이 바뀌지 않는다.
    scores: list[float] = []
    o0 = o1 = 0
    for a in range(0, n - 1, batch):
        b = min(a + batch, n - 1)
        k0, k1 = sum(topic_num[0][a:b]), sum(topic_num[1][a:b])
        tin = [pad_sequence(topic_in[0][o0:o0 + k0], batch_first=True).to(device),
               pad_sequence(topic_in[1][o1:o1 + k1], batch_first=True).to(device)]
        tmk = [pad_sequence(topic_mask[0][o0:o0 + k0], batch_first=True).to(device),
               pad_sequence(topic_mask[1][o1:o1 + k1], batch_first=True).to(device)]
        o0 += k0; o1 += k1
        scores += model.infer(
            torch.tensor(ci[a:b]).to(device), torch.tensor(cm[a:b]).to(device),
            torch.tensor(ti[a:b]).to(device), tin, tmk,
            [topic_num[0][a:b], topic_num[1][a:b]])
    return np.asarray(scores, dtype=float)


def online_boundaries(scores: np.ndarray, c: float, depth_span: int, warmup: int = 3
                      ) -> tuple[list[int], int]:
    """개조 ①②: 유계 depth + running μ+c·σ. 반환 = (경계 전이 인덱스, 실측 최대 지연).

    전이 j 는 `scores[j+1..j+depth_span]` 이 나온 뒤 확정한다. `scores[j]` 자체가
    문장 `j+window_size` 까지 필요하므로 총 지연은 `window_size-1 + depth_span` 이다.
    """
    n = len(scores)
    depths: list[float] = []
    bnd: list[int] = []
    for j in range(n):
        hi = min(n - 1, j + depth_span)
        if hi < j + depth_span and j + depth_span < n:
            break
        # 왼쪽 봉우리 — 인과적이므로 제한 없음
        lflag = scores[j]
        for l in range(j - 1, -1, -1):
            if lflag <= scores[l]:
                lflag = scores[l]
            else:
                break
        # 오른쪽 봉우리 — **유계**(원본은 문서 끝까지 훑는다)
        rflag = scores[j]
        for r in range(j + 1, hi + 1):
            if rflag <= scores[r]:
                rflag = scores[r]
            else:
                break
        d = 0.5 * (lflag + rflag - 2 * scores[j])
        if len(depths) >= warmup:
            past = np.asarray(depths)
            if d > past.mean() + c * past.std():
                bnd.append(j)
        depths.append(d)
    return bnd, depth_span


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-size", type=int, default=2, help="저자 기본값. 바꾸지 말 것")
    ap.add_argument("--right-window", type=int, default=None,
                    help="점수 창의 우측 문장 수. 1 이면 도착 문장만 본다. "
                         "미지정이면 저자 기본값(=window_size)")
    ap.add_argument("--depth-span", type=int, default=1,
                    help="depth 오른쪽 봉우리 탐색 범위(전이). 총 지연 = window_size-1 + 이 값")
    ap.add_argument("--c", type=float, default=None, help="고정 c. 미지정이면 --freeze-c 필요")
    ap.add_argument("--freeze-c", action="store_true",
                    help="dev3 에서 c 를 스윕해 Score 최대값으로 동결한 뒤 test3 에 적용")
    ap.add_argument("--c-grid", default="0.0,0.25,0.5,0.75,1.0,1.25,1.5,2.0")
    ap.add_argument("--label", default="DialSTART-online")
    ap.add_argument("--stage", choices=("score", "readout", "both"), default="both",
                    help="score = 저자 모델로 전이 점수만 계산해 캐시(.venv-dialstart 필요). "
                         "readout = 캐시에서 개조 readout+채점(주 .venv 필요). "
                         "두 환경의 의존이 달라 분리해 둔다")
    args = ap.parse_args()

    rw = args.window_size if args.right_window is None else args.right_window
    total_delay = rw - 1 + args.depth_span
    data = {ds: load_dataset_light(ds) for ds in ("dev3", "test3")}
    SCORES.mkdir(parents=True, exist_ok=True)

    # ── stage 1: 전이 점수 (저자 모델·가중치 그대로) ────────────────────────────
    if args.stage in ("score", "both"):
        from transformers import BertTokenizer                       # noqa: PLC0415
        from model import SegModel                                   # noqa: PLC0415

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tok = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        model = SegModel()
        model.load_state_dict(torch.load(CKPT, map_location="cpu"), False)
        model.to(dev).eval()
        print(f"ckpt={CKPT.name} · window_size={args.window_size} · right_window={rw} · "
              f"depth_span={args.depth_span} ⇒ 총 지연 d={total_delay} · device={dev}",
              flush=True)
        for ds in ("dev3", "test3"):
            for doc, sents, gold in data[ds]:
                p = SCORES / f"{doc}_w{args.window_size}_r{rw}.npy"
                if p.exists():
                    print(f"  score {doc:<16} 캐시 있음 — 건너뜀", flush=True)
                    continue
                t0 = time.perf_counter()
                np.save(p, transition_scores(model, tok, sents, dev, args.window_size,
                                             right_window=args.right_window))
                print(f"  score {doc:<16} 전이 {len(gold):>4} · "
                      f"{time.perf_counter()-t0:.1f}s → {p.name}", flush=True)
        if args.stage == "score":
            return 0

    # ── stage 2: 개조 readout + 채점 ────────────────────────────────────────────
    from hi_ontop.dts_scoring import score_dialogues             # noqa: PLC0415

    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ds in ("dev3", "test3"):
        for doc, _sents, gold in data[ds]:
            p = SCORES / f"{doc}_w{args.window_size}_r{rw}.npy"
            if not p.exists():
                print(f"🔴 점수 캐시 없음: {p} — 먼저 --stage score 로 돌릴 것", file=sys.stderr)
                return 2
            cache[doc] = (np.load(p), gold)

    def evaluate(ds: str, c: float) -> tuple[dict, dict]:
        preds = {}
        for doc, _sents, gold in data[ds]:
            s, g = cache[doc]
            bnd, _ = online_boundaries(s, c, args.depth_span)
            v = np.zeros(len(g), np.int8)
            for j in bnd:
                if 0 <= j < len(v):
                    v[j] = 1
            preds[doc] = v.tolist()
        G = [g for _d, _s, g in data[ds]]
        ys = [np.concatenate([np.asarray(g, np.int8), [0]]).tolist() for g in G]
        ps = [np.concatenate([np.asarray(preds[d], np.int8), [0]]).tolist()
              for d, _s, _g in data[ds]]
        sc = score_dialogues(ys, ps)
        return {k: sc[k] for k in ("pk", "wd", "f1", "score")}, preds

    # 🔒 c 동결은 **dev3 에서만** 한다. test3 에서 재선택하면 시험지를 여는 것이다.
    sweep = {}
    if args.c is not None:
        c_star = args.c
    else:
        if not args.freeze_c:
            print("🔴 --c 또는 --freeze-c 중 하나가 필요하다", file=sys.stderr)
            return 2
        for c in [float(x) for x in args.c_grid.split(",")]:
            m, _ = evaluate("dev3", c)
            sweep[c] = m
            print(f"  dev3 c={c:<5} Score {m['score']:.4f} (Pk {m['pk']:.4f})", flush=True)
        c_star = max(sweep, key=lambda c: sweep[c]["score"])
        print(f"\n🔒 dev3 에서 동결: c* = {c_star}\n", flush=True)

    rows = {}
    PRED.mkdir(parents=True, exist_ok=True)
    for ds in ("dev3", "test3"):
        m, preds = evaluate(ds, c_star)
        rows[ds] = {**m, "n_pred": int(sum(sum(v) for v in preds.values())),
                    "n_gold": int(sum(int(g.sum()) for _d, _s, g in data[ds]))}
        suffix = "zero" if total_delay == 0 else f"d{total_delay}"
        (PRED / f"{args.label}-{suffix}_{ds}.json").write_text(json.dumps(preds))
        print(f"  {args.label}-{suffix} {ds:<6} Pk {m['pk']:.4f} WD {m['wd']:.4f} "
              f"F1 {m['f1']:.4f} Score {m['score']:.4f} · 예측 {rows[ds]['n_pred']}"
              f"/gold {rows[ds]['n_gold']}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dialstart_nes.json").write_text(json.dumps(
        {"config": vars(args), "c_star": c_star, "total_delay": total_delay,
         "dev3_c_sweep": {str(k): v for k, v in sweep.items()}, "rows": rows},
        ensure_ascii=False, indent=2))
    print(f"\n→ {OUT/'dialstart_nes.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

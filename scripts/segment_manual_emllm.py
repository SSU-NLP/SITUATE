#!/usr/bin/env python3
"""EM-LLM 충실 구현 화제분절 (사용자 직접-입력, 문장=turn 근사).

EM-LLM (Fountas et al. 2025, github.com/em-llm/EM-LLM-model) 을 이 저장소의
sentence-as-turn 비교 프레임에 맞춰 옮긴 것. **원본 알고리즘 그대로**:

  1) Surprise = 생성형 LLM 의 토큰 NLL  −log P(x_t | x_<t)  (단일 forward pass 부산물).
     문장 surprise = 그 문장 토큰들의 mean NLL.
  2) 경계(surprise) : s_i > mean(past) + γ·std(past)         [context_manager.py:923]
     γ = surprisal_threshold_gamma (원본 기본 1.1).
  3) 그래프 리파인먼트: 각 경계를 '원래 세그먼트 2배' 국소 윈도우 안에서 이동시켜
     [0:t]/[t:T] 이분할의 modularity 최대(argmax) 또는 conductance 최소(argmin)
     지점으로 스냅.  [similarity_refinement/segmentation.py, similarity.py]
     유사도 행렬 A[i,j] = Σ_{layer≥refine_from_layer, head} ⟨repr_k_i, repr_k_j⟩
     (블록 대표 key = 문장 토큰 key 들의 평균; post-RoPE key = past_key_values).
                                                              [em_llm.py:245,776]

원본과의 (문서화된) 차이 — 비교 프레임 때문에 불가피:
  · 토큰 단위가 아니라 **문장 단위**로 surprise 집계/경계 (89문장 vs A/Bfull/Bk6 대조용).
  · repr_topk: 원본은 attention repr_score 상위 1토큰; 여기선 attention 내부 점수를
    밖에서 못 뽑아 **문장 전체 토큰 평균**으로 대표 key 계산 (repr_topk→all 근사).
  · surprise 임계 윈도우: 원본은 local memory(n_local) 구간; 여기선 online full-past prefix
    (--win 으로 trailing 윈도우 지정 가능).

usage:
  uv run python scripts/segment_manual_emllm.py --file icml_sentences.txt
  uv run python scripts/segment_manual_emllm.py --file icml_sentences.txt --gamma 1.1 --metric modularity
  uv run python scripts/segment_manual_emllm.py --file icml_sentences.txt --metric conductance
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_LM = "Qwen/Qwen2.5-7B"     # base(비-instruct) 다국어·한국어. EM-LLM 논문 스케일(7B급) 정합.
DEFAULT_GAMMA = 1.1                # surprisal_threshold_gamma 원본 기본값
DEFAULT_MIN_SIZE = 1               # min_block_size 원본 기본값
DEFAULT_REFINE_FROM_LAYER = 0      # 원본 기본: 전체 레이어 key 합산


# ── EM-LLM similarity.py 원본 수식 (verbatim, 저자: em-llm) ─────────────────────
def modularity(A, communities):
    m = A.sum() / 2
    k_i = A.sum(dim=-2)
    k_j = A.sum(dim=-1)
    expected = torch.einsum("i,j->ij", k_i, k_j) / (2 * m)
    Q = 0.0
    for c in communities:
        sub_A = A[c, :][:, c]
        sub_e = expected[c, :][:, c]
        Q += (sub_A - sub_e).sum()
    return Q / (4 * m)


def conductance_min(A, communities):
    total_vol = torch.sum(A)
    vals = []
    for c in communities:
        c_bar = [i for i in range(A.shape[-1]) if i not in c]
        cut = torch.sum(A[c, :][:, c_bar])
        vol_S = torch.sum(A[c, :][:, c])
        vol_bar = total_vol - vol_S
        min_vol = torch.minimum(vol_S, vol_bar)
        if min_vol == 0:
            min_vol = torch.tensor(1e-15)
        vals.append(cut / min_vol)
    return torch.stack(vals).min()


def calc_adjacent_similarity(A, first_indx, last_indx, sim_func):
    """sub-matrix A 안에서 split t ∈ [first,last) 각각의 이분할 점수. (원본 verbatim 로직)"""
    T = A.shape[0]
    out = torch.zeros(last_indx - first_indx)
    for t in range(first_indx, last_indx):
        communities = [list(range(0, t)), list(range(t, T))]
        out[t - first_indx] = sim_func(A, communities)
    return out


def refine_boundaries(events_base, A, metric="modularity", min_size=DEFAULT_MIN_SIZE):
    """events_with_similarity_adjustment 원본 로직(offset=0). 문장 경계 리스트 → 리파인 경계 리스트."""
    if metric == "modularity":
        sim_func = modularity
    elif metric == "conductance":
        sim_func = conductance_min
    else:
        raise NotImplementedError(metric)
    N = A.shape[0]
    events_temp = [0]
    for event in events_base:
        if event - events_temp[-1] > min_size and event > min_size:
            orig = event - events_temp[-1]
            half = orig // 2
            start_from = max(0, events_temp[-1] - half)
            end_to = min(N, event + half)
            first_idx = max(0, events_temp[-1] - start_from)
            last_idx = event - start_from
            sub_A = A[start_from:end_to, :][:, start_from:end_to]
            adj = calc_adjacent_similarity(sub_A, first_idx, last_idx, sim_func)
            tail = adj[min_size:]
            arg = int(torch.argmin(tail) if metric == "conductance" else torch.argmax(tail))
            events_temp.append(start_from + first_idx + min_size + arg)
        else:
            events_temp.append(event)
    return events_temp[1:]
# ──────────────────────────────────────────────────────────────────────────────


def read_sentences(path: str) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


@torch.no_grad()
def encode_doc(sents, tok, model, device, sep="\n", refine_from_layer=0):
    """단일 forward pass → (문장 surprise[N], 문장 대표key로 만든 A[N,N])."""
    input_ids: list[int] = []
    spans: list[tuple[int, int]] = []
    for s, sent in enumerate(sents):
        chunk = (sep if s > 0 else "") + sent
        ids = tok.encode(chunk, add_special_tokens=False)
        spans.append((len(input_ids), len(input_ids) + len(ids)))
        input_ids += ids

    ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)
    out = model(input_ids=ids_t, use_cache=True, output_hidden_states=False)
    logits = out.logits[0].float()                       # [T, V]

    # 토큰 NLL: nll_tok[j] = surprise of token j+1
    logp = F.log_softmax(logits[:-1], dim=-1)
    labels = ids_t[0, 1:]
    nll_tok = (-logp.gather(1, labels[:, None]).squeeze(1)).cpu().numpy()   # len T-1

    # post-RoPE keys: past_key_values[l].key  → [1, n_kv_head, T, head_dim]
    pkv = out.past_key_values
    layers = range(refine_from_layer, len(pkv))
    # 문장 대표 key = 문장 토큰 key 평균 (레이어별) → [N, n_layer_used, n_kv_head, head_dim]
    repr_per_layer = []
    for l in layers:
        k = pkv[l][0][0].float()                         # [n_kv_head, T, head_dim]
        k = k.permute(1, 0, 2)                            # [T, n_kv_head, head_dim]
        reps = torch.stack([k[a:b].mean(dim=0) for (a, b) in spans])  # [N, n_kv_head, head_dim]
        repr_per_layer.append(reps)
    R = torch.stack(repr_per_layer, dim=1)               # [N, L, H, D]
    N = R.shape[0]
    Rf = R.reshape(N, -1)                                # concat layer·head·dim
    A = (Rf @ Rf.T).cpu()                                # A[i,j]=Σ_{l,h,d}⟨repr_i,repr_j⟩ (원본 einsum과 동치)

    # 문장 surprise = 그 문장 토큰들의 mean NLL (position0 은 정의 없음 → 제외)
    surprise = np.zeros(N)
    for i, (a, b) in enumerate(spans):
        idx = [p - 1 for p in range(a, b) if p >= 1]       # nll_tok index = token pos -1
        surprise[i] = float(np.mean([nll_tok[j] for j in idx])) if idx else 0.0
    return surprise, A


def surprise_boundaries(surprise, gamma, win=0, warmup=3):
    """s_i > mean(past)+γ·std(past). win=0 → full past prefix, else trailing window."""
    N = len(surprise)
    bnd = []
    for i in range(1, N):
        past = surprise[max(0, i - win):i] if win > 0 else surprise[:i]
        if len(past) < warmup:
            continue
        thr = past.mean() + gamma * past.std()
        if surprise[i] > thr:
            bnd.append(i)
    return bnd


def main() -> int:
    ap = argparse.ArgumentParser(description="EM-LLM 충실 구현 화제분절 (LLM 토큰 NLL surprise + 그래프 리파인먼트)")
    ap.add_argument("--file", required=True, help="문장 파일(줄당 1문장).")
    ap.add_argument("--lm", default=DEFAULT_LM)
    ap.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help="surprisal_threshold_gamma (기본 1.1)")
    ap.add_argument("--metric", choices=["modularity", "conductance", "none"], default="modularity",
                    help="리파인먼트 지표. none=surprise 경계만(리파인 없음).")
    ap.add_argument("--min-size", type=int, default=DEFAULT_MIN_SIZE)
    ap.add_argument("--refine-from-layer", type=int, default=DEFAULT_REFINE_FROM_LAYER)
    ap.add_argument("--win", type=int, default=0, help="surprise 임계 윈도우(0=full past)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--dump", default=None, help="surprise/경계 결과 JSON 저장 경로")
    args = ap.parse_args()

    sents = read_sentences(args.file)
    print(f"[입력] {args.file}: {len(sents)}문장", file=sys.stderr)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[LM 로딩] {args.lm} (device={device}) ...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.lm)
    model = AutoModelForCausalLM.from_pretrained(
        args.lm, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    surprise, A = encode_doc(sents, tok, model, device,
                             refine_from_layer=args.refine_from_layer)

    events = surprise_boundaries(surprise, args.gamma, win=args.win, warmup=args.warmup)
    if args.metric == "none" or not events:
        refined = list(events)
    else:
        refined = refine_boundaries(events, A, metric=args.metric, min_size=args.min_size)
    refined = sorted(set(int(x) for x in refined if 0 < int(x) < len(sents)))

    def segs(bnds):
        pts = [0] + bnds + [len(sents)]
        return [f"[{pts[k]}..{pts[k+1]-1}]" for k in range(len(pts) - 1)]

    print(f"\n[방법] EM-LLM 충실 (LM={args.lm}, γ={args.gamma}, "
          f"refine={args.metric}, win={'full' if args.win==0 else args.win})", file=sys.stderr)
    print("\n" + "=" * 70)
    print(f"surprise 경계 (리파인 전, {len(events)}개): {events}")
    print(f"세그먼트(전) : {' '.join(segs(events))}")
    print("-" * 70)
    print(f"{args.metric} 리파인 경계 ({len(refined)}개): {refined}")
    print(f"세그먼트(후) : {' '.join(segs(refined))}")
    print("=" * 70)
    print("\n문장별 surprise(mean-token NLL) + 경계:")
    ev_set, rf_set = set(events), set(refined)
    for i, s in enumerate(sents):
        m = []
        if i in ev_set:
            m.append("▶sur")
        if i in rf_set:
            m.append(f"▶{args.metric[:4]}")
        tag = " ".join(m) if m else "—"
        print(f"[{i:>2}] nll={surprise[i]:.3f} {tag:<12} | {s[:60]}")

    if args.dump:
        import json
        Path(args.dump).write_text(json.dumps({
            "lm": args.lm, "gamma": args.gamma, "metric": args.metric,
            "surprise": [round(float(x), 4) for x in surprise],
            "events_surprise": events, "events_refined": refined,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[저장] {args.dump}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

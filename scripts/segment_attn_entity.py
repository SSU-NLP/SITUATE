#!/usr/bin/env python3
"""M9-AttnEntityDrift — LLM attention 상위 토큰(=엔티티) 거리 기반 문장 분절.

EM-LLM(Fountas et al. 2025)은 경계를 **토큰 surprisal** `−log P(x_t|x_<t)` 로 잡는다
(`scripts/segment_manual_emllm.py` 가 그 충실 재구현). 본 모듈은 그 신호를
**"LLM 이 집중하는 토큰(=엔티티) 사이의 거리"** 로 교체한다.

배경: EM-LLM 원본은 per-key attention mass 를 **이미 계산한다** —
`em_llm/attention/context_manager.py:716` `exc_repr_score`,
`em_llm/attention/dot_product_attention/torch_impl.py:26` `tmp.sum(dim=-2)`.
그런데 그 값을 검색(retrieval)과 `repr_topk` 에만 쓰고 **경계 판정에는 전혀 쓰지 않는다.**
본 모듈은 그 미사용 신호를 경계 판정으로 돌린다.

파이프라인 (단일 forward pass):
  1) attention mass  a_t = mean_layers mean_heads mean_{q∈Q(i)} p̃[q,t]
     p̃ = 각 쿼리 행을 core(i)(= 문장 i 의 sep·구두점 제외 토큰) 위에서 재정규화.
  2) 엔티티        E_i = a 상위 k 토큰 (core(i) ∩ allow)
  3) 거리 신호     embed / embed_ctx / setdrift / recency  (§signal_*)
  4) 경계          동결 작동점 3종 (OP-1 adaptive_boundaries / OP-2 EM-LLM μ+γσ / OP-3 분위수)

⚠️ **`norm="rowsent"` 는 전체 causal prefix softmax 가 불필요하다** — 전역 분모 Z_q 가 정확히
소거되기 때문(`_mass_rowsent` 도크스트링 참조). `count`/`none` 은 필요하므로 `_mass_full` 로 간다.
두 경로의 일치는 `tests/test_attn_mass.py` 가 강제한다.

usage:
  uv run python scripts/segment_attn_entity.py --file icml_sentences.txt --signal setdrift
  uv run python scripts/segment_attn_entity.py --file icml_sentences.txt --signal embed --dump out.json
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from hi_ontop.hi_ontop_v2 import adaptive_boundaries  # noqa: E402
from segment_manual_emllm import (  # noqa: E402  (전부 EM-LLM 원본 수식 — 복사 금지, import)
    DEFAULT_LM,
    encode_doc,
    read_sentences,
    refine_boundaries,
    surprise_boundaries,
)

DEFAULT_LAYERS = "18-27"      # round(0.625*28); EM-LLM refine_from_layer=20/32 비율
DEFAULT_REPR_LAYER = 18       # hidden_states[18] — 점수 대역의 첫 레이어와 같은 깊이
DEFAULT_TOPK = 3
DEFAULT_CAP = 20
DEFAULT_GAMMA = 1.1
ANCHOR_R = 0.0642             # 🔒 D-B30 앵커 동결 경계율 (OP-3)
M1 = dict(m=2, rho=0.7, a=0.5)  # method-registry §2 canonical (embed_ctx 용)

SCOPES = ("sentnext", "sent", "next")   # window:W 는 문자열 파싱
NORMS = ("rowsent", "count", "none")
SIGNALS = ("embed", "embed_ctx", "setdrift", "setdrift_w", "recency")


# ── 토큰화 ────────────────────────────────────────────────────────────────────
def _is_punct_or_space(s: str) -> bool:
    """공백만이거나 전부 유니코드 구두점/기호이면 True (= 엔티티 후보 제외)."""
    t = s.strip()
    if not t:
        return True
    return all(unicodedata.category(c)[0] in ("P", "S") for c in t)


@dataclass
class DocTok:
    """문서 1편의 토큰화 결과.

    Attributes:
        input_ids: 토큰 id [T].
        spans: 문장별 (start, end) — **sep 포함**. `encode_doc` 와 동일해야 A행렬/surprise 가 호환된다.
        core_spans: 문장별 (start, end) — sep 토큰 제외 (연속 구간).
        core_mask: [T] bool. sep·공백·구두점-only 토큰을 뺀 엔티티 후보.
        sent_of: [T] int. 토큰 → 문장 index.
        char_start: [T] int. 문장-로컬 char offset (sep 부분은 -1).
        text_of: [T] 토큰 원문 조각.
        sents: 원문 문장 리스트.
    """

    input_ids: list[int]
    spans: list[tuple[int, int]]
    core_spans: list[tuple[int, int]]
    core_mask: np.ndarray
    sent_of: np.ndarray
    char_start: np.ndarray
    text_of: list[str]
    sents: list[str]

    @property
    def n_sent(self) -> int:
        return len(self.spans)

    @property
    def n_tok(self) -> int:
        return len(self.input_ids)


def tokenize_doc(sents: list[str], tok, sep: str = "\n") -> DocTok:
    """문장 리스트 → `DocTok`. 청킹 규약은 `segment_manual_emllm.encode_doc:124-128` 과 동일.

    문장 0 은 sep 없이, 문장 1+ 는 `sep + sent` 로 인코딩한다(원본과 like-for-like 유지).
    """
    ids: list[int] = []
    spans, core_spans, sent_of, char_start, text_of = [], [], [], [], []
    core_flags: list[bool] = []
    for s, sent in enumerate(sents):
        pre = sep if s > 0 else ""
        chunk = pre + sent
        enc = tok(chunk, add_special_tokens=False, return_offsets_mapping=True)
        n0 = len(ids)
        core_lo = None
        for tid, (o_s, o_e) in zip(enc["input_ids"], enc["offset_mapping"]):
            piece = chunk[o_s:o_e]
            ids.append(int(tid))
            sent_of.append(s)
            text_of.append(piece)
            # char_start = 토큰의 **첫 비공백 문자**의 문장-로컬 offset.
            # Qwen BPE 토큰은 선행 공백을 포함한다(" McGowan"). o_s 를 그대로 쓰면 공백 위치를
            # 가리켜 spaCy 토큰 span(t.idx 는 단어 시작)과 매칭이 통째로 실패한다.
            # 이 처리는 sep 이 토큰 안으로 섞인 경우("\nAnd")도 함께 해결한다.
            j = next((x for x in range(o_s, o_e) if not chunk[x].isspace()), None)
            if j is None or j < len(pre):             # 순수 공백/sep 토큰
                char_start.append(-1)
                core_flags.append(False)
                continue
            local = j - len(pre)
            char_start.append(local)
            if core_lo is None:
                core_lo = len(ids) - 1
            core_flags.append(not _is_punct_or_space(piece))
        spans.append((n0, len(ids)))
        core_spans.append((core_lo if core_lo is not None else n0, len(ids)))
    return DocTok(
        input_ids=ids,
        spans=spans,
        core_spans=core_spans,
        core_mask=np.asarray(core_flags, bool),
        sent_of=np.asarray(sent_of, np.int32),
        char_start=np.asarray(char_start, np.int32),
        text_of=text_of,
        sents=list(sents),
    )


# ── attention mass 추출 ───────────────────────────────────────────────────────
def parse_layers(spec: str, n_layer: int) -> list[int]:
    """`"18-27"` / `"all"` / `"3,7,11"` → 레이어 index 리스트."""
    if spec == "all":
        return list(range(n_layer))
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    bad = [l for l in out if not 0 <= l < n_layer]
    if bad:
        raise ValueError(f"레이어 범위 밖: {bad} (n_layer={n_layer})")
    return sorted(set(out))


def query_positions(dt: DocTok, i: int, scope: str) -> np.ndarray:
    """문장 i 의 attention mass 를 재는 **쿼리** 토큰 위치 Q(i).

    EM-LLM `repr_score` 는 local window 안의 쿼리로부터 받은 mass 다
    (`context_manager.py:665-668` `get_score=True`, `torch_impl.py:26` `tmp.sum(dim=-2)`).
    `sentnext` 는 그것을 담화-문장 단위로 축소한 것.
    """
    a, b = dt.spans[i]
    nxt = dt.spans[i + 1] if i + 1 < dt.n_sent else None
    if scope == "sent":
        q = np.arange(a, b)
    elif scope == "next":
        q = np.arange(*nxt) if nxt else np.arange(a, b)          # 마지막 문장 폴백
    elif scope == "sentnext":
        q = np.arange(a, nxt[1]) if nxt else np.arange(a, b)
    elif scope.startswith("window:"):
        w = int(scope.split(":")[1])
        q = np.arange(a, min(dt.n_tok, b + w))
    else:
        raise ValueError(f"unknown scope {scope!r}")
    return q


def _post_rope_qk(attn, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """모듈 자신의 `q_proj`/`k_proj` + RoPE + `repeat_kv` 로 post-RoPE Q,K 재구성.

    `transformers/models/qwen2/modeling_qwen2.py:162-167` 를 그대로 미러링한다.
    Returns: (q [H,T,D], k [H,T,D]) — batch 1 squeeze.
    """
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

    shape = (*h.shape[:-1], -1, attn.head_dim)
    q = attn.q_proj(h).view(shape).transpose(1, 2)
    k = attn.k_proj(h).view(shape).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    k = repeat_kv(k, attn.num_key_value_groups)
    return q[0], k[0]


def _mass_rowsent(dt: DocTok, q: torch.Tensor, k: torch.Tensor, scaling: float,
                  scope: str) -> torch.Tensor:
    """`norm="rowsent"` — 전역 softmax 분모가 **정확히 소거**되므로 문장 국소 계산으로 충분.

        p̃[q,t] = p[q,t] / Σ_{t'∈core(i)} p[q,t']
                = (e^{s_qt}/Z_q) / (Σ_{t'∈core(i)} e^{s_qt'}/Z_q)
                = e^{s_qt} / Σ_{t'∈core(i)} e^{s_qt'}      ← Z_q 소거

    즉 core(i) 위의 restricted softmax 와 동일. 인과 마스크(t>q)는 그대로 적용한다.
    쿼리 개수 편향도 같이 사라진다 — 각 쿼리가 문장에 정확히 1.0 을 기여하므로.

    Returns: [H, T] — core(i) 밖 토큰은 0.
    """
    H, T, _ = q.shape
    out = torch.zeros(H, T, dtype=torch.float32, device=q.device)
    for i in range(dt.n_sent):
        tgt = np.nonzero(dt.core_mask[dt.spans[i][0]:dt.spans[i][1]])[0] + dt.spans[i][0]
        if tgt.size == 0:
            continue
        qs = query_positions(dt, i, scope)
        if qs.size == 0:
            continue
        ti = torch.as_tensor(tgt, device=q.device, dtype=torch.long)
        qi = torch.as_tensor(qs, device=q.device, dtype=torch.long)
        s = torch.einsum("hqd,htd->hqt", q[:, qi].float(), k[:, ti].float()) * scaling
        causal = qi[None, :, None] >= ti[None, None, :]          # [1, nq, nt]
        s = s.masked_fill(~causal, float("-inf"))
        alive = causal.squeeze(0).any(-1)                        # 쿼리가 볼 수 있는 타깃이 있나
        if not bool(alive.any()):
            continue
        p = torch.softmax(s[:, alive], dim=-1)                   # core(i) 위 restricted softmax
        out[:, ti] += p.mean(dim=1)                              # 쿼리축 평균
    return out


def _mass_full(dt: DocTok, q: torch.Tensor, k: torch.Tensor, scaling: float,
               scope: str, norm: str, block: int) -> torch.Tensor:
    """`norm ∈ {count, none}` — 전체 causal prefix softmax 가 필요한 경로 (블록 단위 열합).

    쿼리 블록마다 `[H, Bq, T]` 를 만들고 **열합만 누적**한 뒤 행렬은 버린다.
    `count` = 그 토큰을 본 쿼리 수로 나눔(평균), `none` = 합 그대로.
    """
    H, T, _ = q.shape
    acc = torch.zeros(H, T, dtype=torch.float32, device=q.device)
    cnt = torch.zeros(T, dtype=torch.float32, device=q.device)
    # 토큰 t 를 재는 쿼리 집합을 [T] 마스크들의 합으로 만들지 않고, (쿼리 → 그 쿼리가 기여하는 타깃 구간)
    # 으로 뒤집어 처리한다. scope 는 전부 "문장 i 의 타깃을 문장 i(또는 i+1) 쿼리가 잰다" 꼴이다.
    tgt_lo = np.zeros(T, np.int64)
    tgt_hi = np.zeros(T, np.int64)
    for i in range(dt.n_sent):
        for p in query_positions(dt, i, scope):
            a, b = dt.spans[i]
            tgt_lo[p] = min(tgt_lo[p], a) if tgt_hi[p] else a
            tgt_hi[p] = max(tgt_hi[p], b)
    lo_t = torch.as_tensor(tgt_lo, device=q.device)
    hi_t = torch.as_tensor(tgt_hi, device=q.device)
    ar = torch.arange(T, device=q.device)
    for qs in range(0, T, block):
        qe = min(T, qs + block)
        qi = ar[qs:qe]
        s = torch.einsum("hqd,htd->hqt", q[:, qs:qe].float(), k.float()) * scaling
        s = s.masked_fill(ar[None, None, :] > qi[None, :, None], float("-inf"))
        p = torch.softmax(s, dim=-1)                             # 전체 prefix softmax
        keep = (ar[None, :] >= lo_t[qs:qe, None]) & (ar[None, :] < hi_t[qs:qe, None])
        keep &= (ar[None, :] <= qi[:, None])
        p = p * keep[None].to(p.dtype)
        acc += p.sum(dim=1)
        cnt += keep.float().sum(dim=0)
        del s, p
    if norm == "count":
        acc = acc / cnt.clamp(min=1.0)
    return acc


@torch.no_grad()
def attention_mass(dt: DocTok, model, device: str, *, layers: list[int],
                   query_scope: str = "sentnext", norm: str = "rowsent",
                   repr_layers: tuple[int, ...] = (14, 18, 28),
                   mem_budget: int = 8 << 30):
    """단일 forward pass → (per-layer·per-head attention mass, repr hidden states).

    attention 모듈에 `register_forward_pre_hook(with_kwargs=True)` 를 걸어 모듈 입력
    (post-`input_layernorm` hidden) 과 `position_embeddings=(cos,sin)` 을 캡처한 뒤,
    forward 후 모듈 자신의 `q_proj`/`k_proj`/RoPE 로 post-RoPE Q,K 를 재구성한다.
    `output_attentions=True` 는 불가 — T=6,637 에서 한 레이어 `[1,28,T,T]` fp16 = 2.47 GB,
    28 레이어 전부 보관 = 69 GB. 게다가 sdpa 가 조용히 eager 로 폴백한다.

    Returns:
        A_lht: [n_layer, n_head, T] float32 — 레이어·헤드별로 남긴다(대역/집계 ablation 재추출 불필요).
        H_repr: {layer: np.ndarray [T, D] float16}
    """
    cap: dict[int, tuple] = {}
    handles = []

    def mk(li: int):
        def hook(mod, args, kwargs):
            h = kwargs.get("hidden_states", args[0] if args else None)
            pe = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
            assert h is not None and pe is not None, "Qwen2Attention 시그니처가 예상과 다르다"
            cap[li] = (h, pe[0], pe[1])
            return None
        return hook

    lyrs = model.model.layers
    for li in layers:
        handles.append(lyrs[li].self_attn.register_forward_pre_hook(mk(li), with_kwargs=True))
    try:
        ids_t = torch.tensor([dt.input_ids], dtype=torch.long, device=device)
        out = model(input_ids=ids_t, use_cache=False, output_hidden_states=True)
    finally:
        for h in handles:
            h.remove()

    T = dt.n_tok
    H = model.config.num_attention_heads
    A = torch.zeros(len(layers), H, T, dtype=torch.float32, device=device)
    for j, li in enumerate(layers):
        h, cos, sin = cap[li]
        attn = lyrs[li].self_attn
        q, k = _post_rope_qk(attn, h, cos, sin)
        if norm == "rowsent":
            A[j] = _mass_rowsent(dt, q, k, attn.scaling, query_scope)
        else:
            blk = int(np.clip(mem_budget // max(6 * H * T, 1), 128, 1024))
            A[j] = _mass_full(dt, q, k, attn.scaling, query_scope, norm, blk)
        del q, k
        cap[li] = None
    hs = {int(l): out.hidden_states[l][0].to(torch.float16).cpu().numpy() for l in repr_layers}
    del out, cap
    torch.cuda.empty_cache() if device.startswith("cuda") else None
    return A.cpu().numpy(), hs


def aggregate(A_lht: np.ndarray, *, layer_idx: list[int] | None = None,
              head_agg: str = "mean") -> np.ndarray:
    """[n_layer, n_head, T] → [T]. `perhead-vote` 는 헤드별 argmax 투표(원본 `get_block_k` 계보)."""
    A = A_lht if layer_idx is None else A_lht[layer_idx]
    if head_agg == "mean":
        return A.mean(axis=(0, 1))
    if head_agg == "perhead-vote":
        v = np.zeros(A.shape[-1], np.float64)
        flat = A.reshape(-1, A.shape[-1])
        np.add.at(v, flat.argmax(axis=-1), 1.0)
        return v + 1e-9 * A.mean(axis=(0, 1))          # 동점은 평균 mass 로 미세 정렬
    raise ValueError(head_agg)


# ── 엔티티 선택 ───────────────────────────────────────────────────────────────
def select_topk(a_t: np.ndarray, dt: DocTok, *, k: int = DEFAULT_TOPK, mode: str = "fixed",
                frac: float = 0.25, allow: np.ndarray | None = None,
                rank: str = "attn", rng: np.random.Generator | None = None):
    """문장별 엔티티 토큰 위치.

    Args:
        rank: `attn`(상위 k) / `all`(pool 전체 = attention 없는 쌍둥이) / `shuffled`(pool 무작위 k).
        allow: [T] bool 추가 필터(spaCy 내용어/NER). 걸러 비면 core 로 폴백.

    Returns: (picks, stats)
    """
    picks: list[np.ndarray] = []
    n_fb_allow = n_trunc = n_empty = 0
    for i in range(dt.n_sent):
        a, b = dt.spans[i]
        base = np.nonzero(dt.core_mask[a:b])[0] + a
        if base.size == 0:                                        # 구두점만 있는 문장
            base = np.arange(*dt.core_spans[i])
            n_empty += 1
        pool = base
        if allow is not None:
            f = base[allow[base]]
            if f.size:
                pool = f
            else:
                n_fb_allow += 1
        kk = k if mode == "fixed" else int(np.clip(round(frac * pool.size), 1, 8))
        # 절단은 **top-k 를 실제로 쓰는 경우에만** 의미가 있다. rank="all" 은 pool 전체를 쓰므로
        # "pool ≤ k" 가 퇴화 신호가 아니다 (이걸 세면 all-* 셀에 가짜 DEGEN 이 붙는다).
        if rank != "all" and pool.size <= kk:
            n_trunc += 1
        if rank == "all":
            sel = pool
        elif rank == "shuffled":
            assert rng is not None, "rank='shuffled' 는 rng 필요"
            sel = rng.permutation(pool)[:kk]
        else:
            sel = pool[np.argsort(-a_t[pool], kind="stable")[:kk]]
        picks.append(np.sort(sel))
    stats = {"frac_truncated": n_trunc / dt.n_sent, "n_fallback_allow": n_fb_allow,
             "n_empty_core": n_empty}
    return picks, stats


def _word_spans(sent: str) -> list[tuple[int, int, str]]:
    """문장의 공백구분 단어 (start, end, word)."""
    out, i = [], 0
    for w in sent.split():
        j = sent.index(w, i)
        out.append((j, j + len(w), w))
        i = j + len(w)
    return out


def _strip_punct(w: str) -> str:
    return w.strip("".join(c for c in w if unicodedata.category(c)[0] in ("P", "S"))) or ""


def entity_keys(dt: DocTok, picks: list[np.ndarray], *, unit: str = "word",
                lemma: dict[int, dict[int, str]] | None = None) -> list[list[str]]:
    """엔티티 위치 → 동일성 판정용 키.

    `word`(기본): 서브워드를 **원문 단어로 병합** 후 소문자·구두점 strip·중복제거.
    `" McG"+"owan"` 같은 BPE 분할이 Jaccard 를 그냥 죽이므로 병합이 기본이다.
    `token`: 병합 없이 서브워드 문자열 (병합이 필요함을 보이는 ablation).
    """
    ws_cache: dict[int, list[tuple[int, int, str]]] = {}
    out: list[list[str]] = []
    for i, sel in enumerate(picks):
        keys: list[str] = []
        for p in sel:
            if unit == "token":
                k = _strip_punct(dt.text_of[p].strip().lower())
            else:
                c = int(dt.char_start[p])
                if c < 0:
                    continue
                if i not in ws_cache:
                    ws_cache[i] = _word_spans(dt.sents[i])
                w = next((w for s, e, w in ws_cache[i] if s <= c < e), None)
                if w is None:
                    w = next((w for s, e, w in ws_cache[i] if s >= c), None)
                if w is None:
                    continue
                k = _strip_punct(w.lower())
                if lemma is not None:
                    k = lemma.get(i, {}).get(c, k)
            if k:
                keys.append(k)
        seen: set[str] = set()
        out.append([k for k in keys if not (k in seen or seen.add(k))])
    return out


# ── 거리 신호 ─────────────────────────────────────────────────────────────────
def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def signal_embed(Hr: np.ndarray, picks: list[np.ndarray], *, ctx: bool = False,
                 l2: bool = True, center: bool = False) -> np.ndarray:
    """A/A' — 엔티티 hidden state 중심 벡터 간 코사인 거리.

    토큰별 L2 → 평균 → 다시 L2. Qwen2 의 massive-activation 이상치 토큰 때문에
    비정규화 평균은 사실상 그 토큰 하나가 된다. post-RoPE key 는 쓰지 않는다 —
    회전 위상이 내용보다 코사인을 지배하기 때문.

    `ctx=True` 면 `baseline_compare.preds_m1:127-134` 의 M1-CosDrift 수식(m=2, ρ=0.7, a=0.5)을
    엔티티 벡터 e 위에서 그대로 적용한다.
    """
    X = Hr.astype(np.float32)
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    n = len(picks)
    e = np.zeros((n, X.shape[1]), np.float32)
    prev = None
    for i, sel in enumerate(picks):
        if sel.size == 0:
            e[i] = prev if prev is not None else 0.0
            continue
        v = X[sel]
        v = np.stack([_l2(x) for x in v]) if l2 else v
        e[i] = _l2(v.mean(0)) if l2 else v.mean(0)
        prev = e[i]
    d = np.zeros(n)
    if not ctx:
        for i in range(1, n):
            d[i] = 1.0 - float(e[i - 1] @ e[i])
        return d
    for t in range(1, n):                                   # preds_m1 수식 verbatim
        dp = 1.0 - float(e[t - 1] @ e[t])
        lo = max(0, t - M1["m"])
        w = np.array([M1["rho"] ** j for j in range(t - lo)])[::-1]
        c = (w[:, None] * e[lo:t]).sum(0)
        nrm = float(np.linalg.norm(c))
        dc = 1.0 - float(c @ e[t] / nrm) if nrm > 0 else dp
        d[t] = M1["a"] * dp + (1 - M1["a"]) * dc
    return d


def signal_setdrift(keys: list[list[str]]) -> np.ndarray:
    """B — 엔티티 집합 Jaccard 거리. 둘 다 공집합이면 0."""
    n = len(keys)
    d = np.zeros(n)
    for i in range(1, n):
        a, b = set(keys[i - 1]), set(keys[i])
        u = a | b
        d[i] = 0.0 if not u else 1.0 - len(a & b) / len(u)
    return d


DEFAULT_SETW = 6     # NLTK TextTiling 발표 기본 블록크기 k=6 (run_emllm_dts.py --sanity 가 쓰는 값).
                     # 외부·발표값이므로 dev-3 누출이 없다. PREREG 수정이력 A1 참조.


def signal_setdrift_w(keys: list[list[str]], w: int = DEFAULT_SETW) -> np.ndarray:
    """B' — **직전 W문장 블록 대비 신규 엔티티 비율** (인과적, PREREG 수정이력 A1 로 추가).

    사전등록된 `setdrift`(인접 1문장 Jaccard)는 사전점검에서 퇴화했다:
    인접 문장이 엔티티를 하나도 공유하지 않는 비율이 dev-3 세 문서 모두 0.73~0.86 이라
    d 가 1.0 에 뭉쳐 μ+σ 가 1 을 넘고 OP-1 이 경계를 0개 낸다.

    고치는 지점은 **비교 대상 창**이지 신호의 취지가 아니다:
        d_i = 1 − |E_i ∩ (∪_{j=i−W..i−1} E_j)| / |E_i|
    분모를 |E_i| 로 두는 이유 — 과거 블록은 계속 커지므로 Jaccard 로 두면 창 크기에 편향된다.
    E_i 가 비면 0.
    """
    n = len(keys)
    d = np.zeros(n)
    for i in range(1, n):
        cur = set(keys[i])
        if not cur:
            continue
        past: set[str] = set()
        for j in range(max(0, i - w), i):
            past |= set(keys[j])
        d[i] = 1.0 - len(cur & past) / len(cur)
    return d


def signal_recency(keys: list[list[str]], cap: int = DEFAULT_CAP) -> np.ndarray:
    """C — 엔티티 재등장 거리. 신규 엔티티는 CAP → 1.0. CAP 로 나눠 d∈(0,1]."""
    last: dict[str, int] = {}
    n = len(keys)
    d = np.zeros(n)
    for i, ks in enumerate(keys):
        if i > 0:
            vals = [min(i - last.get(k, i - cap), cap) for k in ks] if ks else [cap]
            d[i] = float(np.mean(vals)) / cap
        for k in ks:
            last[k] = i
    return d


def build_signal(name: str, *, Hr=None, picks=None, keys=None, cap: int = DEFAULT_CAP,
                 l2: bool = True, center: bool = False, setw: int = DEFAULT_SETW) -> np.ndarray:
    if name == "embed":
        return signal_embed(Hr, picks, ctx=False, l2=l2, center=center)
    if name == "embed_ctx":
        return signal_embed(Hr, picks, ctx=True, l2=l2, center=center)
    if name == "setdrift":
        return signal_setdrift(keys)
    if name == "setdrift_w":
        return signal_setdrift_w(keys, w=setw)
    if name == "recency":
        return signal_recency(keys, cap=cap)
    raise ValueError(name)


# ── 작동점 (셋 다 사전 동결) ──────────────────────────────────────────────────
DEGENERATE_STD = 1e-9


def is_degenerate(d: np.ndarray) -> bool:
    """상수 신호. 막지 않으면 `adaptive_boundaries` 의 `>=` 가 전 구간 경계로 터진다."""
    return float(np.std(np.asarray(d, np.float64)[1:])) < DEGENERATE_STD


def signal_to_boundaries(d: np.ndarray, op: str) -> np.ndarray:
    """거리 신호 → 전이 라벨 [n-1] (끝-turn 규약).

    OP-1 `adaptive_boundaries(c=1.0, online_nf)` — 🔒 D-B39, headline.
        `online_nf` 는 `warmup` 인자를 무시한다(`cnt<2` 하드코딩, hi_ontop_v2.py:199).
    OP-2 EM-LLM `μ + 1.1σ` online full-past — surprisal 과 like-for-like.
    OP-3 문서별 분위수 `1 − ANCHOR_R` — 🔒 D-B30. ⚠️ dev-3 gold 밀도(0.0657)와 사실상 정합.
    """
    d = np.asarray(d, np.float64)
    n = len(d)
    if is_degenerate(d):
        return np.zeros(n - 1, np.int8)
    if op == "OP-1":
        # ⚠️ `adaptive_boundaries` 는 `return out + [0]` (hi_ontop_v2.py:221) 이라 길이 n 의
        # **턴 단위** 예측을 준다. `score_A` 가 자기 후행 0 을 또 붙이므로 여기서 떼야 한다.
        # out[i-1] ↔ seq[i] = d[i] = dist(i-1,i) → 전이 i-1. 즉 out[:n-1] 이 전이 벡터다.
        return np.asarray(adaptive_boundaries(d.tolist(), c=1.0, mode="online_nf"),
                          np.int8)[:n - 1]
    if op == "OP-2":
        # d[0] 은 의미 없는 센티넬이므로 **과거 통계에서 뺀다**. s = d[1:] 의 index j 는
        # 문장 j+1 이고, 문장 j+1 이 새 세그먼트를 열면 전이 j 가 경계다 → v[j] = 1.
        # (surprisal 통제군은 surprise[0] 이 실제 값이므로 `surprisal_to_boundaries` 로 따로 간다.)
        v = np.zeros(n - 1, np.int8)
        for j in surprise_boundaries(d[1:], DEFAULT_GAMMA, win=0, warmup=3):
            if 0 <= j <= n - 2:
                v[j] = 1
        return v
    if op == "OP-3":
        thr = np.quantile(d[1:], 1 - ANCHOR_R)
        return (d[1:] >= thr).astype(np.int8)
    raise ValueError(op)


def surprisal_to_boundaries(surprise: np.ndarray, op: str) -> np.ndarray:
    """EM-LLM surprisal 통제군 — **원본 규약 verbatim**.

    우리 신호와 달리 `surprise[0]` 은 실제 값(문장 0 의 mean token NLL)이므로 센티넬 제거를
    하지 않는다. `segment_manual_emllm.py:204` 와 동일하게 전체 배열을 넘긴다.
    """
    s = np.asarray(surprise, np.float64)
    n = len(s)
    if is_degenerate(np.concatenate([[0.0], s[1:]])):
        return np.zeros(n - 1, np.int8)
    if op == "OP-2":
        v = np.zeros(n - 1, np.int8)
        for b in surprise_boundaries(s, DEFAULT_GAMMA, win=0, warmup=3):
            if 1 <= b <= n - 1:
                v[b - 1] = 1
        return v
    if op == "OP-1":
        return np.asarray(adaptive_boundaries(s.tolist(), c=1.0, mode="online_nf"),
                          np.int8)[:n - 1]          # 후행 0 제거 (위 signal_to_boundaries 참조)
    if op == "OP-3":
        return (s[1:] >= np.quantile(s[1:], 1 - ANCHOR_R)).astype(np.int8)
    raise ValueError(op)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="M9-AttnEntityDrift — attention 엔티티 거리 분절")
    ap.add_argument("--file", required=True, help="문장 파일(줄당 1문장)")
    ap.add_argument("--lm", default=DEFAULT_LM)
    ap.add_argument("--signal", choices=SIGNALS, default="setdrift")
    ap.add_argument("--scope", default="sentnext")
    ap.add_argument("--norm", choices=NORMS, default="rowsent")
    ap.add_argument("--layers", default=DEFAULT_LAYERS)
    ap.add_argument("--repr-layer", type=int, default=DEFAULT_REPR_LAYER)
    ap.add_argument("--head-agg", default="mean", choices=["mean", "perhead-vote"])
    ap.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    ap.add_argument("--entity-unit", default="word", choices=["word", "token"])
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP)
    ap.add_argument("--op", default="OP-1", choices=["OP-1", "OP-2", "OP-3"])
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    sents = read_sentences(args.file)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[입력] {args.file}: {len(sents)}문장 / [LM] {args.lm} ({device})", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.lm)
    model = AutoModelForCausalLM.from_pretrained(args.lm, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    dt = tokenize_doc(sents, tok)
    lyr = parse_layers(args.layers, model.config.num_hidden_layers)
    A, hs = attention_mass(dt, model, device, layers=lyr, query_scope=args.scope,
                           norm=args.norm, repr_layers=(args.repr_layer,))
    a_t = aggregate(A, head_agg=args.head_agg)
    picks, st = select_topk(a_t, dt, k=args.topk)
    keys = entity_keys(dt, picks, unit=args.entity_unit)
    d = build_signal(args.signal, Hr=hs[args.repr_layer], picks=picks, keys=keys, cap=args.cap)
    pred = signal_to_boundaries(d, args.op)
    bnds = [int(i + 1) for i, v in enumerate(pred) if v]

    print(f"\n[방법] M9-AttnEntityDrift signal={args.signal} scope={args.scope} "
          f"norm={args.norm} layers={args.layers} k={args.topk} op={args.op}")
    print(f"[진단] frac_truncated={st['frac_truncated']:.3f} "
          f"degenerate={is_degenerate(d)} n_distinct={len(np.unique(np.round(d, 9)))}")
    print(f"경계 ({len(bnds)}개): {bnds}")
    for i, s in enumerate(sents):
        mark = "▶" if i in set(bnds) else " "
        print(f"[{i:>3}] d={d[i]:.4f} {mark} {'/'.join(keys[i]):<28} | {s[:52]}")

    if args.dump:
        import json
        Path(args.dump).write_text(json.dumps({
            "lm": args.lm, "signal": args.signal, "scope": args.scope, "norm": args.norm,
            "layers": args.layers, "topk": args.topk, "op": args.op,
            "d": [round(float(x), 6) for x in d], "entities": keys, "boundaries": bnds,
            "diag": st,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[저장] {args.dump}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

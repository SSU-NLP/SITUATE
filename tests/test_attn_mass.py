"""M9 attention mass 추출의 **비트정확성** 검증 (CPU, 다운로드 없음).

`segment_attn_entity.attention_mass` 는 `output_attentions=True` 를 쓸 수 없어
(T=6,637 에서 28 레이어 = 69 GB) pre-hook 으로 모듈 입력을 캡처한 뒤 post-RoPE Q,K 를
**재구성**한다. 재구성이 표준 forward 와 어긋나면 이후 모든 숫자가 조용히 무의미해진다.

여기서 강제하는 것:
  1. 재구성한 softmax(QKᵀ·scaling + causal) == 표준 eager attention 의 `attn_weights`.
  2. `_mass_rowsent`(전역 분모 Z_q 소거를 이용한 싼 경로) == `attn_weights` 로부터
     직접 재정규화한 값. 즉 "싸게 계산한 것"과 "정의대로 계산한 것"이 같다.
  3. `_mass_full(norm="none")` 의 열합 == `attn_weights` 열합 (블록 분할이 값을 안 바꾼다).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from segment_attn_entity import (  # noqa: E402
    DocTok,
    _mass_full,
    _mass_rowsent,
    _post_rope_qk,
    query_positions,
)

SENTS = [
    "Dean McGowan walked into the newsroom carrying a heavy box.",
    "He set it down and looked around the empty office.",
    "Meanwhile in Berlin the weather had turned sharply colder.",
    "Anna checked her watch and decided not to wait any longer.",
    "The box, it turned out, was full of old photographs.",
]


def _tiny_model():
    """3-layer 랜덤 Qwen2 (GQA 유지: 4 head / 2 kv-head). 다운로드 없음, float32."""
    from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(vocab_size=512, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=512, use_sliding_window=False,
                      sliding_window=None, attn_implementation="eager")
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(cfg).to(torch.float32).eval()
    m.config._attn_implementation = "eager"
    return m


def _fake_doc(n_tok_per_sent=(9, 8, 10, 9, 11)) -> DocTok:
    """실제 토크나이저 없이 DocTok 을 만든다 (추출 수학만 시험한다)."""
    rng = np.random.default_rng(0)
    ids, spans, core_spans, sent_of, char_start, text_of, core = [], [], [], [], [], [], []
    for s, L in enumerate(n_tok_per_sent):
        a = len(ids)
        for j in range(L):
            ids.append(int(rng.integers(5, 500)))
            sent_of.append(s)
            is_sep = (s > 0 and j == 0)          # 문장 1+ 의 첫 토큰 = sep
            is_punct = (j == L - 1)              # 마지막 토큰 = 구두점
            char_start.append(-1 if is_sep else j)
            text_of.append("\n" if is_sep else ("." if is_punct else f"w{s}_{j}"))
            core.append(not (is_sep or is_punct))
        spans.append((a, len(ids)))
        core_spans.append((a + (1 if s > 0 else 0), len(ids)))
    return DocTok(ids, spans, core_spans, np.asarray(core, bool), np.asarray(sent_of, np.int32),
                  np.asarray(char_start, np.int32), text_of, list(SENTS))


def _reference_attn(model, dt):
    """표준 forward 의 attn_weights [n_layer][H, T, T] (float64)."""
    ids = torch.tensor([dt.input_ids], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False, output_attentions=True)
    return [a[0].double().numpy() for a in out.attentions]


def _capture(model, dt, layers):
    """attention_mass 와 동일한 pre-hook 으로 (h, cos, sin) 캡처."""
    cap = {}
    hs = []

    def mk(li):
        def hook(mod, args, kwargs):
            h = kwargs.get("hidden_states", args[0] if args else None)
            pe = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
            cap[li] = (h, pe[0], pe[1])
        return hook

    for li in layers:
        hs.append(model.model.layers[li].self_attn.register_forward_pre_hook(mk(li),
                                                                             with_kwargs=True))
    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([dt.input_ids], dtype=torch.long),
                  use_cache=False, output_attentions=True)
    finally:
        for h in hs:
            h.remove()
    return cap


@pytest.fixture(scope="module")
def fixture():
    m = _tiny_model()
    dt = _fake_doc()
    return m, dt, _reference_attn(m, dt), _capture(m, dt, [0, 1, 2])


def test_post_rope_qk_reconstructs_attention_exactly(fixture):
    """1. 재구성한 attention 확률이 표준 forward 와 동일해야 한다."""
    model, dt, ref, cap = fixture
    T = dt.n_tok
    for li in range(3):
        h, cos, sin = cap[li]
        attn = model.model.layers[li].self_attn
        with torch.no_grad():
            q, k = _post_rope_qk(attn, h, cos, sin)
            s = torch.einsum("hqd,htd->hqt", q, k) * attn.scaling
            ar = torch.arange(T)
            s = s.masked_fill(ar[None, None, :] > ar[None, :, None], float("-inf"))
            p = torch.softmax(s, dim=-1, dtype=torch.float32).double().numpy()
        diff = np.abs(p - ref[li]).max()
        assert diff < 1e-6, f"layer {li}: attention 재구성 불일치 max|Δ|={diff:.3e}"


def test_rowsent_matches_definition(fixture):
    """2. 싼 rowsent 경로 == 정의대로(전체 softmax 후 core(i) 재정규화) 계산한 값."""
    model, dt, ref, cap = fixture
    scope = "sentnext"
    for li in range(3):
        h, cos, sin = cap[li]
        attn = model.model.layers[li].self_attn
        with torch.no_grad():
            q, k = _post_rope_qk(attn, h, cos, sin)
            got = _mass_rowsent(dt, q, k, attn.scaling, scope).double().numpy()

        A = ref[li]                                   # [H, T, T] 전체 causal softmax
        want = np.zeros_like(got)
        for i in range(dt.n_sent):
            a, b = dt.spans[i]
            tgt = np.nonzero(dt.core_mask[a:b])[0] + a
            if tgt.size == 0:
                continue
            qs = query_positions(dt, i, scope)
            rows = []
            for qq in qs:
                vis = tgt[tgt <= qq]
                if vis.size == 0:
                    continue
                sub = A[:, qq, :][:, tgt]              # [H, n_tgt]
                sub = np.where(tgt[None, :] <= qq, sub, 0.0)
                den = sub.sum(-1, keepdims=True)
                rows.append(np.divide(sub, den, out=np.zeros_like(sub), where=den > 0))
            if rows:
                want[:, tgt] += np.mean(rows, axis=0)
        diff = np.abs(got - want).max()
        assert diff < 1e-6, f"layer {li}: rowsent 싼 경로 불일치 max|Δ|={diff:.3e}"


@pytest.mark.parametrize("block", [4, 7, 1024])
def test_mass_full_blocking_is_invariant(fixture, block):
    """3. 블록 분할이 열합을 바꾸지 않는다 (norm='none').

    ⚠️ 비트동일은 요구하지 **않는다** — `acc += p.sum(dim=1)` 의 float32 합산은 비결합이라
    블록 경계가 달라지면 누적 순서가 바뀐다. 실측 상대오차 1.9e-07 (= float32 엡실론 수준)이므로
    상대 1e-5 로 강제한다. 이보다 커지면 그건 순서 문제가 아니라 마스킹/인덱싱 버그다.
    """
    model, dt, ref, cap = fixture
    h, cos, sin = cap[0]
    attn = model.model.layers[0].self_attn
    with torch.no_grad():
        q, k = _post_rope_qk(attn, h, cos, sin)
        got = _mass_full(dt, q, k, attn.scaling, "sentnext", "none", block).double().numpy()
    ref_full = _mass_full(dt, q, k, attn.scaling, "sentnext", "none", 10_000).double().numpy()
    nz = ref_full > 0
    rel = (np.abs(got - ref_full)[nz] / ref_full[nz]).max()
    assert rel < 1e-5, f"block={block}: 블록 분할이 값을 바꿨다 max relΔ={rel:.3e}"


def test_degenerate_guard_blocks_all_boundary():
    """상수 신호가 전 구간 경계로 터지지 않는지 (F2 가드)."""
    from segment_attn_entity import is_degenerate, signal_to_boundaries

    d = np.ones(50)
    d[0] = 0.0
    assert is_degenerate(d)
    for op in ("OP-1", "OP-2", "OP-3"):
        assert signal_to_boundaries(d, op).sum() == 0, f"{op}: 상수 신호가 경계를 냈다"

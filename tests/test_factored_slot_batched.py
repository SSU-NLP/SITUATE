"""배치 경로가 기존 단일-문서 경로와 수치 일치하는지 검증 (Phase 2 재현성 보존)."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hi_ontop.factored_slot import FactoredSlot, STATE_AXES


def _model(mono: bool, d1: bool = True, shared_dim: int | None = None) -> FactoredSlot:
    torch.manual_seed(0)
    m = FactoredSlot(slot_dim=16, mono=mono, bottleneck=4, d1=d1, shared_dim=shared_dim)
    m.eval()
    return m


# (mono, d1, shared_dim) — d1=False = 공유 투영 한 벡터를 전 슬롯에 브로드캐스트(D-B33 애블레이션)
CONFIGS = [(False, True, None), (True, True, None), (False, False, None), (False, False, 20)]


def test_encode_seq_batched_matches_single():
    for mono, d1, sd in CONFIGS:
        m = _model(mono, d1, sd)
        torch.manual_seed(1)
        z = torch.randn(7, 768)
        with torch.no_grad():
            single = m.encode_seq(z)                       # [T, n_slots, d]
            batched = m.encode_seq_batched(z.unsqueeze(0))  # [1, T, n_slots, d]
        assert torch.allclose(single, batched[0], atol=1e-5), f"mono={mono} d1={d1} sd={sd}"
        assert single.shape == (7, 1 if mono else 5, 16)
        if not mono:
            # 공유 입력이라도 key 와 자기 상태로 슬롯이 갈라져야 한다(브로드캐스트 ≠ 5개 복제).
            assert not torch.allclose(single[:, 0], single[:, 1], atol=1e-4), f"d1={d1} sd={sd}"


def test_forward_batched_matches_single_unpadded():
    """B=1·패딩 없음이면 loss 가 기존 forward 와 일치해야 한다."""
    for mono, d1, sd in CONFIGS:
        m = _model(mono, d1, sd)
        torch.manual_seed(2)
        T = 9
        z = torch.randn(T, 768)
        resets = torch.randint(0, 2, (T - 1, 5))
        cols = list(STATE_AXES)  # [0, 2, 3]
        states_mat = torch.stack(
            [torch.randint(0, STATE_AXES[a][1], (T,)) for a in cols], dim=1)  # [T,3]
        states_dict = {STATE_AXES[a][0]: states_mat[:, i] for i, a in enumerate(cols)}
        with torch.no_grad():
            out1 = m(z, resets, states_dict)
            out2 = m.forward_batched(
                z.unsqueeze(0), resets.unsqueeze(0), states_mat.unsqueeze(0),
                torch.ones(1, T), torch.ones(1, T - 1))
        assert torch.allclose(out1["loss"], out2["loss"], atol=1e-5), f"mono={mono} d1={d1} sd={sd}"


def test_padding_is_ignored():
    """뒤쪽 패딩을 붙이고 마스크로 제외하면 loss 가 패딩 없는 경우와 같아야 한다."""
    m = _model(False)
    torch.manual_seed(3)
    T, P = 8, 5
    z = torch.randn(T, 768)
    resets = torch.randint(0, 2, (T - 1, 5))
    cols = list(STATE_AXES)
    states_mat = torch.stack([torch.randint(0, STATE_AXES[a][1], (T,)) for a in cols], dim=1)
    zp = torch.cat([z, torch.randn(P, 768)])
    rp = torch.cat([resets, torch.randint(0, 2, (P, 5))])
    sp = torch.cat([states_mat, torch.zeros(P, 3, dtype=torch.long)])
    sm = torch.cat([torch.ones(T), torch.zeros(P)]).unsqueeze(0)
    tm = torch.cat([torch.ones(T - 1), torch.zeros(P)]).unsqueeze(0)
    with torch.no_grad():
        ref = m.forward_batched(z.unsqueeze(0), resets.unsqueeze(0), states_mat.unsqueeze(0),
                                torch.ones(1, T), torch.ones(1, T - 1))
        pad = m.forward_batched(zp.unsqueeze(0), rp.unsqueeze(0), sp.unsqueeze(0), sm, tm)
    assert torch.allclose(ref["loss"], pad["loss"], atol=1e-5)


# ── D-B35 태거 입력 경로 (설계문서 §4.6 D1) ───────────────────────────────────
def _tag_model(tagger: str, axes: int) -> FactoredSlot:
    torch.manual_seed(0)
    m = FactoredSlot(slot_dim=16, bottleneck=8, tagger_dim=8, tagger=tagger, tagger_axes=axes,
                     dynamics="per_slot", pred_space="slot")
    return m.eval()


def test_tagger_paths_shapes_and_batch_equivalence():
    """4 조건 전부 [.,.,5,slot_dim] 을 내고 B=1 배치가 단일 경로와 일치해야 한다."""
    torch.manual_seed(1)
    z, feat = torch.randn(2, 7, 768), torch.randn(2, 7, 5, 8)
    for tagger, axes, want_in in (("replace", 5, 8), ("replace", 3, 8),
                                  ("hybrid", 5, 16), ("hybrid", 3, 16)):
        m = _tag_model(tagger, axes)
        assert m.C.in_features == want_in, f"{tagger}{axes}"
        with torch.no_grad():
            S = m.encode_seq_batched(z, feat)
            single = m.encode_seq(z[0], feat[0])
        assert S.shape == (2, 7, 5, 16)
        assert torch.allclose(single, S[0], atol=1e-5), f"{tagger}{axes}"


def test_replace5_ignores_embedding_entirely():
    """replace-5 = 순수 D1 — 학습형 extractor 가 없고 z 를 바꿔도 상태가 변하지 않아야 한다."""
    m = _tag_model("replace", 5)
    assert not any(n.startswith("extractors") for n, _ in m.named_parameters())
    torch.manual_seed(2)
    feat = torch.randn(1, 6, 5, 8)
    with torch.no_grad():
        a = m.encode_seq_batched(torch.randn(1, 6, 768), feat)
        b = m.encode_seq_batched(torch.randn(1, 6, 768), feat)
    assert torch.equal(a, b), "replace-5 인데 임베딩이 상태에 영향을 준다 = 누출"


def test_replace3_keeps_embedding_for_weak_axes():
    """replace-3 = causality/intentionality(슬롯 3·4)만 z 를 쓴다 — 앞 3축은 z 와 무관."""
    m = _tag_model("replace", 3)
    torch.manual_seed(3)
    feat = torch.randn(1, 6, 5, 8)
    with torch.no_grad():
        U1 = m._axis_input(torch.randn(1, 6, 768), feat)
        U2 = m._axis_input(torch.randn(1, 6, 768), feat)
    assert torch.equal(U1[..., :3, :], U2[..., :3, :]), "앞 3축이 z 에 반응한다"
    assert not torch.equal(U1[..., 3:, :], U2[..., 3:, :]), "뒤 2축이 z 를 안 쓴다"


def test_tagger_requires_features():
    m = _tag_model("hybrid", 5)
    with pytest.raises(ValueError):
        m.encode_seq_batched(torch.randn(1, 4, 768), None)


# ── dynamics (registry §3 사양) ────────────────────────────────────────────────
def test_dynamics_off_by_default_reproduces_phase2():
    """dynamics=None 이 기본 -> Phase 2 수치 완전 재현(예측 항이 loss 에 안 들어감)."""
    m = _model(False)
    assert m.dynamics is None and not hasattr(m, "forward_models")
    torch.manual_seed(5)
    T = 6
    z = torch.randn(T, 768)
    resets = torch.randint(0, 2, (T - 1, 5))
    cols = list(STATE_AXES)
    sm = torch.stack([torch.randint(0, STATE_AXES[a][1], (T,)) for a in cols], dim=1)
    with torch.no_grad():
        out = m(z, resets, {STATE_AXES[a][0]: sm[:, i] for i, a in enumerate(cols)})
    assert "l_pred" not in out and "pred_err" not in out


def test_dynamics_shapes_and_slot_counts():
    """M5=축별 5개 / M6=공유 1개 / M4=1개 (registry §3 dynamics 열)."""
    torch.manual_seed(0)
    m5 = FactoredSlot(slot_dim=16, mono=False, bottleneck=4, dynamics="per_slot")
    m6 = FactoredSlot(slot_dim=16, mono=False, bottleneck=4, dynamics="shared")
    m4 = FactoredSlot(slot_dim=16, mono=True, bottleneck=4, dynamics="per_slot")
    assert len(m5.forward_models) == 5, "M5 = 축별 dynamics 5개"
    assert len(m6.forward_models) == 1, "M6 = 공유 dynamics 1개"
    assert len(m4.forward_models) == 1, "M4 = 단일 슬롯이므로 1개"
    torch.manual_seed(6)
    z = torch.randn(1, 7, 768)
    with torch.no_grad():
        S = m5.encode_seq_batched(z)
        Sh, zh = m5.predict_next(S)
    assert Sh.shape == (1, 6, 5, 16) and zh.shape == (1, 7 - 1, 768)


def test_dynamics_adds_prediction_error_and_matches_batched():
    """예측오차가 loss 에 들어가고, B=1 에서 단일 경로와 배치 경로가 일치."""
    for dyn in ("per_slot", "shared"):
        torch.manual_seed(0)
        m = FactoredSlot(slot_dim=16, mono=False, bottleneck=4, dynamics=dyn)
        m.eval()
        torch.manual_seed(7)
        T = 8
        z = torch.randn(T, 768)
        z = z / z.norm(dim=-1, keepdim=True)
        resets = torch.randint(0, 2, (T - 1, 5))
        cols = list(STATE_AXES)
        sm = torch.stack([torch.randint(0, STATE_AXES[a][1], (T,)) for a in cols], dim=1)
        with torch.no_grad():
            o1 = m(z, resets, {STATE_AXES[a][0]: sm[:, i] for i, a in enumerate(cols)})
            o2 = m.forward_batched(z.unsqueeze(0), resets.unsqueeze(0), sm.unsqueeze(0),
                                   torch.ones(1, T), torch.ones(1, T - 1))
        assert "pred_err" in o1 and o1["pred_err"].shape == (T - 1,)
        assert torch.allclose(o1["loss"], o2["loss"], atol=1e-5), dyn
        assert torch.allclose(o1["pred_err"], o2["pred_err"][0], atol=1e-5), dyn


# ── §4.6 Axis2: slot-native 예측 공간 ─────────────────────────────────────────
def _fixture(dyn="per_slot", space="obs", T=8):
    torch.manual_seed(0)
    m = FactoredSlot(slot_dim=16, mono=False, bottleneck=4, dynamics=dyn, pred_space=space)
    m.eval()
    torch.manual_seed(11)
    z = torch.randn(T, 768)
    z = z / z.norm(dim=-1, keepdim=True)
    resets = torch.randint(0, 2, (T - 1, 5))
    cols = list(STATE_AXES)
    sm = torch.stack([torch.randint(0, STATE_AXES[a][1], (T,)) for a in cols], dim=1)
    return m, z, resets, sm, cols, T


def test_pred_space_default_is_obs():
    """기본값 유지 = 기존 체크포인트/수치 재현."""
    m = FactoredSlot(slot_dim=16, mono=False, bottleneck=4, dynamics="per_slot")
    assert m.pred_space == "obs"


def test_pred_space_slot_uses_slot_space_error():
    """slot-native 면 pred_err 가 ‖Ŝ−S‖² 의 슬롯 평균과 일치해야 한다 (768D 항 미포함)."""
    m, z, resets, sm, cols, T = _fixture(space="slot")
    with torch.no_grad():
        S = m.encode_seq(z)
        Sh, _ = m.predict_next(S)
        expect = ((Sh - S[1:]) ** 2).sum(-1).mean(-1)
        out = m(z, resets, {STATE_AXES[a][0]: sm[:, i] for i, a in enumerate(cols)})
    assert torch.allclose(out["pred_err"], expect, atol=1e-5)


def test_pred_space_variants_differ_and_batched_matches():
    """obs / slot / both 가 서로 다른 loss 를 주고, 각각 배치 경로와 일치."""
    losses = {}
    for space in ("obs", "slot", "both"):
        m, z, resets, sm, cols, T = _fixture(space=space)
        with torch.no_grad():
            o1 = m(z, resets, {STATE_AXES[a][0]: sm[:, i] for i, a in enumerate(cols)})
            o2 = m.forward_batched(z.unsqueeze(0), resets.unsqueeze(0), sm.unsqueeze(0),
                                   torch.ones(1, T), torch.ones(1, T - 1))
        assert torch.allclose(o1["loss"], o2["loss"], atol=1e-5), space
        losses[space] = float(o1["loss"])
    assert losses["obs"] != losses["slot"]
    assert min(losses["obs"], losses["slot"]) <= losses["both"] <= max(losses["obs"], losses["slot"]) + 1e-4


def test_pred_space_invalid_raises():
    try:
        FactoredSlot(slot_dim=8, dynamics="per_slot", pred_space="nope")
    except ValueError:
        return
    raise AssertionError("잘못된 pred_space 가 통과됨")


# ── D-B29 누출 방지 회귀 (P/Ψ 묶음) ──────────────────────────────────────────
def _mk(**kw):
    import torch
    from hi_ontop.factored_slot import FactoredSlot
    torch.manual_seed(0)
    return FactoredSlot(slot_dim=16, dynamics="per_slot", pred_space="slot", **kw).eval()


def test_prediction_does_not_see_the_future():
    """ŝ_t = f(s_{t-1}) 는 s_t 를 보면 안 된다 — z 의 미래를 흔들어도 ŝ 가 안 변해야 한다."""
    import torch
    m = _mk()
    z = torch.randn(1, 8, 768)
    with torch.no_grad():
        S = m.encode_seq_batched(z)
        Sh, _ = m.predict_next(S)
        z2 = z.clone()
        z2[:, 5:] = torch.randn(1, 3, 768)          # 시점 5 이후를 완전히 교체
        Sh2, _ = m.predict_next(m.encode_seq_batched(z2))
    # 전이 j 의 예측 ŝ_j = f(s_j) 는 j <= 4 인 동안 미래와 무관해야 한다
    assert torch.allclose(Sh[:, :5], Sh2[:, :5], atol=1e-6)


def test_psi_is_causal():
    """ψₑ[t] 는 t 까지의 정보만 담아야 한다."""
    import torch
    m = _mk(psi="cummean")
    z = torch.randn(1, 8, 768)
    bnd = torch.zeros(1, 7)
    with torch.no_grad():
        P1 = m.psi_seq(m.encode_seq_batched(z), bnd)
        z2 = z.clone()
        z2[:, 5:] = torch.randn(1, 3, 768)
        P2 = m.psi_seq(m.encode_seq_batched(z2), bnd)
    assert torch.allclose(P1[:, :5], P2[:, :5], atol=1e-6)


def test_psi_resets_at_boundary():
    """경계 직후 ψ 는 그 시점 상태 자체여야 한다(이전 에피소드가 안 섞임)."""
    import torch
    m = _mk(psi="cummean")
    z = torch.randn(1, 6, 768)
    bnd = torch.zeros(1, 5)
    bnd[0, 2] = 1.0                                  # 전이 2 가 경계 -> s_3 부터 새 에피소드
    with torch.no_grad():
        S = m.encode_seq_batched(z)
        Psi = m.psi_seq(S, bnd)
    assert torch.allclose(Psi[:, 3], S[:, 3], atol=1e-6)


def test_legacy_path_unchanged():
    """head_feat='delta', psi=None 은 기존 입력 구성과 동일해야 한다."""
    import torch
    m = _mk()
    a, b = torch.randn(2, 3, 16), torch.randn(2, 3, 16)
    got = m.head_input(a, b, None, None)
    assert torch.allclose(got, torch.cat([a, b, b - a, b * a], -1))
    assert m.head_in == 4 * 16

"""배치 경로가 기존 단일-문서 경로와 수치 일치하는지 검증 (Phase 2 재현성 보존)."""
from __future__ import annotations
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hi_ontop.factored_slot import FactoredSlot, STATE_AXES


def _model(mono: bool) -> FactoredSlot:
    torch.manual_seed(0)
    m = FactoredSlot(slot_dim=16, mono=mono, bottleneck=4)
    m.eval()
    return m


def test_encode_seq_batched_matches_single():
    for mono in (False, True):
        m = _model(mono)
        torch.manual_seed(1)
        z = torch.randn(7, 768)
        with torch.no_grad():
            single = m.encode_seq(z)                       # [T, n_slots, d]
            batched = m.encode_seq_batched(z.unsqueeze(0))  # [1, T, n_slots, d]
        assert torch.allclose(single, batched[0], atol=1e-5), f"mono={mono}"


def test_forward_batched_matches_single_unpadded():
    """B=1·패딩 없음이면 loss 가 기존 forward 와 일치해야 한다."""
    for mono in (False, True):
        m = _model(mono)
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
        assert torch.allclose(out1["loss"], out2["loss"], atol=1e-5), f"mono={mono}"


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

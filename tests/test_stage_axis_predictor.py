"""Numerical and causal regression tests for the extracted SITUATE predictor."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/stage"))
from axis_predictor import AxisPredictor, FactoredSlot


def fixture_model():
    torch.manual_seed(73)
    head = FactoredSlot(z_dim=16, slot_dim=8, bottleneck=4,
                        dynamics="per_slot").eval()
    return head, AxisPredictor(head), F.normalize(torch.randn(13, 16), dim=-1)


def test_matches_original_forward():
    head, model, z = fixture_model()
    with torch.no_grad():
        labels = {name: torch.zeros(len(z), dtype=torch.long)
                  for name in ("perspective", "temporal_shift_marker", "event_type")}
        original = head(z, torch.zeros(12, 5), labels)
        expected = torch.stack([x.sigmoid() for x in original["change_logits"].values()], -1)
        actual, state = model(z)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(state, original["S"][-1])
    assert all(not p.requires_grad for p in model.parameters())
    assert not any(k.startswith(("state_heads", "dynamics", "readout"))
                   for k in model.state_dict())


def test_chunks_and_future_do_not_change_past():
    _, model, z = fixture_model()
    whole, final = model(z)
    first, state = model(z[:5])
    second, state = model(z[5:6], state)
    third, state = model(z[6:], state)
    torch.testing.assert_close(torch.cat([first, second, third]), whole)
    torch.testing.assert_close(state, final)
    altered = z.clone()
    altered[5:] = torch.randn_like(altered[5:])
    torch.testing.assert_close(model(altered)[0][:4], first)


def test_initial_empty_and_bad_inputs():
    _, model, z = fixture_model()
    assert model(z[:1])[0].shape == (0, 5)
    probabilities, state = model(z[:0])
    assert probabilities.shape == (0, 5) and state is None
    with pytest.raises(ValueError):
        model(torch.full_like(z, float("nan")))
    with pytest.raises(ValueError):
        model(z[:, :3])
    with pytest.raises(ValueError):
        model(z, torch.zeros(1))
    model.train()
    with pytest.raises(RuntimeError):
        model(z)

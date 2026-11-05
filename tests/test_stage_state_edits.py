from __future__ import annotations

from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/stage"))
from state_edits import apply_edits


def patch(**kwargs):
    result = dict(remove=[], replace=[], add=[], developments_since_previous_checkpoint=[])
    result.update(kwargs)
    return result


def test_preserve_goal_update_relationship():
    prior = ["A wants to escape", "A trusts B"]
    assert apply_edits(prior, patch(), {2})["current_state"][1]["claim"] == "A trusts B"
    out = apply_edits(prior, patch(replace=[{
        "previous_claim": "A trusts B", "claim": "A distrusts B",
        "evidence_scene_orders": [3]}]), {3})
    assert out["current_state"] == [
        {"claim": "A wants to escape", "evidence_scene_orders": []},
        {"claim": "A distrusts B", "evidence_scene_orders": [3]}]


def test_reject_future_evidence_and_unknown_removal():
    with pytest.raises(ValueError):
        apply_edits(["A trusts B"], patch(remove=[{
            "claim": "A trusts B", "evidence_scene_orders": [4]}]), {3})
    with pytest.raises(ValueError):
        apply_edits(["A trusts B"], patch(remove=[{
            "claim": "A wants to escape", "evidence_scene_orders": [3]}]), {3})


def test_no_silent_overflow_or_duplicate():
    with pytest.raises(ValueError):
        apply_edits([str(i) for i in range(8)], patch(add=[{
            "claim": "new", "evidence_scene_orders": [3]}]), {3})
    with pytest.raises(ValueError):
        apply_edits(["old"], patch(add=[{
            "claim": "old", "evidence_scene_orders": [3]}]), {3})

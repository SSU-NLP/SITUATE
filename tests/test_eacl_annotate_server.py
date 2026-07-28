import importlib.util
import json
from pathlib import Path

import pytest


P = Path(__file__).parents[1] / "scripts/eacl_phase0/annotate_server.py"
spec = importlib.util.spec_from_file_location("annotate_server", P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def write_pack(root, axis="protagonist", value=None):
    row = {"doc_id": "d", "transition": 1, "previous": "English A.",
           "previous_ko_reference": "영어 A.", "current": "English B.",
           "current_ko_reference": "영어 B.", axis: value, "notes": ""}
    (root / f"{axis}.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n")


def test_public_rows_are_allowlisted_and_blinded(tmp_path, monkeypatch):
    write_pack(tmp_path)
    monkeypatch.setattr(m, "_DIR", tmp_path)
    row = m.public_rows("protagonist")[0]
    assert row["previous_ko_reference"] == "영어 A."
    assert row["value"] is None
    assert not ({"boundary", "idm_score", "place", "time", "causality", "goal"} & set(row))
    with pytest.raises(ValueError):
        m.public_rows("boundary")


def test_save_is_atomic_and_null_restores(tmp_path, monkeypatch):
    write_pack(tmp_path)
    monkeypatch.setattr(m, "_DIR", tmp_path)
    out = m.save_annotation("protagonist", "d", 1, 1, "changed")
    assert out == {"ok": True, "filled": 1, "total": 1}
    assert m.read_jsonl(tmp_path / "protagonist.jsonl")[0]["protagonist"] == 1
    out = m.save_annotation("protagonist", "d", 1, None, "undo")
    assert out["filled"] == 0
    assert m.read_jsonl(tmp_path / "protagonist.jsonl")[0]["protagonist"] is None
    assert not (tmp_path / "protagonist.jsonl.tmp").exists()


def test_rejects_invalid_value_and_unavailable_boundary(tmp_path, monkeypatch):
    write_pack(tmp_path)
    monkeypatch.setattr(m, "_DIR", tmp_path)
    with pytest.raises(ValueError):
        m.save_annotation("protagonist", "d", 1, 2, "bad")
    with pytest.raises(ValueError):
        m.save_annotation("boundary", "d", 1, 1, "leak")


def test_korean_reference_is_always_visible():
    assert '<div class="ko"><b>한국어 참고</b>' in m.PAGE
    assert '<details class="ko">' not in m.PAGE


def test_writingprompts_english_is_detokenized_for_display():
    text = m.display_english("I ca n't believe it 's true . `` Really ? ''")
    assert text == 'I can\'t believe it\'s true. "Really?"'

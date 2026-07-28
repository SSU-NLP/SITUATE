import importlib.util
import json
from pathlib import Path


P = Path(__file__).parents[1] / "scripts/eacl_phase0/split_blinded_packs.py"
spec = importlib.util.spec_from_file_location("split_blinded", P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_write_jsonl_exposes_only_one_label(tmp_path):
    row = {"doc_id": "d", "transition": 1, "previous": "a", "current": "b"}
    out = tmp_path / "protagonist.jsonl"
    item = {**row, "protagonist": None, "notes": ""}
    m.write_jsonl(out, [item])
    loaded = json.loads(out.read_text())
    assert set(loaded) == {*row, "protagonist", "notes"}
    assert not ({"boundary", "place", "time", "causality", "goal"} & set(loaded))

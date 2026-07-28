import importlib.util
from pathlib import Path


P = Path(__file__).parents[1] / "scripts/eacl_phase0/evaluate_resets.py"
spec = importlib.util.spec_from_file_location("phase0", P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_auc_ties():
    assert m.auc([0, 0, 1, 1], [0, 1, 1, 2]) == 0.875


def test_summary_coordinated_resets_win():
    rows = []
    for i in range(10):
        boundary = int(i >= 5)
        r = {"doc_id": "d", "boundary": boundary, "idm_score": i % 2}
        r.update({s: boundary for s in m.SLOTS})
        rows.append(r)
    out = m.summarize(rows)
    assert out["auc_resets"] == 1.0
    assert out["auc_gain"] > .10
    assert out["mean_resets_boundary"] == 5
    assert out["mean_resets_internal"] == 0

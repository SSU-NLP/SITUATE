import importlib.util
from pathlib import Path


P = Path(__file__).parents[1] / "scripts/eacl_phase0/build_narrative_pack.py"
spec = importlib.util.spec_from_file_location("narrative_pack", P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def story(i, text, tr):
    return {"story_id": str(i), "title": f"s{i}", "text": text,
            "source_url": f"https://example.test/{i}", "license": "test", "translations": tr}


def test_free_boundary_and_bilingual_alignment():
    rows = m.build([
        story(1, "I ran home. It was late.", ["집으로 달렸다.", "늦었다."]),
        story(2, "She found a key. It opened the box.", ["열쇠를 찾았다.", "상자를 열었다."]),
    ])
    assert len(rows) == 3
    assert [r["boundary"] for r in rows] == [0, 1, 0]
    assert rows[1]["previous"] == "It was late."
    assert rows[1]["current"] == "She found a key."
    assert rows[1]["current_ko_reference"] == "열쇠를 찾았다."

"""Accept legacy screenplay normalization only when the released SHA-256 matches."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.release_root / "evaluation/src"))
    from stage_narrative.screenplay_titles import normalize_public_scene_titles
    found, missing = [], []
    for source in sorted((args.source_root / "STAGE").glob("*/*/script.json")):
        relative = source.relative_to(args.source_root)
        asset = args.release_root / relative.parent / "task_1_reference_state_update.json"
        expected = json.loads(asset.read_text())["script"]
        original = source.read_bytes()
        records = json.loads(original)
        numbered = deepcopy(records)
        for i, row in enumerate(numbered, 1):
            row["_id"] = i
        variants = [("original_records", records), ("one_based_ids", numbered)]
        variants += [(name + "+official_titles", normalize_public_scene_titles(rows)[0])
                     for name, rows in variants]
        match = None
        if hashlib.sha256(original).hexdigest() == expected["sha256"]:
            match = ("original_bytes", original)
        for name, rows in variants:
            if match:
                break
            for indent, ascii_, sort, newline in (
                    (indent, ascii_, sort, newline) for indent in (2, 4, None)
                    for ascii_ in (False, True) for sort in (True, False)
                    for newline in ("\n", "")):
                raw = (json.dumps(rows, ensure_ascii=ascii_, indent=indent,
                                  sort_keys=sort) + newline).encode()
                if hashlib.sha256(raw).hexdigest() == expected["sha256"]:
                    match = (f"{name};indent={indent};ascii={ascii_};"
                             f"sort={sort};newline={bool(newline)}", raw)
                    break
        if match:
            target = args.output_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.read_bytes() != match[1]:
                    raise FileExistsError(f"Refusing to overwrite different bytes: {target}")
            else:
                with target.open("xb") as stream:
                    stream.write(match[1])
            found.append({"path": str(relative), "source": str(source),
                          "transform": match[0], "sha256": expected["sha256"]})
        else:
            missing.append({"path": str(relative), "scenes": len(records),
                            "expected_scenes": expected["scene_count"]})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"verified": found, "unmatched": missing}, indent=2) + "\n")
    print(json.dumps({"verified": len(found), "unmatched": len(missing),
                      "english_verified": sum("/English/" in x["path"] for x in found)}))


if __name__ == "__main__":
    main()

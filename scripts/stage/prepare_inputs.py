"""Materialize hash-verified ordinary Task I inputs without mutating upstream assets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from preflight import sha256


def order_movies(rows: list[dict], root: Path) -> list[dict]:
    """Choose two small bilingual smoke cases by input bytes, without score selection."""
    first = [min((r for r in rows if r["language"] == language),
                 key=lambda r: (root / r["movie_id"] / "script.json").stat().st_size)
             for language in ("en", "zh")]
    selected = {row["movie_id"] for row in first}
    return first + [row for row in rows if row["movie_id"] not in selected]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--screenplays", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for ref in sorted((args.release_root / "STAGE").glob("*/*/task_1_reference_state_update.json")):
        reference = json.loads(ref.read_text())
        relative = ref.parent.relative_to(args.release_root)
        script = args.screenplays / relative / "script.json"
        if sha256(script) != reference["script"]["sha256"]:
            raise ValueError(f"Unverified screenplay: {script}")
        target = args.output_root / reference["movie_id"]
        target.mkdir(parents=True, exist_ok=False)
        sources = {}
        for name in ("task_1_reference_state_update.json", "task_1_autoregressive_state_update.json"):
            source = ref.parent / name
            asset = json.loads(source.read_text())
            language = {"English": "en", "Chinese": "zh", "en": "en", "zh": "zh"}[asset["language"]]
            asset["language"] = language
            (target / name).write_text(json.dumps(asset, ensure_ascii=False, indent=2) + "\n")
            sources[name] = {"source_sha256": sha256(source),
                             "derived_sha256": sha256(target / name)}
        shutil.copyfile(script, target / "script.json")
        rows.append({"movie_id": reference["movie_id"], "language": language,
                     "checkpoint_count": reference["counts"]["checkpoints"],
                     "trajectory_count": reference["counts"]["trajectories"],
                     "script_sha256": sha256(script), "assets": sources})
    manifest = {"condition": "ordinary", "movies": order_movies(rows, args.output_root),
                "asset_transform": "language only: English->en, Chinese->zh",
                "max_concurrent_jobs": 2, "max_total_gpus": 4,
                "smoke_selection": "smallest screenplay bytes per language; en then zh",
                "anonymous_condition": "not acquired; not included in this run"}
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(rows)} hash-verified movies")


if __name__ == "__main__":
    main()

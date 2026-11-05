"""Movie-macro aggregation and paired movie bootstrap for Task I interventions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_tracking import METHODS


def leaves(value: dict, prefix: str = "") -> dict:
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(leaves(item, name))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            result[name] = item
        elif item is None:
            result[name] = float("nan")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    movies = json.loads((args.inputs / "manifest.json").read_text())["movies"]
    movies = movies[args.shard::args.num_shards]
    result = {"condition": "ordinary", "judge": "Qwen/Qwen3-32B",
              "bootstrap_seed": 20260727, "bootstrap_replicates": 1000,
              "movie_count": len(movies), "groups": {}}
    for language in ("all", "en", "zh"):
        subset = [m for m in movies if language == "all" or m["language"] == language]
        if not subset:
            continue
        matrices = {}
        metrics = None
        for method in METHODS:
            rows = []
            for movie in subset:
                path = args.evaluation / movie["movie_id"] / method / "summary.json"
                summary = json.loads(path.read_text())
                row = {}
                for setting in ("reference_state_update", "autoregressive_state_update", "accumulation_gap"):
                    row.update(leaves(summary[setting], setting))
                if metrics is None:
                    metrics = sorted(row)
                if sorted(row) != metrics:
                    raise ValueError(f"Metric coverage differs: {path}")
                rows.append([row[key] for key in metrics])
            matrices[method] = np.asarray(rows)
        rng = np.random.default_rng(20260727)
        resamples = rng.integers(0, len(subset), (1000, len(subset)))
        group = {"movies": [m["movie_id"] for m in subset], "methods": {}, "paired_differences": {}}
        def aggregate(values):
            output = {}
            for i, key in enumerate(metrics):
                column = values[:, i]
                valid = np.isfinite(column)
                sampled = column[resamples]
                counts = np.isfinite(sampled).sum(1)
                usable = counts > 0
                means = np.nansum(sampled[usable], axis=1) / counts[usable]
                output[key] = {"mean": float(column[valid].mean()) if valid.any() else None,
                               "ci95": np.quantile(means, [.025, .975]).tolist() if len(means) else None,
                               "defined_movies": int(valid.sum())}
            return output
        for method, values in matrices.items():
            group["methods"][method] = aggregate(values)
        for left, right in (("edit", "rewrite"), ("edit_axes", "edit"),
                            ("edit_axes", "edit_permuted_axes")):
            group["paired_differences"][f"{left}-minus-{right}"] = aggregate(matrices[left] - matrices[right])
        result["groups"][language] = group
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"SUMMARY complete: {len(movies)} movies x {len(METHODS)} methods")


if __name__ == "__main__":
    main()

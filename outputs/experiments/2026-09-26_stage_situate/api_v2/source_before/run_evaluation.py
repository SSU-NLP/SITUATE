"""Apply the unmodified official judge and aggregation runner to all four methods."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from run_tracking import METHODS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("evaluation", "inputs", "config", "predictions", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    movies = json.loads((args.inputs / "manifest.json").read_text())["movies"]
    movies = movies[args.shard::args.num_shards]
    if args.limit:
        movies = movies[:args.limit]
    for movie in movies:
        movie_id = movie["movie_id"]
        for method in METHODS:
            cmd = [sys.executable, str(args.evaluation / "scripts/run_task1_state_update_evaluation.py"),
                   "--reference-asset", str(args.inputs / movie_id / "task_1_reference_state_update.json"),
                   "--prediction-run", str(args.predictions / movie_id / method),
                   "--script", str(args.inputs / movie_id / "script.json"),
                   "--config", str(args.config), "--output-root", str(args.output / movie_id / method),
                   "--workers", "24", "--resume"]
            subprocess.run(cmd + ["--dry-run"], check=True)
            subprocess.run(cmd, check=True)
            print(f"EVALUATED {movie_id} {method}", flush=True)


if __name__ == "__main__":
    main()

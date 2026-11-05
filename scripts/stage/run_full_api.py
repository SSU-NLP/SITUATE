"""Resume the full paired experiment, with persistent per-movie progress."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
CLI = "/home/namchailin/.local/bin/vesslctl"


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--movie-workers", type=int, default=16)
    args = parser.parse_args()
    run = args.run.resolve()
    lock = (run / "full_run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.chdir(ROOT)
    os.environ["STAGE_USAGE_LEDGER"] = str(run / "usage.jsonl")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    inputs = ROOT / "data/stage_task1_inputs"
    movies = json.loads((inputs / "manifest.json").read_text())["movies"]
    status = {"status": "running", "started_at": time.time(), "pid": os.getpid(),
              "total_movies": len(movies), "completed": [], "failed": [],
              "methods": ["rewrite", "rewrite_axes"], "seed": 1,
              "movie_workers": args.movie_workers}
    write_json(run / "progress.json", status)
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (ROOT / "scripts/stage").glob("*.py")}
    write_json(run / "source_hashes.json", hashes)
    (run / "logs").mkdir(exist_ok=True)

    def execute_movie(movie):
        mid = movie["movie_id"]
        if all((run / "evaluation" / mid / method / "summary.json").exists()
               for method in status["methods"]):
            return mid, None
        axis = run / "axes" / f"{mid}.json"
        for _ in range(60):
            if axis.exists():
                break
            result = subprocess.run([
                CLI, "volume", "download", "objvol-gsvyr0eu87wt", str(run / "axes"),
                "--remote-prefix", "hiontop/stage-situate-api-v2/axes/"],
                capture_output=True, text=True, timeout=120)
            if axis.exists():
                break
            time.sleep(30)
        if not axis.exists():
            return mid, "axis artifact unavailable"
        common = ["--evaluation", str(ROOT / "benchmarks/STAGE_v0/evaluation"),
                  "--inputs", str(inputs), "--config", str(run / "config.json"),
                  "--num-shards", "1", "--movie-id", mid]
        commands = [
            [sys.executable, "scripts/stage/run_tracking.py", *common,
             "--axes", str(run / "axes"), "--output", str(run / "predictions"), "--workers", "64"],
            [sys.executable, "scripts/stage/run_evaluation.py", *common,
             "--predictions", str(run / "predictions"), "--output", str(run / "evaluation")]]
        with (run / "logs" / f"{mid}.log").open("a") as log:
            for command in commands:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    return mid, f"{Path(command[1]).name}: exit {result.returncode}"
        for method in status["methods"]:
            if not (run / "evaluation" / mid / method / "summary.json").exists():
                return mid, "missing evaluation summary"
        return mid, None

    for phase, batch in (("smoke", movies[:2]), ("full", movies[2:])):
        status["phase"] = phase
        write_json(run / "progress.json", status)
        with ThreadPoolExecutor(max_workers=args.movie_workers) as pool:
            futures = [pool.submit(execute_movie, movie) for movie in batch]
            for future in as_completed(futures):
                try:
                    mid, error = future.result()
                except Exception as exc:
                    mid, error = "unknown", f"{type(exc).__name__}: {exc}"
                if error:
                    status["failed"].append({"movie_id": mid, "error": error})
                else:
                    status["completed"].append(mid)
                status["updated_at"] = time.time()
                write_json(run / "progress.json", status)
                print(f"Completed {len(status['completed'])}/{len(movies)}; failures={len(status['failed'])}", flush=True)
        if phase == "smoke" and status["failed"]:
            status["status"] = "smoke_failed"
            write_json(run / "progress.json", status)
            return
    if not status["failed"]:
        result = subprocess.run([sys.executable, "scripts/stage/summarize.py", "--inputs", str(inputs),
                                 "--evaluation", str(run / "evaluation"), "--output", str(run / "summary.json")])
        status["status"] = "completed" if result.returncode == 0 else "aggregation_failed"
    else:
        status["status"] = "incomplete"
    status["finished_at"] = time.time()
    write_json(run / "progress.json", status)


if __name__ == "__main__":
    main()

"""Submit at most two two-GPU jobs, with an idempotent local submission ledger."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import subprocess

CLI = "/home/namchailin/.local/bin/vesslctl"
TERMINAL = {"succeeded", "failed", "terminated", "deleted", "cancelled", "canceled"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--remote-package", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    package = json.loads(args.manifest.read_text())
    args.state.mkdir(parents=True, exist_ok=True)
    with (args.state / "submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for shard in range(2):
            ledger = args.state / f"submission{shard}.json"
            if ledger.exists():
                print(f"Already attempted shard {shard}; inspect ledger before any retry")
                continue
            name = f"situate-stage-20260926-shard{shard}"
            command = [CLI, "job", "create", "-n", name, "-r", "resourcespec-a100x2",
                       "-i", "quay.io/vessl-ai/torch@sha256:8bd418d49eeed841a607582b614c9d3ba86590b07d1be671816176c47253344a",
                       "--object-volume", "objvol-gsvyr0eu87wt:/shared",
                       "--secret", "HF_TOKEN=HIONTOP_HF_CHAILIN",
                       "-e", f"PACKAGE={args.remote_package}",
                       "-e", f"PACKAGE_SHA256={package['sha256']}",
                       "-e", f"OUT={args.remote_root}/shard{shard}", "-e", f"SHARD={shard}",
                       "--cmd", f"bash {args.remote_root}/job_stage_tracking.sh", "-o", "json"]
            if not args.execute:
                print(json.dumps(command))
                continue
            jobs = json.loads(subprocess.check_output([CLI, "job", "list", "-o", "json"], text=True))
            if any(job["name"] == name for job in jobs):
                raise RuntimeError(f"A remote job already has this name: {name}")
            active = [job for job in jobs if job["workloadState"] not in TERMINAL]
            gpu_count = sum(int(job.get("resourceSpecAcceleratorCount") or 0) for job in active)
            if len(active) >= 2 or gpu_count + 2 > 4:
                raise RuntimeError(f"User resource limit: {len(active)} active jobs, {gpu_count} GPUs")
            ledger.write_text(json.dumps({"status": "submission_started", "command": command}, indent=2))
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            ledger.write_text(json.dumps({"returncode": result.returncode,
                                           "stdout": result.stdout, "stderr": result.stderr,
                                           "command": command}, indent=2))
            result.check_returncode()
            print(result.stdout, flush=True)


if __name__ == "__main__":
    main()

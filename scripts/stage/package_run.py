"""Build an immutable VESSL experiment bundle with only required code and assets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    artifacts = root / "outputs/experiments/2026-09-26_qwen17b_lora_e1_s1/fast/artifacts"
    report = root / "outputs/experiments/2026-09-26_stage_situate"
    tokenizer = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json"
    entries = []
    for directory, dest in ((root / "scripts/stage", "scripts/stage"),
                            (root / "src/hi_ontop", "src/hi_ontop"),
                            (root / "benchmarks/STAGE_v0/evaluation", "evaluation"),
                            (root / "data/stage_task1_inputs", "inputs"),
                            (artifacts / "adapter_ep001", "artifacts/adapter_ep001")):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                entries.append((path, str(Path(dest) / path.relative_to(directory))))
    for source, dest in (
            (root / "scripts/vessl/job_stage_tracking.sh", "scripts/vessl/job_stage_tracking.sh"),
            (artifacts / "M5_lora_ep001_seed1.pt", "artifacts/M5_lora_ep001_seed1.pt"),
            (artifacts / "manifest.json", "artifacts/manifest.json"),
            (report / "model_revisions.json", "model_revisions.json"),
            (root / "tests/test_stage_axis_predictor.py", "tests/test_stage_axis_predictor.py"),
            (root / "tests/test_stage_state_edits.py", "tests/test_stage_state_edits.py"),
            (tokenizer, "tokenizer.json")):
        entries.append((source, dest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    with tarfile.open(args.output, "w:gz", dereference=True) as archive:
        for source, dest in entries:
            archive.add(source, arcname=dest)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest = {"package": str(args.output), "sha256": digest,
                "files": [{"path": dest, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
                          for source, dest in entries],
                "max_concurrent_jobs": 2, "max_total_gpus": 4}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"package": str(args.output), "sha256": digest,
                      "file_count": len(entries), "bytes": args.output.stat().st_size}))


if __name__ == "__main__":
    main()

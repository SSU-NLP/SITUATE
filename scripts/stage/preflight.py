"""Audit released Task I assets and runtime screenplay readiness without GPU calls."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(root: Path, scripts: Path | None = None) -> dict:
    """Check upstream checksums, paired assets, language and exact screenplay bytes."""
    release = json.loads((root / "manifest.json").read_text())
    mismatches = []
    for item in release["artifacts"]["files"]:
        path = root / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            mismatches.append(item["path"])
    sys.path.insert(0, str(root / "evaluation/src"))
    spec = importlib.util.spec_from_file_location(
        "stage_asset_builder", root / "evaluation/scripts/build_task1_state_update_assets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    counts, languages, missing, drift, errors = Counter(), Counter(), [], [], []
    for condition in ("STAGE", "STAGE_Anon"):
        for path in sorted((root / condition).glob("*/*/task_1_reference_state_update.json")):
            ref = json.loads(path.read_text())
            ar = json.loads(path.with_name("task_1_autoregressive_state_update.json").read_text())
            result = module.validate_assets(reference=ref, autoregressive=ar)
            if result["status"] != "passed":
                errors.append({"asset": str(path.relative_to(root)), "errors": result["errors"]})
            languages[ref["language"]] += 1
            counts[f"{condition}_movies"] += 1
            for field in ("trajectories", "checkpoints"):
                counts[f"{condition}_{field}"] += ref["counts"][field]
            relative = path.parent.relative_to(root) / "script.json"
            script = (scripts or root) / relative
            if not script.is_file():
                missing.append(str(relative))
            elif sha256(script) != ref["script"]["sha256"]:
                drift.append(str(relative))
    language_errors = sorted(set(languages) - {"en", "zh"})
    asset_ok = not mismatches and not errors
    return {
        "status": "ready" if asset_ok and not missing and not drift and not language_errors
        else "blocked",
        "upstream_commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "counts": dict(counts), "languages": dict(languages),
        "release_files_checked": len(release["artifacts"]["files"]),
        "release_checksum_mismatches": mismatches, "paired_asset_errors": errors,
        "runner_unsupported_language_values": language_errors,
        "missing_scripts": missing, "script_hash_mismatches": drift,
        "gpu_jobs_submitted": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--scripts-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.release_root.resolve(), args.scripts_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in
                      {"missing_scripts", "paired_asset_errors", "release_checksum_mismatches"}},
                     ensure_ascii=False))
    print(f"Missing screenplays: {len(result['missing_scripts'])}; "
          f"asset errors: {len(result['paired_asset_errors'])}; "
          f"checksum mismatches: {len(result['release_checksum_mismatches'])}")
    raise SystemExit(0 if result["status"] == "ready" else 2)


if __name__ == "__main__":
    main()

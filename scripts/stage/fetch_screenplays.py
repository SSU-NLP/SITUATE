"""Fetch public legacy screenplay bytes and compare with current Task I hashes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repository", default="zhuiwen-ai/STAGE")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    repo = args.repository
    with urllib.request.urlopen(
            f"https://api.github.com/repos/{repo}/git/trees/{args.revision}?recursive=1", timeout=60) as response:
        tree = json.load(response)
    if tree.get("truncated"):
        raise ValueError("Incomplete upstream tree")
    revision = tree["sha"]
    paths = [item["path"] for item in tree["tree"] if item["path"].endswith("/script.json")]

    def fetch(relative: str) -> dict:
        url = f"https://raw.githubusercontent.com/{repo}/{revision}/{relative}"
        output = args.output_root / "STAGE" / relative
        try:
            if output.exists():
                raw = output.read_bytes()
            else:
                with urllib.request.urlopen(url, timeout=60) as response:
                    raw = response.read()
                json.loads(raw)
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("xb") as stream:
                    stream.write(raw)
            expected_path = args.release_root / "STAGE" / Path(relative).parent / "task_1_reference_state_update.json"
            expected = json.loads(expected_path.read_text())["script"]
            digest = hashlib.sha256(raw).hexdigest()
            return {"path": str(output), "url": url, "sha256": digest,
                    "expected_sha256": expected["sha256"], "exact_match": digest == expected["sha256"],
                    "scenes": len(json.loads(raw)), "expected_scenes": expected["scene_count"]}
        except Exception as exc:
            return {"path": str(output), "error": f"{type(exc).__name__}: {exc}"}

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(fetch, paths))
    report = {"repository": repo, "revision": revision, "files": rows,
              "exact_matches": sum(row.get("exact_match", False) for row in rows),
              "errors": sum("error" in row for row in rows)}
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "acquisition.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "files"}))


if __name__ == "__main__":
    main()

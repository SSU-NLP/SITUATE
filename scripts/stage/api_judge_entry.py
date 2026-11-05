"""Run official metrics with an explicitly substituted GPT-OSS API judge."""
import asyncio
import json
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys

from api_client import build_api_client


def main():
    evaluation = Path(sys.argv[1]).resolve()
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    sys.path.insert(0, str(evaluation / "src"))
    spec = importlib.util.spec_from_file_location(
        "stage_judge", evaluation / "scripts/run_task1_state_update_evaluation.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.build_json_client = build_api_client
    original = runner._evidence_excerpts
    cache = {}
    def excerpts(**kwargs):
        key = json.dumps([kwargs["reference"]["checkpoint_id"], kwargs["row"],
                          kwargs["prediction"].get("previous_state_evidence_scene_orders", [])], sort_keys=True)
        if key not in cache:
            cache[key] = original(**kwargs)
        return cache[key]
    runner._evidence_excerpts = excerpts
    # GPT-OSS uses reasoning. Preserve its billed tokens in all output metadata;
    # this is a declared judge substitution, not the paper's non-reasoning protocol.
    def check_judge(metadata):
        if metadata.get("requested_model") != "openrouter/openai/gpt-oss-120b":
            raise ValueError("Unexpected evaluation model")
    runner._assert_no_reasoning = check_judge
    asyncio.run(runner.main_async())


if __name__ == "__main__":
    main()

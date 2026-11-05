"""Freeze the ordinary STAGE internal-comparison actor and judge configuration."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.template.read_text())
    config["track_id"] = "situate-stage-ordinary-frozen-lora"
    for field, model in (("prediction_llm", "Qwen/Qwen3-8B"),
                         ("evaluation_llm", "Qwen/Qwen3-32B")):
        config[field] = {"model": model, "base_url": "http://127.0.0.1:8001/v1",
                         "api_key": "local", "temperature": 0, "max_attempts": 3,
                         "semantic_attempts": 3, "json_response_format": True,
                         "timeout_seconds": 600,
                         "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    config["evaluation_llm"]["fallback_llm"] = deepcopy(config["evaluation_llm"])
    config["tokenizer"] = {"backend": "tokenizer_json", "path": str(args.tokenizer.resolve())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()

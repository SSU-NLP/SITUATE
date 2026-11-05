"""Zero-call integration checks for input isolation and official prediction contracts."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from run_tracking import METHODS, load_runner, renderer
from state_edits import apply_edits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    r = load_runner(args.evaluation.resolve())
    config = r.BenchmarkRuntimeConfig.load(args.config.resolve())
    counter = config.build_token_counter()
    from stage_narrative.evaluation.task1_schemas import validate_task1_prediction
    checked = []
    for language in ("en", "zh"):
        movies = json.loads((args.inputs / "manifest.json").read_text())["movies"]
        movie = next(row for row in movies if row["language"] == language)
        ref = r._object(args.inputs / movie["movie_id"] / "task_1_reference_state_update.json")
        trajectory = ref["trajectories"][0]
        checkpoint = trajectory["checkpoints"][1]
        prior = r._previous_state_content(checkpoint["model_input"]["previous_state"])
        scene = checkpoint["screenplay_interval"]["end_scene_order_inclusive"]
        memory = [{"claim": "The character sees new evidence.", "evidence_scene_orders": [scene]}]
        scores = {"axes": ["protagonist", "space", "time", "causality", "intentionality"],
                  "scenes": {str(scene): {"mean": [.1, .2, .3, .4, .5]},
                             str(scene + 1): {"mean": [.8] * 5}}}
        kwargs = dict(trajectory=trajectory, checkpoint=checkpoint, previous_state=prior,
                      memory=memory, language=language, config=config, counter=counter)
        for method in METHODS:
            render = renderer(r, r._render_prediction, method, scores)
            prompt = render(**kwargs)
            mutated = deepcopy(kwargs)
            mutated["checkpoint"]["evaluation_target"] = {"forbidden": "GOLD_SECRET_SENTINEL"}
            if render(**mutated) != prompt:
                raise AssertionError("Evaluator target influenced actor prompt")
            future = deepcopy(scores)
            future["scenes"][str(scene + 1)]["mean"] = [99] * 5
            if renderer(r, r._render_prediction, method, future)(**kwargs) != prompt:
                raise AssertionError("Future scene scores influenced actor prompt")
            if r.INTERNAL_ID.search(prompt["user_prompt"]):
                raise AssertionError("Private identifiers leaked to actor")
            checked.append(f"{language}/{method}/gold_isolation+future_isolation+budget")
        output = apply_edits(prior, {"remove": [], "replace": [], "add": [],
                                   "developments_since_previous_checkpoint": []}, {scene})
        validate_task1_prediction(output)
        if [row["claim"] for row in output["current_state"]] != prior:
            raise AssertionError("Untouched prior claims were lost")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"status": "passed_zero_call", "checks": checked}, indent=2) + "\n")
    print(f"PASS {len(checked)} bilingual integration checks")


if __name__ == "__main__":
    main()

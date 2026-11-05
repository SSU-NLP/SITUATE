"""Matched state-update interventions around the official STAGE Task I runner."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import re
import sys

from state_edits import EDIT_INSTRUCTIONS, EDIT_INSTRUCTIONS_ZH, apply_edits

METHODS = ("rewrite", "rewrite_axes")


def load_runner(evaluation: Path):
    sys.path.insert(0, str(evaluation / "src"))
    sys.path.insert(0, str(evaluation))
    spec = importlib.util.spec_from_file_location(
        "stage_prediction_runner", evaluation / "scripts/run_task1_state_update_predictions.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def tagged_json(user: str, tag: str):
    match = re.search(rf"<{tag}>(.*?)</{tag}>", user, re.S)
    if not match:
        raise ValueError(f"Missing prompt field: {tag}")
    return json.loads(match.group(1))


class CheckedClient:
    """Use the same bounded format retries for every state-update method."""

    def __init__(self, client, runner, cache: Path):
        self.client, self.runner, self.cache = client, runner, cache
        cache.mkdir(parents=True, exist_ok=True)

    async def generate_json(self, *, system_prompt: str, user_prompt: str, stage: str):
        from stage_narrative.clients import JsonCall, ModelResponseParseError
        from stage_narrative.evaluation.task1_schemas import validate_task1_prediction
        key = self.runner.sha256_json({"system": system_prompt, "user": user_prompt})
        path = self.cache / f"{key}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            return JsonCall(data=saved["data"], metadata=saved["metadata"])
        previous = tagged_json(user_prompt, "previous_state")
        memory = tagged_json(user_prompt, "entity_centric_memory")
        allowed = {int(n) for row in memory for n in row["evidence_scene_orders"]}
        error = None
        last_payload = None
        for attempt in range(3):
            correction = "" if error is None else (
                "\nYour previous response violated the output contract: " + str(error)[:250]
                + ". Allowed evidence scene numbers: " + json.dumps(sorted(allowed))
                + ". Each row needs exactly claim and evidence_scene_orders. "
                "Do not repeat claims within a list. Developments require nonempty evidence "
                "from these allowed scenes; if the observations support no development, "
                "return an empty developments list. A retained prior state may have empty "
                "evidence. Keep descriptions concise and non-graphic. Return complete JSON. "
                "Invalid output excerpt: " + json.dumps(last_payload, ensure_ascii=False)[:500])
            try:
                response = await self.client.generate_json(
                    system_prompt=system_prompt, user_prompt=user_prompt + correction, stage=stage)
                last_payload = response.data
                self.runner._assert_no_reasoning(response.metadata)
                patch = response.data if "<state_edit_contract>" in user_prompt else None
                result = apply_edits(previous, patch, allowed) if patch is not None else response.data
                validate_task1_prediction(result)
                for field, rows in result.items():
                    for row in rows:
                        if not set(row["evidence_scene_orders"]) <= allowed:
                            raise ValueError("Citation outside interval memory")
                        if field == "developments_since_previous_checkpoint" and not row["evidence_scene_orders"]:
                            raise ValueError("Development lacks interval evidence")
                metadata = {**response.metadata, "format_attempts": attempt + 1,
                            "raw_edit_operations": patch}
                final = replace(response, data=result, metadata=metadata)
                self.runner.atomic_write_json(path, {"data": result, "metadata": metadata})
                return final
            except (ValueError, ModelResponseParseError) as exc:
                error = exc
        raise ValueError(f"State update failed after three format attempts: {stage}: {error}")


class MemoryCheckedClient:
    def __init__(self, client, runner):
        self.client, self.runner = client, runner

    async def generate_json(self, *, system_prompt, user_prompt, stage):
        if not stage.startswith("task1_entity_memory:"):
            return await self.client.generate_json(
                system_prompt=system_prompt, user_prompt=user_prompt, stage=stage)
        from stage_narrative.clients import ModelResponseParseError
        allowed = set(tagged_json(user_prompt, "allowed_scene_orders"))
        rejected = []
        last_payload = None
        for attempt in range(3):
            correction = "" if not rejected else (
                "\nRegenerate valid JSON. Error: " + rejected[-1] +
                ". Every evidence_scene_orders entry must come from " + str(sorted(allowed)) +
                ". Each observation must have exactly two keys: claim (string) and "
                "evidence_scene_orders (nonempty integer array). No other keys. "
                "Use concise, non-graphic descriptions of character circumstances. "
                "Finish all strings and JSON brackets. Invalid output excerpt: " +
                json.dumps(last_payload, ensure_ascii=False)[:500])
            try:
                response = await self.client.generate_json(
                    system_prompt=system_prompt, user_prompt=user_prompt + correction, stage=stage)
                last_payload = response.data
                self.runner._assert_no_reasoning(response.metadata)
                self.runner._validate_observations(response.data, allowed_scene_orders=allowed)
                return replace(response, metadata={**response.metadata,
                    "format_attempts": attempt + 1, "rejected_contracts": rejected})
            except (ValueError, ModelResponseParseError) as exc:
                rejected.append(str(exc)[:250])
        raise ValueError(f"Memory extraction failed contract: {stage}: {rejected}")


def renderer(runner, original, method: str, scores: dict):
    if method not in METHODS:
        raise ValueError(method)
    def render(**kwargs):
        base = original(**kwargs)
        if method == "rewrite" or not kwargs["previous_state"]:
            return base
        scenes = sorted({n for row in kwargs["memory"] for n in row["evidence_scene_orders"]})
        cues = []
        for scene in scenes:
            values = scores["scenes"][str(scene)]["mean"]
            if values is not None:
                cues.append({"scene": scene, "change_scores": {
                    axis: round(value, 4) for axis, value in zip(scores["axes"], values)}})
        explanation = (
            "Auxiliary narrative transition scores follow. These are uncalibrated change "
            "signals for protagonist, space, time, causality and intentionality. Use them "
            "to focus review of relevant claims. They are not evidence of character "
            "knowledge and do not prove any state changed. Low scores never override "
            "explicit textual evidence. Keep the original output schema.")
        if kwargs["language"] == "zh":
            explanation = (
                "以下是辅助叙事情境变化分数，对应 protagonist（人物）、space（空间）、"
                "time（时间）、causality（因果）、intentionality（意图）。分数未经校准，"
                "仅帮助确定哪些断言需要检查；不能证明角色知道什么，也不能证明状态已经改变。"
                "低分不能推翻明确的文字证据。保持原始输出格式。")
        user = base["user_prompt"] + "\n<auxiliary_change_scores>\n" + explanation + "\n" + json.dumps(cues) + "\n</auxiliary_change_scores>"
        return runner._budgeted(base["system_prompt"], user,
                                config=kwargs["config"], counter=kwargs["counter"])
    return render


async def run(args) -> None:
    runner = load_runner(args.evaluation.resolve())
    config = runner.BenchmarkRuntimeConfig.load(args.config.resolve())
    counter = config.build_token_counter()
    counter.count = lru_cache(maxsize=10000)(counter.count)
    movies = json.loads((args.inputs / "manifest.json").read_text())["movies"]
    movies = movies[args.shard::args.num_shards]
    if args.movie_id:
        movies = [m for m in movies if m["movie_id"] == args.movie_id]
        if not movies:
            raise ValueError("Requested movie not in selected shard")
    if args.limit:
        movies = movies[:args.limit]
    original = runner._render_prediction
    semaphore = asyncio.Semaphore(args.workers)
    def prepared(movie, ref, ar):
        directory = args.inputs / movie["movie_id"]
        key = runner.sha256_json({
            "reference": runner.sha256_file(directory / "task_1_reference_state_update.json"),
            "autoregressive": runner.sha256_file(directory / "task_1_autoregressive_state_update.json"),
            "script": runner.sha256_file(directory / "script.json"),
            "config": runner.sha256_file(args.config),
            "runner": runner.sha256_file(args.evaluation / "scripts/run_task1_state_update_predictions.py")})
        path = args.output / "prepared_intervals" / f"{movie['movie_id']}-{key}.json"
        if path.exists():
            value = runner._object(path)
            return value["jobs"], value["audit"]
        jobs = runner._memory_jobs(ref, scenes=runner.load_scenes(directory / "script.json"),
                                   counter=counter, chunk_content_tokens=600)
        item = runner._preflight(reference_asset=ref, autoregressive_asset=ar, memory_jobs=jobs,
                                 config=config, counter=counter)
        runner.atomic_write_json(path, {"jobs": jobs, "audit": item})
        counter.count.cache_clear()
        return jobs, item
    # Check all screenplay and paired-asset contracts before any semantic call.
    audit = []
    for movie in movies:
        directory = args.inputs / movie["movie_id"]
        ref = runner._object(directory / "task_1_reference_state_update.json")
        ar = runner._object(directory / "task_1_autoregressive_state_update.json")
        if runner.validate_assets(reference=ref, autoregressive=ar)["status"] != "passed":
            raise ValueError("Invalid paired assets")
        if runner.sha256_file(directory / "script.json") != ref["script"]["sha256"]:
            raise ValueError("Screenplay hash mismatch")
        jobs, item = prepared(movie, ref, ar)
        audit.append({"movie_id": movie["movie_id"], **item})
        print(f"PREFLIGHT {movie['movie_id']} {len(audit)}/{len(movies)}", flush=True)
    if args.preflight_only:
        args.output.mkdir(parents=True, exist_ok=True)
        runner.atomic_write_json(args.output / f"preflight_shard{args.shard}.json", audit)
        print(f"PREFLIGHT {len(movies)} movies passed", flush=True)
        return
    from api_client import build_api_client
    raw = build_api_client({**config.prediction_llm, "json_response_format": True,
                                   "max_tokens": config.call_budgets["task1_prediction"].max_output_tokens})
    for movie in movies:
        movie_id = movie["movie_id"]
        directory = args.inputs / movie_id
        output = args.output / movie_id
        output.mkdir(parents=True, exist_ok=True)
        ref = runner._object(directory / "task_1_reference_state_update.json")
        ar = runner._object(directory / "task_1_autoregressive_state_update.json")
        scores = runner._object(args.axes / f"{movie_id}.json")
        if scores["script_sha256"] != ref["script"]["sha256"]:
            raise ValueError("Axis input hash mismatch")
        jobs, _ = prepared(movie, ref, ar)
        memory_path = output / "shared_entity_memory.json"
        if memory_path.exists():
            memory = runner._object(memory_path)
        else:
            memory = await runner._extract_shared_memory(
                jobs=jobs, client=MemoryCheckedClient(raw, runner), semaphore=semaphore, config=config, counter=counter,
                output_root=output / "memory", resume=True)
            runner.atomic_write_json(memory_path, memory)
        checked = CheckedClient(raw, runner, output / "validated_calls")
        for method in METHODS:
            run_dir = output / method
            run_dir.mkdir(parents=True, exist_ok=True)
            runner._render_prediction = renderer(runner, original, method, scores)
            # Materialize all method prompts before entering the costly rollout.
            for trajectory in ref["trajectories"]:
                by_id = {row["checkpoint_id"]: row for row in memory["slots"]}
                for checkpoint in trajectory["checkpoints"]:
                    runner._render_prediction(
                        trajectory=trajectory, checkpoint=checkpoint,
                        previous_state=runner._previous_state_content(checkpoint["model_input"]["previous_state"]),
                        memory=by_id[checkpoint["checkpoint_id"]]["observations"],
                        language=ref["language"], config=config, counter=counter)
            reference = await runner._run_setting(
                asset=ref, memory_pack=memory, client=checked, semaphore=semaphore,
                config=config, counter=counter, output_root=run_dir / "reference_state_update", resume=True)
            first = {t["checkpoints"][0]["checkpoint_id"]: t["checkpoints"][0]
                     for t in reference["trajectories"]}
            autoregressive = await runner._run_setting(
                asset=ar, memory_pack=memory, client=checked, semaphore=semaphore,
                config=config, counter=counter, output_root=run_dir / "autoregressive_state_update",
                resume=True, seed_predictions=first)
            runner.atomic_write_json(run_dir / "reference_state_update_predictions.json", reference)
            runner.atomic_write_json(run_dir / "autoregressive_state_update_predictions.json", autoregressive)
            runner.atomic_write_json(run_dir / "shared_entity_memory.json", memory)
            runner.atomic_write_json(run_dir / "manifest.json", {
                "method": method, "prediction_model": config.prediction_llm["model"],
                "previous_state_projection": "claim_text_only", "status": "completed",
                "shared_memory_sha256": runner.sha256_file(memory_path),
                "config_sha256": runner.sha256_file(args.config),
                "axis_scores_sha256": runner.sha256_file(args.axes / f"{movie_id}.json"),
                "format_attempts_maximum": 3})
            print(f"PREDICTED {movie_id} {method}", flush=True)
        runner._render_prediction = original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("evaluation", "inputs", "config", "axes", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--movie-id")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

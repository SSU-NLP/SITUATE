"""Compute frozen, causal SITUATE scores on screenplay sentences."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import torch
import torch.nn.functional as F

from axis_predictor import AXES, AxisPredictor, load_head, validate_adapter
from preflight import sha256


def sentences(scene: dict, language: str) -> list[str]:
    text = str(scene.get("content") or "").strip()
    headings = [str(scene.get(k) or "").strip() for k in ("title", "subtitle")]
    if language == "en":
        import nltk
        content = nltk.sent_tokenize(text) if text else []
    else:
        content = re.split(r"(?<=[。！？])|\n\s*\n", text)
    return [s for s in headings + content if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=2)
    args = parser.parse_args()
    from transformers import AutoModel, AutoTokenizer
    from peft import PeftModel
    head, manifest = load_head(args.artifacts / "M5_lora_ep001_seed1.pt")
    adapter = args.artifacts / "adapter_ep001"
    validate_adapter(adapter, manifest)
    predictor = AxisPredictor(head).cuda()
    tokenizer = AutoTokenizer.from_pretrained(adapter, padding_side="left")
    base = AutoModel.from_pretrained(manifest["model"], revision=manifest["revision"],
                                    torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    encoder = PeftModel.from_pretrained(base, adapter).cuda().eval()
    rows = json.loads((args.inputs / "manifest.json").read_text())["movies"]
    rows = rows[args.shard::args.num_shards]
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for movie in rows:
            destination = args.output / f"{movie['movie_id']}.json"
            if destination.exists():
                raise FileExistsError(destination)
            source = args.inputs / movie["movie_id"] / "script.json"
            if sha256(source) != movie["script_sha256"]:
                raise ValueError("Screenplay hash drift")
            scenes = json.loads(source.read_text())
            recurrent = None
            scores = {}
            for order, scene in enumerate(scenes, 1):
                texts = sentences(scene, movie["language"])
                if not texts:
                    scores[str(order)] = {"mean": None, "transitions": 0}
                    continue
                tokens = tokenizer(texts, padding=False, truncation=False)["input_ids"]
                batches, batch = [], []
                for i in sorted(range(len(tokens)), key=lambda k: len(tokens[k])):
                    n = len(tokens[i])
                    if n > encoder.config.max_position_embeddings:
                        raise ValueError(f"Sentence exceeds encoder context: {n}")
                    if batch and (len(batch) >= 128 or (len(batch) + 1) * n > 8192):
                        batches.append(batch)
                        batch = []
                    batch.append(i)
                if batch:
                    batches.append(batch)
                z = torch.empty(len(texts), 2048, device="cuda")
                for indices in batches:
                    inputs = tokenizer.pad({"input_ids": [tokens[i] for i in indices]},
                                           return_tensors="pt").to("cuda")
                    if not inputs.attention_mask[:, -1].all():
                        raise ValueError("Expected left padding")
                    hidden = encoder(**inputs, use_cache=False).last_hidden_state[:, -1].float()
                    z[indices] = F.normalize(hidden, dim=-1)
                probabilities, recurrent = predictor(z, recurrent)
                scores[str(order)] = {"mean": probabilities.mean(0).cpu().tolist()
                                      if len(probabilities) else None,
                                      "transitions": len(probabilities)}
            result = {"movie_id": movie["movie_id"], "axes": AXES, "scenes": scores,
                      "script_sha256": movie["script_sha256"], "model_revision": manifest["revision"],
                      "checkpoint_sha256": sha256(args.artifacts / "M5_lora_ep001_seed1.pt"),
                      "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
                      "aggregation": "scene mean of causal sentence transition scores; no threshold"}
            destination.write_text(json.dumps(result, indent=2) + "\n")
            print(f"AXES {movie['movie_id']} {len(scenes)} scenes", flush=True)


if __name__ == "__main__":
    main()

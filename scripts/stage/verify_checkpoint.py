"""Check the actual LoRA artifact and extracted recurrent head on CPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from axis_predictor import AXES, AxisPredictor, load_head, validate_adapter
from preflight import sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    checkpoint = args.artifacts / "M5_lora_ep001_seed1.pt"
    adapter = args.artifacts / "adapter_ep001"
    head, manifest = load_head(checkpoint)
    validate_adapter(adapter, manifest)
    predictor = AxisPredictor(head)
    torch.manual_seed(17)
    z = F.normalize(torch.randn(17, 2048), dim=-1)
    labels = {name: torch.zeros(len(z), dtype=torch.long)
              for name in ("perspective", "temporal_shift_marker", "event_type")}
    with torch.no_grad():
        original = head(z, torch.zeros(len(z) - 1, 5), labels)
        expected = torch.stack([original["change_logits"][axis].sigmoid() for axis in AXES], -1)
        actual, state = predictor(z)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        first, carried = predictor(z[:5])
        rest, carried = predictor(z[5:], carried)
        chunked = torch.cat([first, rest])
        torch.testing.assert_close(chunked, expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(carried, state, atol=1e-6, rtol=1e-5)
        z[5:] = torch.randn_like(z[5:])
        torch.testing.assert_close(predictor(z)[0][:4], first, atol=1e-6, rtol=1e-5)
    output = {
        "status": "passed_cpu_head_equivalence",
        "checkpoint_sha256": sha256(checkpoint),
        "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
        "adapter_config_sha256": sha256(adapter / "adapter_config.json"),
        "source_sha256": sha256(Path(__file__).resolve().parents[2] / "src/hi_ontop/factored_slot.py"),
        "training_manifest": manifest, "axes": AXES,
        "head_parameters": sum(p.numel() for p in head.parameters()),
        "extracted_parameters": sum(p.numel() for p in predictor.parameters()),
        "max_absolute_error": (actual - expected).abs().max().item(),
        "chunk_max_absolute_error": (chunked - expected).abs().max().item(),
        "future_invariance": True,
        "scope": "Synthetic normalized embeddings; Qwen encoder and STAGE task not evaluated",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "training_manifest"}))


if __name__ == "__main__":
    main()

"""Frozen SITUATE change inference, without the event-boundary decision rule."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from hi_ontop.factored_slot import AXES, FactoredSlot


class AxisPredictor(nn.Module):
    """Keep the trained extractors, recurrence and change heads only.

    Inputs are the original normalized sentence embeddings in narrative order.
    The first sentence has no predecessor and therefore no change prediction.
    Carry state between chunks; reset only at the beginning of a new sequence.
    These axes describe narrative changes, not character-specific knowledge.
    """

    def __init__(self, head: FactoredSlot):
        super().__init__()
        if (head.mono or not head.d1 or head.tagger is not None
                or head.head_feat != "delta" or head.psi is not None
                or head.slot_reset is not None or head.block_readout):
            raise ValueError("Unsupported checkpoint architecture")
        self.extractors = head.extractors
        self.keys = head.keys
        self.gate_in = head.gate_in
        self.A, self.B, self.C, self.phi = head.A, head.B, head.C, head.phi
        self.change_heads = head.change_heads
        self.z_dim = head.extractors[0][0].in_features
        self.requires_grad_(False)
        self.eval()

    @torch.inference_mode()
    def forward(self, z: torch.Tensor, state: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return probabilities [transitions, 5] and the last recurrent state."""
        if self.training:
            raise RuntimeError("AxisPredictor requires eval mode")
        if z.ndim != 2 or z.shape[1] != self.z_dim:
            raise ValueError(f"Expected [sentences, {self.z_dim}] embeddings")
        if not torch.isfinite(z).all():
            raise ValueError("Non-finite sentence embeddings")
        if state is not None and (state.shape != self.keys.shape
                                  or not torch.isfinite(state).all()):
            raise ValueError("Invalid recurrent state")
        if not len(z):
            return z.new_empty((0, len(AXES))), state
        h = self.keys.clone() if state is None else state
        previous_exists = state is not None
        inputs = torch.stack([extractor(z) for extractor in self.extractors], dim=-2)
        probabilities = []
        for u in inputs:
            old = h
            gi = self.gate_in(u)
            gate = torch.sigmoid((gi * h).sum(-1) + (gi * self.keys).sum(-1))
            candidate = self.phi(self.A(h) + self.B(self.keys) + self.C(u))
            h = h + gate.unsqueeze(-1) * candidate
            h = h / (h.norm(dim=-1, keepdim=True) + 1e-8)
            if previous_exists:
                features = torch.cat([old, h, h - old, h * old], dim=-1)
                probabilities.append(torch.stack([
                    torch.sigmoid(layer(features[k]).squeeze(-1))
                    for k, layer in enumerate(self.change_heads)
                ]))
            previous_exists = True
        result = torch.stack(probabilities) if probabilities else z.new_empty((0, len(AXES)))
        return result, h


def load_head(checkpoint: Path) -> tuple[FactoredSlot, dict]:
    """Load the complete trained head strictly before extracting inference parts."""
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
    manifest = bundle["manifest"]
    if manifest["model"] != "Qwen/Qwen3-1.7B" or not manifest.get("revision"):
        raise ValueError("Expected the pinned Qwen3-1.7B LoRA experiment")
    head = FactoredSlot(z_dim=2048, **bundle["args"], mono=bundle["mono"],
                        dynamics=bundle["dynamics"], pos_weight=bundle["pos_weight"])
    head.load_state_dict(bundle["state"], strict=True)
    return head.eval(), manifest


def validate_adapter(adapter: Path, manifest: dict) -> None:
    """Reject a mismatched LoRA configuration instead of using a frozen base."""
    config = json.loads((adapter / "adapter_config.json").read_text())
    expected = {"base_model_name_or_path": manifest["model"],
                "r": manifest["lora_rank"], "lora_alpha": manifest["lora_alpha"],
                "lora_dropout": manifest["lora_dropout"], "task_type": "FEATURE_EXTRACTION"}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Adapter mismatch: {key}")
    if set(config["target_modules"]) != set(manifest["lora_targets"]):
        raise ValueError("Adapter target modules differ")
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(adapter / "adapter_model.safetensors")

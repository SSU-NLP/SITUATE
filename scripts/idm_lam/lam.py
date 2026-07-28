"""LAM = IDM + VQ bottleneck + FDM, minimized for dialogue (per gonogo_expreiment.md §2).

Design levers (each gates Gate-1 success):
  * DIFFERENCE INPUT: IDM sees (z_t, z_{t+1}-z_t) so it encodes the *change*, not content.
  * NARROW BOTTLENECK: information capacity of c_t = log2(K) bits. K small (8/16) makes it
    structurally impossible for c_t to smuggle 768-D content -> anti-leakage is enforced by
    architecture, not hope. code_dim is only the per-code embedding width (does NOT add capacity).
  * PREPEND not ADD: FDM gets concat(z_t, code_vec), never z_t + code_vec (Jasmine/LAPO tip).
  * DEAD-CODE handling: kmeans_init + EMA decay + threshold_ema_dead_code.

Loss = ||FDM(z_t, VQ(IDM(z_t, dz))) - z_{t+1}||^2 + commitment.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize


def mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class LAM(nn.Module):
    def __init__(
        self,
        z_dim: int = 768,
        code_dim: int = 32,
        codebook_size: int = 16,
        commitment_weight: float = 0.25,
        decay: float = 0.99,
        idm_hidden: tuple[int, ...] = (512, 256),
        fdm_hidden: tuple[int, ...] = (512,),
    ):
        super().__init__()
        self.z_dim = z_dim
        self.code_dim = code_dim
        self.codebook_size = codebook_size
        # IDM: (z_t, dz) -> a_t (code_dim)
        self.idm = mlp([2 * z_dim, *idm_hidden, code_dim])
        # VQ bottleneck
        self.vq = VectorQuantize(
            dim=code_dim,
            codebook_size=codebook_size,
            decay=decay,
            commitment_weight=commitment_weight,
            kmeans_init=True,
            kmeans_iters=10,
            threshold_ema_dead_code=2,
        )
        # FDM: prepend(concat) z_t and code_vec -> z_{t+1}
        self.fdm = mlp([z_dim + code_dim, *fdm_hidden, z_dim])

    def encode_transition(self, z_t: torch.Tensor, z_tp1: torch.Tensor):
        """Return (a_t continuous, quantized code_vec, code index, commit_loss)."""
        dz = z_tp1 - z_t
        a = self.idm(torch.cat([z_t, dz], dim=-1))  # (B, code_dim)
        q, idx, commit = self.vq(a.unsqueeze(1))  # VQ wants (B, n, d)
        return a, q.squeeze(1), idx.squeeze(1), commit.mean()

    def forward(self, z_t: torch.Tensor, z_tp1: torch.Tensor):
        a, q, idx, commit = self.encode_transition(z_t, z_tp1)
        zhat = self.fdm(torch.cat([z_t, q], dim=-1))
        recon = ((zhat - z_tp1) ** 2).sum(-1).mean()
        return {
            "recon": recon,
            "commit": commit,
            "loss": recon + commit,
            "idx": idx,
            "a": a,
            "zhat": zhat,
        }

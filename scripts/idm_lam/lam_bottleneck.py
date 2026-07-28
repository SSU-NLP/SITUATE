"""LAM with PLUGGABLE bottleneck (per revised design): continuous+sparsity (1st), continuous+noise
(2nd, VIB/VAE), VQ (control). IDM(z_t, z_{t+1}-z_t) -> a_t -> bottleneck -> ā_t -> FDM(z_t, ā_t).

- sparse : ā_t = a_t (continuous), loss += lam * mean(|a_t|)  [VCM-style: few dims active]
- noise  : IDM outputs (mu, logvar); ā_t = mu + sigma*eps (train), mu (eval); loss += beta*KL  [VIB]
- vq     : ā_t = nearest of K codebook vectors; loss += commitment  [discrete, control]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize


def mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class LAMBottleneck(nn.Module):
    def __init__(self, z_dim=768, latent_dim=64, bottleneck="sparse",
                 lam=0.1, beta=0.1, codebook_size=16, code_dim=32):
        super().__init__()
        self.z_dim = z_dim
        self.bottleneck = bottleneck
        self.lam = lam
        self.beta = beta
        out = 2 * latent_dim if bottleneck == "noise" else (code_dim if bottleneck == "vq" else latent_dim)
        self.idm = mlp([2 * z_dim, 512, 256, out])
        self.latent_dim = latent_dim
        if bottleneck == "vq":
            self.vq = VectorQuantize(dim=code_dim, codebook_size=codebook_size, decay=0.99,
                                     commitment_weight=0.25, kmeans_init=True, kmeans_iters=10,
                                     threshold_ema_dead_code=2)
            fdm_in = z_dim + code_dim
        else:
            fdm_in = z_dim + latent_dim
        self.fdm = mlp([fdm_in, 512, z_dim])

    def encode(self, z_t, z_tp1):
        """Return (abar, reg_loss). abar = constrained transition representation (for FDM & diagnostics)."""
        a = self.idm(torch.cat([z_t, z_tp1 - z_t], -1))
        if self.bottleneck == "sparse":
            abar = a
            reg = self.lam * a.abs().mean()
        elif self.bottleneck == "noise":
            mu, logvar = a.chunk(2, -1)
            logvar = torch.clamp(logvar, -8, 4)
            if self.training:
                abar = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            else:
                abar = mu
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
            reg = self.beta * kl
        elif self.bottleneck == "vq":
            q, idx, commit = self.vq(a.unsqueeze(1))
            abar = q.squeeze(1)
            reg = commit.mean()
            self._last_idx = idx.squeeze(1)
        else:
            raise ValueError(self.bottleneck)
        return abar, reg

    def forward(self, z_t, z_tp1):
        abar, reg = self.encode(z_t, z_tp1)
        zhat = self.fdm(torch.cat([z_t, abar], -1))
        recon = ((zhat - z_tp1) ** 2).sum(-1).mean()
        return {"recon": recon, "reg": reg, "loss": recon + reg, "abar": abar}

    @torch.no_grad()
    def repr_seq(self, emb, device):
        """Return ā_t sequence (n-1, latent/code dim) for one dialogue (eval mode: mu / code / a)."""
        z = torch.tensor(emb, device=device)
        abar, _ = self.encode(z[:-1], z[1:])
        return abar.cpu().numpy()
